"""阶段 1 梯度冲突审计的单元测试（stdlib unittest；不依赖真实数据、不跑真实审计）。

覆盖：损失分解、无副作用（optimizer/grad/RNG）、逐位保真、余弦计算、参数分组、
采样节奏、零梯度计数、门禁判定、输出确定性。
"""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import gradient_audit as GA
from census_benchmark import protocol as P
import run_census_gradient_audit
from run_census_benchmark import build_mptrec
from multitaskrec.model import MPTRec
from multitaskrec.train import MPTRecTrainManager

REPO = Path(__file__).resolve().parents[2]

# 与 test_protocol.tiny_mptrec 同口径：2 个特征 × embedding_size 4 = dnn_input 维度 8
TINY_VOCAB, TINY_INPUT_SIZE = {"a": 3, "b": 2}, 8


class TinyCensus(Dataset):
    """极小 CensusIncome 形状：(income, marital, new_task, features)。"""

    def __init__(self, n, seed):
        gen = torch.Generator().manual_seed(seed)
        self.features = {"a": torch.randint(0, 3, (n,), generator=gen),
                         "b": torch.randint(0, 2, (n,), generator=gen)}
        self.labels = [torch.randint(0, 2, (n,), generator=gen).float() for _ in range(3)]

    def __len__(self):
        return int(self.features["a"].shape[0])

    def __getitem__(self, i):
        return (self.labels[0][i], self.labels[1][i], self.labels[2][i],
                {"a": self.features["a"][i], "b": self.features["b"][i]})


def tiny_model():
    return MPTRec(num_tasks=2, feature_vocabulary=dict(TINY_VOCAB), embedding_size=4,
                  input_size=TINY_INPUT_SIZE, expert_dnn_hidden_units=(8, 4),
                  tower_dnn_hidden_units=(4, 2), reg_embedding=P.REG_EMBEDDING,
                  reg_dnn=P.REG_DNN, device=torch.device("cpu"))


def make_relu_paths_live(model, *, bias=0.5):
    """把每个 ReLU 之前的 `nn.Linear` bias 置为正常数 `bias` —— 确定性避开"死 ReLU"退化。

    tiny fixture 只是生产图的**代理**，关心的是梯度管路是否连通。随机初始化下窄塔
    （`tower_dnn_hidden_units=(4, 2)`）的 ReLU 可能整层落在负半轴 → 该层输出恒为零 →
    **上游梯度精确为零**（图上可达、数值恒零）：此时"trunk 上梯度为零"是 fixture 退化，
    不是图不可达。生产配置（expert (256,128) / tower (64,32)）实测三路分量在 trunk 上
    梯度均非零，故这里是 fixture 需要"养活"，而不是断言需要放宽。
    """
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.named_children())
        for (_, layer), (_, nxt) in zip(children, children[1:]):
            if isinstance(layer, nn.Linear) and isinstance(nxt, nn.ReLU):
                nn.init.constant_(layer.bias, bias)
    return model


def dead_relu_modules(model, features, alpha=0.6):
    """返回在给定批次上输出**恒为零**的 ReLU 模块名（去重、升序；空列表 = 不存在死 ReLU）。

    注意同一个塔在一次 forward 里被调用两次（gen_rep 与 fused_rep 各一次），故按集合去重。
    """
    dead, hooks = set(), []

    def check(name):
        def hook(_module, _inputs, output):
            if not bool((output > 0).any()):
                dead.add(name)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(check(name)))
    try:
        with torch.no_grad():
            model(features, alpha)
    finally:
        for handle in hooks:
            handle.remove()
    return sorted(dead)


def tiny_batch(n=16, seed=7):
    gen = torch.Generator().manual_seed(seed)
    y0 = torch.randint(0, 2, (n,), generator=gen).float()
    y1 = torch.randint(0, 2, (n,), generator=gen).float()
    features = {"a": torch.randint(0, 3, (n,), generator=gen),
                "b": torch.randint(0, 2, (n,), generator=gen)}
    return y0, y1, features


def tiny_env_ids(n=16, seed=P.ENV_SEED):
    return P.make_env_ids(n, seed)


def forward_outputs(model, features, alpha=1.0):
    return model({k: v for k, v in features.items()}, alpha)


def decompose(model, alpha=1.0, n=16, seed=7):
    y0, y1, features = tiny_batch(n, seed)
    output = forward_outputs(model, features, alpha)
    components = GA.decompose_losses(model, output, y_income=y0, y_marital=y1,
                                     env_ids=tiny_env_ids(n), uni_coe=P.UNI_COE, env_coe=P.ENV_COE)
    return components, output, y0, y1, features


class TestLossDecomposition(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = tiny_model()

    def test_components_match_manual_definition(self):
        components, output, y0, y1, _ = decompose(self.model, alpha=0.7)
        bce = torch.nn.BCELoss()
        uni0 = bce(output["gen_preds"][0], y0)
        uni1 = bce(output["gen_preds"][1], y1)
        fused0 = bce(output["fused_preds"][0], y0)
        fused1 = bce(output["fused_preds"][1], y1)
        self.assertTrue(torch.equal(components["income"], fused0 + P.UNI_COE * uni0))
        self.assertTrue(torch.equal(components["marital"], fused1 + P.UNI_COE * uni1))
        self.assertTrue(torch.equal(components["reg"], self.model.get_l2_reg()))
        self.assertEqual(set(components), {"income", "marital", "env", "reg"})

    def test_env_component_is_env_coe_times_nll(self):
        y0, y1, features = tiny_batch()
        env_ids = tiny_env_ids()
        output = forward_outputs(self.model, features, alpha=0.4)
        components = GA.decompose_losses(self.model, output, y_income=y0, y_marital=y1,
                                         env_ids=env_ids, uni_coe=P.UNI_COE, env_coe=P.ENV_COE)
        nll = torch.nn.NLLLoss()(output["env_pred"], env_ids)
        self.assertTrue(torch.equal(components["env"], P.ENV_COE * nll))

    def test_all_components_non_negative_and_sum_matches_total(self):
        components, output, y0, y1, _ = decompose(self.model, alpha=0.5)
        for name, value in components.items():
            self.assertGreaterEqual(float(value), 0.0, name)
        total = GA.total_loss(components)
        bce = torch.nn.BCELoss()
        reference = (bce(output["fused_preds"][0], y0) + bce(output["fused_preds"][1], y1)
                     + P.UNI_COE * (bce(output["gen_preds"][0], y0) + bce(output["gen_preds"][1], y1))
                     + P.ENV_COE * torch.nn.NLLLoss()(output["env_pred"], tiny_env_ids())
                     + self.model.get_l2_reg())
        self.assertTrue(torch.allclose(total, reference, atol=1e-6, rtol=1e-5))

    def test_general_vs_specific_composite_is_not_defined(self):
        """按预注册：gate 权重依赖样本 → 损失空间无 general/specific 可加分解。"""
        self.assertFalse(hasattr(GA, "general_specific_composite"))
        self.assertEqual(GA.UNDEFINED_COMPOSITES, ("general_vs_specific",))


class TestNoSideEffects(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.model = tiny_model()
        self.components, self.output, *_ = decompose(self.model, alpha=0.6)

    def _params(self):
        return [(name, p) for name, p in self.model.named_parameters()]

    def test_grad_attribute_untouched(self):
        self.assertTrue(all(p.grad is None for _, p in self._params()))
        GA.component_grads(self.components, [p for _, p in self._params()])
        self.assertTrue(all(p.grad is None for _, p in self._params()))   # autograd.grad 不写 .grad

    def test_parameters_bit_for_bit_unchanged(self):
        before = [(name, p.detach().clone()) for name, p in self._params()]
        GA.component_grads(self.components, [p for _, p in self._params()])
        for (name, old), (_, new) in zip(before, self._params()):
            self.assertTrue(torch.equal(old, new.detach()), name)

    def test_rng_state_unchanged(self):
        torch.manual_seed(99)
        state = torch.get_rng_state()
        GA.component_grads(self.components, [p for _, p in self._params()])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_optimizer_state_untouched(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=P.LR)
        GA.total_loss(self.components).backward()          # 先让 Adam 产生一次状态（同时释放该图）
        optimizer.step()
        before = {key: {field: (value.clone() if torch.is_tensor(value) else value)
                        for field, value in state.items()}
                  for key, state in optimizer.state_dict()["state"].items()}
        self.assertNotEqual(before, {})                    # 确认 Adam 真的建了状态，否则测的是空气
        fresh, *_ = decompose(self.model, alpha=0.6)       # 新图：setUp 的图已被 backward 释放
        GA.component_grads(fresh, [p for _, p in self._params()])
        after = optimizer.state_dict()["state"]
        self.assertEqual(set(before), set(after))
        for key in before:
            for field, old in before[key].items():
                if torch.is_tensor(old):
                    self.assertTrue(torch.equal(old, after[key][field]), f"{key}.{field}")
                else:
                    self.assertEqual(old, after[key][field])

    def test_graph_survives_for_the_real_backward(self):
        params = [p for _, p in self._params()]
        grads = GA.component_grads(self.components, params)
        self.assertTrue(any(grad is not None and float(grad.abs().sum()) > 0.0
                            for per_component in grads.values() for grad in per_component))
        GA.total_loss(self.components).backward()          # retain_graph 保住了原图
        for name, p in self._params():
            self.assertIsNotNone(p.grad, name)


class TestParameterGroups(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = make_relu_paths_live(tiny_model())      # 确定性：固定 seed + 无死 ReLU
        _, _, self.features = tiny_batch(16, 7)
        self._assert_fixture_is_live()

    def _assert_fixture_is_live(self):
        """显式前置条件：fixture 必须"活"，否则本类的可达性断言会得出错误结论。

        死 ReLU（或与样本无关的预测量）会让图上可达的参数拿到**精确为零**的梯度，
        使"图不可达"与"fixture 退化"无法区分。阈值 1e-6 是 fixture 级护栏：
        实测活模型 std ≈ 1.5e-2，死 ReLU 模型 std ≈ 3e-8，两侧各有 4 个数量级余量。
        """
        dead = dead_relu_modules(self.model, self.features, alpha=0.6)
        self.assertEqual(dead, [], f"tiny fixture 退化（死 ReLU）: {dead}")
        gen_pred = forward_outputs(self.model, self.features, alpha=0.6)["gen_preds"][0]
        self.assertGreater(float(gen_pred.std()), 1e-6, "tiny fixture 退化（预测与样本无关）")

    def test_groups_are_a_partition_of_all_parameters(self):
        groups = GA.parameter_groups(self.model)
        names = [name for name, _ in self.model.named_parameters()]
        flat = [name for group in GA.PARTITION_GROUPS for name in groups[group]]
        self.assertEqual(sorted(flat), sorted(names))              # 每个参数恰好属于一组
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(groups[GA.GROUP_UNCLASSIFIED], [])
        for key in GA.REPORTED_GROUPS:                             # 报告用的组必须都存在
            self.assertIn(key, groups)

    def test_expected_membership(self):
        groups = GA.parameter_groups(self.model)
        self.assertTrue(all(n.startswith("embedding_network.") for n in groups[GA.GROUP_EMBEDDINGS]))
        self.assertTrue(all(n.startswith("shared_expert_network.") for n in groups[GA.GROUP_SHARED_EXPERT]))
        self.assertTrue(all(n.startswith("env_classifier.") for n in groups[GA.GROUP_ENV_CLASSIFIER]))
        for name in groups[GA.GROUP_TASK_EXCLUSIVE]:
            self.assertTrue(name.startswith(("specific_expert_networks.", "gate_networks.",
                                             "tower_networks.", "env_embedding_network.")), name)

    def test_shared_trunk_is_embeddings_plus_shared_expert_only(self):
        groups = GA.parameter_groups(self.model)
        trunk = sorted(groups[GA.GROUP_SHARED_TRUNK])
        self.assertEqual(trunk, sorted(groups[GA.GROUP_EMBEDDINGS] + groups[GA.GROUP_SHARED_EXPERT]))
        for name in trunk:
            self.assertFalse(name.startswith(("tower_networks.", "specific_expert_networks.",
                                               "gate_networks.", "env_embedding_network.",
                                               "env_classifier.")), name)

    def test_shared_trunk_is_reachable_from_all_three_components(self):
        """共享的定义 = 被三个分量同时可达（autograd 实测，不是口头声明）。"""
        components, *_ = decompose(self.model, alpha=0.6)
        params = [(name, p) for name, p in self.model.named_parameters()]
        grads = GA.component_grads(components, [p for _, p in params])
        trunk = set(GA.parameter_groups(self.model)[GA.GROUP_SHARED_TRUNK])
        for component in ("income", "marital", "env"):
            touched = [(name, grad) for (name, _), grad in zip(params, grads[component])
                       if name in trunk]
            self.assertTrue(touched)
            for name, grad in touched:
                self.assertIsNotNone(grad, f"{component} 应可达 {name}")
            self.assertGreater(sum(float(grad.abs().sum()) for _, grad in touched), 0.0,
                               f"{component} 在 shared_trunk 上应整体非零")

    def test_task_exclusive_params_receive_zero_grad_from_other_tasks(self):
        """income 分量在 marital 独有塔上图上不可达（`None`）；env_embedding 仅第 0 行有梯度。"""
        components, *_ = decompose(self.model, alpha=0.6)
        params = [(name, p) for name, p in self.model.named_parameters()]
        grads = GA.component_grads(components, [p for _, p in params])
        for (name, _), grad in zip(params, grads["income"]):
            if name.startswith(("tower_networks.1.", "specific_expert_networks.1.",
                                "gate_networks.1.")):
                self.assertIsNone(grad, f"income 不应可达 {name}")
            if name == "env_embedding_network.weight":
                self.assertIsNotNone(grad)
                self.assertTrue(torch.equal(grad[1], torch.zeros_like(grad[1])))   # 第 1 行
                self.assertGreater(float(grad[0].abs().sum()), 0.0)                # 第 0 行非零

    def test_env_classifier_receives_gradient_only_from_env(self):
        components, *_ = decompose(self.model, alpha=0.6)
        params = [(name, p) for name, p in self.model.named_parameters()]
        grads = GA.component_grads(components, [p for _, p in params])
        for (name, _), grad in zip(params, grads["income"]):
            if name.startswith("env_classifier."):
                self.assertIsNone(grad, f"income 不应可达 {name}")
        for (name, _), grad in zip(params, grads["env"]):
            if name.startswith("env_classifier."):
                self.assertIsNotNone(grad, name)
                self.assertGreater(float(grad.abs().sum()), 0.0, name)

    def test_alpha_zero_kills_env_gradient_on_shared_trunk_only(self):
        """预注册 §5：epoch1 step0 的 alpha=0 → env 在 trunk 上梯度精确为零。"""
        components, *_ = decompose(self.model, alpha=0.0)
        params = [(name, p) for name, p in self.model.named_parameters()]
        grads = GA.component_grads(components, [p for _, p in params])
        trunk = set(GA.parameter_groups(self.model)[GA.GROUP_SHARED_TRUNK])
        for (name, _), grad in zip(params, grads["env"]):
            if name in trunk:
                self.assertTrue(torch.equal(grad, torch.zeros_like(grad)), name)
            if name.startswith("env_classifier."):
                self.assertGreater(float(grad.abs().sum()), 0.0, name)   # 反转点在 trunk 之下


class TestProductionGraphReachability(unittest.TestCase):
    """生产配置的图**确实**三路可达共享主干——tiny fixture 的退化是 fixture 的问题，不是图的问题。

    合成特征即可：可达性是图结构性质，不依赖真实数据；模型走 `build_mptrec`（与正式审计同一条
    构造路径、同 seed），故断言的是**生产图**本身。
    """

    def test_shared_trunk_is_reachable_from_all_three_components(self):
        torch.manual_seed(P.MODEL_SEED)
        model = build_mptrec(torch.device("cpu"))
        vocab = model.embedding_network.feature_names
        gen = torch.Generator().manual_seed(0)
        features = {name: torch.randint(
            0, model.embedding_network.embedding_dict[name].num_embeddings, (256,), generator=gen)
            for name in vocab}
        dense = P.INPUT_SIZE - len(vocab) * P.EMBEDDING_SIZE       # 不在 vocab 里的稠密特征
        features.update({f"dense{i}": torch.randn(256, generator=gen) for i in range(dense)})
        self.assertEqual(len(features), len(vocab) + dense)

        y0 = torch.randint(0, 2, (256,), generator=gen).float()
        y1 = torch.randint(0, 2, (256,), generator=gen).float()
        output = forward_outputs(model, features, alpha=0.5)
        components = GA.decompose_losses(model, output, y_income=y0, y_marital=y1,
                                         env_ids=P.make_env_ids(256, P.ENV_SEED),
                                         uni_coe=P.UNI_COE, env_coe=P.ENV_COE)
        params = [(name, p) for name, p in model.named_parameters()]
        grads = GA.component_grads(components, [p for _, p in params])
        trunk = set(GA.parameter_groups(model)[GA.GROUP_SHARED_TRUNK])
        for component in ("income", "marital", "env"):
            touched = [(name, grad) for (name, _), grad in zip(params, grads[component])
                       if name in trunk]
            self.assertTrue(touched)
            for name, grad in touched:
                self.assertIsNotNone(grad, f"{component} 应可达 {name}")
            self.assertGreater(sum(float(grad.abs().sum()) for _, grad in touched), 0.0,
                               f"生产图上 {component} 在 shared_trunk 上应整体非零")


class TestSampleSchedule(unittest.TestCase):
    def test_deterministic_sorted_unique_in_range(self):
        first = GA.sample_schedule(780)
        self.assertEqual(first, GA.sample_schedule(780))
        self.assertEqual(first, sorted(set(first)))
        self.assertTrue(all(0 <= i < 780 for i in first))

    def test_includes_first_warmup_batches(self):
        schedule = GA.sample_schedule(780)
        self.assertEqual(schedule[:GA.WARMUP_STEPS], list(range(GA.WARMUP_STEPS)))

    def test_evenly_spaced_tail_covers_the_epoch(self):
        schedule = GA.sample_schedule(1000)
        tail = [i for i in schedule if i >= GA.WARMUP_STEPS]
        self.assertEqual(tail, [(k * 1000) // GA.SPACED_STEPS for k in range(1, GA.SPACED_STEPS)])
        gaps = [b - a for a, b in zip(tail, tail[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 1)                  # 均匀

    def test_two_epochs_clear_the_minimum_step_requirement(self):
        self.assertGreaterEqual(2 * len(GA.sample_schedule(780)), GA.MIN_SAMPLED_STEPS)

    def test_degrades_safely_on_tiny_loaders(self):
        self.assertEqual(GA.sample_schedule(0), [])
        self.assertEqual(GA.sample_schedule(3), [0, 1, 2])
        self.assertEqual(GA.sample_schedule(1), [0])
        self.assertEqual(GA.sample_schedule(5, warmup=0, spaced=1), [0])


class TestCosineMath(unittest.TestCase):
    def _record(self, ga, gb, gc, gd, *, names=("p0", "p1"), groups=None):
        groups = groups or {GA.GROUP_SHARED_TRUNK: list(names)}
        grads = {"income": [ga, gb], "marital": [gc, gd],
                 "env": [torch.zeros_like(ga), torch.zeros_like(gb)],
                 "reg": [torch.ones_like(ga), torch.ones_like(gb)]}
        return GA.step_record(epoch=1, step=0, alpha=0.5, grads=grads,
                              param_names=list(names), groups=groups)

    def test_parallel_opposite_orthogonal(self):
        key = f"cos_income_marital__{GA.GROUP_SHARED_TRUNK}"
        v = torch.tensor([1.0, 0.0])
        self.assertAlmostEqual(self._record(v, v, 2 * v, 2 * v)[key], 1.0, places=9)
        self.assertAlmostEqual(self._record(v, v, -3 * v, -3 * v)[key], -1.0, places=9)
        self.assertAlmostEqual(self._record(v, v, torch.tensor([0.0, 5.0]),
                                            torch.tensor([0.0, 5.0]))[key], 0.0, places=9)

    def test_cosine_is_over_the_concatenated_group_vector(self):
        """组内余弦按拼接整体定义，而非逐参数余弦再平均。"""
        key = f"cos_income_marital__{GA.GROUP_SHARED_TRUNK}"
        a0, a1 = torch.tensor([3.0, 0.0]), torch.tensor([1.0, 0.0])
        b0, b1 = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
        per_param_mean = 0.5 * (1.0 + -1.0)                              # = 0.0
        cat_a = torch.cat([a0, a1]).double()
        cat_b = torch.cat([b0, b1]).double()
        expected = float(torch.dot(cat_a, cat_b) / (cat_a.norm() * cat_b.norm()))
        self.assertNotAlmostEqual(expected, per_param_mean, places=6)
        self.assertAlmostEqual(self._record(a0, a1, b0, b1)[key], expected, places=9)

    def test_zero_norm_yields_undefined_cosine_not_nan(self):
        key = f"cos_income_marital__{GA.GROUP_SHARED_TRUNK}"
        zero = torch.zeros(2)
        v = torch.tensor([1.0, 2.0])
        record = self._record(v, v, zero, zero)
        self.assertIsNone(record[key])                                   # 未定义 → None（JSON null）
        self.assertEqual(record[f"n_marital__{GA.GROUP_SHARED_TRUNK}"], 0.0)
        self.assertGreater(record[f"n_income__{GA.GROUP_SHARED_TRUNK}"], 0.0)
        self.assertEqual(record[f"z_marital__{GA.GROUP_SHARED_TRUNK}"], 2)   # 两个参数都精确零梯度
        self.assertEqual(record[f"z_income__{GA.GROUP_SHARED_TRUNK}"], 0)

    def test_norms_are_group_level_frobenius(self):
        a0, a1 = torch.tensor([3.0, 4.0]), torch.tensor([0.0, 12.0])     # 范数 5 与 12
        r = self._record(a0, a1, a0, a1)
        self.assertAlmostEqual(r[f"n_income__{GA.GROUP_SHARED_TRUNK}"], 13.0, places=6)

    def test_unreachable_parameter_counts_as_exact_zero(self):
        groups = {GA.GROUP_SHARED_TRUNK: ["p0", "p1"], GA.GROUP_ENV_CLASSIFIER: ["c"]}
        grads = {"income": [torch.tensor([1.0]), None], "marital": [torch.tensor([1.0]), None],
                 "env": [torch.tensor([1.0]), torch.tensor([2.0])],
                 "reg": [torch.tensor([1.0]), torch.tensor([1.0])]}
        record = GA.step_record(epoch=1, step=0, alpha=1.0, grads=grads,
                                param_names=["p0", "p1"], groups=groups)
        self.assertEqual(record[f"z_income__{GA.GROUP_SHARED_TRUNK}"], 1)
        self.assertEqual(record[f"z_env__{GA.GROUP_ENV_CLASSIFIER}"], 0)
        self.assertIsNone(record[f"cos_income_env__{GA.GROUP_ENV_CLASSIFIER}"])   # income 在该组范数为 0

    def test_vectors_stay_parameter_aligned_when_reachability_differs(self):
        """两个分量可达的参数不同（但元素数恰好相同）时，必须逐位对齐到**参数**而不是碰巧等长。

        income 只在 p0 上有梯度、marital 只在 p1 上：正确语义下二者在 shared_trunk 上正交
        （余弦 0）；若像旧实现那样把 `None` 丢掉再拼接，两个长度为 2 的向量会逐位错位相乘，
        给出余弦 1.0 —— 静默地把不相干的参数当成"方向一致"。
        """
        key = f"cos_income_marital__{GA.GROUP_SHARED_TRUNK}"
        groups = {GA.GROUP_SHARED_TRUNK: ["p0", "p1"]}
        g = torch.tensor([1.0, 1.0])
        grads = {"income": [g, None], "marital": [None, g],
                 "env": [None, None], "reg": [None, None]}
        record = GA.step_record(epoch=1, step=0, alpha=1.0, grads=grads,
                                param_names=["p0", "p1"], groups=groups,
                                params=[torch.zeros(2), torch.zeros(2)])
        self.assertAlmostEqual(record[key], 0.0, places=9)
        self.assertAlmostEqual(record[f"n_income__{GA.GROUP_SHARED_TRUNK}"], 2 ** 0.5, places=9)
        self.assertAlmostEqual(record[f"n_marital__{GA.GROUP_SHARED_TRUNK}"], 2 ** 0.5, places=9)
        self.assertEqual(record[f"z_income__{GA.GROUP_SHARED_TRUNK}"], 1)


class TestStepRecordShape(unittest.TestCase):
    def test_record_keys_are_complete_and_flat(self):
        torch.manual_seed(23)
        model = tiny_model()
        components, *_ = decompose(model, alpha=0.5)
        params = [(n, p) for n, p in model.named_parameters()]
        grads = GA.component_grads(components, [p for _, p in params])
        record = GA.step_record(epoch=2, step=5, alpha=0.5, grads=grads,
                                param_names=[n for n, _ in params],
                                groups=GA.parameter_groups(model))
        expected = {"epoch", "step", "alpha"}
        for component in GA.COMPONENTS:
            for group in GA.REPORTED_GROUPS:
                expected.add(f"n_{component}__{group}")
                expected.add(f"z_{component}__{group}")
        for a, b in GA.CONFLICT_PAIRS:
            for group in GA.REPORTED_GROUPS:
                expected.add(f"cos_{a}_{b}__{group}")
        self.assertEqual(set(record), expected)
        for key, value in record.items():
            self.assertIsInstance(value, (int, float, type(None)), key)
        json.dumps(record)                                                # 可 JSON 序列化

    def test_unclassified_parameters_are_reported_separately_from_trunk(self):
        """未分类参数不得被悄悄算进 shared_trunk。"""
        model = tiny_model()
        groups = GA.parameter_groups(model)
        self.assertEqual(groups[GA.GROUP_UNCLASSIFIED], [])
        trunk = set(groups[GA.GROUP_SHARED_TRUNK])
        for name in groups[GA.GROUP_ENV_CLASSIFIER] + groups[GA.GROUP_TASK_EXCLUSIVE]:
            self.assertNotIn(name, trunk)


class TestAggregateAndGate(unittest.TestCase):
    """合成记录 → 汇总与门禁；阈值取自预注册，不看真实结果。"""

    @staticmethod
    def _records(n_steps, cos_values, *, epoch=1, group=GA.GROUP_SHARED_TRUNK, alpha=0.5):
        records = []
        for step, cos in enumerate(cos_values):
            record = {"epoch": epoch, "step": step, "alpha": alpha}
            for component in GA.COMPONENTS:
                for grp in GA.REPORTED_GROUPS:
                    record[f"n_{component}__{grp}"] = 1.0
                    record[f"z_{component}__{grp}"] = 0
            for a, b in GA.CONFLICT_PAIRS:
                for grp in GA.REPORTED_GROUPS:
                    record[f"cos_{a}_{b}__{grp}"] = 0.5
            record[f"cos_income_marital__{group}"] = cos
            records.append(record)
        return records

    def test_aggregate_counts_and_rates(self):
        records = self._records(4, [0.5, -0.5, -0.2, 0.1]) + \
            self._records(4, [-0.5, -0.5, 0.3, 0.4], epoch=2)
        agg = GA.aggregate(records)
        self.assertEqual(agg["n_steps"], 8)
        self.assertEqual(agg["n_steps_by_epoch"]["1"], 4)
        self.assertEqual(agg["n_steps_by_epoch"]["2"], 4)
        stats = agg["pairs"]["income_vs_marital"][GA.GROUP_SHARED_TRUNK]
        self.assertEqual(stats["n_defined"], 8)
        self.assertEqual(stats["n_undefined"], 0)
        self.assertEqual(stats["n_negative"], 4)
        self.assertAlmostEqual(stats["neg_cos_rate"], 0.5, places=9)
        self.assertAlmostEqual(stats["mean_cos"], (0.5 - 0.5 - 0.2 + 0.1 - 0.5 - 0.5 + 0.3 + 0.4) / 8, places=9)
        self.assertAlmostEqual(stats["nonzero_rate_a"], 1.0, places=9)
        self.assertAlmostEqual(stats["mean_of_ratio"], 1.0, places=9)
        self.assertAlmostEqual(stats["ratio_of_means"], 1.0, places=9)
        self.assertAlmostEqual(stats["mean_dominance_ratio"], 0.5, places=9)

    def test_aggregate_skips_undefined_cosines(self):
        records = self._records(3, [None, -0.5, -0.5])
        stats = GA.aggregate(records)["pairs"]["income_vs_marital"][GA.GROUP_SHARED_TRUNK]
        self.assertEqual((stats["n_defined"], stats["n_undefined"]), (2, 1))
        self.assertAlmostEqual(stats["mean_cos"], -0.5, places=9)
        self.assertAlmostEqual(stats["neg_cos_rate"], 1.0, places=9)

    def test_aggregate_reports_zero_gradient_counts_by_epoch_and_group(self):
        records = self._records(2, [0.1, 0.1])
        records[0][f"n_env__{GA.GROUP_SHARED_TRUNK}"] = 0.0
        records[0][f"z_env__{GA.GROUP_SHARED_TRUNK}"] = 3
        agg = GA.aggregate(records)
        zero = agg["zero_norm_steps"]["env"][GA.GROUP_SHARED_TRUNK]
        self.assertEqual(zero["by_epoch"]["1"], 1)
        self.assertEqual(zero["total"], 1)
        self.assertAlmostEqual(zero["rate"], 0.5, places=9)
        tensors = agg["zero_tensor"]["env"][GA.GROUP_SHARED_TRUNK]
        self.assertEqual(tensors["total"], 3)
        self.assertGreater(tensors["frac"], 0.0)

    def test_gate_requires_all_four_conditions(self):
        passing = self._records(30, [-0.4] * 30) + self._records(30, [-0.3] * 30, epoch=2)
        verdict = GA.judge_gate(GA.aggregate(passing))
        self.assertEqual(verdict["status"], "ACTIONABLE")
        self.assertTrue(verdict["actionable"])
        self.assertEqual(verdict["reasons"], [])

    def test_gate_fails_on_low_negative_rate(self):
        records = self._records(30, [-1.0] * 8 + [0.5] * 22) + self._records(30, [0.5] * 30, epoch=2)
        verdict = GA.judge_gate(GA.aggregate(records))
        self.assertEqual(verdict["status"], "NO_ACTION")
        self.assertIn("G1", " ".join(verdict["reasons"]))
        self.assertFalse(verdict["criteria"]["G1"]["pass"])

    def test_gate_fails_on_shallow_mean_cosine(self):
        records = self._records(30, [-0.01] * 30) + self._records(30, [-0.01] * 30, epoch=2)
        verdict = GA.judge_gate(GA.aggregate(records))
        self.assertEqual(verdict["status"], "NO_ACTION")
        self.assertFalse(verdict["criteria"]["G2"]["pass"])
        self.assertTrue(verdict["criteria"]["G1"]["pass"])            # 负率 100% 但幅度不够

    def test_gate_fails_when_a_norm_is_often_zero(self):
        records = self._records(30, [-0.5] * 30) + self._records(30, [-0.5] * 30, epoch=2)
        for record in records[:10]:
            record[f"n_income__{GA.GROUP_SHARED_TRUNK}"] = 0.0
            record[f"cos_income_marital__{GA.GROUP_SHARED_TRUNK}"] = None
        verdict = GA.judge_gate(GA.aggregate(records))
        self.assertEqual(verdict["status"], "NO_ACTION")
        self.assertFalse(verdict["criteria"]["G3"]["pass"])

    def test_gate_fails_below_minimum_sampled_steps(self):
        records = self._records(20, [-0.5] * 20) + self._records(20, [-0.5] * 20, epoch=2)
        verdict = GA.judge_gate(GA.aggregate(records))
        self.assertEqual(verdict["status"], "NO_ACTION")
        self.assertFalse(verdict["criteria"]["G0"]["pass"])
        self.assertIn("步数", " ".join(verdict["reasons"]))

    def test_env_conflict_alone_never_triggers_action(self):
        records = self._records(30, [0.5] * 30) + self._records(30, [0.5] * 30, epoch=2)
        for record in records:                                        # income vs env 强负，income vs marital 正
            record[f"cos_income_env__{GA.GROUP_SHARED_TRUNK}"] = -0.9
            record[f"cos_marital_env__{GA.GROUP_SHARED_TRUNK}"] = -0.9
        verdict = GA.judge_gate(GA.aggregate(records))
        self.assertEqual(verdict["status"], "NO_ACTION")
        self.assertTrue(verdict["env_diagnostic_only"])
        self.assertFalse(verdict["criteria"]["G1"]["pass"])

    def test_env_diagnostics_are_reported_separately(self):
        records = self._records(30, [-0.4] * 30) + self._records(30, [-0.4] * 30, epoch=2)
        for record in records:
            record[f"cos_income_env__{GA.GROUP_SHARED_TRUNK}"] = -0.7
        verdict = GA.judge_gate(GA.aggregate(records))
        self.assertAlmostEqual(verdict["env_diagnostic"]["income_vs_env"]["shared_trunk"]["mean_cos"],
                               -0.7, places=6)

    def test_gate_thresholds_match_preregistration(self):
        self.assertEqual((GA.MIN_SAMPLED_STEPS, GA.GATE_NEG_COS_RATE, GA.GATE_MEAN_COS,
                          GA.GATE_NONZERO_STEP_RATE), (50, 0.25, -0.05, 0.95))
        self.assertEqual((GA.WARMUP_STEPS, GA.SPACED_STEPS), (10, 25))
        self.assertEqual(GA.GATE_GROUP, GA.GROUP_SHARED_TRUNK)
        self.assertEqual(GA.GATE_PAIR, ("income", "marital"))


class TestDeterministicOutput(unittest.TestCase):
    def test_aggregate_and_gate_are_byte_identical_on_repeat(self):
        records = TestAggregateAndGate._records(30, [-0.3] * 15 + [0.4] * 15)
        first = P.canonical_json({"agg": GA.aggregate(records), "gate": GA.judge_gate(GA.aggregate(records))})
        second = P.canonical_json({"agg": GA.aggregate(records), "gate": GA.judge_gate(GA.aggregate(records))})
        self.assertEqual(first, second)

    def test_payload_has_no_timestamp_or_absolute_path(self):
        records = TestAggregateAndGate._records(12, [-0.3] * 12)
        payload = GA.build_payload(step_records=records, meta={"model_seed": P.MODEL_SEED}, smoke=False)
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        for token in ("created", "timestamp", "C:\\", ":/", "\\\\"):
            self.assertNotIn(token, text)
        self.assertIn("prereg", payload)
        self.assertEqual(payload["prereg"]["min_sampled_steps"], GA.MIN_SAMPLED_STEPS)

    def test_smoke_payload_never_carries_a_gate_verdict(self):
        records = TestAggregateAndGate._records(30, [-0.4] * 30) + \
            TestAggregateAndGate._records(30, [-0.4] * 30, epoch=2)
        payload = GA.build_payload(step_records=records, meta={}, smoke=True)
        self.assertEqual(payload["gate"]["status"], "NOT_EVALUATED_SMOKE")
        self.assertFalse(payload["gate"]["actionable"])


class TestInstrumentationFidelity(unittest.TestCase):
    """插桩版 train_two_task 必须与父类逐位一致（同进程、同 seed）。"""

    N_TRAIN, N_VAL, BATCH = 64, 32, 16

    def _fixture(self, seed):
        torch.manual_seed(seed)
        model = tiny_model()
        train_ds, val_ds = TinyCensus(self.N_TRAIN, 3), TinyCensus(self.N_VAL, 4)
        loaders = {"train": DataLoader(train_ds, batch_size=self.BATCH),
                   "val": DataLoader(val_ds, batch_size=self.BATCH)}
        return model, loaders

    def _run(self, seed, *, manager_cls=MPTRecTrainManager, audit=None, schedule=None, epochs=2):
        model, loaders = self._fixture(seed)
        env_ids = tiny_env_ids(self.N_TRAIN)
        kwargs = dict(model=model, train_loader=loaders["train"], val_loader=loaders["val"],
                      env_ids=env_ids, task_name=["Income", "Marital"], lr=P.LR,
                      batch_size=self.BATCH, uni_coe=P.UNI_COE, env_coe=P.ENV_COE, epochs=epochs)
        if manager_cls is GA.GradientAuditTrainManager:
            manager = manager_cls(audit=audit, audit_schedule=schedule, **kwargs)
        else:
            manager = manager_cls(**kwargs)
        manager.train_two_task()
        state = {name: p.detach().clone() for name, p in model.named_parameters()}
        return state, manager

    def _assert_same_state(self, left, right):
        self.assertEqual(sorted(left), sorted(right))
        for name in left:
            self.assertTrue(torch.equal(left[name], right[name]), f"参数不逐位一致: {name}")

    def test_audit_manager_without_collector_is_bit_for_bit_identical(self):
        baseline, _ = self._run(P.MODEL_SEED)
        replayed, _ = self._run(P.MODEL_SEED, manager_cls=GA.GradientAuditTrainManager, audit=None)
        self._assert_same_state(baseline, replayed)

    def test_collecting_does_not_change_training_trajectory(self):
        baseline, _ = self._run(P.MODEL_SEED)
        n_batches = self.N_TRAIN // self.BATCH
        collected, manager = self._run(P.MODEL_SEED, manager_cls=GA.GradientAuditTrainManager,
                                       audit=GA.GradientAudit(model=None, uni_coe=P.UNI_COE,
                                                              env_coe=P.ENV_COE),
                                       schedule=list(range(n_batches)))
        self._assert_same_state(baseline, collected)
        self.assertEqual(len(manager.audit.records), 2 * n_batches)          # 两个 epoch 各采全量
        self.assertEqual([r["epoch"] for r in manager.audit.records],
                         [1] * n_batches + [2] * n_batches)
        self.assertEqual([r["step"] for r in manager.audit.records],
                         list(range(n_batches)) * 2)
        self.assertEqual(manager.audit.records[0]["alpha"], 0.0)             # 预注册 §5 的构造性零点
        self.assertGreater(manager.audit.records[n_batches]["alpha"], 0.0)

    def test_sampled_trajectory_equals_full_replay_on_unsampled_steps_only(self):
        """未采样步不应被记录，但训练轨迹与全量采集版一致 → 采样不影响轨迹。"""
        n_batches = self.N_TRAIN // self.BATCH
        full, _ = self._run(P.MODEL_SEED, manager_cls=GA.GradientAuditTrainManager,
                            audit=GA.GradientAudit(model=None, uni_coe=P.UNI_COE, env_coe=P.ENV_COE),
                            schedule=list(range(n_batches)))
        sparse, manager = self._run(P.MODEL_SEED, manager_cls=GA.GradientAuditTrainManager,
                                    audit=GA.GradientAudit(model=None, uni_coe=P.UNI_COE,
                                                           env_coe=P.ENV_COE),
                                    schedule=[0, 2])
        self._assert_same_state(full, sparse)
        self.assertEqual(len(manager.audit.records), 4)

    def test_schedule_uses_loaders_batch_count(self):
        model, loaders = self._fixture(P.MODEL_SEED)
        manager = GA.GradientAuditTrainManager(
            model=model, train_loader=loaders["train"], val_loader=loaders["val"],
            env_ids=tiny_env_ids(self.N_TRAIN), task_name=["Income", "Marital"], lr=P.LR,
            batch_size=self.BATCH, uni_coe=P.UNI_COE, env_coe=P.ENV_COE, epochs=2,
            audit=GA.GradientAudit(model=None, uni_coe=P.UNI_COE, env_coe=P.ENV_COE))
        self.assertEqual(manager.audit_schedule, GA.sample_schedule(self.N_TRAIN // self.BATCH))


class TestCollectorWiring(unittest.TestCase):
    def test_collector_binds_to_model_and_records_real_gradients(self):
        torch.manual_seed(31)
        model = tiny_model()
        audit = GA.GradientAudit(model=model, uni_coe=P.UNI_COE, env_coe=P.ENV_COE)
        y0, y1, features = tiny_batch()
        output = forward_outputs(model, features, alpha=0.8)
        record = audit.record(epoch=1, step=3, alpha=0.8, output=output,
                              y_income=y0, y_marital=y1, env_ids=tiny_env_ids())
        self.assertEqual((record["epoch"], record["step"], record["alpha"]), (1, 3, 0.8))
        self.assertGreater(record[f"n_income__{GA.GROUP_SHARED_TRUNK}"], 0.0)
        self.assertGreater(record[f"n_env__{GA.GROUP_SHARED_TRUNK}"], 0.0)
        # income 在 marital 独有参数上必须精确为零 → 该组零梯度计数 > 0（而非被悄悄忽略）
        self.assertGreater(record[f"z_income__{GA.GROUP_TASK_EXCLUSIVE}"], 0)
        self.assertEqual(record[f"z_income__{GA.GROUP_SHARED_TRUNK}"], 0)
        self.assertEqual(len(audit.records), 1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_collector_defers_model_binding_until_attach(self):
        audit = GA.GradientAudit(model=None, uni_coe=P.UNI_COE, env_coe=P.ENV_COE)
        self.assertFalse(audit.ready)
        audit.attach(tiny_model())
        self.assertTrue(audit.ready)

    def test_unbound_collector_raises_rather_than_silently_skipping(self):
        audit = GA.GradientAudit(model=None, uni_coe=P.UNI_COE, env_coe=P.ENV_COE)
        y0, y1, features = tiny_batch()
        output = forward_outputs(tiny_model(), features)
        with self.assertRaises(RuntimeError):
            audit.record(epoch=1, step=0, alpha=1.0, output=output,
                         y_income=y0, y_marital=y1, env_ids=tiny_env_ids())


class TestMarkdownReport(unittest.TestCase):
    def test_report_states_gradient_surgery_is_prior_work_and_no_claim(self):
        records = TestAggregateAndGate._records(30, [-0.4] * 30) + \
            TestAggregateAndGate._records(30, [-0.4] * 30, epoch=2)
        payload = GA.build_payload(step_records=records, meta={"model_seed": P.MODEL_SEED}, smoke=False)
        text = GA.render_markdown(payload)
        self.assertIn("既有文献", text)
        self.assertIn("NO_ACTION", text.replace("ACTIONABLE", ""))
        self.assertIn("shared_trunk", text)
        for key in ("neg_cos_rate", "mean_cos", "nonzero_rate", "dominance"):
            self.assertIn(key, text)

    @staticmethod
    def _payload(cos_values):
        records = (TestAggregateAndGate._records(30, list(cos_values[:30]))
                   + TestAggregateAndGate._records(30, list(cos_values[30:]), epoch=2))
        return GA.build_payload(step_records=records, meta={"model_seed": P.MODEL_SEED}, smoke=False)

    def test_interpretation_records_cancellation_not_persistent_conflict(self):
        """半数步负余弦、均值≈0 的形态：必须写明"双向抵消"，而不是说成持续的方向性对立。"""
        payload = self._payload([-0.6] * 15 + [0.61] * 15 + [-0.6] * 15 + [0.6] * 15)
        self.assertEqual(payload["gate"]["status"], "NO_ACTION")
        self.assertTrue(payload["gate"]["criteria"]["G1"]["pass"])       # 冲突确实出现
        self.assertFalse(payload["gate"]["criteria"]["G2"]["pass"])      # 但均值不过门禁
        text = GA.render_markdown(payload)
        self.assertIn("## 结论解读", text)
        lines = GA.interpretation_lines(payload)
        joined = "\n".join(lines)
        self.assertIn("50.0%", joined)                                   # 用了 payload 里的负余弦占比
        self.assertIn("抵消", joined)
        self.assertIn("不构成 PCGrad 的依据", joined)
        self.assertIn("NO_ACTION", joined)
        self.assertIn("不做梯度手术", joined)
        self.assertIn("env", joined)                                     # env 只作诊断
        self.assertIn("30/60", joined)                                   # n_negative / n_defined 取自 payload

    def test_interpretation_does_not_call_half_rate_conflict_persistent(self):
        """反向守护：均值确实为负且过门禁时，不得再写"抵消"。"""
        payload = self._payload([-0.4] * 30 + [-0.3] * 30)
        self.assertEqual(payload["gate"]["status"], "ACTIONABLE")
        joined = "\n".join(GA.interpretation_lines(payload))
        self.assertIn("ACTIONABLE", joined)
        self.assertNotIn("抵消", joined)
        self.assertNotIn("不做梯度手术", joined)

    def test_interpretation_flags_insufficient_steps(self):
        payload = self._payload([-0.4] * 30)          # 单 epoch：30 步 < 50
        self.assertFalse(payload["gate"]["criteria"]["G0"]["pass"])
        self.assertIn("采样步数不足", "\n".join(GA.interpretation_lines(payload)))

    def test_interpretation_smoke_never_implies_a_verdict(self):
        records = TestAggregateAndGate._records(30, [-0.6] * 30)
        payload = GA.build_payload(step_records=records, meta={}, smoke=True)
        joined = "\n".join(GA.interpretation_lines(payload))
        self.assertIn("不产生门禁判定", joined)
        self.assertNotIn("ACTIONABLE", joined)
        self.assertNotIn("NO_ACTION", joined)


class TestLossIdentity(unittest.TestCase):
    """审计用的分解必须**就是**训练目标本身；否则整份审计无效。"""

    def test_identity_gap_is_tiny_and_recorded_once(self):
        torch.manual_seed(41)
        model = tiny_model()
        audit = GA.GradientAudit(model=model, uni_coe=P.UNI_COE, env_coe=P.ENV_COE)
        self.assertIsNone(audit.identity_gap)
        y0, y1, features = tiny_batch()
        output = forward_outputs(model, features, alpha=0.5)
        audit.record(epoch=1, step=0, alpha=0.5, output=output, y_income=y0, y_marital=y1,
                     env_ids=tiny_env_ids())
        first_gap = audit.identity_gap
        self.assertIsNotNone(first_gap)
        self.assertLess(first_gap, 1e-5)
        audit.record(epoch=1, step=1, alpha=0.5, output=output, y_income=y0, y_marital=y1,
                     env_ids=tiny_env_ids())
        self.assertEqual(audit.identity_gap, first_gap)          # 只测一次，不重复消耗

    def test_identity_can_be_disabled(self):
        audit = GA.GradientAudit(model=tiny_model(), uni_coe=P.UNI_COE, env_coe=P.ENV_COE,
                                 verify_identity=False)
        self.assertFalse(audit.verify_identity)


def audit_loaders(n_train=64, n_test=32, batch_size=16):
    """train 全量；test 按 SPLIT_SEED 切成 val（与真实路径同构：Subset + 同 batch_size）。"""
    train_ds, test_ds = TinyCensus(n_train, 3), TinyCensus(n_test, 4)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=n_train), (val_idx, test_idx)


class TestRunnerSmoke(unittest.TestCase):
    """端到端走通 run_audit（tiny + CPU），确认产物齐全且 smoke 不产生判定。"""

    def _run(self, root, **kwargs):
        loaders, stats, indices = audit_loaders()
        # 显式播种后再建模型：否则两次调用会拿到不同的全局 RNG 状态 → 模型初值不同 → 产物不可复现
        torch.manual_seed(P.MODEL_SEED)
        return run_census_gradient_audit.run_audit(
            root, epochs=2, device=torch.device("cpu"), model=tiny_model(),
            loaders=loaders, stats=stats, indices=indices, **kwargs)

    def test_full_run_writes_artifacts_and_gate_verdict(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            out = self._run(root)
            run_dir = Path(out["run_dir"])
            for name in ("audit.json", "report.md", "prereg.md", "stdout.log"):
                self.assertTrue((run_dir / name).exists(), name)
            payload = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], GA.SCHEMA)
            self.assertFalse(payload["smoke"])
            self.assertEqual(len(payload["steps"]), payload["aggregate"]["n_steps"])
            self.assertEqual(payload["meta"]["n_sampled_steps"], len(payload["steps"]))
            self.assertTrue(payload["meta"]["loss_identity_ok"])
            self.assertEqual(payload["meta"]["unclassified_parameters"], [])
            self.assertEqual(payload["gate"]["status"], "NO_ACTION")     # tiny 上必然 < 50 步
            self.assertFalse(payload["gate"]["criteria"]["G0"]["pass"])
            self.assertEqual(payload["meta"]["sampled_steps_per_epoch"],
                             len(GA.sample_schedule(payload["meta"]["n_batches"])))
            summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(summary), 3)
            self.assertIn(out["run_id"], summary[2])

    def test_smoke_run_never_produces_a_gate_verdict(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            out = self._run(root, max_batches=2)
            payload = json.loads((Path(out["run_dir"]) / "audit.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["smoke"])
            self.assertEqual(payload["gate"]["status"], GA.SMOKE_STATUS)
            self.assertFalse(payload["gate"]["actionable"])
            self.assertIsNone(payload["gate"]["criteria"])
            self.assertEqual(payload["meta"]["n_batches"], 2)
            self.assertIn("smoke", out["run_id"])

    def test_audit_json_is_byte_identical_across_reruns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            first = self._run(root, now=datetime(2026, 9, 30, 10, 0))
            second = self._run(root, now=datetime(2026, 9, 30, 10, 5))
            self.assertNotEqual(first["run_id"], second["run_id"])   # run_id 含时间戳，只在目录名
            left = (Path(first["run_dir"]) / "audit.json").read_bytes()
            right = (Path(second["run_dir"]) / "audit.json").read_bytes()
            self.assertEqual(left, right)                            # 同输入 → 逐字节一致

    def test_prereg_spec_file_exists_and_matches_constants(self):
        spec = (REPO / GA.PREREG_SPEC).read_text(encoding="utf-8")
        for token in (str(GA.MIN_SAMPLED_STEPS), str(GA.GATE_NEG_COS_RATE),
                      str(GA.GATE_MEAN_COS), str(GA.GATE_NONZERO_STEP_RATE),
                      str(GA.WARMUP_STEPS), str(GA.SPACED_STEPS)):
            self.assertIn(token, spec)
        for token in ("NO_ACTION", "shared_trunk", "income-vs-marital", "既有文献"):
            self.assertIn(token, spec)

    def test_no_checkpoint_and_no_stage2_artifacts_written(self):
        """审计分支不得产出 backbone/newtask checkpoint（不产出模型）。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            self._run(root)
            written = {path.name for path in root.rglob("*") if path.is_file()}
            self.assertIn("audit.json", written)
            self.assertNotIn("backbone.pt", written)
            self.assertNotIn("newtask.pt", written)


if __name__ == "__main__":
    unittest.main()
