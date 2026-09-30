"""条件化 Scale+Bias（消融）测试。

唯一事实来源：docs/superpowers/specs/2026-09-30-stage2-cond-scale-bias-design.md（第 5 节）。
本文件在实现之前写成（TDD：先失败），锁定七件事：
  1. 零初始化 ⇒ 与基线逐位一致（共享参数 / 前向 / 首个训练步梯度 / 全局 RNG 终点）
  2. 公式实现：γ(x)=W@Γ、β(x)=W@B、env_aware=s⊙(e_new+γ)+β（用独立循环式复算）
  3. 源 router W 与既有 scale s 原样保留（与 router_probe.router_weights、基线 forward 同源）
  4. 调制诊断定义（范数 / 有效调制比）与流式累加口径
  5. 预注册判据 C1 `AUC-Test-Education >= 0.8521`、C2 `modulation_ratio ∈ [0.01, 1.0]`（含端点）
  6. 接线：`run_stage2(variant="cond-scale-bias")` 的 run_id / config / metrics；基线臂键集不变
  7. 模块源码可被本解释器编译（3.10 下跨物理行 f-string 是语法错误，import 即崩）

定位：本方向只是**消融 / 工程诊断**（条件化调制是已有大类），不得据此主张方法新颖性。
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import run_census_benchmark
from census_benchmark import cond_scale_bias as CSB
from census_benchmark import protocol as P
from census_benchmark import router_probe
from multitaskrec.model import NewTask

# 与 test_protocol / test_smoke / test_router_probe 同口径：dnn_input 维度 = 2 个特征 × embedding_size 4 = 8
TINY_VOCAB, TINY_INPUT_SIZE, TINY_EMBEDDING, TINY_REP_DIM = {"a": 3, "b": 2}, 8, 4, 4
SOURCE = Path(__file__).resolve().parents[1] / "cond_scale_bias.py"


def tiny_backbone(device=None):
    """与 test_router_probe.tiny_backbone 同口径（dnn_input 8 维、rep_dim 4、2 个 env）。"""
    return run_census_benchmark.MPTRec(
        num_tasks=2, feature_vocabulary=dict(TINY_VOCAB), embedding_size=TINY_EMBEDDING,
        input_size=TINY_INPUT_SIZE, expert_dnn_hidden_units=(8, TINY_REP_DIM),
        tower_dnn_hidden_units=(4, 2), device=device)


def tiny_newtask(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, device=None):
    """基线头（不得改动）：超参与 stage2 一致。"""
    return NewTask(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
                   reg_dnn=P.REG_DNN, device=device)


def tiny_variant(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, device=None, num_envs=P.NUM_ENVS):
    """处理臂头：条件化 Scale+Bias（子类，不动 model.py）。"""
    return CSB.CondScaleBiasNewTask(input_size=input_size, rep_dim=rep_dim,
                                    tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN,
                                    device=device, num_envs=num_envs)


def tiny_reps(device=None, n=8, seed=3):
    """抽一批表征：(dnn_input, gen_rep, spec_reps, env_embs)，形状与真实路径一致。

    与真实路径同款"三件套之 3"：`no_grad` 下抽表征——否则表征会挂回 backbone 的计算图，
    同一批表征被两次 `backward()` 就会报 "backward through the graph a second time"。
    """
    gen = torch.Generator().manual_seed(seed)
    features = {"a": torch.randint(0, TINY_VOCAB["a"], (n,), generator=gen),
                "b": torch.randint(0, TINY_VOCAB["b"], (n,), generator=gen)}
    backbone = tiny_backbone(device)
    with torch.no_grad():
        return backbone.get_infos(features)


def with_shared_weights_baseline(variant, device=None):
    """构造一个与 variant 共享全部权重（γ/β 为零初始化的原样）的基线头，用于逐位比对。"""
    baseline = tiny_newtask(device=device)
    baseline.load_state_dict({name: value for name, value in variant.state_dict().items()
                              if name in baseline.state_dict()})
    return baseline


def dezero_variant(variant, *, seed=7):
    """把 γ/β 从零初始化推到非零（模拟训练后），并给共享参数一点扰动，使测试非平凡。"""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        variant.gamma.copy_(torch.randn(variant.gamma.shape, generator=gen) * 0.1)
        variant.beta.copy_(torch.randn(variant.beta.shape, generator=gen) * 0.1)
    return variant


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


def tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


def run_tiny_stage1(root, device):
    loaders, stats, indices = tiny_inputs()
    stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_backbone(device),
                                             loaders=loaders, stats=stats, indices=indices, tag="short")
    return stage1, loaders, stats, indices


class TestModuleSourceCompiles(unittest.TestCase):
    """静态守卫：源码必须能被本解释器编译（最便宜的用例，先跑它；venv 是 3.10）。"""

    def test_source_compiles(self):
        self.assertTrue(SOURCE.is_file(), f"未找到被测源码: {SOURCE}")
        compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec")


class TestPreregisteredCriteria(unittest.TestCase):
    """预注册判据（spec 5.3）——只允许在看到结果之前修改。"""

    def test_threshold_values(self):
        self.assertAlmostEqual(CSB.AUC_TEST_MIN, 0.8521, places=12)
        self.assertEqual(tuple(CSB.MODULATION_RATIO_BOUNDS), (0.01, 1.0))
        self.assertEqual(CSB.VARIANT, "cond-scale-bias")
        self.assertEqual(CSB.BASELINE_VARIANT, "baseline")
        self.assertEqual(CSB.RUN_ID_SUFFIX, "-csb")

    def test_c1_inclusive_boundary(self):
        self.assertTrue(CSB.arm_verdict(0.8521, 0.5)["C1"]["pass"])          # 含等号
        self.assertTrue(CSB.arm_verdict(0.8522, 0.5)["C1"]["pass"])
        self.assertFalse(CSB.arm_verdict(0.850069, 0.5)["C1"]["pass"])       # 基线实测值不达标

    def test_c2_band_inclusive_boundaries(self):
        for ratio in (0.01, 0.5, 1.0):
            self.assertTrue(CSB.modulation_ratio_verdict(ratio)["pass"], f"带内应通过: {ratio}")
        for ratio in (0.0099, 1.0001, 0.0, float("inf")):
            self.assertFalse(CSB.modulation_ratio_verdict(ratio)["pass"], f"带外应失败: {ratio}")

    def test_arm_verdict_requires_both_conditions(self):
        both = CSB.arm_verdict(0.86, 0.3)
        self.assertTrue(both["pass"])
        self.assertFalse(CSB.arm_verdict(0.86, 0.005)["pass"])               # C2 出带
        self.assertFalse(CSB.arm_verdict(0.85, 0.3)["pass"])                 # C1 未达

    def test_verdict_payload_is_json_serialisable(self):
        verdict = CSB.arm_verdict(0.80, 0.02)
        self.assertEqual(verdict["C1"]["rule"], "auc_test_education >= 0.8521")
        self.assertEqual(verdict["C1"]["threshold"], 0.8521)
        self.assertAlmostEqual(verdict["C1"]["observed"], 0.80, places=12)
        self.assertEqual(verdict["C2"]["rule"], "0.01 <= modulation_ratio <= 1.0")
        self.assertEqual(verdict["C2"]["bounds"], [0.01, 1.0])
        self.assertAlmostEqual(verdict["C2"]["observed"], 0.02, places=12)
        self.assertIsInstance(verdict["pass"], bool)
        json.dumps(verdict)                                                   # 必须可落盘


class TestZeroInitIdentity(unittest.TestCase):
    """零初始化 ⇒ 初始模型与基线**逐位一致**（spec 5.1 恒等回退）。

    这是本消融的立身之本：初始状态若与基线有任何差别，"提升"就无法归因到条件化本身。
    """

    def test_shared_params_and_rng_match_baseline_construction(self):
        device = torch.device("cpu")
        P.seed_model(P.MODEL_SEED)                       # stage2 顺序：seed → backbone → 新任务头
        tiny_backbone(device)
        baseline = tiny_newtask()
        rng_after_baseline = torch.get_rng_state().clone()

        P.seed_model(P.MODEL_SEED)
        tiny_backbone(device)
        variant = tiny_variant()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_after_baseline),
                        "γ/β 的构造消耗了全局 RNG——初始化不再与基线同序")

        got, want = variant.state_dict(), baseline.state_dict()
        self.assertEqual(sorted(set(got) - set(want)), ["beta", "gamma"])     # 唯一新增量
        self.assertEqual(sorted(set(want) - set(got)), [])
        for name in sorted(want):
            self.assertTrue(torch.equal(got[name], want[name]), f"共享参数偏离基线初始化: {name}")
        for name in ("gamma", "beta"):
            self.assertEqual(tuple(got[name].shape), (P.NUM_ENVS, TINY_REP_DIM))
            self.assertTrue(torch.equal(got[name], torch.zeros_like(got[name])), f"{name} 必须零初始化")

    def test_forward_is_bit_identical_to_baseline_at_init(self):
        device = torch.device("cpu")
        variant = tiny_variant(device=device)
        baseline = with_shared_weights_baseline(variant, device=device)
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)
        with torch.no_grad():
            got = variant(dnn_input, gen_rep, spec_reps, env_embs)
            want = baseline(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertTrue(torch.equal(got, want), "零初始化下前向与基线不逐位一致")

    def test_first_optimizer_step_keeps_shared_params_identical_to_baseline(self):
        """更强的一致性：γ=β=0 时首个训练步对**共享参数**的梯度也逐位相同（训练轨迹同起点）。"""
        device = torch.device("cpu")
        variant, baseline = tiny_variant(device=device), None
        baseline = with_shared_weights_baseline(variant, device=device)
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)
        labels = torch.randint(0, 2, (dnn_input.shape[0],)).float()
        loss_func = torch.nn.BCELoss()

        for module in (variant, baseline):
            module.train()
            module.zero_grad()
            pred = module(dnn_input, gen_rep, spec_reps, env_embs)
            (loss_func(pred, labels) + module.get_l2_reg()).backward()

        shared = [name for name in baseline.state_dict() if name in dict(baseline.named_parameters())]
        for name in shared:
            got = dict(variant.named_parameters())[name].grad
            want = dict(baseline.named_parameters())[name].grad
            self.assertIsNotNone(got, f"共享参数无梯度: {name}")
            self.assertTrue(torch.equal(got, want), f"首个训练步梯度偏离基线: {name}")

        # 新增量必须真的吃梯度（否则"可学习"是假的）
        for name in ("gamma", "beta"):
            grad = dict(variant.named_parameters())[name].grad
            self.assertIsNotNone(grad, f"{name} 未参与计算图")
            self.assertFalse(torch.equal(grad, torch.zeros_like(grad)), f"{name} 梯度恒为零")


class TestFormula(unittest.TestCase):
    """spec 5.1：γ(x)=W@Γ、β(x)=W@B、env_aware=s⊙(e_new+γ)+β；W 与 s 原样保留。"""

    def test_router_weights_preserve_the_source_router(self):
        """源 router = 基线表达式（与前置诊断 router_probe.router_weights 同一式），逐位一致。"""
        device = torch.device("cpu")
        variant = dezero_variant(tiny_variant(device=device))
        baseline = with_shared_weights_baseline(variant, device=device)
        dnn_input, _, _, env_embs = tiny_reps(device)

        with torch.no_grad():
            got = variant.router_weights(dnn_input, env_embs)
            want = F.softmax(torch.mm(baseline.projection_network(dnn_input),
                                      torch.stack(env_embs, dim=1)) / baseline.temperature, dim=-1)
            probed = router_probe.router_weights(baseline, dnn_input, env_embs)
        self.assertTrue(torch.equal(got, want), "router W 与基线表达式不一致")
        self.assertTrue(torch.equal(got, probed), "router W 与前置诊断复算式不一致")
        self.assertTrue(torch.allclose(got.sum(dim=-1), torch.ones(got.shape[0]), atol=1e-6))

        with torch.no_grad():                                                 # 温度必须取自模块本身
            variant.temperature = 7
            hotter = variant.router_weights(dnn_input, env_embs)
        self.assertFalse(torch.equal(got, hotter), "router 未读取 NewTask.temperature")

    def test_modulation_terms_match_hand_computation(self):
        device = torch.device("cpu")
        variant = dezero_variant(tiny_variant(device=device))
        dnn_input, _, spec_reps, env_embs = tiny_reps(device)

        with torch.no_grad():
            terms = variant.modulation_terms(dnn_input, spec_reps, env_embs)
            w = F.softmax(torch.mm(variant.projection_network(dnn_input),
                                   torch.stack(env_embs, dim=1)) / variant.temperature, dim=-1)
            # 独立复算：逐 env 循环求和（实现用矩阵乘，两者应数值一致）
            want_e_new = sum(w[:, k].unsqueeze(1) * spec_reps[k] for k in range(P.NUM_ENVS))
            want_gamma = sum(w[:, k].unsqueeze(1) * variant.gamma[k] for k in range(P.NUM_ENVS))
            want_beta = sum(w[:, k].unsqueeze(1) * variant.beta[k] for k in range(P.NUM_ENVS))

        self.assertTrue(torch.equal(terms["w"], w))
        for got, want, name in ((terms["e_new"], want_e_new, "e_new"),
                                (terms["gamma_x"], want_gamma, "gamma_x"),
                                (terms["beta_x"], want_beta, "beta_x")):
            self.assertTrue(torch.allclose(got, want, atol=1e-6), f"{name} 与手算不一致")

    def test_forward_is_the_preserved_pipeline_on_env_aware(self):
        """前向 = 既有 scale s ⊙ (e_new+γ) + β，再走既有 gate 融合与 tower（不引入任何其它改动）。"""
        device = torch.device("cpu")
        variant = dezero_variant(tiny_variant(device=device))
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)
        with torch.no_grad():
            terms = variant.modulation_terms(dnn_input, spec_reps, env_embs)
            scale = variant.env_embedding_network(variant.new_env_idx).squeeze(0)          # 既有 s
            env_aware = scale * (terms["e_new"] + terms["gamma_x"]) + terms["beta_x"]
            gate_out = variant.gate_network(dnn_input).unsqueeze(dim=2)                    # 既有 gate
            fused = torch.matmul(torch.stack([env_aware, gen_rep], dim=2), gate_out).squeeze()
            want = variant.tower_network(fused).squeeze()                                  # 既有 tower
            got = variant(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertTrue(torch.equal(got, want), "前向不是既有管线 + γ/β 调制的组合")

    def test_gamma_beta_actually_change_the_output(self):
        """非平凡性：γ/β 离开零点必须改变输出；再归零必须回到基线。"""
        device = torch.device("cpu")
        variant = tiny_variant(device=device)
        variant.eval()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)
        with torch.no_grad():
            at_zero = variant(dnn_input, gen_rep, spec_reps, env_embs)
            dezero_variant(variant)
            modulated = variant(dnn_input, gen_rep, spec_reps, env_embs)
            variant.gamma.zero_()
            variant.beta.zero_()
            back_to_zero = variant(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertFalse(torch.equal(at_zero, modulated), "γ/β 未接入前向")
        self.assertTrue(torch.equal(at_zero, back_to_zero), "归零后未回到基线行为")

    def test_env_count_mismatch_is_rejected(self):
        """env 数与本模块的 γ/β 形状不符 → 拒绝出数（同 probe_loader 的口径）。"""
        device = torch.device("cpu")
        variant = tiny_variant(device=device)
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)
        wrong = env_embs + [torch.zeros(TINY_REP_DIM)]
        with self.assertRaises(ValueError) as ctx:
            variant(dnn_input, gen_rep, spec_reps, wrong)
        self.assertIn("env", str(ctx.exception))


class TestModulationStats(unittest.TestCase):
    """spec 5.2：范数与有效调制比的定义、流式累加、退化输入。"""

    @staticmethod
    def terms(gamma_x, beta_x, e_new):
        return {"gamma_x": torch.tensor(gamma_x, dtype=torch.float32),
                "beta_x": torch.tensor(beta_x, dtype=torch.float32),
                "e_new": torch.tensor(e_new, dtype=torch.float32)}

    def test_hand_computed_norms_and_ratio(self):
        stats = CSB.ModulationStats()
        stats.update(self.terms(gamma_x=[[3.0, 4.0], [0.0, 0.0]],                    # 范数 5, 0 → 均值 2.5
                                beta_x=[[0.0, 0.0], [0.0, 3.0]],                     # 范数 0, 3 → 均值 1.5
                                e_new=[[1.0, 0.0], [2.0, 0.0]]))                     # 范数 1, 2 → 均值 1.5
        got = stats.result()
        self.assertAlmostEqual(got["gamma_x_norm_mean"], 2.5, places=6)
        self.assertAlmostEqual(got["beta_x_norm_mean"], 1.5, places=6)
        self.assertAlmostEqual(got["e_new_norm_mean"], 1.5, places=6)
        self.assertAlmostEqual(got["modulation_ratio"], 2.5 / 1.5, places=6)
        self.assertAlmostEqual(got["bias_ratio"], 1.0, places=6)
        self.assertEqual(got["n_samples"], 2)

    def test_streaming_update_equals_single_batch(self):
        first = self.terms(gamma_x=[[3.0, 4.0]], beta_x=[[0.0, 1.0]], e_new=[[1.0, 0.0]])
        second = self.terms(gamma_x=[[0.0, 0.0], [0.0, 6.0]], beta_x=[[2.0, 0.0], [0.0, 0.0]],
                            e_new=[[3.0, 0.0], [0.0, 4.0]])
        streamed = CSB.ModulationStats()
        streamed.update(first)
        streamed.update(second)
        single = CSB.ModulationStats()
        merged = {key: torch.cat([first[key], second[key]]) for key in first}
        single.update(merged)
        self.assertEqual(streamed.result(), single.result())

    def test_degenerate_zero_e_new_ratio(self):
        zero = CSB.ModulationStats()
        zero.update(self.terms(gamma_x=[[0.0, 0.0]], beta_x=[[0.0, 0.0]], e_new=[[0.0, 0.0]]))
        self.assertAlmostEqual(zero.result()["modulation_ratio"], 0.0, places=12)     # 0/0 → 0.0

        degenerate = CSB.ModulationStats()
        degenerate.update(self.terms(gamma_x=[[1.0, 0.0]], beta_x=[[0.0, 0.0]], e_new=[[0.0, 0.0]]))
        self.assertEqual(degenerate.result()["modulation_ratio"], float("inf"))       # 有调制无表征 → 判失败

    def test_param_norms_per_env(self):
        variant = tiny_variant()
        with torch.no_grad():
            variant.gamma.copy_(torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
            variant.beta.copy_(torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 12.0]]))
        got = CSB.param_norms(variant)
        self.assertAlmostEqual(got["gamma_param_norm_per_env"][0], 5.0, places=6)
        self.assertAlmostEqual(got["gamma_param_norm_per_env"][1], 0.0, places=6)
        self.assertAlmostEqual(got["beta_param_norm_per_env"][1], 12.0, places=6)
        json.dumps(got)                                                       # 必须可落盘

    def test_result_is_json_serialisable_and_count_zero_safe(self):
        empty = CSB.ModulationStats()
        json.dumps(empty.result())                                            # 未 update 也不得炸
        self.assertEqual(empty.result()["n_samples"], 0)


class TestEvaluateModulation(unittest.TestCase):
    """诊断在选点之后、只用 val 跑一遍 no_grad 前向：不改参数、不训练、不参与任何选择。"""

    def test_matches_recomputation_and_has_no_side_effects(self):
        device = torch.device("cpu")
        variant = dezero_variant(tiny_variant(device=device))
        backbone = tiny_backbone(device)
        loaders, _, _ = tiny_inputs()
        before = {name: value.detach().clone() for name, value in variant.state_dict().items()}
        sha_before = P.backbone_sha256(backbone)

        got = CSB.evaluate_modulation(variant, backbone, loaders["val"], device)

        stats = CSB.ModulationStats()
        for _, _, _, features in loaders["val"]:
            dnn_input, _, spec_reps, env_embs = backbone.get_infos(features)
            stats.update(variant.modulation_terms(dnn_input, spec_reps, env_embs))
        for key, value in stats.result().items():
            self.assertAlmostEqual(got[key], value, places=9, msg=f"诊断口径不一致: {key}")
        for key, value in CSB.param_norms(variant).items():
            self.assertEqual(got[key], value)

        for name, value in variant.state_dict().items():                      # 参数逐位不变
            self.assertTrue(torch.equal(before[name], value), f"诊断改动了参数: {name}")
        self.assertEqual(P.backbone_sha256(backbone), sha_before)
        P.assert_no_grads(backbone)
        self.assertFalse(variant.training)

    def test_baseline_newtask_is_rejected(self):
        """诊断只对处理臂有意义：喂基线头必须报错，而不是静默出数。"""
        device = torch.device("cpu")
        loaders, _, _ = tiny_inputs()
        with self.assertRaises(ValueError):
            CSB.evaluate_modulation(tiny_newtask(device=device), tiny_backbone(device), loaders["val"], device)


class TestVariantFactory(unittest.TestCase):
    def test_baseline_variant_is_the_untouched_newtask(self):
        head = CSB.build_newtask(CSB.BASELINE_VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                 tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        self.assertIs(type(head), NewTask)                                    # 不是子类，就是基线类本身

    def test_treatment_variant_is_the_subclass(self):
        head = CSB.build_newtask(CSB.VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                 tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        self.assertIsInstance(head, CSB.CondScaleBiasNewTask)
        self.assertIsInstance(head, NewTask)
        self.assertIn("gamma", head.state_dict())

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            CSB.build_newtask("no_such_variant", input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                              tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)


class TestRunnerIntegration(unittest.TestCase):
    """接线（spec 5.5）：标准 short 命令 + `--variant` 即跑本消融，并记录 config/metrics。"""

    def test_treatment_arm_records_variant_and_modulation(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2, device=device,
                                                  model=tiny_backbone(device), loaders=loaders, stats=stats,
                                                  indices=indices, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                                  variant=CSB.VARIANT)
            run_path = Path(out["run_dir"])
            self.assertTrue(out["run_id"].endswith(CSB.RUN_ID_SUFFIX), out["run_id"])

            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], CSB.VARIANT)
            self.assertEqual(config["num_envs"], P.NUM_ENVS)

            payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["variant"], CSB.VARIANT)
            for key in ("gamma_x_norm_mean", "beta_x_norm_mean", "e_new_norm_mean", "modulation_ratio",
                        "bias_ratio", "gamma_param_norm_per_env", "beta_param_norm_per_env", "n_samples"):
                self.assertIn(key, payload["mechanism"], f"机制诊断缺字段: {key}")
            self.assertEqual(len(payload["mechanism"]["gate_mean"]), 2)       # 基线 M3 仍在
            self.assertEqual(payload["mechanism"]["n_samples"], 16)           # = val 集大小
            # 训练后 γ/β 必须真的离开零点（零初始化本身由 TestZeroInitIdentity 锁定）
            self.assertGreater(payload["mechanism"]["gamma_param_norm_per_env"][0], 0.0)
            self.assertGreater(payload["mechanism"]["beta_param_norm_per_env"][0], 0.0)
            arm = payload["csb_arm"]
            self.assertEqual(arm["C1"]["threshold"], 0.8521)
            self.assertEqual(arm["C2"]["bounds"], [0.01, 1.0])
            self.assertIsInstance(arm["pass"], bool)

            report = json.loads((run_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["A1"]["pass"])                             # 冻结三件套不受影响
            self.assertEqual(report["A3"]["status"], "on_demand")
            self.assertNotIn("pass", report["A3"])                            # 臂级判定不混入 A/B 门禁
            self.assertNotIn("csb_arm", report)

            state = torch.load(run_path / "newtask.pt", map_location="cpu")
            head = tiny_variant(device=device)
            head.load_state_dict(state)                                       # strict：γ/β 必须在 checkpoint 里
            self.assertIn("gamma", state)
            self.assertIn("beta", state)

    def test_baseline_arm_is_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2, device=device,
                                                  model=tiny_backbone(device), loaders=loaders, stats=stats,
                                                  indices=indices, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM)
            run_path = Path(out["run_dir"])
            self.assertFalse(out["run_id"].endswith(CSB.RUN_ID_SUFFIX))

            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], CSB.BASELINE_VARIANT)
            self.assertNotIn("num_envs", config)                              # 处理臂专属字段不得出现

            payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["variant"], CSB.BASELINE_VARIANT)
            self.assertNotIn("csb_arm", payload)
            self.assertNotIn("modulation_ratio", json.dumps(payload))
            self.assertEqual(sorted(payload["mechanism"]),
                             ["cos_gen_spec", "env_acc_stage1", "gate_mean", "gen_std"])   # M3/M4 键集不变
            state = torch.load(run_path / "newtask.pt", map_location="cpu")
            self.assertNotIn("gamma", state)                                  # 基线 checkpoint 不含新增量
            head = tiny_newtask(device=device)
            head.load_state_dict(state)                                       # strict 载入

    def test_cli_exposes_variant_with_baseline_default(self):
        parser = run_census_benchmark.build_parser()
        args = parser.parse_args(["stage2", "--stage1-dir", "x"])
        self.assertEqual(args.variant, CSB.BASELINE_VARIANT)
        args = parser.parse_args(["stage2", "--stage1-dir", "x", "--variant", CSB.VARIANT])
        self.assertEqual(args.variant, CSB.VARIANT)
        with self.assertRaises(SystemExit):
            parser.parse_args(["stage2", "--stage1-dir", "x", "--variant", "nope"])


if __name__ == "__main__":
    unittest.main()
