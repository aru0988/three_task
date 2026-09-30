"""逐样本 CGR（per-sample confidence-gated routing）修正实现的测试。

唯一事实来源：docs/superpowers/experiments/2026-09-30-stage2-cgr-instance.md
上游协议：docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md（划分 / 种子 / 冻结 / A-B 门禁零改动）
被修正的对象：exp/stage2-cgr-repro 的忠实复现（判定 CONFIRMED_DEGENERATE，三条退化判据全部命中）

本文件在实现之前写成（TDD：先失败），锁定七件事：

  1. **D1 修复——逐样本门控**：门控输入只含逐样本量（H_out、attention 置信度、agreement），
     同一样本在任意批上下文下 g 恒定（批级门控下会随批变化）；
     静态守卫：源码不得出现历史批级统计量表达式。
  2. **D2 修复——构造顺序**：`super().__init__()` 先行 ⇒ 同一 model seed 下所有**共享**参数
     （projection / gate / tower / env_embedding）与基线逐位相同；额外参数在共享参数之后才消耗 RNG。
  3. **D3 修复——无常量行对称、无正偏置饱和**：隐层标准初始化（行互不相同）、
     门控输出层零初始化 ⇒ z ≡ 0 ⇒ g ≡ 0.5（精确、非饱和、sigmoid′(0) = 0.25）。
  4. **无 warmup**：g 就是 σ(z)，不存在 alpha 插值（历史 D4 的时序怪癖在结构上不可能出现）。
  5. **D6 修复——B=1 安全**：逐样本量、无批内方差；forward 用 `squeeze(-1)` 而非 `squeeze()`，
     单样本批不 NaN 且形状正确（基线在此处形状报错，作为对照留痕）。
     "B=1 能跑通"（形状 / 有限 / 反传走通）与"B=1 可学"（梯度非零）是两个测试：后者需要
     **注意力非均匀**这一前置条件，见下条。
  6. **机制可学**：门控输出层在初始步即收到非零梯度；一步优化后门控全部参数梯度非零。
     前置条件纪律（本文件第二版修正）：`W = g·W_attn + (1−g)·W_fw` ⇒ `dW/dg = W_attn − W_fw`；
     注意力塌到均匀（投影瓶颈全灭 ⇒ H_out ≡ 0 ⇒ W_attn 逐位 = W_fw）时 `dL/dg ≡ 0`，
     门控**全部**参数梯度**精确**为 0——这是混合权重的结构性质，不是实现缺陷。
     随机夹具采到全灭瓶颈并不罕见（B=1 夹具扫描 60 个全局 RNG 状态，20 个如此），
     所以凡断言"梯度非零"的地方都用显式播种 / 构造性夹具，并先验收 `W_attn != W_fw`。
  7. **接线**：`--variant cgr-instance` / `--cgr-instance`、run_id 后缀 `-cgrinst`、
     `metrics.json:cgr_instance_arm`；**基线臂逐键不变**（构造不建处理头、payload 键集与改动前相同）。

预注册判据（**只允许在看到结果之前修改**）：

    机制门（任一不满足 ⇒ MECHANISM_FAILED ⇒ STOP，与 AUC 无关）：
      g_within_batch_std_mean >= 0.01   且   grad_norm > 0
      0.05 <= g_mean <= 0.95            且   weight_l1_mean >= 1e-4
    效应门（仅机制通过时才解释）：auc_test_education >= 0.8521（基线 0.8500685307175756）

定位：这不是新方法。置信门控 / 条件化路由 / 门控融合（MoE 门控、FiLM、prompt 调制）是已有大类，
本臂只是一次**修正后的工程消融**——把 exp/stage2-cgr-repro 审计出的表格缺陷逐条修好后重跑一次。
"""
import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import run_census_benchmark
from census_benchmark import cgr_instance as CGI
from census_benchmark import protocol as P
from multitaskrec.model import NewTask

# 与 test_protocol / test_smoke / test_cgr 同口径：dnn_input = 2 特征 × embedding 4 = 8
TINY_VOCAB, TINY_INPUT_SIZE, TINY_EMBEDDING, TINY_REP_DIM = {"a": 3, "b": 2}, 8, 4, 4
TOWER_HIDDEN = list(P.TOWER_HIDDEN)
SOURCE = Path(__file__).resolve().parents[1] / "cgr_instance.py"
REPO = Path(__file__).resolve().parents[2]
BRANCH_BASE = "infra/fair-stage2-benchmark"      # 本分支的基点（冻结文件的对照基准，见静态守卫）


def tiny_backbone(device=None):
    """dnn_input 8 维、rep_dim 4、2 个 env（与其它测试文件同一夹具口径）。"""
    return run_census_benchmark.MPTRec(
        num_tasks=2, feature_vocabulary=dict(TINY_VOCAB), embedding_size=TINY_EMBEDDING,
        input_size=TINY_INPUT_SIZE, expert_dnn_hidden_units=(8, TINY_REP_DIM),
        tower_dnn_hidden_units=(4, 2), device=device)


def tiny_newtask(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, device=None):
    """基线头（不得改动）。"""
    return NewTask(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=TOWER_HIDDEN,
                   reg_dnn=P.REG_DNN, device=device)


def tiny_head(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, device=None, gate_hidden=CGI.GATE_HIDDEN):
    """处理臂头：CGRInstanceNewTask（NewTask 子类，不动 model.py）。"""
    return CGI.CGRInstanceNewTask(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=TOWER_HIDDEN,
                                  reg_dnn=P.REG_DNN, device=device, gate_hidden=gate_hidden)


def tiny_reps(device=None, n=8, seed=3, tasks=2):
    """抽一批表征：(dnn_input, gen_rep, spec_reps, env_embs)——与真实路径同款 no_grad 抽取。"""
    gen = torch.Generator().manual_seed(seed)
    features = {"a": torch.randint(0, TINY_VOCAB["a"], (n,), generator=gen),
                "b": torch.randint(0, TINY_VOCAB["b"], (n,), generator=gen)}
    with torch.no_grad():
        dnn_input, gen_rep, spec_reps, env_embs = tiny_backbone(device).get_infos(features)
    return dnn_input, gen_rep, spec_reps[:tasks], env_embs[:tasks]


def pick(reps, indices):
    """取给定索引构成的批（保持批维度）。

    **必须从同一份 reps 里取**：`tiny_reps` 每次调用都会新建一个 backbone，而 `MPTRec.__init__`
    消耗全局 RNG ⇒ 两次 `tiny_reps` 得到的是**两个不同初始化的 backbone**，直接比对会引入
    与批上下文无关的巨大差异（这正是本文件第一版的一个真实缺陷）。
    """
    dnn_input, gen_rep, spec_reps, env_embs = reps
    selection = list(indices)
    return (dnn_input[selection], gen_rep[selection], [s[selection] for s in spec_reps], env_embs)


def take(reps, index):
    """取第 index 个样本，保持批维度 = 1（单样本批：D6 的靶场）。"""
    return pick(reps, [index])


def live_head(head, seed=11, scale=0.5):
    """把门控置成确定性的、非退化的映射（打破零初始化），只用于"门控随样本变化"类断言。

    这不是改实现：零初始化下 z ≡ 0 ⇒ g ≡ 0.5（逐样本也不变，属**预期**行为），
    要断言"门控具备逐样本表达力"就必须先给它一组非零权重。
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        w1 = head.confidence_mlp[0].weight
        w1.copy_(torch.randn(w1.shape, generator=gen) * scale)
        head.confidence_mlp[0].bias.copy_(torch.randn(head.confidence_mlp[0].bias.shape, generator=gen) * scale)
        w2 = head.confidence_mlp[2].weight
        w2.copy_(torch.randn(w2.shape, generator=gen) * scale)
        head.confidence_mlp[2].bias.zero_()
    return head


def force_g(head, value):
    """把门控输出层偏置推到 ±1e6 ⇒ σ(z) 在 float32 下精确等于 1.0 / 0.0（用于端点等价性断言）。"""
    with torch.no_grad():
        head.confidence_mlp[2].weight.zero_()
        head.confidence_mlp[2].bias.fill_(1e6 if value == 1.0 else -1e6)
    return head


FIXTURE_SEED = 20260930


def seeded_head(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, device=None, gate_hidden=CGI.GATE_HIDDEN):
    """显式播种的处理头夹具：初值只由 `FIXTURE_SEED` 决定，与执行顺序、前序 RNG 消耗无关。

    `tiny_head()` **故意**不自己播种（`TestHeadConstruction` 要观察它对全局 RNG 的消耗量），
    所以凡需要"确定性初值"的测试都走这一层，而不是依赖运行时恰好排在前面的测试。
    """
    torch.manual_seed(FIXTURE_SEED)
    return tiny_head(input_size=input_size, rep_dim=rep_dim, device=device, gate_hidden=gate_hidden)


def deterministic_reps(n=8, tasks=2, seed=FIXTURE_SEED):
    """完全由夹具给定的四元表征（形状与 `tiny_reps` 同构，但不经过随机骨干）。

    `tiny_reps` 每次调用都新建骨干，而骨干初值取自全局 RNG ⇒ 表征本身也依赖执行顺序。
    梯度类断言要的是**受控输入**，故这里用局部 generator 直接给出确定性张量。

    环境向量取标准基：配 `force_live_attention` 得到的 `H_out = [√3, −1/√3, …]`，注意力 logits
    是 `[√3, −1/√3]/T`，间隔固定 ≈ 0.0154 ⇒ `|W − W_attn|` 有可观幅度，不会出现"logits 恰好抵消、
    混合≈恒等、幅度阈值变成毛刺"的情况。前置条件仍要验收（`assert_attention_is_live`）。
    """
    gen = torch.Generator().manual_seed(seed)
    env_embs = []
    for index in range(tasks):
        vec = torch.zeros(TINY_REP_DIM)
        vec[index % TINY_REP_DIM] = 1.0
        env_embs.append(vec)
    return (torch.randn(n, TINY_INPUT_SIZE, generator=gen),
            torch.randn(n, TINY_REP_DIM, generator=gen),
            [torch.randn(n, TINY_REP_DIM, generator=gen) for _ in range(tasks)],
            env_embs)


def force_live_attention(head, dnn_input):
    """把投影网络写成**构造上必然存活**的确定性状态：注意力非均匀不再取决于 seed 运气。

    瓶颈第 0 行 = 批内首个样本的单位方向 ⇒ 该样本预激活 = ‖x‖ > 0，ReLU 必然通过；
    第二层只留 (0, 0) = 1 ⇒ H_out = LayerNorm([r, 0, …, 0])，与 r 的尺度无关（LayerNorm 归一化）。
    刻意不掷骰子：随机初值下约 1/3 的 RNG 状态会让瓶颈全灭，那时 `W_attn ≡ W_fw`、
    门控梯度**精确**为 0（对照测试 `test_collapsed_attention_is_a_gradient_fixed_point`）。
    """
    with torch.no_grad():
        bottleneck = head.projection_network[0].weight           # [rep_dim // 2, input_size]，无 bias
        bottleneck.zero_()
        bottleneck[0].copy_(dnn_input[0] / dnn_input[0].norm())
        second = head.projection_network[2].weight               # [rep_dim, rep_dim // 2]，无 bias
        second.zero_()
        second[0, 0] = 1.0
    return head


def attention_is_uniform(terms):
    """读数：W_attn 是否逐位等于均匀权重 W_fw（塌到均匀不动点 ⇒ dW/dg ≡ 0）。"""
    return bool(torch.equal(terms["w_attn"], terms["w_fw"].expand_as(terms["w_attn"])))


def assert_attention_is_live(case, head, reps):
    """前置条件断言：注意力未塌到均匀不动点；返回门控中间量供调用方复用。

    `W = g·W_attn + (1−g)·W_fw` ⇒ `dW/dg = W_attn − W_fw`；两者逐位相等时 `dL/dg ≡ 0`，
    于是 `dL/dz = dL/dg·σ′(z) ≡ 0`，门控全部参数梯度精确为 0。也就是说：夹具一旦采到
    全灭的投影瓶颈，"grad > 0" 在数学上就不可能成立。先验收前置条件，失败信息才不会指错人。
    """
    terms = head.gate_terms(*reps)
    case.assertFalse(attention_is_uniform(terms),
                     "夹具退化：W_attn 逐位均匀 ⇒ dW/dg ≡ 0 ⇒ 门控梯度必然为 0；"
                     "此失败属夹具问题，不是实现问题（对照见 test_collapsed_attention_is_a_gradient_fixed_point）")
    return terms


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
    """train_ds 是「全量训练集」，test_ds 按 P.SPLIT_SEED 切成 val / test，与真实路径同构。"""
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


def _git(*args):
    import subprocess
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


class TestModuleSourceCompiles(unittest.TestCase):
    def test_source_compiles(self):
        import py_compile
        self.assertTrue(SOURCE.exists(), f"模块必须存在: {SOURCE}")
        py_compile.compile(str(SOURCE), doraise=True)

    def test_no_batch_global_reductions_in_source(self):
        """D1 静态守卫：历史批级统计量的表达式不得出现（行为守卫见 TestPerSampleGate）。"""
        src = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("gen_rep.mean", "gen_rep.var", "exist_env_embs.mean", "exist_env_embs.var",
                          "env_embs.mean", "env_embs.var", ".expand("):
            self.assertNotIn(forbidden, src, f"门控不得使用批级统计量: {forbidden}")

    def test_no_warmup_machinery_in_source(self):
        """无 warmup：不得有 alpha 插值的任何残留（D4 在结构上不可能出现）。"""
        src = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("gate_warmup_alpha", "warmup_epochs", "set_warmup_alpha"):
            self.assertNotIn(forbidden, src)


class TestPreregisteredCriteria(unittest.TestCase):
    def test_threshold_values(self):
        self.assertEqual(CGI.AUC_TEST_MIN, 0.8521)
        self.assertEqual(CGI.BASELINE_TEST_AUC, 0.8500685307175756)
        self.assertEqual(CGI.GATE_STD_MIN, 0.01)
        self.assertEqual(CGI.GRAD_NORM_MIN, 0.0)
        self.assertEqual((CGI.G_MEAN_MIN, CGI.G_MEAN_MAX), (0.05, 0.95))
        self.assertEqual(CGI.WEIGHT_L1_MIN, 1e-4)
        self.assertFalse(CGI.WARMUP)

    def test_rule_texts_are_the_preregistered_originals(self):
        self.assertEqual(CGI.RULE_GATE_STD, "g_within_batch_std_mean >= 0.01")
        self.assertEqual(CGI.RULE_GRAD_NORM, "grad_norm > 0")
        self.assertEqual(CGI.RULE_G_MEAN, "0.05 <= g_mean <= 0.95")
        self.assertEqual(CGI.RULE_WEIGHT_L1, "weight_l1_mean >= 1e-4")
        self.assertEqual(CGI.RULE_EFFECT, "auc_test_education >= 0.8521")

    def test_prereg_snapshot_is_json_serialisable(self):
        snapshot = CGI.preregistered_criteria()
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)
        self.assertEqual(snapshot["rules"]["mechanism"],
                         [CGI.RULE_GATE_STD, CGI.RULE_GRAD_NORM, CGI.RULE_G_MEAN, CGI.RULE_WEIGHT_L1])
        self.assertEqual(snapshot["rules"]["effect"], CGI.RULE_EFFECT)
        self.assertFalse(snapshot["warmup"])

    def test_provenance_names_the_audited_defects_and_the_spec(self):
        prov = CGI.provenance()
        self.assertEqual(json.loads(json.dumps(prov)), prov)
        for defect in ("D1", "D2", "D3", "D6"):
            self.assertIn(defect, prov["fixes"])
        self.assertIn("2026-09-30-stage2-cgr-instance.md", prov["spec"])
        self.assertIn("stage2-cgr-repro", prov["corrective_source"])
        self.assertEqual(prov["gate_input"], "per_sample")
        self.assertEqual(prov["warmup"], "removed")


class TestMechanismVerdict(unittest.TestCase):
    OK = dict(g_std=0.05, grad_norm=1e-3, g_mean=0.5, weight_l1=0.05)

    def test_all_mechanism_gates_pass(self):
        verdict = CGI.mechanism_verdict(**self.OK)
        self.assertEqual(verdict["status"], "MECHANISM_OK")
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["failed_rules"], [])

    def test_gate_std_boundary_is_inclusive(self):
        self.assertEqual(CGI.mechanism_verdict(**{**self.OK, "g_std": 0.01})["status"], "MECHANISM_OK")
        failed = CGI.mechanism_verdict(**{**self.OK, "g_std": 0.0099})
        self.assertEqual(failed["status"], "MECHANISM_FAILED")
        self.assertIn(CGI.RULE_GATE_STD, failed["failed_rules"])

    def test_grad_norm_must_be_strictly_positive(self):
        self.assertEqual(CGI.mechanism_verdict(**{**self.OK, "grad_norm": 1e-12})["status"], "MECHANISM_OK")
        failed = CGI.mechanism_verdict(**{**self.OK, "grad_norm": 0.0})
        self.assertEqual(failed["status"], "MECHANISM_FAILED")
        self.assertIn(CGI.RULE_GRAD_NORM, failed["failed_rules"])

    def test_g_mean_bounds_are_inclusive(self):
        for value in (0.05, 0.95):
            self.assertEqual(CGI.mechanism_verdict(**{**self.OK, "g_mean": value})["status"], "MECHANISM_OK")
        for value in (0.0499, 0.9501):
            failed = CGI.mechanism_verdict(**{**self.OK, "g_mean": value})
            self.assertEqual(failed["status"], "MECHANISM_FAILED")
            self.assertIn(CGI.RULE_G_MEAN, failed["failed_rules"])

    def test_weight_l1_boundary_is_inclusive(self):
        self.assertEqual(CGI.mechanism_verdict(**{**self.OK, "weight_l1": 1e-4})["status"], "MECHANISM_OK")
        failed = CGI.mechanism_verdict(**{**self.OK, "weight_l1": 9.9e-5})
        self.assertEqual(failed["status"], "MECHANISM_FAILED")
        self.assertIn(CGI.RULE_WEIGHT_L1, failed["failed_rules"])

    def test_every_failed_rule_is_reported(self):
        verdict = CGI.mechanism_verdict(g_std=0.0, grad_norm=0.0, g_mean=0.0, weight_l1=0.0)
        self.assertEqual(sorted(verdict["failed_rules"]),
                         sorted([CGI.RULE_GATE_STD, CGI.RULE_GRAD_NORM, CGI.RULE_G_MEAN, CGI.RULE_WEIGHT_L1]))

    def test_observed_values_are_recorded(self):
        verdict = CGI.mechanism_verdict(**self.OK)
        self.assertEqual(verdict["observed"]["g_std"], 0.05)
        self.assertEqual(verdict["checks"]["gate_std_ge_min"]["observed"], 0.05)


class TestArmVerdict(unittest.TestCase):
    def _ok(self):
        return CGI.mechanism_verdict(g_std=0.05, grad_norm=1e-3, g_mean=0.5, weight_l1=0.05)

    def test_mechanism_failure_stops_regardless_of_auc(self):
        bad = CGI.mechanism_verdict(g_std=0.0, grad_norm=1e-3, g_mean=0.5, weight_l1=0.05)
        arm = CGI.arm_verdict(0.99, mechanism=bad)                 # 即便 AUC 远高于阈值
        self.assertEqual(arm["status"], "STOP")
        self.assertFalse(arm["pass"])
        self.assertFalse(arm["effect"]["evaluated"])
        self.assertIsNone(arm["effect"]["pass"])

    def test_effect_threshold_boundary(self):
        self.assertEqual(CGI.arm_verdict(0.8521, mechanism=self._ok())["status"], "EFFECT_CONFIRMED")
        below = CGI.arm_verdict(0.8520, mechanism=self._ok())
        self.assertEqual(below["status"], "EFFECT_NOT_CONFIRMED")
        self.assertFalse(below["pass"])
        self.assertTrue(below["effect"]["evaluated"])

    def test_arm_payload_is_json_serialisable(self):
        arm = CGI.arm_verdict(0.853, mechanism=self._ok())
        self.assertEqual(json.loads(json.dumps(arm)), arm)


class TestHeadConstruction(unittest.TestCase):
    def test_shared_parameters_are_bit_identical_to_baseline_same_seed(self):
        """D2：同一 seed 下，共享参数（projection / gate / tower / env_embedding）逐位相同。"""
        torch.manual_seed(7)
        baseline = tiny_newtask()
        torch.manual_seed(7)
        head = tiny_head()
        base_state, head_state = baseline.state_dict(), head.state_dict()
        self.assertEqual(set(base_state), set(head_state) - set(self._extra_keys(head)))
        for key, value in base_state.items():
            self.assertTrue(torch.equal(value, head_state[key]), f"共享参数不一致: {key}")

    def test_extra_parameters_are_appended_after_the_shared_ones(self):
        """D2：额外参数在共享参数之后构造 ⇒ 不会位移基线的初始化 RNG。"""
        torch.manual_seed(11)
        baseline = tiny_newtask()
        after_baseline = torch.get_rng_state()
        torch.manual_seed(11)
        head = tiny_head()
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, head.state_dict()[key]), key)
        self.assertFalse(torch.equal(after_baseline, torch.get_rng_state()),
                         "处理头必须额外消耗 RNG（否则说明它没建新参数）")
        self.assertEqual(len(self._extra_keys(head)), 4)

    def test_state_dict_keys_are_baseline_keys_plus_the_gate(self):
        head = tiny_head()
        extra = set(self._extra_keys(head))
        self.assertEqual(extra, {"confidence_mlp.0.weight", "confidence_mlp.0.bias",
                                 "confidence_mlp.2.weight", "confidence_mlp.2.bias"})
        w1 = head.confidence_mlp[0].weight
        self.assertEqual(tuple(w1.shape), (CGI.GATE_HIDDEN, head.gate_input_dim))

    def test_gate_input_dim_is_rep_dim_plus_the_scalar_features(self):
        head = tiny_head(rep_dim=TINY_REP_DIM)
        self.assertEqual(head.gate_input_dim, TINY_REP_DIM + len(CGI.GATE_SCALAR_FEATURES))
        self.assertEqual(CGI.GATE_SCALAR_FEATURES, ("p_max", "entropy_norm", "cos_agreement"))

    def test_no_sigmoid_module_inside_the_gate(self):
        """z（presigmoid）必须可读 ⇒ sigmoid 在 gate_terms 里显式施加，不作为模块藏在 Sequential 中。"""
        head = tiny_head()
        self.assertFalse(any(isinstance(m, torch.nn.Sigmoid) for m in head.confidence_mlp.modules()))

    def test_l2_regularisation_is_unchanged_from_baseline(self):
        """get_l2_reg 不重写：只正则 tower（与基线、与历史实现同一口径）。"""
        torch.manual_seed(3)
        baseline = tiny_newtask()
        torch.manual_seed(3)
        head = tiny_head()
        self.assertEqual(type(head).get_l2_reg, NewTask.get_l2_reg)
        self.assertTrue(torch.equal(baseline.get_l2_reg(), head.get_l2_reg()))

    def test_attention_temperature_constant_matches_the_baseline_head(self):
        """ATTENTION_TEMPERATURE 只是留给落盘的常量；漂移必须被测试挡住（不调参）。"""
        self.assertEqual(CGI.ATTENTION_TEMPERATURE, tiny_newtask().temperature)

    @staticmethod
    def _extra_keys(head):
        return [key for key in head.state_dict() if key.startswith("confidence_mlp.")]


class TestGateInitialisation(unittest.TestCase):
    """D3 修复：标准初始化破对称 + 零初始化末层 ⇒ g ≡ 0.5 且不饱和。"""

    def test_output_layer_is_zero_initialised(self):
        head = tiny_head()
        self.assertEqual(float(head.confidence_mlp[2].weight.abs().max()), 0.0)
        self.assertEqual(float(head.confidence_mlp[2].bias.abs().max()), 0.0)

    def test_hidden_rows_are_not_tied(self):
        head = tiny_head()
        reading = CGI.hidden_row_symmetry(head)
        self.assertFalse(reading["hidden_rows_identical"])
        self.assertGreater(reading["hidden_row_symmetry_deviation"], 0.0)
        self.assertGreater(reading["hidden_row_symmetry_deviation_max"], 0.0)

    def test_symmetry_reading_discriminates_a_constant_init_control(self):
        """读数有效性对照：历史常量初始化（32 行逐位相同）下读数恒为 0。"""
        head = tiny_head()
        with torch.no_grad():
            head.confidence_mlp[0].weight.fill_(0.01)
        reading = CGI.hidden_row_symmetry(head)
        self.assertTrue(reading["hidden_rows_identical"])
        self.assertEqual(reading["hidden_row_symmetry_deviation"], 0.0)

    def test_presigmoid_is_exactly_zero_at_init(self):
        head = tiny_head()
        terms = head.gate_terms(*tiny_reps())
        self.assertEqual(float(terms["presigmoid"].abs().max()), 0.0)
        self.assertGreaterEqual(float(terms["presigmoid"].min()), -CGI.PRESIGMOID_SATURATION_ABS)
        self.assertLessEqual(float(terms["presigmoid"].max()), CGI.PRESIGMOID_SATURATION_ABS)

    def test_g_is_exactly_one_half_at_init_with_full_sigmoid_slope(self):
        head = tiny_head()
        g = head.gate_terms(*tiny_reps())["g"]
        self.assertTrue(torch.equal(g, torch.full_like(g, 0.5)))
        self.assertAlmostEqual(CGI.sigmoid_slope(0.0), 0.25, places=12)

    def test_historical_positive_bias_constant_is_absent(self):
        """历史常量初始化是 0.01/0.01 + 0.1/2.0；这里只允许标准初始化 + 零末层。"""
        head = tiny_head()
        self.assertGreater(float(head.confidence_mlp[0].weight.std()), 0.0)
        self.assertGreater(float(head.confidence_mlp[0].bias.std()), 0.0)


class TestPerSampleGate(unittest.TestCase):
    """D1 修复的核心证据：g 是逐样本量，且与批上下文无关。"""

    def test_gate_is_per_sample_when_live(self):
        head = live_head(tiny_head())
        terms = head.gate_terms(*tiny_reps(n=8))
        g = terms["g"].reshape(-1)
        self.assertEqual(tuple(g.shape), (8,))
        self.assertGreater(float(g.std(unbiased=False)), 0.005)
        self.assertGreater(int(torch.unique(g).numel()), 1)

    def test_same_sample_keeps_the_same_g_in_a_single_sample_batch(self):
        """D1 的靶心：同一样本单独成批时的 g 必须与成批时一致（批级门控会变）。

        用 1e-5 而非逐位相等：不同批大小下 BLAS 归约顺序不同，末位可能有 ~1e-7 的舍入差；
        批级门控造成的差异是 O(0.1)（见 exp/stage2-cgr-repro 的实测），量级差 4 个数量级。
        """
        head = live_head(tiny_head())
        reps = tiny_reps(n=8)
        batch_g = head.gate_terms(*reps)["g"].reshape(-1)
        for index in (0, 3, 7):
            single_g = head.gate_terms(*take(reps, index))["g"].reshape(-1)
            self.assertLess(abs(float(single_g.item()) - float(batch_g[index].item())), 1e-5,
                            f"样本 {index} 的 g 随批大小变化 ⇒ 门控仍是批级量")

    def test_same_sample_keeps_the_same_g_when_other_samples_change(self):
        """批组成改变（同一样本 + **不同的同伴、不同的批内位置**）不得改变该样本的 g。

        必须从**同一份** `reps`（同一个 backbone）里取：换 seed 重取会同时换掉 backbone 初值，
        那样比的是两个不同的模型，测不到"批上下文无关"这件事。这里让样本 2 分别与
        `{3}` 和 `{5, 7, 1}` 同批、且批内位置从 0 变到 1。
        """
        head = live_head(tiny_head())
        reps = tiny_reps(n=8)
        index = 2
        g_first = head.gate_terms(*pick(reps, [index, 3]))["g"][0]
        g_second = head.gate_terms(*pick(reps, [5, index, 7, 1]))["g"][1]
        self.assertLess(abs(float(g_first.item()) - float(g_second.item())), 1e-5)

    def test_gate_reads_every_sample_feature_shape(self):
        head = live_head(tiny_head())
        terms = head.gate_terms(*tiny_reps(n=5))
        self.assertEqual(tuple(terms["p_max"].shape), (5,))
        self.assertEqual(tuple(terms["entropy_norm"].shape), (5,))
        self.assertEqual(tuple(terms["cos_agreement"].shape), (5,))
        self.assertEqual(tuple(terms["gate_input"].shape), (5, head.gate_input_dim))

    def test_gate_depends_on_the_attention_projection(self):
        """门控输入含 H_out ⇒ 换一组 projection 权重必须改变 g（门控确实读到了逐样本表征）。

        这里**重抽**权重而不是等比缩放：`projection_network` 末端是 LayerNorm，
        等比缩放会被 LayerNorm 抹掉（只差一个 eps 量级），不构成有效扰动。
        """
        head = live_head(tiny_head())
        reps = tiny_reps(n=4)
        before = head.gate_terms(*reps)["g"].detach().clone()
        generator = torch.Generator().manual_seed(4242)
        with torch.no_grad():
            head.projection_network[0].weight.copy_(
                torch.randn(head.projection_network[0].weight.shape, generator=generator) * 0.5)
        after = head.gate_terms(*reps)["g"].detach()
        self.assertGreater(float((before - after).abs().max()), 1e-6)


class TestGateFeatures(unittest.TestCase):
    def test_attention_weights_match_the_baseline_expression(self):
        head = tiny_head()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps()
        exist_env_embs = torch.stack(env_embs, dim=1)
        h_out = head.projection_network(dnn_input)
        expected = torch.nn.functional.softmax(torch.mm(h_out, exist_env_embs) / head.temperature, dim=-1)
        terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertTrue(torch.equal(terms["w_attn"], expected))
        self.assertEqual(tuple(terms["w_attn"].shape), (dnn_input.shape[0], len(spec_reps)))

    def test_confidence_features_match_hand_computation(self):
        head = tiny_head()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(n=6)
        terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        probs = terms["w_attn"].double().tolist()
        for i, row in enumerate(probs):
            self.assertAlmostEqual(float(terms["p_max"][i]), max(row), places=6)
            entropy = -sum(p * math.log(p) for p in row) / math.log(len(row))
            self.assertAlmostEqual(float(terms["entropy_norm"][i]), entropy, places=5)

    def test_agreement_feature_is_cosine_between_attention_fused_and_general_rep(self):
        head = tiny_head()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps()
        terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        mix = torch.matmul(torch.stack(spec_reps, dim=2), terms["w_attn"].unsqueeze(2)).squeeze(-1)
        self.assertTrue(torch.equal(terms["spec_mix"], mix))
        expected = torch.nn.functional.cosine_similarity(mix.double(), gen_rep.double(), dim=-1)
        self.assertTrue(torch.allclose(terms["cos_agreement"].double(), expected, atol=1e-6))

    def test_features_are_bounded(self):
        head = live_head(tiny_head())
        terms = head.gate_terms(*tiny_reps(n=16))
        self.assertGreaterEqual(float(terms["p_max"].min()), 0.5)
        self.assertLessEqual(float(terms["p_max"].max()), 1.0)
        self.assertGreaterEqual(float(terms["entropy_norm"].min()), -1e-6)
        self.assertLessEqual(float(terms["entropy_norm"].max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(terms["cos_agreement"].min()), -1.0 - 1e-6)
        self.assertLessEqual(float(terms["cos_agreement"].max()), 1.0 + 1e-6)

    def test_single_source_task_is_handled(self):
        """K=1 时归一化熵的分母为 log(1)=0 ⇒ 必须显式退化，不得产生 NaN。"""
        head = live_head(tiny_head())
        terms = head.gate_terms(*tiny_reps(n=4, tasks=1))
        self.assertTrue(torch.isfinite(terms["gate_input"]).all())
        self.assertEqual(float(terms["entropy_norm"].abs().max()), 0.0)
        self.assertEqual(float(terms["p_max"].min()), 1.0)


class TestBlend(unittest.TestCase):
    def test_w_is_a_convex_combination_of_attention_and_uniform(self):
        head = live_head(tiny_head())
        terms = head.gate_terms(*tiny_reps())
        g = terms["g"].unsqueeze(-1)
        expected = g * terms["w_attn"].unsqueeze(-1) + (1.0 - g) * terms["w_fw"].view(1, -1, 1)
        self.assertTrue(torch.equal(terms["w"], expected))
        self.assertTrue(torch.allclose(terms["w"].sum(dim=1), torch.ones_like(terms["w"][:, :1]), atol=1e-6))

    def test_uniform_weights_are_exactly_one_over_k(self):
        terms = tiny_head().gate_terms(*tiny_reps())
        self.assertTrue(torch.equal(terms["w_fw"], torch.full((2,), 0.5)))

    def test_g_one_reproduces_attention_exactly(self):
        head = force_g(tiny_head(), 1.0)
        terms = head.gate_terms(*tiny_reps())
        self.assertTrue(torch.equal(terms["g"], torch.ones_like(terms["g"])))
        self.assertTrue(torch.equal(terms["w"], terms["w_attn"].unsqueeze(-1)))

    def test_g_zero_reproduces_uniform_exactly(self):
        head = force_g(tiny_head(), 0.0)
        terms = head.gate_terms(*tiny_reps())
        self.assertTrue(torch.equal(terms["w"], terms["w_fw"].view(1, -1, 1).expand_as(terms["w"])))

    def test_forward_matches_the_baseline_pipeline_bit_for_bit_when_g_is_one(self):
        """W ≡ W_attn 时，本头的 forward 必须与基线 forward 逐位相同（无静默管线漂移）。"""
        reps = tiny_reps(n=8)
        torch.manual_seed(7)
        baseline = tiny_newtask()
        torch.manual_seed(7)
        head = force_g(tiny_head(), 1.0)
        self.assertTrue(torch.equal(baseline(*reps), head(*reps)))

    def test_forward_differs_from_baseline_at_the_neutral_gate(self):
        """g ≡ 0.5（初始化）时前向必须**不同于**基线：混合确实生效了。

        两个头在同一 seed 下共享参数（D2），故唯一差异来源是混合权重；而 `W_attn = W_fw` 时
        `W = W_attn`，前向会与基线**逐位相同** —— 所以必须先确定性地把注意力从均匀不动点移开，
        否则夹具随机采到全灭瓶颈时，这条断言会因为"混合恰好是恒等映射"而误报。
        受控投影**同一份**写进两个头，差异因此仍然只来自混合。
        """
        reps = deterministic_reps(n=8)
        torch.manual_seed(7)
        baseline = tiny_newtask()
        torch.manual_seed(7)
        head = tiny_head()
        force_live_attention(baseline, reps[0])
        force_live_attention(head, reps[0])
        terms = assert_attention_is_live(self, head, reps)
        self.assertFalse(torch.equal(baseline(*reps), head(*reps)))
        self.assertGreater(float((terms["w"] - terms["w_attn"].unsqueeze(-1)).abs().sum(dim=1).mean()), 1e-6)


class TestBatchSizeOne(unittest.TestCase):
    """D6 修复：单样本批不得 NaN、不得形状错误。

    覆盖与可学性分开：本类只管"跑得通"；"梯度非零"需要一个受控夹具
    （注意力非均匀），单独成测试（`test_gate_receives_nonzero_gradient_at_batch_size_one`）。
    """

    def test_gate_terms_are_finite_at_batch_size_one(self):
        head = live_head(tiny_head())
        terms = head.gate_terms(*take(tiny_reps(), 0))
        for key in ("gate_input", "presigmoid", "g", "w"):
            self.assertTrue(torch.isfinite(terms[key]).all(), f"{key} 在 B=1 下必须有限")

    def test_forward_and_backward_work_at_batch_size_one(self):
        """B=1 的形状 / 有限性 / 反传覆盖（D6）。

        这里**只**管"跑得通"：形状正确、数值有限、反传不报错、门控四个参数都拿到形状正确的梯度。
        "梯度非零"（D5 的可学性）另立测试——它需要注意力非均匀这一前置条件。分开的理由：随机夹具
        下瓶颈可能全灭，那时梯度是**精确** 0（数学性质），若把非零断言混在这里，B=1 覆盖失败与
        可学性失败就无法区分。
        """
        head = live_head(tiny_head())
        reps = take(tiny_reps(), 0)
        pred = head(*reps)
        self.assertEqual(tuple(pred.shape), (1,))
        self.assertTrue(torch.isfinite(pred).all())
        torch.nn.functional.binary_cross_entropy(pred, torch.ones(1)).backward()
        for name, param in head.confidence_mlp.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} 未参与反传")
            self.assertEqual(tuple(param.grad.shape), tuple(param.shape), name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_gate_receives_nonzero_gradient_at_batch_size_one(self):
        """B=1 下的可学性（受控夹具）：注意力非均匀 ⇒ dL/dg ≠ 0 ⇒ 门控梯度 > 0。

        夹具不掷骰子：投影 / 环境 / 源表征全部由 `deterministic_reps` + `force_live_attention` 给定，
        因此本断言与执行顺序、前序 RNG 消耗无关（这正是第一版在整套用例里随机失败的原因）。
        取的是第 2 个样本构成的单样本批，避免"恰好只有首样本特殊"的假象。
        """
        head = live_head(seeded_head())
        reps = take(deterministic_reps(n=4), 2)
        force_live_attention(head, reps[0])
        terms = assert_attention_is_live(self, head, reps)
        hidden = head.confidence_mlp[1](head.confidence_mlp[0](terms["gate_input"]))
        self.assertGreater(float(hidden.abs().max()), 0.0,
                           "门控隐层全灭 ⇒ ∂z/∂W_out = h ≡ 0，输出层梯度必然为 0（夹具问题）")
        pred = head(*reps)
        self.assertEqual(tuple(pred.shape), (1,))
        torch.nn.functional.binary_cross_entropy(pred, torch.ones(1)).backward()
        self.assertGreater(CGI.confidence_grad_norm(head), 0.0)

    def test_baseline_head_breaks_at_batch_size_one(self):
        """对照留痕：基线 NewTask.forward 在 B=1 下 `squeeze()` 连批维一起压掉 ⇒ `torch.stack(dim=2)` 越界。

        基线 `model.py:926` 的 `torch.stack([env_aware_rep, gen_rep], dim=2)` 中，`env_aware_rep` 已被
        上一行的 `.squeeze()` 压成 1 维（B=1 时批维与末维同为 1，二者一并消失）；对 1 维张量取 dim=2
        抛的是 `IndexError`（维度越界），而非形状不匹配的 `RuntimeError`。本头改用 `squeeze(-1)` 正是
        为此，故此处按实测的异常类型与消息锁定，不因"基线本来就坏"而放宽。
        """
        with self.assertRaises(IndexError) as ctx:
            tiny_newtask()(*take(tiny_reps(), 0))
        self.assertEqual(
            str(ctx.exception),
            "Dimension out of range (expected to be in range of [-2, 1], but got 2)",
        )


class TestGateGradient(unittest.TestCase):
    def test_output_layer_receives_gradient_at_the_first_step(self):
        """初始步的门控梯度（D5）：需要 dL/dz ≠ 0（前置条件）且隐层非全灭（h ≢ 0）。"""
        head = seeded_head()
        reps = deterministic_reps(n=8)
        force_live_attention(head, reps[0])
        terms = assert_attention_is_live(self, head, reps)
        hidden = head.confidence_mlp[1](head.confidence_mlp[0](terms["gate_input"]))
        self.assertGreater(float(hidden.abs().max()), 0.0,
                           "门控隐层全灭 ⇒ ∂z/∂W_out = h ≡ 0，输出层梯度必然为 0（夹具问题）")
        torch.nn.functional.binary_cross_entropy(head(*reps), torch.ones(8)).backward()
        self.assertGreater(float(head.confidence_mlp[2].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.confidence_mlp[2].bias.grad.abs().sum()), 0.0)

    def test_hidden_layer_gradient_is_zero_at_init_then_nonzero_after_one_step(self):
        """零初始化末层的已知代价：第一步不给隐层梯度；一步之后必须全参数有梯度。"""
        head = seeded_head()
        reps = deterministic_reps(n=8)
        force_live_attention(head, reps[0])
        assert_attention_is_live(self, head, reps)
        optimizer = torch.optim.Adam(head.parameters(), lr=P.LR)
        for step in range(2):
            optimizer.zero_grad()
            torch.nn.functional.binary_cross_entropy(head(*reps), torch.ones(8)).backward()
            if step == 0:
                self.assertEqual(float(head.confidence_mlp[0].weight.grad.abs().sum()), 0.0)
            else:
                self.assertGreater(float(head.confidence_mlp[0].weight.grad.abs().sum()), 0.0)
            optimizer.step()

    def test_confidence_grad_norm_matches_manual_computation(self):
        head = live_head(tiny_head())
        torch.nn.functional.binary_cross_entropy(head(*tiny_reps(n=8)), torch.ones(8)).backward()
        manual = sum(float(p.grad.detach().double().norm(2)) ** 2 for p in head.confidence_mlp.parameters()) ** 0.5
        self.assertAlmostEqual(CGI.confidence_grad_norm(head), manual, places=6)

    def test_confidence_grad_norm_is_zero_when_no_gradients_exist(self):
        self.assertEqual(CGI.confidence_grad_norm(tiny_head()), 0.0)

    def test_collapsed_attention_is_a_gradient_fixed_point(self):
        """确定性对照：投影全灭 ⇒ H_out ≡ 0 ⇒ W_attn 逐位 = W_fw ⇒ 门控梯度**精确**为 0。

        混合权重对 g 的导数是 `W_attn − W_fw`：注意力塌到均匀时它恒为 0，g 对 loss 没有任何
        一阶影响 —— 这是结构性质，不是实现缺陷。本测试锁住的是**夹具纪律**：随机初值下瓶颈
        全灭（B=1 夹具扫描 60 个 RNG 状态里有 20 个）会让任何"grad > 0"断言必然失败，
        所以那类断言必须先验收前置条件（`assert_attention_is_live`）。

        构造不掷骰子：瓶颈权重置零 ⇒ 预激活 = 0 ⇒ ReLU(0) = 0 ⇒ 第二个 Linear（无 bias）= 0
        ⇒ LayerNorm(全零) = 0（weight=1、bias=0 是 LayerNorm 的确定性初值）。
        """
        head = live_head(seeded_head())
        reps = deterministic_reps(n=4)
        with torch.no_grad():
            head.projection_network[0].weight.zero_()
        terms = head.gate_terms(*reps)
        self.assertTrue(attention_is_uniform(terms))
        self.assertTrue(torch.equal(terms["w_attn"], torch.full_like(terms["w_attn"], 0.5)))
        self.assertTrue(torch.equal(terms["h_out"], torch.zeros_like(terms["h_out"])))
        torch.nn.functional.binary_cross_entropy(head(*reps), torch.ones(4)).backward()
        self.assertIsNotNone(head.confidence_mlp[2].weight.grad, "门控仍在计算图上，只是导数为零")
        self.assertEqual(float(head.confidence_mlp[2].weight.grad.abs().sum()), 0.0)
        self.assertEqual(CGI.confidence_grad_norm(head), 0.0)

    def test_gate_parameters_are_trained_by_the_optimizer(self):
        head = live_head(seeded_head())
        reps = deterministic_reps(n=8)
        force_live_attention(head, reps[0])
        assert_attention_is_live(self, head, reps)
        before = head.confidence_mlp[2].weight.detach().clone()
        optimizer = torch.optim.Adam(head.parameters(), lr=P.LR)
        optimizer.zero_grad()
        torch.nn.functional.binary_cross_entropy(head(*reps), torch.ones(8)).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, head.confidence_mlp[2].weight))


class TestStatsAccumulator(unittest.TestCase):
    def _terms(self, w_attn, g, presigmoid):
        w_attn = torch.as_tensor(w_attn, dtype=torch.float64)
        g = torch.as_tensor(g, dtype=torch.float64).view(-1, 1)
        return {"w_attn": w_attn, "g": g, "presigmoid": torch.as_tensor(presigmoid, dtype=torch.float64).view(-1, 1),
                "w": g.unsqueeze(-1) * w_attn.unsqueeze(-1) + (1 - g).unsqueeze(-1) * torch.full((1, 2, 1), 0.5, dtype=torch.float64),
                "p_max": w_attn.max(dim=-1).values, "entropy_norm": torch.zeros(w_attn.shape[0], dtype=torch.float64),
                "cos_agreement": torch.zeros(w_attn.shape[0], dtype=torch.float64)}

    def test_hand_computed_batch_metrics(self):
        stats = CGI.CGRInstanceStats()
        stats.update(self._terms([[0.8, 0.2], [0.5, 0.5]], [0.25, 0.75], [-1.0986, 1.0986]))
        result = stats.result()
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(result["n_batches"], 1)
        self.assertAlmostEqual(result["g_mean"], 0.5)
        self.assertAlmostEqual(result["g_min"], 0.25)
        self.assertAlmostEqual(result["g_max"], 0.75)
        self.assertAlmostEqual(result["g_within_batch_std_mean"], 0.25)
        self.assertAlmostEqual(result["g_within_batch_unique_frac_mean"], 1.0)
        self.assertAlmostEqual(result["weight_l1_mean"], 0.225)          # (0.45 + 0.0) / 2
        self.assertAlmostEqual(result["presigmoid_abs_mean"], 1.0986)

    def test_batch_constant_g_yields_zero_std(self):
        """D1 的签名读数：批内同一个标量 ⇒ std 恒为 0、唯一值占比 = 1/批大小。"""
        stats = CGI.CGRInstanceStats()
        stats.update(self._terms([[0.8, 0.2], [0.5, 0.5]], [0.6, 0.6], [0.4, 0.4]))
        result = stats.result()
        self.assertEqual(result["g_within_batch_std_mean"], 0.0)
        self.assertAlmostEqual(result["g_within_batch_unique_frac_mean"], 0.5)

    def test_streaming_preserves_sample_level_aggregates(self):
        batched, split = CGI.CGRInstanceStats(), CGI.CGRInstanceStats()
        terms = self._terms([[0.8, 0.2], [0.5, 0.5], [0.9, 0.1], [0.6, 0.4]], [0.2, 0.4, 0.6, 0.8], [0.1, 0.2, 0.3, 0.4])
        batched.update(terms)
        for start in range(0, 4, 2):
            split.update({key: value[start:start + 2] for key, value in terms.items()})
        for key in ("g_mean", "g_min", "g_max", "presigmoid_abs_mean"):
            self.assertAlmostEqual(batched.result()[key], split.result()[key], places=9)
        self.assertEqual(split.result()["n_samples"], 4)
        self.assertEqual(split.result()["n_batches"], 2)

    def test_weight_l1_is_zero_when_g_is_one(self):
        stats = CGI.CGRInstanceStats()
        stats.update(self._terms([[0.8, 0.2], [0.5, 0.5]], [1.0, 1.0], [10.0, 10.0]))
        self.assertEqual(stats.result()["weight_l1_mean"], 0.0)

    def test_saturation_reading_counts_samples_beyond_the_dead_zone(self):
        stats = CGI.CGRInstanceStats()
        stats.update(self._terms([[0.8, 0.2], [0.5, 0.5]], [0.5, 0.5], [1.0, 30.0]))
        result = stats.result()
        self.assertAlmostEqual(result["presigmoid_saturation_frac"], 0.5)
        self.assertEqual(result["presigmoid_saturation_abs"], CGI.PRESIGMOID_SATURATION_ABS)
        self.assertAlmostEqual(result["presigmoid_abs_max"], 30.0)

    def test_empty_accumulator_is_zero_safe(self):
        result = CGI.CGRInstanceStats().result()
        self.assertEqual(result["n_samples"], 0)
        self.assertEqual(result["g_mean"], 0.0)
        self.assertEqual(result["g_within_batch_std_mean"], 0.0)
        self.assertEqual(json.loads(json.dumps(result)), result)


class TestTrainGateTrace(unittest.TestCase):
    def test_trace_records_per_epoch_grad_stats_and_gate_readings(self):
        head = live_head(seeded_head())
        reps = deterministic_reps(n=8)
        force_live_attention(head, reps[0])
        assert_attention_is_live(self, head, reps)
        optimizer = torch.optim.Adam(head.parameters(), lr=P.LR)
        trace = CGI.TrainGateTrace()
        trace.start_epoch(1)
        for _ in range(3):
            optimizer.zero_grad()
            torch.nn.functional.binary_cross_entropy(head(*reps), torch.ones(8)).backward()
            norm = trace.record_step(head, *reps)
            self.assertGreater(norm, 0.0)
            optimizer.step()
        record = trace.end_epoch()
        self.assertEqual(record["epoch"], 1)
        self.assertEqual(record["steps"], 3)
        self.assertEqual(record["grad_nonzero_steps"], 3)
        self.assertGreater(record["grad_norm_max"], 0.0)
        self.assertIn("g_mean", record)
        result = trace.result()
        self.assertEqual(result["epochs"], 1)
        self.assertEqual(result["steps"], 3)
        self.assertEqual(len(result["per_epoch"]), 1)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_empty_trace_reports_zero_grad_norm(self):
        result = CGI.TrainGateTrace().result()
        self.assertEqual(result["grad_norm_max"], 0.0)
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["per_epoch"], [])
        self.assertFalse(CGI.mechanism_verdict(
            g_std=result["g_within_batch_std_mean"], grad_norm=result["grad_norm_max"],
            g_mean=result["g_mean"], weight_l1=result["weight_l1_mean"])["pass"])

    def test_zero_gradient_forever_fires_the_grad_rule(self):
        head = force_g(tiny_head(), 1.0)                     # σ(z) 饱和在 1.0 ⇒ σ′ 下溢为 0
        head.confidence_mlp[2].bias.requires_grad_(False)    # 冻结输出层 ⇒ 门控收不到任何梯度
        head.confidence_mlp[2].weight.requires_grad_(False)
        reps = tiny_reps(n=8)
        trace = CGI.TrainGateTrace()
        trace.start_epoch(1)
        torch.nn.functional.binary_cross_entropy(head(*reps), torch.ones(8)).backward()
        trace.record_step(head, *reps)
        trace.end_epoch()
        self.assertEqual(trace.result()["grad_norm_max"], 0.0)
        self.assertIn(CGI.RULE_GRAD_NORM, CGI.mechanism_verdict(
            g_std=0.0, grad_norm=trace.result()["grad_norm_max"], g_mean=0.5,
            weight_l1=0.0)["failed_rules"])

    def test_epoch_bookkeeping_errors_are_explicit(self):
        trace, head = CGI.TrainGateTrace(), tiny_head()
        with self.assertRaises(RuntimeError):
            trace.end_epoch()
        with self.assertRaises(RuntimeError):
            trace.record_step(head, *tiny_reps(n=2))
        trace.start_epoch(1)
        with self.assertRaises(RuntimeError):
            trace.start_epoch(2)
        with self.assertRaises(ValueError):
            trace.record_step(tiny_newtask(), *tiny_reps(n=2))


class TestEvaluateGate(unittest.TestCase):
    def test_matches_recomputation_and_leaves_parameters_unchanged(self):
        head = live_head(tiny_head())
        backbone = tiny_backbone()
        loaders, _, _ = tiny_inputs()
        before = {key: value.clone() for key, value in head.state_dict().items()}
        readings = CGI.evaluate_gate(head, backbone, loaders["val"], torch.device("cpu"))
        self.assertEqual(readings["n_samples"], 16)
        self.assertEqual(readings["n_batches"], 1)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, head.state_dict()[key]), key)
        self.assertTrue(all(param.grad is None for param in head.parameters()))
        expected = CGI.CGRInstanceStats()
        for _, _, _, features in loaders["val"]:
            with torch.no_grad():
                expected.update(head.gate_terms(*backbone.get_infos(features)))
        self.assertAlmostEqual(readings["g_mean"], expected.result()["g_mean"], places=9)
        self.assertAlmostEqual(readings["weight_l1_mean"], expected.result()["weight_l1_mean"], places=9)

    def test_reports_initialisation_and_gate_readings(self):
        head = live_head(tiny_head())
        readings = CGI.evaluate_gate(head, tiny_backbone(), tiny_inputs()[0]["val"], torch.device("cpu"))
        for key in ("hidden_row_symmetry_deviation", "hidden_rows_identical", "gate_output_weight_absmax",
                    "gate_output_bias", "gate_input_dim", "gate_hidden"):
            self.assertIn(key, readings)
        self.assertFalse(readings["hidden_rows_identical"])
        self.assertEqual(readings["gate_input_dim"], TINY_REP_DIM + 3)
        self.assertEqual(json.loads(json.dumps(readings)), readings)

    def test_baseline_newtask_is_rejected(self):
        with self.assertRaises(ValueError):
            CGI.evaluate_gate(tiny_newtask(), tiny_backbone(), tiny_inputs()[0]["val"], torch.device("cpu"))


class TestVariantFactory(unittest.TestCase):
    def test_baseline_variant_is_the_untouched_newtask(self):
        torch.manual_seed(5)
        built = CGI.build_newtask(CGI.BASELINE_VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                  tower_dnn_hidden_units=TOWER_HIDDEN, reg_dnn=P.REG_DNN)
        self.assertIs(type(built), NewTask)
        torch.manual_seed(5)
        reference = tiny_newtask()
        for key, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, built.state_dict()[key]), key)

    def test_baseline_variant_consumes_no_extra_rng(self):
        torch.manual_seed(13)
        CGI.build_newtask(CGI.BASELINE_VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                          tower_dnn_hidden_units=TOWER_HIDDEN, reg_dnn=P.REG_DNN)
        after_build = torch.get_rng_state()
        torch.manual_seed(13)
        tiny_newtask()
        self.assertTrue(torch.equal(after_build, torch.get_rng_state()))

    def test_treatment_variant_is_the_corrective_subclass(self):
        built = CGI.build_newtask(CGI.VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                  tower_dnn_hidden_units=TOWER_HIDDEN, reg_dnn=P.REG_DNN)
        self.assertIsInstance(built, CGI.CGRInstanceNewTask)
        self.assertIsInstance(built, NewTask)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            CGI.build_newtask("nope", input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                              tower_dnn_hidden_units=TOWER_HIDDEN, reg_dnn=P.REG_DNN)

    def test_stage2_config_is_empty_for_baseline(self):
        self.assertEqual(CGI.stage2_config(CGI.BASELINE_VARIANT, rep_dim=TINY_REP_DIM), {})

    def test_stage2_config_records_the_gate_design_for_the_treatment(self):
        config = CGI.stage2_config(CGI.VARIANT, rep_dim=TINY_REP_DIM)
        self.assertEqual(config["variant"], CGI.VARIANT)
        self.assertEqual(config["gate_hidden"], CGI.GATE_HIDDEN)
        self.assertEqual(config["gate_input_dim"], TINY_REP_DIM + 3)
        self.assertFalse(config["warmup"])
        self.assertEqual(json.loads(json.dumps(config)), config)


class TestRunnerIntegration(unittest.TestCase):
    """接线测试：基线臂逐键不变；处理臂只多出 cgr_instance_arm 段。"""

    LEGACY_PAYLOAD_KEYS = {"run_id", "stage1_id", "commit", "split_sha256", "env_ids_sha256",
                           "backbone_sha256_before", "backbone_sha256_after", "stage1", "stage2", "mechanism"}
    LEGACY_CONFIG_KEYS = {"run_id", "stage1_id", "commit", "tag", "frozen", "split_seed", "model_seed",
                          "env_seed", "epochs", "patience", "lr", "batch_size", "input_size", "rep_dim"}

    def _stage1(self, root, device):
        loaders, stats, indices = tiny_inputs()
        return run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_backbone(),
                                               loaders=loaders, stats=stats, indices=indices)

    def _stage2(self, root, stage1, device, **kwargs):
        loaders, stats, indices = tiny_inputs()
        return run_census_benchmark.run_stage2(
            root, stage1_dir=Path(stage1["dir"]), epochs=2, device=device, model=tiny_backbone(),
            loaders=loaders, stats=stats, indices=indices, input_size=TINY_INPUT_SIZE,
            rep_dim=TINY_REP_DIM, **kwargs)

    def test_baseline_arm_payload_keys_are_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1 = self._stage1(root, device)
            out = self._stage2(root, stage1, device)
            payload = json.loads((Path(out["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
            config = json.loads((Path(out["run_dir"]) / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(set(payload), self.LEGACY_PAYLOAD_KEYS)
            self.assertEqual(set(config), self.LEGACY_CONFIG_KEYS)
            self.assertNotIn(CGI.RUN_ID_SUFFIX, out["run_id"])
            self.assertIsNone(out["arm"])
            self.assertEqual(set(payload["mechanism"]),
                             {"gate_mean", "cos_gen_spec", "gen_std", "env_acc_stage1"})

    def test_baseline_arm_never_constructs_the_treatment_head(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1 = self._stage1(root, device)
            boom = mock.Mock(side_effect=AssertionError("基线臂不得构造处理头"))
            with mock.patch.object(CGI, "CGRInstanceNewTask", boom):
                out = self._stage2(root, stage1, device)
            boom.assert_not_called()
            self.assertIsNone(out["arm"])

    def test_baseline_arm_is_deterministic_across_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1 = self._stage1(root, device)
            first = self._stage2(root, stage1, device, now=datetime(2026, 9, 30, 9, 0))
            second = self._stage2(root, stage1, device, now=datetime(2026, 9, 30, 9, 1))
            self.assertEqual(first["test_auc"], second["test_auc"])
            self.assertNotEqual(first["run_id"], second["run_id"])

    def test_treatment_arm_records_arm_block_trace_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1 = self._stage1(root, device)
            out = self._stage2(root, stage1, device, variant=CGI.VARIANT)
            arm = out["arm"]
            self.assertEqual(out["variant"], CGI.VARIANT)
            self.assertTrue(out["run_id"].endswith(CGI.RUN_ID_SUFFIX))
            self.assertEqual(set(arm), {"status", "pass", "mechanism", "effect", "diagnostics",
                                        "train_gate_trace", "confidence_grad_norm", "prereg", "provenance"})
            self.assertIn(arm["status"], ("STOP", "EFFECT_CONFIRMED", "EFFECT_NOT_CONFIRMED"))
            self.assertEqual(len(arm["train_gate_trace"]), 2)             # epochs=2，patience=2 不触发
            self.assertEqual(arm["diagnostics"]["n_samples"], 16)         # val 集
            self.assertEqual(arm["prereg"]["auc_test_min"], CGI.AUC_TEST_MIN)
            self.assertGreater(arm["confidence_grad_norm"]["grad_norm_max"], 0.0)
            payload = json.loads((Path(out["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["variant"], CGI.VARIANT)
            self.assertEqual(payload["cgr_instance_arm"], arm)
            self.assertEqual(set(payload["mechanism"]),
                             {"gate_mean", "cos_gen_spec", "gen_std", "env_acc_stage1"})
            config = json.loads((Path(out["run_dir"]) / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], CGI.VARIANT)
            self.assertFalse(config["warmup"])
            state = torch.load(Path(out["run_dir"]) / "newtask.pt", map_location="cpu")
            self.assertIn("confidence_mlp.2.weight", state)
            self.assertIn(out["run_id"], (root / "SUMMARY.md").read_text(encoding="utf-8"))

    def test_treatment_arm_mechanism_verdict_uses_the_val_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1 = self._stage1(root, device)
            arm = self._stage2(root, stage1, device, variant=CGI.VARIANT)["arm"]
            observed = arm["mechanism"]["observed"]
            self.assertEqual(observed["g_std"], arm["diagnostics"]["g_within_batch_std_mean"])
            self.assertEqual(observed["weight_l1"], arm["diagnostics"]["weight_l1_mean"])
            self.assertEqual(observed["g_mean"], arm["diagnostics"]["g_mean"])
            self.assertEqual(observed["grad_norm"], arm["confidence_grad_norm"]["grad_norm_max"])
            self.assertFalse(arm["mechanism"]["pass"] and arm["status"] == "STOP")

    def test_cli_exposes_variant_and_alias(self):
        parser = run_census_benchmark.build_parser()
        base = ["stage2", "--stage1-dir", "x"]
        self.assertEqual(parser.parse_args(base).variant, CGI.BASELINE_VARIANT)
        self.assertEqual(parser.parse_args(base + ["--variant", CGI.VARIANT]).variant, CGI.VARIANT)
        self.assertEqual(parser.parse_args(base + ["--cgr-instance"]).variant, CGI.VARIANT)
        self.assertEqual(parser.parse_args(base + ["--cgr-instance", "--variant", "baseline"]).variant,
                         CGI.BASELINE_VARIANT)

    def test_static_guard_protocol_model_and_metrics_untouched(self):
        """冻结文件不得改动。

        基准**不能**一刀切用 master：`census_benchmark/protocol.py` / `metrics.py` 是本分支祖先
        （`be20ff9` / `904f8d0`）**新增**的文件，在 master 上根本不存在，`git diff --name-only master`
        会把它们列成 added 而不是 modified，于是守卫恒失败（这是本文件第一版的一个真实缺陷）。
        故：模型/配置对 master 比（两者在 master 上已存在），协议/指标对本分支基点比
        （`infra/fair-stage2-benchmark`；该 ref 缺失时退化为 HEAD，仍能挡住工作区改动）。
        """
        model_diff = _git("diff", "--name-only", "master", "--", "multitaskrec", "config.py").stdout.strip()
        self.assertEqual(model_diff, "", f"模型/配置文件不得改动: {model_diff}")
        base = BRANCH_BASE
        if _git("rev-parse", "--verify", "--quiet", base).returncode != 0:
            base = "HEAD"
        frozen_diff = _git("diff", "--name-only", base, "--", "census_benchmark/protocol.py",
                           "census_benchmark/metrics.py").stdout.strip()
        self.assertEqual(frozen_diff, "", f"协议/指标文件不得改动（基准 {base}）: {frozen_diff}")


if __name__ == "__main__":
    unittest.main()
