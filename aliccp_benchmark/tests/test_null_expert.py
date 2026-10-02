"""Null Expert（AliCCP 跨数据集复现）：阶段 2 源任务路由追加"零候选"。

对应实验说明 `docs/superpowers/specs/2026-10-03-aliccp-stage2-null-expert-design.md`。

- 默认 `use_null_expert=False`：NewTask 必须与分支基点 `8133d32:multitaskrec/model.py`（= 基线 run
  实际使用的模型代码）**逐位一致**（前向 + 反向 + 参数集合）。
- `use_null_expert=True`：在既有 K 个 spec_rep 候选之后追加一个**值恒为零**的候选，
  router key 追加一个可学习 `null_key`（rep_dim），与既有 H_out 点积后同温度 softmax 得 K+1 权重；
  其余 new_env_emb / gate / tower / loss 完全不变。

测试用 CPU 极小夹具，不构成任何性能证据，只验证语义、指标与接线。
"""
import json
import math
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from aliccp_benchmark import bench, metrics, protocol as P
from multitaskrec.model import NewTask
import run_aliccp_benchmark

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "superpowers" / "specs" / "2026-10-03-aliccp-stage2-null-expert-design.md"
BASE_COMMIT = "8133d32"          # 本分支基点（infra/aliccp-fair-benchmark 顶端 = 基线 run 的模型代码）
BATCH, INPUT_SIZE, REP_DIM, NUM_SRC = 6, 8, 4, 2


def _base_newtask_class():
    """从 git 基点提交动态加载 NewTask 作为逐位对照的参考实现（不落地任何文件）。

    显式 `encoding="utf-8"`：Windows 下 `text=True` 默认走 locale 编码（GBK），
    model.py 含中文注释 → UnicodeDecodeError 会把源码变成 None；errors="strict" 保证解码问题不被静默吞掉。
    """
    src = subprocess.run(["git", "show", f"{BASE_COMMIT}:multitaskrec/model.py"], cwd=REPO, check=True,
                         capture_output=True, text=True, encoding="utf-8", errors="strict").stdout
    module = types.ModuleType("model_base")
    exec(compile(src, f"{BASE_COMMIT}:multitaskrec/model.py", "exec"), module.__dict__)
    return module.NewTask


def _ours(use_null_expert=False):
    return NewTask(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                   reg_dnn=P.REG_DNN, device=torch.device("cpu"), use_null_expert=use_null_expert)


def _base():
    return _base_newtask_class()(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                                 reg_dnn=P.REG_DNN, device=torch.device("cpu"))


def _inputs(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(BATCH, INPUT_SIZE, generator=gen), torch.randn(BATCH, REP_DIM, generator=gen),
            [torch.randn(BATCH, REP_DIM, generator=gen) for _ in range(NUM_SRC)],
            [torch.randn(REP_DIM, generator=gen) for _ in range(NUM_SRC)],
            torch.randint(0, 2, (BATCH,), generator=gen).float())


def _forward_backward(module, inputs):
    dnn_input, gen_rep, spec_reps, env_embs, target = inputs
    module.zero_grad(set_to_none=True)
    pred = module(dnn_input, gen_rep, spec_reps, env_embs)
    F.binary_cross_entropy(pred, target).backward()
    return pred


class TestDefaultArmBitIdenticalToBase(unittest.TestCase):
    """默认臂（use_null_expert=False）必须是基点实现的逐位复制。"""

    def test_default_arm_has_no_null_key(self):
        ours = _ours()
        self.assertNotIn("null_key", dict(ours.named_parameters()))
        self.assertEqual(set(ours.state_dict()), set(_base().state_dict()))
        ours.load_state_dict(_base().state_dict())                        # strict=True：参数集合必须一致

    def test_default_forward_bit_identical(self):
        base, ours, inputs = _base(), _ours(), _inputs(1)
        ours.load_state_dict(base.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(*inputs[:4]), base(*inputs[:4])))

    def test_default_backward_bit_identical(self):
        base, ours, inputs = _base(), _ours(), _inputs(2)
        ours.load_state_dict(base.state_dict())
        pred_base, pred_ours = _forward_backward(base, inputs), _forward_backward(ours, inputs)
        self.assertTrue(torch.equal(pred_base, pred_ours))
        grads_ours = dict(ours.named_parameters())
        for name, param in base.named_parameters():
            self.assertIsNotNone(param.grad, f"基点该参数无梯度，反向对照无效: {name}")
            self.assertTrue(torch.equal(param.grad, grads_ours[name].grad), f"梯度非逐位一致: {name}")

    def test_null_key_does_not_consume_rng(self):
        """两臂共享参数初始化必须逐位一致：零初始化 + 末尾创建不得消耗全局 RNG。"""
        torch.manual_seed(0)
        off = _ours()
        torch.manual_seed(0)
        on = _ours(use_null_expert=True)
        for name, param in off.named_parameters():
            self.assertTrue(torch.equal(param, dict(on.named_parameters())[name]), f"共享参数初始化漂移: {name}")
        self.assertEqual(tuple(on.null_key.shape), (REP_DIM,))


class TestNullCandidateMechanics(unittest.TestCase):
    """开启 Null Expert 后的候选/权重语义。"""

    def test_routing_weights_shape_is_k_then_k_plus_one(self):
        off, on, inputs = _ours(), _ours(use_null_expert=True), _inputs(3)
        self.assertEqual(tuple(off.routing_weights(inputs[0], inputs[3]).shape), (BATCH, NUM_SRC))
        self.assertEqual(tuple(on.routing_weights(inputs[0], inputs[3]).shape), (BATCH, NUM_SRC + 1))

    def test_routing_weights_are_probabilities(self):
        on, inputs = _ours(use_null_expert=True), _inputs(4)
        weights = on.routing_weights(inputs[0], inputs[3])
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(BATCH), atol=1e-6))
        self.assertTrue(bool((weights >= 0).all()))

    def test_null_value_is_exactly_zero_and_other_columns_untouched(self):
        on, inputs = _ours(use_null_expert=True), _inputs(5)
        values = on.routing_values(inputs[2])
        self.assertEqual(tuple(values.shape), (BATCH, REP_DIM, NUM_SRC + 1))
        self.assertTrue(torch.equal(values[:, :, -1], torch.zeros(BATCH, REP_DIM)))   # 精确零，非近似零
        self.assertTrue(torch.equal(values[:, :, :NUM_SRC], torch.stack(inputs[2], dim=2)))
        off = _ours()
        self.assertTrue(torch.equal(off.routing_values(inputs[2]), torch.stack(inputs[2], dim=2)))

    def test_null_key_is_the_only_new_parameter(self):
        ours = _ours(use_null_expert=True)
        extra = set(ours.state_dict()) - set(_base().state_dict())
        self.assertEqual(extra, {"null_key"})
        self.assertEqual(set(_base().state_dict()) - set(ours.state_dict()), set())
        self.assertEqual(tuple(ours.null_key.shape), (REP_DIM,))
        self.assertTrue(ours.null_key.requires_grad)

    def test_null_key_receives_gradient(self):
        # 固定初始化：部分随机初始化下 tower 的 ReLU 对全部样本死亡 → specific 路径梯度恒为 0
        # （master 既有结构性质：样本不激活任何 ReLU 时 out 与输入无关，全路径梯度精确为零，非本臂引入）。
        # 固定种子保证本测试确定性地落在"活跃初始化"上。
        torch.manual_seed(0)
        ours, inputs = _ours(use_null_expert=True), _inputs(6)
        _forward_backward(ours, inputs)
        self.assertIsNotNone(ours.null_key.grad)
        self.assertGreater(float(ours.null_key.grad.abs().sum()), 0.0)
        for name, param in ours.named_parameters():
            if name != "null_key":
                self.assertIsNotNone(param.grad, f"开启 Null Expert 后该参数丢失梯度: {name}")

    def test_enabled_forward_matches_reference_formula(self):
        """K+1 前向必须等于实验说明写死的公式（同温度 softmax，最后一列为零值）。"""
        ours, inputs = _ours(use_null_expert=True), _inputs(7)
        dnn_input, gen_rep, spec_reps, env_embs, _ = inputs
        keys = torch.cat([torch.stack(env_embs, dim=1), ours.null_key.unsqueeze(1)], dim=1)
        logits = torch.mm(ours.projection_network(dnn_input), keys) / ours.temperature
        weights = F.softmax(logits, dim=-1).unsqueeze(2)
        values = torch.cat([torch.stack(spec_reps, dim=2), torch.zeros(BATCH, REP_DIM, 1)], dim=2)
        new_spec_rep = torch.matmul(values, weights).squeeze()
        env_aware_rep = new_spec_rep * ours.env_embedding_network(ours.new_env_idx).squeeze(0)
        fused_rep = torch.matmul(torch.stack([env_aware_rep, gen_rep], dim=2),
                                 ours.gate_network(dnn_input).unsqueeze(dim=2)).squeeze()
        expected = ours.tower_network(fused_rep).squeeze()
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(*inputs[:4]), expected))


# ---- 诊断指标用的构造数据（float64：断言在累加精度下可精确表示）----
WEIGHTS = torch.tensor([[0.5, 0.3, 0.2],
                        [0.1, 0.2, 0.7],
                        [0.4, 0.4, 0.2]], dtype=torch.float64)
NULL_VALUES = [0.2, 0.7, 0.2]


def _mean(values):
    return sum(values) / len(values)


def _pvar(values):
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _quantile(values, p):
    """linear 插值分位数（与 numpy/torch 默认口径一致）：pos = p*(n-1)。"""
    ordered = sorted(values)
    pos = p * (len(ordered) - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    if lo + 1 >= len(ordered):
        return ordered[lo]
    return ordered[lo] + frac * (ordered[lo + 1] - ordered[lo])


def _entropy(row):
    return -sum(w * math.log(w) for w in row if w > 0)


class TestNullRouteStats(unittest.TestCase):
    """M7：null 候选的样本均值权重与 top1 占比（判定量，逐位可复现）。"""

    def test_null_mean_and_top1_rate(self):
        stats = metrics.NullRouteStats()
        stats.update(WEIGHTS)
        out = stats.result()
        self.assertAlmostEqual(out["null_mean"], _mean(NULL_VALUES), places=9)
        self.assertAlmostEqual(out["null_top1_rate"], 1 / 3, places=9)   # 平局取首个 → null 不算 top1

    def test_streaming_updates_equal_single_update(self):
        single, streamed = metrics.NullRouteStats(), metrics.NullRouteStats()
        single.update(WEIGHTS)
        streamed.update(WEIGHTS[:2])
        streamed.update(WEIGHTS[2:])
        self.assertEqual(single.result(), streamed.result())

    def test_single_row_batch(self):
        stats = metrics.NullRouteStats()
        stats.update(torch.tensor([[0.1, 0.2, 0.7]]))
        self.assertAlmostEqual(stats.result()["null_top1_rate"], 1.0, places=9)


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 才能复现累加器 device mismatch")
class TestNullRouteStatsCuda(unittest.TestCase):
    def test_cuda_weights_accumulate_on_cpu(self):
        stats = metrics.NullRouteStats()
        stats.update(WEIGHTS.cuda())
        out = stats.result()
        self.assertAlmostEqual(out["null_mean"], _mean(NULL_VALUES), places=9)
        self.assertAlmostEqual(out["null_top1_rate"], 1 / 3, places=9)


class TestNullRouteDiagnostics(unittest.TestCase):
    """C 类：null 质量分布/分位数/熵/样本方差（记录，不判定）。"""

    def test_distribution_stats(self):
        diag = metrics.NullRouteDiagnostics()
        diag.update(WEIGHTS)
        out = diag.result()
        self.assertAlmostEqual(out["null_var"], _pvar(NULL_VALUES), places=9)
        self.assertAlmostEqual(out["null_std"], math.sqrt(_pvar(NULL_VALUES)), places=9)
        for key, p in (("null_q10", 0.10), ("null_q25", 0.25), ("null_q50", 0.50),
                       ("null_q75", 0.75), ("null_q90", 0.90)):
            self.assertAlmostEqual(out[key], _quantile(NULL_VALUES, p), places=9, msg=key)

    def test_entropy_and_route_variance(self):
        diag = metrics.NullRouteDiagnostics()
        diag.update(WEIGHTS)
        out = diag.result()
        rows = WEIGHTS.tolist()
        entropies = [_entropy(row) for row in rows]
        self.assertAlmostEqual(out["route_entropy_mean"], _mean(entropies), places=9)
        self.assertAlmostEqual(out["route_entropy_std"], math.sqrt(_pvar(entropies)), places=9)
        self.assertAlmostEqual(out["route_var_mean"], _mean([_pvar(row) for row in rows]), places=9)

    def test_uniform_distribution_entropy_is_log_k(self):
        diag = metrics.NullRouteDiagnostics()
        diag.update(torch.full((4, 3), 1 / 3, dtype=torch.float64))
        self.assertAlmostEqual(diag.result()["route_entropy_mean"], math.log(3), places=9)
        self.assertAlmostEqual(diag.result()["route_var_mean"], _pvar([1 / 3, 1 / 3, 1 / 3]), places=9)


class TestSourceGateStats(unittest.TestCase):
    """源任务 gate 均值（backbone gate_networks 每任务 specific 分支，val）。"""

    def test_specific_branch_means(self):
        stats = metrics.SourceGateStats(NUM_SRC)
        # dtype=float64：断言与十进制字面量在 9 位小数下可比（float32 字面量有 ~1e-8 表示误差）
        stats.update([torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float64),
                      torch.tensor([[0.1, 0.9], [0.3, 0.7]], dtype=torch.float64)])
        stats.update([torch.tensor([[0.4, 0.6]], dtype=torch.float64),
                      torch.tensor([[0.5, 0.5]], dtype=torch.float64)])
        out = stats.result()
        self.assertAlmostEqual(out[0], (0.8 + 0.6 + 0.4) / 3, places=9)
        self.assertAlmostEqual(out[1], (0.1 + 0.3 + 0.5) / 3, places=9)
        self.assertEqual(len(out), NUM_SRC)


class TestNullSupervisionStats(unittest.TestCase):
    """分层与相关（BSI 标签 / 预测 / |误差|）与预测离散度。"""

    NULL = [0.1, 0.2, 0.3, 0.4]
    Y = [1.0, 0.0, 1.0, 0.0]
    PRED = [0.4, 0.3, 0.2, 0.1]

    def test_stratification_and_correlation(self):
        # float64：断言与十进制字面量的精确算术可比（float32 输入有 ~1e-8 表示误差）
        out = metrics.null_supervision_stats(self.NULL, torch.tensor(self.Y, dtype=torch.float64),
                                             torch.tensor(self.PRED, dtype=torch.float64))
        self.assertAlmostEqual(out["null_mean_bsi_pos"], 0.2, places=9)
        self.assertAlmostEqual(out["null_mean_bsi_neg"], 0.3, places=9)
        self.assertAlmostEqual(out["null_label_gap"], -0.1, places=9)
        self.assertAlmostEqual(out["corr_null_pred"], -1.0, places=9)     # 构造的完全负相关
        err = [abs(y - p) for y, p in zip(self.Y, self.PRED)]
        self.assertAlmostEqual(out["corr_null_abs_err"], _corr(self.NULL, err), places=9)
        self.assertAlmostEqual(out["pred_std"], math.sqrt(_pvar(self.PRED)), places=9)

    def test_single_class_and_degenerate_cases(self):
        out = metrics.null_supervision_stats([0.5, 0.5], torch.tensor([1.0, 1.0]),
                                             torch.tensor([0.2, 0.4]))
        self.assertIsNone(out["null_mean_bsi_neg"])
        self.assertIsNone(out["null_label_gap"])
        self.assertIsNone(out["corr_null_pred"])                          # 常量 null 值 → 相关无定义


def _corr(xs, ys):
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))


class TestNullArmVerdict(unittest.TestCase):
    """预注册接受标准（实验说明 §5）：PI Δ +0.0055 且 val 同向 且 top1 ∈ [0.05, 0.95]。"""

    def test_constants_pin_baseline_and_thresholds(self):
        self.assertEqual(metrics.NULL_ARM_BASELINE_TEST_AUC, 0.5988392178311113)
        self.assertEqual(metrics.NULL_ARM_BASELINE_VAL_AUC, 0.5781533414372665)
        self.assertEqual(metrics.NULL_ARM_AUC_MIN, 0.6043392178311113)               # +0.0055
        self.assertEqual(metrics.NULL_ARM_PROTOCOL_AUC_MIN, 0.6038392178311113)      # +0.005（协议 10.1）
        self.assertEqual((metrics.NULL_ARM_TOP1_MIN, metrics.NULL_ARM_TOP1_MAX), (0.05, 0.95))

    def test_boundaries_inclusive(self):
        ok = metrics.null_arm_verdict(0.6043392178311113, 0.5781533414372666, 0.05)
        self.assertTrue(ok["pass"])
        self.assertTrue(metrics.null_arm_verdict(0.6043392178311113, 0.5781533414372666, 0.95)["pass"])

    def test_below_threshold_stops(self):
        self.assertFalse(metrics.null_arm_verdict(0.6043392178311112, 0.5781533414372666, 0.5)["pass"])  # AUC 未达标
        self.assertFalse(metrics.null_arm_verdict(0.99, 0.5781533414372665, 0.5)["pass"])  # val 无正方向
        self.assertFalse(metrics.null_arm_verdict(0.99, 0.59, 0.049)["pass"])              # null 从不被选
        self.assertFalse(metrics.null_arm_verdict(0.99, 0.59, 0.951)["pass"])              # null 总被选

    def test_protocol_checks_recorded_separately(self):
        """主判据未过、协议 0.005 口径过 → pass=false 但 protocol_checks 如实记录（不得用宽松口径重判）。"""
        verdict = metrics.null_arm_verdict(0.6040, 0.5781533414372666, 0.5)
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["checks"]["auc"])
        self.assertTrue(verdict["protocol_checks"]["auc"])
        self.assertTrue(verdict["protocol_checks"]["val_direction"])

    def test_verdict_records_evidence(self):
        verdict = metrics.null_arm_verdict(0.61, 0.58, 0.3)
        self.assertEqual(verdict["test_auc"], 0.61)
        self.assertEqual(verdict["val_auc_best"], 0.58)
        self.assertEqual(verdict["null_top1_rate"], 0.3)
        self.assertEqual(verdict["baseline_test_auc"], 0.5988392178311113)
        self.assertEqual(verdict["baseline_val_auc"], 0.5781533414372665)
        self.assertEqual(verdict["top1_range"], [0.05, 0.95])
        self.assertAlmostEqual(verdict["delta_test_auc"], 0.61 - 0.5988392178311113, places=12)
        self.assertEqual(verdict["checks"], {"auc": True, "val_direction": True, "top1": True})

    def test_doc_preregisters_same_numbers(self):
        text = DOC.read_text(encoding="utf-8")
        for token in ("0.5988392178311113", "0.5781533414372665", "0.6043392178311113",
                      "0.0055", "[0.05, 0.95]",
                      "20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9",
                      "s1-5c060b9c-m1688723512-e3-3a30e2c0"):
            self.assertIn(token, text, f"实验说明未写死接受标准: {token}")


class TestCliFlag(unittest.TestCase):
    def test_null_expert_flag_defaults_to_false(self):
        parser = run_aliccp_benchmark.build_parser()
        self.assertFalse(parser.parse_args(["stage2", "--stage1-id", "x"]).null_expert)
        self.assertTrue(parser.parse_args(["stage2", "--stage1-id", "x", "--null-expert"]).null_expert)


# ---- 极小 AliCCP 夹具（与 test_smoke.py 同构，自包含）----
HEADER = "click,purchase,X,121,122,301"
TINY_VOCAB = {"121": 5, "122": 4}
TRAIN_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2], [0, 0, 9, 2, 2, 1], [1, 1, 9, 0, 1, 3],
    [0, 0, 8, 3, 0, 2], [1, 0, 8, 1, 1, 1], [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2],
]
VAL_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2],
]
TEST_ROWS = [
    [0, 0, 9, 2, 2, 1], [1, 0, 9, 0, 1, 3], [0, 0, 8, 3, 0, 2], [1, 1, 8, 1, 1, 1],
    [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2], [0, 0, 9, 1, 0, 3], [1, 0, 8, 4, 1, 2],
]


def _write_aliccp_file(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


def _make_tiny_root(td: str):
    root = Path(td) / "artifacts"
    data_dir = Path(td) / "data"
    data_dir.mkdir()
    _write_aliccp_file(data_dir / "train.csv", TRAIN_ROWS)
    _write_aliccp_file(data_dir / "val.csv", VAL_ROWS)
    _write_aliccp_file(data_dir / "test.csv", TEST_ROWS)
    data_files = {
        "train": str(data_dir / "train.csv"),
        "val": str(data_dir / "val.csv"),
        "test": str(data_dir / "test.csv"),
    }
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return root, data_files, budgets


class TestRunnerNullArm(unittest.TestCase):
    """runner 接线：两臂共享同一 Stage-1 产物，默认臂形状不变，处理臂落盘完整，冻结仍在。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root, cls.data_files, cls.budgets = _make_tiny_root(cls._tmp.name)
        cls.device = torch.device("cpu")
        cls.noop = staticmethod(lambda *a, **k: None)
        cls.meta = bench.run_stage1(
            root=cls.root, data_files=cls.data_files, budgets=cls.budgets, prefix_tag="tiny",
            model_seed=123, env_seed=456, epochs=2, patience=2, device=cls.device, vocab=TINY_VOCAB,
            expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4, log=cls.noop,
        )
        cls.baseline = cls._stage2(null_expert=False)
        cls.null = cls._stage2(null_expert=True)

    @classmethod
    def _stage2(cls, null_expert):
        return bench.run_stage2(
            root=cls.root, stage1_id=cls.meta["stage1_id"], data_files=cls.data_files, budgets=cls.budgets,
            prefix_tag="tiny", model_seed=123, epochs=1, patience=1, tag="smoke", device=cls.device,
            vocab=TINY_VOCAB, expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
            enforce_b=False, null_expert=null_expert, log=cls.noop,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _json(self, out, name):
        return json.loads((Path(out["run_dir"]) / name).read_text(encoding="utf-8"))

    def test_baseline_arm_artifact_shape_unchanged(self):
        self.assertNotIn("nullx", self.baseline["run_id"])
        self.assertFalse(self._json(self.baseline, "config.json")["null_expert"])
        recorded = self._json(self.baseline, "metrics.json")
        self.assertFalse(recorded["null_expert"])
        self.assertNotIn("null_arm", recorded)
        self.assertNotIn("mechanism", recorded)
        self.assertNotIn("null_mean", recorded)
        self.assertEqual(set(recorded) & {"null_top1_rate", "trainable_params"}, set())

    def test_null_arm_records_config_metrics_and_run_id(self):
        self.assertTrue(self.null["run_id"].endswith("-nullx"))
        self.assertEqual(self._json(self.null, "config.json")["null_expert"], True)
        recorded = self._json(self.null, "metrics.json")
        self.assertTrue(recorded["null_expert"])
        mech = recorded["mechanism"]
        for key in ("null_mean", "null_top1_rate", "null_std", "null_var", "null_q10", "null_q25",
                    "null_q50", "null_q75", "null_q90", "route_entropy_mean", "route_entropy_std",
                    "route_var_mean", "null_mean_bsi_pos", "null_mean_bsi_neg", "null_label_gap",
                    "corr_null_pred", "corr_null_abs_err", "source_gate_mean", "pred_std"):
            self.assertIn(key, mech, key)
        self.assertTrue(0.0 <= mech["null_mean"] <= 1.0)
        self.assertTrue(0.0 <= mech["null_top1_rate"] <= 1.0)
        self.assertEqual(len(mech["source_gate_mean"]), 2)
        self.assertTrue(all(0.0 <= g <= 1.0 for g in mech["source_gate_mean"]))
        self.assertEqual(recorded["trainable_params"] - recorded["trainable_params_default"], 4)  # = rep_dim
        self.assertEqual(recorded["null_arm"]["test_auc"], recorded["test_auc_bsi"])
        self.assertEqual(recorded["null_arm"]["top1_range"], [0.05, 0.95])
        self.assertIn("pass", recorded["null_arm"])

    def test_null_arm_keeps_freeze_and_shared_checkpoint(self):
        null_metrics, base_metrics = self._json(self.null, "metrics.json"), self._json(self.baseline, "metrics.json")
        self.assertEqual(null_metrics["stage1_id"], self.meta["stage1_id"])
        self.assertEqual(null_metrics["backbone_sha256_before"], base_metrics["backbone_sha256_before"])
        self.assertEqual(null_metrics["backbone_sha256_before"], null_metrics["backbone_sha256_after"])
        self.assertTrue(null_metrics["backbone_grads_none"])
        self.assertEqual(self.null["gates"]["A1"]["verdict"], "PASS")
        self.assertEqual(self.null["gates"]["A4"]["verdict"], "PASS")

    def test_default_head_param_count_matches_null_minus_key(self):
        off = NewTask(input_size=4, rep_dim=4, tower_dnn_hidden_units=(4,), reg_dnn=P.REG_DNN,
                      device=self.device)
        on = NewTask(input_size=4, rep_dim=4, tower_dnn_hidden_units=(4,), reg_dnn=P.REG_DNN,
                     device=self.device, use_null_expert=True)
        count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
        self.assertEqual(count(on) - count(off), 4)


class TestStaticGuards(unittest.TestCase):
    """I11：模型侧唯一允许的改动是 model.py 的 Null Expert 分支；协议文件零改动。"""

    def _git(self, *args):
        return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                              text=True, encoding="utf-8").stdout

    def test_only_model_py_differs_from_master(self):
        diff = self._git("diff", "--name-only", "master", "--", "multitaskrec", "config.py",
                         "AliCCP_MPTRec.py", "AliCCP_NewTask.py").split()
        self.assertEqual(diff, ["multitaskrec/model.py"],
                         f"除 model.py 的 Null Expert 外不得改动模型/master 文件: {diff}")

    def test_protocol_file_untouched_since_branch_base(self):
        diff = self._git("diff", "--name-only", BASE_COMMIT, "--", "aliccp_benchmark/protocol.py").split()
        self.assertEqual(diff, [], f"协议文件不得修改: {diff}")


if __name__ == "__main__":
    unittest.main()
