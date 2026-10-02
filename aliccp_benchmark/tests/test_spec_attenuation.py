"""特定混合固定衰减对照（post-hoc 归因控制）：对照说明
`docs/superpowers/specs/2026-10-03-aliccp-stage2-specific-attenuation-control-design.md`。

- 默认 `spec_attenuation=1.0`：NewTask 必须与分支基点 `8133d32:multitaskrec/model.py`（= 基线 run
  实际使用的模型代码）**逐位一致**（前向 + 反向 + 参数集合）。
- `spec_attenuation=c`（0 < c ≤ 1）：仅把 specific 混合 `new_spec_rep = Σ_k W_k·spec_rep_k` 乘以固定
  常数 c（对照臂 c = 0.6972233730330467）；不新增 router 类别、不新增参数、不改 loss。

测试用 CPU 极小夹具，不构成任何性能证据，只验证语义、指标与接线。
"""
import json
import math
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from aliccp_benchmark import bench, compare, metrics, protocol as P
from multitaskrec.model import NewTask

import run_aliccp_benchmark

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "superpowers" / "specs" / "2026-10-03-aliccp-stage2-specific-attenuation-control-design.md"
BASE_COMMIT = "8133d32"          # 本分支基点（infra/aliccp-fair-benchmark 顶端 = 基线 run 的模型代码）
COEF = 0.6972233730330467        # 预注册系数（spec §1.3）
BATCH, INPUT_SIZE, REP_DIM, NUM_SRC = 6, 8, 4, 2
NULL_RUN_DIR = REPO / "artifacts" / "aliccp_bench" / "runs" / metrics.NULL_RUN_ID
BASELINE_RUN_DIR = REPO / "artifacts" / "aliccp_bench" / "runs" / metrics.BASELINE_RUN_ID


def _base_newtask_class():
    """从 git 基点提交动态加载 NewTask 作为逐位对照的参考实现（不落地任何文件）。

    显式 `encoding="utf-8"`：Windows 下默认 locale 编码（GBK）会把含中文注释的 model.py 解码弄坏；
    errors="strict" 保证解码问题不被静默吞掉。
    """
    src = subprocess.run(["git", "show", f"{BASE_COMMIT}:multitaskrec/model.py"], cwd=REPO, check=True,
                         capture_output=True, text=True, encoding="utf-8", errors="strict").stdout
    module = types.ModuleType("model_base")
    exec(compile(src, f"{BASE_COMMIT}:multitaskrec/model.py", "exec"), module.__dict__)
    return module.NewTask


def _ours(spec_attenuation=1.0):
    return NewTask(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                   reg_dnn=P.REG_DNN, device=torch.device("cpu"), spec_attenuation=spec_attenuation)


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
    """默认臂（spec_attenuation=1.0）必须是基点实现的逐位复制。"""

    def test_default_state_dict_matches_base(self):
        ours = _ours()
        self.assertEqual(set(ours.state_dict()), set(_base().state_dict()))
        ours.load_state_dict(_base().state_dict())                        # strict=True：参数集合必须一致
        self.assertNotIn("spec_attenuation", ours.state_dict())           # 衰减不是参数、不进 state_dict

    def test_default_forward_bit_identical(self):
        base, ours, inputs = _base(), _ours(), _inputs(1)
        ours.load_state_dict(base.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(*inputs[:4]), base(*inputs[:4])))

    def test_default_backward_bit_identical(self):
        torch.manual_seed(0)  # 固定构造 RNG：部分随机初始化下 tower 的 ReLU 全灭会零化整条 specific 路径梯度
        base, inputs = _base(), _inputs(2)
        torch.manual_seed(0)
        ours = _ours()                                                    # 构造顺序相同 → 初始化逐位一致
        pred_base, pred_ours = _forward_backward(base, inputs), _forward_backward(ours, inputs)
        self.assertTrue(torch.equal(pred_base, pred_ours))
        grads_ours = dict(ours.named_parameters())
        for name, param in base.named_parameters():
            self.assertIsNotNone(param.grad, f"基点该参数无梯度，反向对照无效: {name}")
            self.assertTrue(torch.equal(param.grad, grads_ours[name].grad), f"梯度非逐位一致: {name}")

    def test_attenuation_does_not_consume_rng(self):
        """同一种子下开启/关闭衰减的共享参数初始化必须逐位一致（无参数量、无 RNG 消耗）。"""
        torch.manual_seed(3)
        off = _ours()
        torch.manual_seed(3)
        on = _ours(spec_attenuation=COEF)
        for name, param in off.named_parameters():
            self.assertTrue(torch.equal(param, dict(on.named_parameters())[name]), f"共享参数初始化漂移: {name}")
        self.assertEqual(on.spec_attenuation, COEF)

    def test_default_spec_attenuation_is_one(self):
        self.assertEqual(_ours().spec_attenuation, 1.0)


class TestAttenuationSemantics(unittest.TestCase):
    """开启衰减后的语义：公式、参数集合、梯度、合法性。"""

    def test_coefficient_literals(self):
        self.assertEqual(metrics.SPEC_ATTENUATION_COEF, 1.0 - 0.3027766269669533)
        self.assertEqual(metrics.SPEC_ATTENUATION_COEF, COEF)
        self.assertEqual(1.0 - metrics.NULL_RUN_NULL_MEAN, COEF)
        self.assertEqual(repr(metrics.SPEC_ATTENUATION_COEF_F32), repr(float(torch.tensor(COEF, dtype=torch.float32))))
        self.assertTrue(0.0 < metrics.SPEC_ATTENUATION_COEF <= 1.0)

    def test_invalid_coefficients_raise(self):
        for bad in (0.0, -0.5, 1.0000001, 2.0):
            with self.assertRaises(ValueError, msg=f"c={bad} 应拒绝"):
                _ours(spec_attenuation=bad)
        _ours(spec_attenuation=1.0)          # 合法边界
        _ours(spec_attenuation=COEF)

    def test_enabled_forward_matches_reference_formula(self):
        """写死公式：new_spec_rep' = c × Σ_k W_k spec_rep_k（同温度 softmax，其余同基点）。"""
        ours, inputs = _ours(spec_attenuation=COEF), _inputs(7)
        dnn_input, gen_rep, spec_reps, env_embs, _ = inputs
        keys = torch.stack(env_embs, dim=1)
        logits = torch.mm(ours.projection_network(dnn_input), keys) / ours.temperature
        weights = F.softmax(logits, dim=-1).unsqueeze(2)
        new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), weights).squeeze()
        new_spec_rep = new_spec_rep * ours.spec_attenuation
        env_aware_rep = new_spec_rep * ours.env_embedding_network(ours.new_env_idx).squeeze(0)
        fused_rep = torch.matmul(torch.stack([env_aware_rep, gen_rep], dim=2),
                                 ours.gate_network(dnn_input).unsqueeze(dim=2)).squeeze()
        expected = ours.tower_network(fused_rep).squeeze()
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(*inputs[:4]), expected))

    def test_enabled_differs_from_default_same_weights(self):
        torch.manual_seed(0)
        base, inputs = _base(), _inputs(8)
        torch.manual_seed(0)
        ours = _ours(spec_attenuation=COEF)
        with torch.no_grad():
            self.assertFalse(torch.equal(ours(*inputs[:4]), base(*inputs[:4])))

    def test_no_new_parameters_and_params_count_equal(self):
        base, ours = _base(), _ours(spec_attenuation=COEF)
        self.assertEqual(set(ours.state_dict()), set(base.state_dict()))
        self.assertEqual(set(dict(ours.named_parameters())), set(dict(base.named_parameters())))
        count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.assertEqual(count(ours), count(base))

    def test_get_l2_reg_unchanged(self):
        base, ours, inputs = _base(), _ours(spec_attenuation=COEF), _inputs(9)
        ours.load_state_dict(base.state_dict())
        self.assertTrue(torch.equal(ours.get_l2_reg(), base.get_l2_reg()))

    def test_enabled_backward_gradients_all_present(self):
        # 固定种子保证确定性地落在"tower ReLU 活跃"的初始化上（基点既有结构性质，非本臂引入）。
        torch.manual_seed(0)
        ours, inputs = _ours(spec_attenuation=COEF), _inputs(6)
        _forward_backward(ours, inputs)
        for name, param in ours.named_parameters():
            self.assertIsNotNone(param.grad, f"开启衰减后该参数丢失梯度: {name}")
            self.assertGreater(float(param.grad.abs().sum()), 0.0, f"该参数梯度全零: {name}")


# ---- 诊断指标用的构造数据（float64：断言在累加精度下可精确表示）----
GATE_OUTS = [torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.5, 0.5]], dtype=torch.float64),
             torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.8, 0.2]], dtype=torch.float64)]
PRED_A = torch.tensor([0.10, 0.20, 0.30, 0.40, 0.50], dtype=torch.float64)
PRED_B = torch.tensor([0.12, 0.18, 0.33, 0.41, 0.55], dtype=torch.float64)


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


class TestDiagnosticMetrics(unittest.TestCase):
    """C 类诊断纯函数（记录，不判定）。"""

    def test_source_gate_stats_matches_manual(self):
        stats = metrics.SourceGateStats(num_tasks=2)
        stats.update(GATE_OUTS)
        expected = [float(GATE_OUTS[0][:, 0].mean()), float(GATE_OUTS[1][:, 0].mean())]
        self.assertEqual(stats.result(), expected)

    def test_source_gate_stats_streaming_equals_single(self):
        single, streamed = metrics.SourceGateStats(2), metrics.SourceGateStats(2)
        single.update(GATE_OUTS)
        streamed.update([g[:1] for g in GATE_OUTS])
        streamed.update([g[1:] for g in GATE_OUTS])
        self.assertEqual(single.result(), streamed.result())

    def test_pred_dispersion_matches_manual(self):
        out = metrics.pred_dispersion(PRED_A)
        values = PRED_A.tolist()
        self.assertEqual(out["pred_mean"], _mean(values))
        self.assertEqual(out["pred_var"], _pvar(values))
        self.assertEqual(out["pred_std"], math.sqrt(_pvar(values)))
        self.assertEqual(out["pred_min"], min(values))
        self.assertEqual(out["pred_max"], max(values))
        for key, p in (("pred_q10", 0.10), ("pred_q25", 0.25), ("pred_q50", 0.50),
                       ("pred_q75", 0.75), ("pred_q90", 0.90)):
            self.assertAlmostEqual(out[key], _quantile(values, p), places=12, msg=key)

    def test_pearson_matches_numpy(self):
        import numpy as np
        a, b = PRED_A.numpy(), PRED_B.numpy()
        expected = float(np.corrcoef(a, b)[0, 1])
        self.assertAlmostEqual(metrics.pearson_corr(PRED_A, PRED_B), expected, places=12)
        self.assertIsNone(metrics.pearson_corr(PRED_A, torch.full_like(PRED_A, 0.3)))  # 零方差

    def test_spearman_matches_scipy_with_ties(self):
        from scipy.stats import rankdata, spearmanr
        a = torch.tensor([0.1, 0.5, 0.5, 0.9, -0.3], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 0.5, 3.0, -1.0], dtype=torch.float64)
        self.assertTrue(torch.equal(metrics.average_rank(a),
                                    torch.from_numpy(rankdata(a.numpy(), method="average"))))
        self.assertAlmostEqual(metrics.spearman_corr(a, b), float(spearmanr(a.numpy(), b.numpy()).statistic),
                               places=12)

    def test_logit_clip(self):
        out = metrics.logit_clip(torch.tensor([0.5, 0.0, 1.0], dtype=torch.float64))
        self.assertEqual(float(out[0]), 0.0)
        self.assertTrue(torch.isfinite(out).all())              # 0/1 被 eps 截断 → 有限
        self.assertLess(float(out[1]), -20.0)                   # 0.0 → 大负 logit
        self.assertGreater(float(out[2]), 20.0)                 # 1.0 → 大正 logit

    def test_paired_delta_stats_matches_manual(self):
        out = metrics.paired_delta_stats(PRED_A, PRED_B)
        deltas = (PRED_A - PRED_B).tolist()
        self.assertEqual(out["delta_mean"], _mean(deltas))
        self.assertEqual(out["delta_std"], math.sqrt(_pvar(deltas)))
        self.assertEqual(out["delta_min"], min(deltas))
        self.assertEqual(out["delta_max"], max(deltas))
        self.assertEqual(out["frac_pos"], sum(1 for d in deltas if d > 0) / len(deltas))
        self.assertEqual(out["frac_neg"], sum(1 for d in deltas if d < 0) / len(deltas))
        self.assertAlmostEqual(out["delta_q50"], _quantile(deltas, 0.5), places=12)

    def test_cross_arm_stats_keys_and_values(self):
        out = metrics.cross_arm_stats(PRED_A, PRED_B)
        for key in ("pearson_logit", "pearson_pred", "spearman_pred", "delta_mean", "delta_std",
                    "delta_min", "delta_max", "frac_pos", "frac_neg", "delta_q10", "delta_q50", "delta_q90"):
            self.assertIn(key, out, key)
        self.assertAlmostEqual(out["pearson_pred"], metrics.pearson_corr(PRED_A, PRED_B), places=15)
        self.assertAlmostEqual(out["pearson_logit"],
                               metrics.pearson_corr(metrics.logit_clip(PRED_A), metrics.logit_clip(PRED_B)),
                               places=15)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 才能复现累加器 device mismatch")
    def test_source_gate_stats_cuda_accumulates_on_cpu(self):
        stats = metrics.SourceGateStats(2)
        stats.update([g.cuda() for g in GATE_OUTS])
        expected = [float(GATE_OUTS[0][:, 0].mean()), float(GATE_OUTS[1][:, 0].mean())]
        self.assertEqual(stats.result(), expected)


class TestAttenuationArmVerdict(unittest.TestCase):
    """预注册判定（spec §5）：reproduces_most（AUC 字面量，闭）× val_direction（严格 >）。"""

    def test_boundary_repro_min(self):
        ok = metrics.attenuation_arm_verdict(metrics.ATTN_ARM_AUC_MIN_REPRO, metrics.BASELINE_VAL_AUC + 1e-6)
        self.assertTrue(ok["checks"]["reproduces_most"])
        below = metrics.attenuation_arm_verdict(metrics.ATTN_ARM_AUC_MIN_REPRO - 1e-12, metrics.BASELINE_VAL_AUC + 1e-6)
        self.assertFalse(below["checks"]["reproduces_most"])

    def test_boundary_val_direction_is_strict(self):
        flat = metrics.attenuation_arm_verdict(metrics.ATTN_ARM_AUC_MIN_REPRO, metrics.BASELINE_VAL_AUC)
        self.assertFalse(flat["checks"]["val_direction"])
        above = metrics.attenuation_arm_verdict(metrics.ATTN_ARM_AUC_MIN_REPRO, metrics.BASELINE_VAL_AUC + 1e-12)
        self.assertTrue(above["checks"]["val_direction"])

    def test_supported_requires_both(self):
        v = metrics.attenuation_arm_verdict(metrics.ATTN_ARM_AUC_MIN_REPRO, metrics.BASELINE_VAL_AUC + 1e-6)
        self.assertEqual(v["classification"], "ATTENUATION_SUPPORTED")
        v2 = metrics.attenuation_arm_verdict(metrics.ATTN_ARM_AUC_MIN_REPRO, metrics.BASELINE_VAL_AUC - 1e-6)
        self.assertEqual(v2["classification"], "ATTENUATION_FALSIFIED")

    def test_pi_recorded_separately(self):
        """主判据未过、PI +0.0055 口径过 → FALSIFIED 但 pi_utility 如实记录（不得用宽松口径重判）。"""
        auc = metrics.ATTN_ARM_AUC_MIN_PI + 1e-6
        v = metrics.attenuation_arm_verdict(auc, metrics.BASELINE_VAL_AUC + 1e-6)
        self.assertEqual(v["classification"], "ATTENUATION_FALSIFIED")
        self.assertTrue(v["checks"]["pi_utility"])
        self.assertFalse(v["checks"]["reproduces_most"])

    def test_band_values(self):
        dt = metrics.NULL_DELTA_TEST_AUC
        base_t, base_v = metrics.BASELINE_TEST_AUC, metrics.BASELINE_VAL_AUC + 1e-6
        self.assertEqual(metrics.attenuation_arm_verdict(base_t + 0.70 * dt, base_v)["band"], "MOST")
        self.assertEqual(metrics.attenuation_arm_verdict(base_t + 0.69 * dt, base_v)["band"], "PARTIAL")
        self.assertEqual(metrics.attenuation_arm_verdict(base_t + 0.29 * dt, base_v)["band"], "MINOR")
        self.assertEqual(metrics.attenuation_arm_verdict(base_t - 0.01 * dt, base_v)["band"], "NEGATIVE")

    def test_evidence_fields(self):
        v = metrics.attenuation_arm_verdict(0.61, 0.58)
        self.assertEqual(v["baseline_test_auc"], 0.5988392178311113)
        self.assertEqual(v["baseline_val_auc"], 0.5781533414372665)
        self.assertEqual(v["null_arm_test_auc"], 0.6126857532817985)
        self.assertEqual(v["null_arm_val_auc"], 0.5919398413138859)
        self.assertEqual(v["null_delta_test_auc"], 0.013846535450687258)
        self.assertEqual(v["null_delta_val_auc"], 0.013786499876619396)
        self.assertEqual(v["auc_min_repro"], 0.6085317926465923)
        self.assertEqual(v["auc_min_pi"], 0.6043392178311113)
        self.assertAlmostEqual(v["delta_test_auc"], 0.61 - 0.5988392178311113, places=15)
        self.assertAlmostEqual(v["reproduction_ratio_test"], (0.61 - 0.5988392178311113) / 0.013846535450687258,
                               places=12)
        self.assertEqual(v["checks"], {"reproduces_most": True, "val_direction": True, "pi_utility": True})

    def test_doc_preregisters_same_numbers(self):
        text = DOC.read_text(encoding="utf-8")
        for token in ("0.3027766269669533", "0.6972233730330467", "0.6972233653068542",
                      "0.6085317926465923", "0.6043392178311113",
                      "0.5988392178311113", "0.5781533414372665",
                      "0.6126857532817985", "0.5919398413138859",
                      "0.013846535450687258", "0.013786499876619396",
                      "0.008530222646916882", "0.004380239336642408",
                      "0.176159685842067", "0.2537776756367646",
                      "0.48325546875", "0.5167445625", "0.851149", "0.1488510625",
                      "20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9",
                      "20261003-0221-p2M-v500k-t1M-m1688723512-short-6224c0f-nullx",
                      "s1-5c060b9c-m1688723512-e3-3a30e2c0",
                      "836db5327b550cdcd07dee004c1c30e75010df43b38d5f950a41f1977539786c",
                      "6224c0f", "aba8ee9", "0.70", "0.30"):
            self.assertIn(token, text, f"实验说明未写死: {token}")


# ---- 极小 AliCCP 夹具（与 test_null_expert.py 同构，自包含）----
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
TINY_DIMS = dict(expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4)


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


class TestRunnerAttenuationArm(unittest.TestCase):
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
            **TINY_DIMS, log=cls.noop,
        )
        cls.baseline = cls._stage2(spec_attenuation=1.0)
        cls.arm = cls._stage2(spec_attenuation=metrics.SPEC_ATTENUATION_COEF)

    @classmethod
    def _stage2(cls, spec_attenuation):
        return bench.run_stage2(
            root=cls.root, stage1_id=cls.meta["stage1_id"], data_files=cls.data_files, budgets=cls.budgets,
            prefix_tag="tiny", model_seed=123, epochs=1, patience=1, tag="smoke", device=cls.device,
            vocab=TINY_VOCAB, **TINY_DIMS, enforce_b=False, spec_attenuation=spec_attenuation, log=cls.noop,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _json(self, out, name):
        return json.loads((Path(out["run_dir"]) / name).read_text(encoding="utf-8"))

    def test_baseline_arm_artifact_shape_unchanged(self):
        self.assertNotIn("sattn", self.baseline["run_id"])
        self.assertEqual(self._json(self.baseline, "config.json")["spec_attenuation"], 1.0)
        recorded = self._json(self.baseline, "metrics.json")
        self.assertEqual(recorded["spec_attenuation"], 1.0)
        self.assertNotIn("attenuation_arm", recorded)
        self.assertNotIn("probe", recorded)
        self.assertNotIn("trainable_params", recorded)

    def test_arm_records_config_metrics_and_run_id(self):
        self.assertTrue(self.arm["run_id"].endswith("-sattn"))
        self.assertEqual(self._json(self.arm, "config.json")["spec_attenuation"], metrics.SPEC_ATTENUATION_COEF)
        recorded = self._json(self.arm, "metrics.json")
        self.assertEqual(recorded["spec_attenuation"], metrics.SPEC_ATTENUATION_COEF)
        probe = recorded["probe"]
        for key in ("pred_mean", "pred_std", "pred_var", "pred_min", "pred_max",
                    "pred_q10", "pred_q25", "pred_q50", "pred_q75", "pred_q90", "source_gate_mean"):
            self.assertIn(key, probe, key)
        self.assertEqual(len(probe["source_gate_mean"]), 2)
        self.assertTrue(all(0.0 <= g <= 1.0 for g in probe["source_gate_mean"]))
        self.assertEqual(recorded["trainable_params"], recorded["trainable_params_default"])
        verdict = recorded["attenuation_arm"]
        self.assertIn(verdict["classification"], {"ATTENUATION_SUPPORTED", "ATTENUATION_FALSIFIED"})
        self.assertEqual(verdict["test_auc"], recorded["test_auc_bsi"])
        self.assertEqual(verdict["val_auc_best"], recorded["best_val_auc_bsi"])
        self.assertEqual(verdict["auc_min_repro"], 0.6085317926465923)

    def test_arm_keeps_freeze_and_shared_checkpoint(self):
        arm_metrics = self._json(self.arm, "metrics.json")
        base_metrics = self._json(self.baseline, "metrics.json")
        self.assertEqual(arm_metrics["stage1_id"], self.meta["stage1_id"])
        self.assertEqual(arm_metrics["backbone_sha256_before"], base_metrics["backbone_sha256_before"])
        self.assertEqual(arm_metrics["backbone_sha256_before"], arm_metrics["backbone_sha256_after"])
        self.assertTrue(arm_metrics["backbone_grads_none"])
        self.assertEqual(self.arm["gates"]["A1"]["verdict"], "PASS")
        self.assertEqual(self.arm["gates"]["A4"]["verdict"], "PASS")


class TestComparePipeline(unittest.TestCase):
    """三方对照（spec §6.2）：predict/evaluate 等价、null 头类重构、端到端完整性。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root, cls.data_files, cls.budgets = _make_tiny_root(cls._tmp.name)
        cls.device = torch.device("cpu")
        cls.noop = staticmethod(lambda *a, **k: None)
        cls.meta = bench.run_stage1(
            root=cls.root, data_files=cls.data_files, budgets=cls.budgets, prefix_tag="tiny",
            model_seed=123, env_seed=456, epochs=2, patience=2, device=cls.device, vocab=TINY_VOCAB,
            **TINY_DIMS, log=cls.noop,
        )
        cls.baseline = bench.run_stage2(
            root=cls.root, stage1_id=cls.meta["stage1_id"], data_files=cls.data_files, budgets=cls.budgets,
            prefix_tag="tiny", model_seed=123, epochs=1, patience=1, tag="smoke", device=cls.device,
            vocab=TINY_VOCAB, **TINY_DIMS, enforce_b=False, log=cls.noop,
        )
        cls.control = bench.run_stage2(
            root=cls.root, stage1_id=cls.meta["stage1_id"], data_files=cls.data_files, budgets=cls.budgets,
            prefix_tag="tiny", model_seed=123, epochs=1, patience=1, tag="smoke", device=cls.device,
            vocab=TINY_VOCAB, **TINY_DIMS, enforce_b=False, spec_attenuation=COEF, log=cls.noop,
        )
        cls.null_run_id = cls._fabricate_null_run()
        cls.result = compare.run_comparison(
            root=cls.root, stage1_id=cls.meta["stage1_id"],
            run_ids={"baseline": cls.baseline["run_id"], "null": cls.null_run_id, "control": cls.control["run_id"]},
            data_files=cls.data_files, budgets=cls.budgets, prefix_tag="tiny", device=cls.device,
            vocab=TINY_VOCAB, **TINY_DIMS, log=cls.noop,
        )

    @classmethod
    def _fabricate_null_run(cls):
        """构造一个"前臂形态"的 null 头 run（随机权重 + 自洽 metrics.json），供对照管线练通。"""
        model, _ = compare.build_frozen_backbone(
            cls.root, cls.meta["stage1_id"], cls.device, vocab=TINY_VOCAB, **TINY_DIMS,
        )
        _, loaders = bench._loaders(cls.data_files, cls.budgets, P.BATCH_SIZE)
        null_cls, _ = compare.load_null_head_class()
        torch.manual_seed(7)
        head = null_cls(input_size=4, rep_dim=4, tower_dnn_hidden_units=(4,), reg_dnn=P.REG_DNN,
                        device=cls.device, use_null_expert=True)
        val_auc = bench.evaluate_newtask(head, model, loaders["val"], cls.device)
        test_auc = bench.evaluate_newtask(head, model, loaders["test"], cls.device)
        gate_mean = bench.newtask_gate_mean(head, model, loaders["val"], cls.device)
        source_gate = compare.source_gate_probe(model, loaders["val"], cls.device)
        run_id = "tiny-null-fabricated"
        run_dir = P.run_dir(cls.root, run_id)
        run_dir.mkdir(parents=True)
        torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, run_dir / "newtask.pt")
        (run_dir / "metrics.json").write_text(json.dumps({
            "run_id": run_id, "commit": "fabricated", "test_auc_bsi": test_auc, "best_val_auc_bsi": val_auc,
            "gate_mean": gate_mean, "mechanism": {"source_gate_mean": source_gate},
        }), encoding="utf-8")
        return run_id

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_predict_auc_equals_evaluate_auc(self):
        model, _ = compare.build_frozen_backbone(self.root, self.meta["stage1_id"], self.device,
                                                 vocab=TINY_VOCAB, **TINY_DIMS)
        _, loaders = bench._loaders(self.data_files, self.budgets, P.BATCH_SIZE)
        torch.manual_seed(11)
        head = NewTask(input_size=4, rep_dim=4, tower_dnn_hidden_units=(4,), reg_dnn=P.REG_DNN, device=self.device)
        via_evaluate = bench.evaluate_newtask(head, model, loaders["val"], self.device)
        y_true, preds = compare.predict_newtask(head, model, loaders["val"], self.device)
        self.assertEqual(metrics.auc_score(y_true, preds), via_evaluate)          # 逐位一致

    def test_null_head_class_source_pinned(self):
        null_cls, sha = compare.load_null_head_class()
        self.assertEqual(sha, compare.NULL_HEAD_MODEL_SHA256)
        self.assertEqual(sha, "836db5327b550cdcd07dee004c1c30e75010df43b38d5f950a41f1977539786c")
        head = null_cls(input_size=4, rep_dim=4, tower_dnn_hidden_units=(4,), reg_dnn=P.REG_DNN,
                        device=self.device, use_null_expert=True)
        self.assertIn("null_key", dict(head.named_parameters()))

    @unittest.skipUnless(NULL_RUN_DIR.exists() and BASELINE_RUN_DIR.exists(), "真实 run 产物不存在（本地未跑过）")
    def test_real_artifacts_strict_load(self):
        import torch as _torch
        null_cls, _ = compare.load_null_head_class()
        null_head = null_cls(input_size=80, rep_dim=64, tower_dnn_hidden_units=(32, 32), reg_dnn=P.REG_DNN,
                             device=self.device, use_null_expert=True)
        null_head.load_state_dict(_torch.load(NULL_RUN_DIR / "newtask.pt", map_location="cpu"), strict=True)
        self.assertEqual(len(null_head.null_key), 64)
        base_head = NewTask(input_size=80, rep_dim=64, tower_dnn_hidden_units=(32, 32), reg_dnn=P.REG_DNN,
                            device=self.device)
        base_head.load_state_dict(_torch.load(BASELINE_RUN_DIR / "newtask.pt", map_location="cpu"), strict=True)

    def test_comparison_result_integrity_and_shape(self):
        result = self.result
        self.assertTrue(result["integrity"]["pass"], result["integrity"]["checks"])
        checks = result["integrity"]["checks"]
        for arm in ("baseline", "null", "control"):
            self.assertTrue(checks[f"{arm}.test_auc_matches_recorded"]["pass"], arm)
            self.assertTrue(checks[f"{arm}.val_auc_matches_recorded"]["pass"], arm)
            self.assertTrue(checks[f"{arm}.gate_mean_matches_recorded"]["pass"], arm)
        self.assertTrue(checks["y_true_identical_across_arms"]["pass"])
        self.assertTrue(checks["null_head_source_sha256"]["pass"])
        for pair in ("control_vs_null", "control_vs_baseline", "null_vs_baseline"):
            stats = result["comparisons"][pair]
            self.assertIn("pearson_logit", stats)
            self.assertIn("delta_mean", stats)
        self.assertIn("val_dispersion", result["arms"]["control"])
        self.assertIn("test_dispersion", result["arms"]["null"])
        # control 重算 val pred_std 与 in-run 探针逐位一致
        control_probe = json.loads((P.run_dir(self.root, self.control["run_id"]) / "metrics.json")
                                   .read_text(encoding="utf-8"))["probe"]
        self.assertEqual(result["arms"]["control"]["val_dispersion"]["pred_std"], control_probe["pred_std"])
        self.assertTrue(checks["control.pred_std_matches_probe"]["pass"])
        out_path = P.run_dir(self.root, self.control["run_id"]) / "three_way_compare.json"
        self.assertTrue(out_path.exists())
        json.loads(out_path.read_text(encoding="utf-8"))

    def test_integrity_failure_is_flagged_not_raised(self):
        tampered = "tiny-baseline-tampered"
        shutil.copytree(P.run_dir(self.root, self.baseline["run_id"]), P.run_dir(self.root, tampered))
        metrics_path = P.run_dir(self.root, tampered) / "metrics.json"
        doc = json.loads(metrics_path.read_text(encoding="utf-8"))
        doc["test_auc_bsi"] = doc["test_auc_bsi"] + 1e-6
        metrics_path.write_text(json.dumps(doc), encoding="utf-8")
        result = compare.run_comparison(
            root=self.root, stage1_id=self.meta["stage1_id"],
            run_ids={"baseline": tampered, "null": self.null_run_id, "control": self.control["run_id"]},
            data_files=self.data_files, budgets=self.budgets, prefix_tag="tiny", device=self.device,
            vocab=TINY_VOCAB, **TINY_DIMS, log=self.noop,
        )
        self.assertFalse(result["integrity"]["pass"])
        self.assertFalse(result["integrity"]["checks"]["baseline.test_auc_matches_recorded"]["pass"])


class TestCliFlags(unittest.TestCase):
    def test_spec_attenuation_flag(self):
        parser = run_aliccp_benchmark.build_parser()
        self.assertEqual(parser.parse_args(["stage2", "--stage1-id", "x"]).spec_attenuation, 1.0)
        parsed = parser.parse_args(["stage2", "--stage1-id", "x", "--spec-attenuation", "0.6972233730330467"])
        self.assertEqual(parsed.spec_attenuation, 0.6972233730330467)

    def test_compare_subcommand_defaults(self):
        args = run_aliccp_benchmark.build_parser().parse_args(["compare", "--stage1-id", "x", "--arm-run", "y"])
        self.assertEqual(args.baseline_run, metrics.BASELINE_RUN_ID)
        self.assertEqual(args.null_run, metrics.NULL_RUN_ID)
        self.assertEqual(args.arm_run, "y")


class TestStaticGuards(unittest.TestCase):
    """I10：模型侧唯一允许的改动是 model.py 的衰减分支；协议文件零改动；跟踪改动 ⊆ 白名单。"""

    WHITELIST = {
        "multitaskrec/model.py",
        "aliccp_benchmark/metrics.py",
        "aliccp_benchmark/bench.py",
        "aliccp_benchmark/compare.py",
        "aliccp_benchmark/tests/test_spec_attenuation.py",
        "run_aliccp_benchmark.py",
        "docs/superpowers/specs/2026-10-03-aliccp-stage2-specific-attenuation-control-design.md",
        "artifacts/aliccp_bench/SUMMARY.md",
    }

    def _git(self, *args):
        return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                              text=True, encoding="utf-8").stdout

    def test_only_model_py_differs_model_side(self):
        diff = self._git("diff", "--name-only", BASE_COMMIT, "--", "multitaskrec", "config.py",
                         "AliCCP_MPTRec.py", "AliCCP_NewTask.py").split()
        self.assertEqual(diff, ["multitaskrec/model.py"],
                         f"除 model.py 的 spec_attenuation 外不得改动模型/master 文件: {diff}")

    def test_protocol_file_untouched_since_branch_base(self):
        diff = self._git("diff", "--name-only", BASE_COMMIT, "--", "aliccp_benchmark/protocol.py").split()
        self.assertEqual(diff, [], f"协议文件不得修改: {diff}")

    def test_tracked_diff_within_whitelist(self):
        changed = self._git("diff", "--name-only", BASE_COMMIT).split()
        extra = [p for p in changed if p not in self.WHITELIST]
        self.assertEqual(extra, [], f"白名单外的跟踪文件被改动: {extra}")


if __name__ == "__main__":
    unittest.main()
