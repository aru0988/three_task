"""Null Expert C 类诊断（移植自姊妹线 exp/aliccp-stage2-null-expert 6224c0f）+ runner 探针与落盘接线。

预注册：`docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-design.md` §5：
- 诊断只在**处理臂**、只在 **val** 上算；基线臂 mechanism 键集必须逐字不变。
- 探针的 M7 值（null_mean / null_top1_rate）必须与 `evaluate_newtask(mechanism=True)` **逐位一致**
  （同一权重、同一 loader、同一遍历顺序）。
- `null_arm`（0.8521 历史判据）字段逐字保留，不受本文件任何改动影响。
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import metrics
from census_benchmark import protocol as P
from multitaskrec.model import MPTRec, NewTask
import run_census_benchmark

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "superpowers" / "specs" / "2026-10-05-census-stage2-null-expert-multiseed-design.md"
BATCH, INPUT_SIZE, REP_DIM, NUM_SRC = 6, 8, 4, 2

WEIGHTS = torch.tensor([[0.5, 0.3, 0.2],
                        [0.1, 0.2, 0.7],
                        [0.4, 0.4, 0.2]], dtype=torch.float64)

DIAGNOSTIC_KEYS = {"null_std", "null_var", "route_entropy_mean", "route_entropy_std", "route_var_mean",
                   "null_q10", "null_q25", "null_q50", "null_q75", "null_q90",
                   "null_mean_edu_pos", "null_mean_edu_neg", "null_label_gap",
                   "corr_null_pred", "corr_null_abs_err", "pred_std"}


class TestNullRouteDiagnostics(unittest.TestCase):
    """C 类：null 质量分布、router 熵、逐样本路由方差、分位数。"""

    def test_known_values_match_independent_float64_reference(self):
        diag = metrics.NullRouteDiagnostics()
        diag.update(WEIGHTS)
        out = diag.result()
        values = WEIGHTS[:, -1]
        self.assertAlmostEqual(out["null_std"], float(values.std(unbiased=False)), places=12)
        self.assertAlmostEqual(out["null_var"], float(values.var(unbiased=False)), places=12)
        entropy = -(WEIGHTS.clamp_min(1e-12).log() * WEIGHTS).sum(dim=1)
        self.assertAlmostEqual(out["route_entropy_mean"], float(entropy.mean()), places=12)
        self.assertAlmostEqual(out["route_entropy_std"], float(entropy.std(unbiased=False)), places=12)
        self.assertAlmostEqual(out["route_var_mean"], float(WEIGHTS.var(dim=1, unbiased=False).mean()), places=12)
        for q in (10, 25, 50, 75, 90):
            self.assertAlmostEqual(out[f"null_q{q:02d}"], float(torch.quantile(values, q / 100)), places=12)

    def test_streaming_updates_match_single_update(self):
        single, streamed = metrics.NullRouteDiagnostics(), metrics.NullRouteDiagnostics()
        single.update(WEIGHTS)
        streamed.update(WEIGHTS[:2])
        streamed.update(WEIGHTS[2:])
        one, two = single.result(), streamed.result()
        for key in ("null_std", "null_var", "route_entropy_mean", "route_entropy_std", "route_var_mean",
                    "null_q10", "null_q50", "null_q90"):
            self.assertAlmostEqual(one[key], two[key], places=12, msg=key)

    def test_exact_zero_weight_does_not_produce_nan(self):
        diag = metrics.NullRouteDiagnostics()
        diag.update(torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64))     # 精确 0 → log(0) 需被 clamp 防住
        out = diag.result()
        for key, value in out.items():
            self.assertFalse(value != value, f"NaN: {key}")                   # NaN != NaN


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 才能复现累加器 device mismatch")
class TestNullRouteDiagnosticsCuda(unittest.TestCase):
    def test_cuda_weights_produce_cpu_accumulators(self):
        diag = metrics.NullRouteDiagnostics()
        diag.update(WEIGHTS.cuda())
        out = diag.result()
        self.assertAlmostEqual(out["null_std"], float(WEIGHTS[:, -1].std(unbiased=False)), places=12)
        self.assertAlmostEqual(out["null_q50"], float(torch.quantile(WEIGHTS[:, -1], 0.5)), places=12)


class TestNullSupervisionStats(unittest.TestCase):
    """标签分层与相关性（education 标签）。"""

    Y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    PRED = torch.tensor([0.9, 0.8, 0.2, 0.1])
    NULL = [0.4, 0.5, 0.3, 0.2]

    def test_known_values(self):
        out = metrics.null_supervision_stats(self.NULL, self.Y, self.PRED)
        self.assertAlmostEqual(out["null_mean_edu_pos"], 0.45, places=12)
        self.assertAlmostEqual(out["null_mean_edu_neg"], 0.25, places=12)
        self.assertAlmostEqual(out["null_label_gap"], 0.20, places=12)
        self.assertAlmostEqual(out["pred_std"], float(self.PRED.double().std(unbiased=False)), places=12)
        null_t = torch.tensor(self.NULL, dtype=torch.float64)
        expected = float(torch.corrcoef(torch.stack([null_t, self.PRED.double()]))[0, 1])
        self.assertAlmostEqual(out["corr_null_pred"], expected, places=12)
        err = (self.Y.double() - self.PRED.double()).abs()
        expected_err = float(torch.corrcoef(torch.stack([null_t, err]))[0, 1])
        self.assertAlmostEqual(out["corr_null_abs_err"], expected_err, places=12)

    def test_constant_null_gives_none_correlation_but_keeps_strata(self):
        out = metrics.null_supervision_stats([0.3, 0.3, 0.3, 0.3], self.Y, self.PRED)
        self.assertIsNone(out["corr_null_pred"])
        self.assertIsNone(out["corr_null_abs_err"])
        self.assertAlmostEqual(out["null_mean_edu_pos"], 0.3, places=12)
        self.assertAlmostEqual(out["null_label_gap"], 0.0, places=12)


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


def _tiny_model():
    return MPTRec(num_tasks=2, feature_vocabulary={"a": 3, "b": 2}, embedding_size=4, input_size=8,
                  expert_dnn_hidden_units=(8, 4), tower_dnn_hidden_units=(4, 2), device=torch.device("cpu"))


def _tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


class TestProbeMatchesEvaluate(unittest.TestCase):
    """探针与既有机制 pass 必须给出**逐位相同**的 M7 值；探针补齐 C 类键。"""

    def test_probe_m7_bit_equal_and_diagnostic_keys_present(self):
        backbone = _tiny_model()
        newtask = NewTask(input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN,
                          device=torch.device("cpu"), use_null_expert=True)
        _, _, (val_idx, _) = _tiny_inputs()
        dataset = TinyCensus(32, 7)
        val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=16)
        probe = run_census_benchmark.newtask_null_probe(newtask, backbone, val_loader, torch.device("cpu"))
        evaluate = metrics.evaluate_newtask(newtask, backbone, val_loader, torch.device("cpu"), mechanism=True)
        self.assertEqual(probe["null_mean"], evaluate["null_mean"])                    # 逐位一致：同 class、同左结合
        self.assertEqual(probe["null_top1_rate"], evaluate["null_top1_rate"])
        self.assertEqual(DIAGNOSTIC_KEYS, set(probe) - {"null_mean", "null_top1_rate"})


class TestRunnerDiagnosticsWiring(unittest.TestCase):
    """runner：处理臂 mechanism 追加诊断键；基线臂键集逐字不变；test 不参与机制诊断。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root, cls.device = Path(cls._tmp.name) / "artifacts", torch.device("cpu")
        cls.loaders, cls.stats, cls.indices = _tiny_inputs()
        cls.stage1 = run_census_benchmark.run_stage1(cls.root, epochs=2, device=cls.device, model=_tiny_model(),
                                                     loaders=cls.loaders, stats=cls.stats,
                                                     indices=cls.indices, tag="short")
        cls.baseline = cls._stage2(null_expert=False)
        cls.null = cls._stage2(null_expert=True)

    @classmethod
    def _stage2(cls, null_expert):
        return run_census_benchmark.run_stage2(cls.root, stage1_dir=Path(cls.stage1["dir"]), epochs=2,
                                               device=cls.device, model=_tiny_model(), loaders=cls.loaders,
                                               stats=cls.stats, indices=cls.indices, input_size=8, rep_dim=4,
                                               null_expert=null_expert)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _json(self, out, name):
        return json.loads((Path(out["run_dir"]) / name).read_text(encoding="utf-8"))

    def test_null_arm_mechanism_has_full_diagnostics(self):
        mechanism = self._json(self.null, "metrics.json")["mechanism"]
        for key in DIAGNOSTIC_KEYS | {"null_mean", "null_top1_rate"}:
            self.assertIn(key, mechanism, f"处理臂 mechanism 缺少诊断键: {key}")
        self.assertTrue(0.0 <= mechanism["null_std"] <= 0.5)
        self.assertTrue(0.0 <= mechanism["route_entropy_mean"] <= 1.2)          # ln(3) ≈ 1.0986（3 候选）

    def test_baseline_arm_mechanism_key_set_unchanged(self):
        mechanism = self._json(self.baseline, "metrics.json")["mechanism"]
        self.assertEqual(set(mechanism), {"gate_mean", "cos_gen_spec", "gen_std", "env_acc_stage1"})

    def test_null_arm_legacy_verdict_fields_verbatim(self):
        recorded = self._json(self.null, "metrics.json")
        verdict = recorded["null_arm"]
        self.assertEqual(verdict["auc_min"], 0.8521)
        self.assertEqual(verdict["baseline_test_auc"], 0.8500685307)
        self.assertEqual(verdict["top1_range"], [0.05, 0.95])
        self.assertEqual(set(verdict["checks"]), {"auc", "top1"})
        self.assertEqual(verdict["test_auc"], recorded["stage2"]["test_auc"])


class TestPreregDocPins(unittest.TestCase):
    def test_prereg_doc_contains_frozen_thresholds(self):
        text = DOC.read_text(encoding="utf-8")
        for token in ("+0.001", "-0.02", "0.05", "0.95", "4.303", "1e-9", "ddof=1", "1e-9"):
            self.assertIn(token, text, f"预注册文档缺少判据 token: {token}")
