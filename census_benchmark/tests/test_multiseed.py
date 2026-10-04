"""三 seed 配对验证 + 二级分类（预注册 `2026-10-05-census-stage2-null-expert-multiseed-design.md` §6）。

- 配对硬约束：每 seed 的基线臂与处理臂必须引用**同一** stage1_id、同一 split 指纹（§4.1）。
- 二级分类与 headroom 规则为预注册机械规则，逐字实现（§6.1、§6.2）。
- 检查点统计：mean / 样本 std(ddof=1) / positive_count / worst_seed / Student t 95% CI（§6.3）。
- 纯度：本模块只**读** run 目录，不写任何 artifacts（检查点 JSON 由 CLI 显式 --out 指定）。
"""
import json
import statistics
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import multiseed as M
from census_benchmark import protocol as P
from multitaskrec.model import MPTRec
import run_census_benchmark

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "superpowers" / "specs" / "2026-10-05-census-stage2-null-expert-multiseed-design.md"


class TestCanonicalSeeds(unittest.TestCase):
    def test_canonical_list_from_committed_evidence(self):
        self.assertEqual(M.CANONICAL_MODEL_SEEDS,
                         (1685480945, 1685463909, 1685477428, 1685459668, 1685496394))
        self.assertEqual(M.CHECKPOINT_SEEDS, M.CANONICAL_MODEL_SEEDS[:3])

    def test_threshold_constants(self):
        self.assertEqual(M.RECLASS_POSITIVE_MIN, 0.001)
        self.assertEqual(M.RECLASS_DEGRADE_MAX, -0.02)
        self.assertEqual(M.REPRO_TOL, 1e-9)


class TestReclassifyDelta(unittest.TestCase):
    def test_boundaries_inclusive_as_preregistered(self):
        self.assertEqual(M.reclassify_delta(+0.001), M.POSITIVE_IMPROVEMENT)          # >= +0.001
        self.assertEqual(M.reclassify_delta(+0.2), M.POSITIVE_IMPROVEMENT)
        self.assertEqual(M.reclassify_delta(+0.000999999), M.NO_CLEAR_IMPROVEMENT)
        self.assertEqual(M.reclassify_delta(0.0), M.NO_CLEAR_IMPROVEMENT)
        self.assertEqual(M.reclassify_delta(-0.0199999), M.NO_CLEAR_IMPROVEMENT)
        self.assertEqual(M.reclassify_delta(-0.02), M.CLEAR_DEGRADATION)              # <= -0.02
        self.assertEqual(M.reclassify_delta(-0.5), M.CLEAR_DEGRADATION)

    def test_class_names_are_the_preregistered_literals(self):
        self.assertEqual({M.POSITIVE_IMPROVEMENT, M.NO_CLEAR_IMPROVEMENT, M.CLEAR_DEGRADATION},
                         {"POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION"})


class TestCrossSeedConsistent(unittest.TestCase):
    def test_majority_positive_and_no_degradation(self):
        self.assertTrue(M.cross_seed_consistent([0.001, 0.002, -0.0001]))    # 2/3 正、无退化
        self.assertTrue(M.cross_seed_consistent([0.001, 0.002, 0.003]))      # 3/3
        self.assertFalse(M.cross_seed_consistent([0.001, -0.001, -0.002]))   # 仅 1/3 正
        self.assertFalse(M.cross_seed_consistent([0.001, 0.002, -0.03]))     # 出现 CLEAR_DEGRADATION

    def test_zero_is_not_positive(self):
        self.assertFalse(M.cross_seed_consistent([0.0, 0.0, 0.0]))


class TestHeadroom(unittest.TestCase):
    """预注册规则：headroom_remains = 四维度全真（机制活性、val 方向、预算、跨 seed 一致性）。"""

    ALL_TRUE = dict(mechanism_active=True, delta_val=0.001, budget_right_censored=True,
                    budget_last_delta_positive=True, cross_seed_consistent=True)

    def test_all_true(self):
        out = M.seed_headroom(**self.ALL_TRUE)
        self.assertTrue(out["headroom_remains"])
        self.assertEqual(out["mechanism_active"], True)
        self.assertEqual(out["val_direction_positive"], True)

    def test_each_component_false_blocks_headroom(self):
        for key, value in (("mechanism_active", False), ("delta_val", -0.001),
                           ("budget_right_censored", False), ("budget_last_delta_positive", False),
                           ("cross_seed_consistent", False)):
            out = M.seed_headroom(**dict(self.ALL_TRUE, **{key: value}))
            self.assertFalse(out["headroom_remains"], f"组件 {key} 为假时 headroom 必须为假")
            self.assertIn(key if key != "delta_val" else "val_direction_positive", out)

    def test_unknown_cross_seed_gives_none_not_true(self):
        out = M.seed_headroom(**dict(self.ALL_TRUE, cross_seed_consistent=None))
        self.assertIsNone(out["headroom_remains"])           # 未知不能当"有 headroom"

    def test_raw_values_recorded(self):
        out = M.seed_headroom(**dict(self.ALL_TRUE, delta_val=+0.00042))
        self.assertEqual(out["delta_val"], 0.00042)


class TestCheckpointStats(unittest.TestCase):
    DELTAS = [0.0013, 0.0005, -0.0002]

    def test_known_values_and_method(self):
        stats = M.checkpoint_stats(self.DELTAS)
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean"], sum(self.DELTAS) / 3, places=15)
        self.assertAlmostEqual(stats["std"], statistics.stdev(self.DELTAS), places=15)   # ddof=1
        self.assertEqual(stats["positive_count"], 1)                                     # 仅 0.0013 >= +0.001
        self.assertEqual(stats["worst_index"], 2)
        self.assertAlmostEqual(stats["worst_delta"], -0.0002, places=15)
        half = 4.303 * statistics.stdev(self.DELTAS) / (3 ** 0.5)
        self.assertAlmostEqual(stats["ci95_low"], stats["mean"] - half, places=15)
        self.assertAlmostEqual(stats["ci95_high"], stats["mean"] + half, places=15)
        self.assertIn("Student t", stats["ci_method"])
        self.assertIn("4.303", stats["ci_method"])

    def test_t_table_lookup_for_five_seeds(self):
        stats = M.checkpoint_stats([0.001, 0.002, 0.003, 0.004, 0.005])
        self.assertEqual(stats["n"], 5)
        self.assertIn("2.776", stats["ci_method"])

    def test_unsupported_sample_sizes_raise(self):
        with self.assertRaises(ValueError):
            M.checkpoint_stats([0.001])                                     # n=1 → df=0 无方差
        with self.assertRaises(ValueError):
            M.checkpoint_stats([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])  # n=6 → t 表未覆盖


def _write_run(root, run_id, *, model_seed, null_expert, test_auc, best_val_auc, stage1_id="s1-aaaa0000-m0-e2-0000aaaa",
               split_fp="ab" * 32, env_sha="cd" * 32, val_aucs=None, commit="c0ffee0"):
    """伪造 run 目录（仅含有趣字段），用于检查点模块的纯函数测试。"""
    run_dir = Path(root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    val_aucs = val_aucs or [0.80, 0.82, 0.84, 0.85, best_val_auc]
    records = [{"epoch": i + 1, "loss": 0.2, "auc_val_education": auc} for i, auc in enumerate(val_aucs)]
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "stage1_id": stage1_id, "commit": commit, "tag": "short", "frozen": True,
        "null_expert": null_expert, "split_seed": P.SPLIT_SEED, "model_seed": model_seed,
        "env_seed": P.ENV_SEED, "epochs": 5, "patience": 2, "lr": 1e-3, "batch_size": 256,
        "input_size": 123, "rep_dim": 128}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "stage1_id": stage1_id, "commit": commit, "null_expert": null_expert,
        "split_sha256": {"val": "v", "test": "t", "fingerprint": split_fp},
        "env_ids_sha256": env_sha,
        "stage2": {"epoch_records": records, "best_epoch": len(records), "best_val_auc": best_val_auc,
                   "test_auc": test_auc},
        "mechanism": ({"null_mean": 0.34, "null_top1_rate": 0.37} if null_expert
                      else {"gate_mean": [0.5, 0.6], "gen_std": 0.1})}), encoding="utf-8")
    (run_dir / "gate_report.json").write_text(json.dumps({
        "A1": {"pass": True}, "A2": {"pass": True}, "A4": {"pass": True}, "A5": {"pass": True},
        "B1": {"pass": True}, "B2": {"pass": True}, "B3": {"pass": False}, "B4": {"pass": True},
        "overall_pass": False, "failures": ["B3"]}), encoding="utf-8")
    return run_dir


class TestFindRuns(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        return root

    def test_finds_single_paired_runs_and_ignores_other_seeds(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            _write_run(root, "20261005-0100-s20260929-m1685463909-short-abc1234", model_seed=1685463909,
                       null_expert=False, test_auc=0.85, best_val_auc=0.853)
            _write_run(root, "20261005-0110-s20260929-m1685463909-short-abc1234-nullx", model_seed=1685463909,
                       null_expert=True, test_auc=0.8515, best_val_auc=0.8532)
            _write_run(root, "20261005-0120-s20260929-m1685477428-short-abc1234", model_seed=1685477428,
                       null_expert=False, test_auc=0.86, best_val_auc=0.861)
            found = M.find_runs(root, 1685463909)
            self.assertEqual(found["baseline"].name, "20261005-0100-s20260929-m1685463909-short-abc1234")
            self.assertEqual(found["nullx"].name, "20261005-0110-s20260929-m1685463909-short-abc1234-nullx")

    def test_stage1_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            _write_run(root, "20261005-0100-s20260929-m1685463909-short-abc1234", model_seed=1685463909,
                       null_expert=False, test_auc=0.85, best_val_auc=0.853, stage1_id="s1-A")
            _write_run(root, "20261005-0110-s20260929-m1685463909-short-abc1234-nullx", model_seed=1685463909,
                       null_expert=True, test_auc=0.8515, best_val_auc=0.8532, stage1_id="s1-B")
            with self.assertRaises(ValueError):
                M.find_runs(root, 1685463909)

    def test_split_fingerprint_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            _write_run(root, "20261005-0100-s20260929-m1685463909-short-abc1234", model_seed=1685463909,
                       null_expert=False, test_auc=0.85, best_val_auc=0.853, split_fp="ab" * 32)
            _write_run(root, "20261005-0110-s20260929-m1685463909-short-abc1234-nullx", model_seed=1685463909,
                       null_expert=True, test_auc=0.8515, best_val_auc=0.8532, split_fp="ef" * 32)
            with self.assertRaises(ValueError):
                M.find_runs(root, 1685463909)

    def test_duplicate_baseline_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            for suffix in ("abc1234", "def5678"):
                _write_run(root, f"20261005-0100-s20260929-m1685463909-short-{suffix}", model_seed=1685463909,
                           null_expert=False, test_auc=0.85, best_val_auc=0.853)
            with self.assertRaises(ValueError):
                M.find_runs(root, 1685463909)

    def test_missing_arm_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            _write_run(root, "20261005-0100-s20260929-m1685463909-short-abc1234", model_seed=1685463909,
                       null_expert=False, test_auc=0.85, best_val_auc=0.853)
            with self.assertRaises(ValueError):
                M.find_runs(root, 1685463909)


class TestSeedRecord(unittest.TestCase):
    def test_paired_deltas_classification_and_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_run(root, "20261005-0100-s20260929-m1685463909-short-abc1234", model_seed=1685463909,
                       null_expert=False, test_auc=0.8500, best_val_auc=0.8520)
            _write_run(root, "20261005-0110-s20260929-m1685463909-short-abc1234-nullx", model_seed=1685463909,
                       null_expert=True, test_auc=0.8515, best_val_auc=0.8524)
            rec = M.seed_record(root, 1685463909)
            self.assertAlmostEqual(rec["delta_test"], 0.0015, places=15)
            self.assertAlmostEqual(rec["delta_val"], 0.0004, places=15)
            self.assertEqual(rec["classification"], M.POSITIVE_IMPROVEMENT)
            self.assertEqual(rec["model_seed"], 1685463909)
            self.assertEqual(rec["baseline"]["run_id"], "20261005-0100-s20260929-m1685463909-short-abc1234")
            self.assertEqual(rec["arm"]["run_id"], "20261005-0110-s20260929-m1685463909-short-abc1234-nullx")
            self.assertTrue(rec["budget"]["arm_right_censored"])              # best epoch == 最后 epoch
            self.assertEqual(rec["budget"]["common_epochs"], 5)
            self.assertAlmostEqual(rec["budget"]["delta_val_last_common"],
                                   rec["arm"]["epoch_records"][-1]["auc_val_education"]
                                   - rec["baseline"]["epoch_records"][-1]["auc_val_education"], places=15)
            self.assertIn("headroom", rec)
            self.assertEqual(rec["arm"]["mechanism"]["null_top1_rate"], 0.37)
            self.assertFalse(rec["gates"]["arm"]["B3"]["pass"])
            self.assertEqual(rec["baseline"]["config"]["model_seed"], 1685463909)
            self.assertIn("null_expert", rec["arm"]["config"])

    def test_file_hashes_recorded_and_stage1_absent_tolerated(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = _write_run(root, "20261005-0100-s20260929-m1685463909-short-abc1234",
                                 model_seed=1685463909, null_expert=False, test_auc=0.85, best_val_auc=0.853,
                                 stage1_id="s1-deadbeef-m1685463909-e2-cafe0000")
            _write_run(root, "20261005-0110-s20260929-m1685463909-short-abc1234-nullx", model_seed=1685463909,
                       null_expert=True, test_auc=0.8515, best_val_auc=0.8532,
                       stage1_id="s1-deadbeef-m1685463909-e2-cafe0000")
            rec_absent = M.seed_record(root, 1685463909)                      # stage1 目录不存在 → None，不抛错
            self.assertIsNone(rec_absent["stage1"]["files"])
            stage1_dir = root / "stage1" / "s1-deadbeef-m1685463909-e2-cafe0000"
            stage1_dir.mkdir(parents=True)
            (stage1_dir / "backbone.pt").write_bytes(b"dummy-backbone")
            (stage1_dir / "meta.json").write_text(json.dumps({"backbone_sha256": "b" * 64,
                                                              "env_ids_sha256": "e" * 64,
                                                              "split_fingerprint_sha256": "f" * 64,
                                                              "config_hash": "c" * 64}), encoding="utf-8")
            rec = M.seed_record(root, 1685463909)
            expected = hashlib.sha256(b"dummy-backbone").hexdigest()
            self.assertEqual(rec["stage1"]["files"]["backbone.pt"], expected)
            self.assertEqual(rec["stage1"]["meta"]["backbone_sha256"], "b" * 64)
            run_files = rec["baseline"]["files"]
            self.assertEqual(run_files["metrics.json"],
                             hashlib.sha256((run_dir / "metrics.json").read_bytes()).hexdigest())

    def test_headroom_uses_injected_cross_seed_flag(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_run(root, "20261005-0100-s20260929-m1685463909-short-abc1234", model_seed=1685463909,
                       null_expert=False, test_auc=0.8500, best_val_auc=0.8520)
            _write_run(root, "20261005-0110-s20260929-m1685463909-short-abc1234-nullx", model_seed=1685463909,
                       null_expert=True, test_auc=0.8505, best_val_auc=0.8524)   # NO_CLEAR_IMPROVEMENT
            rec = M.seed_record(root, 1685463909, cross_consistent=True)
            self.assertEqual(rec["classification"], M.NO_CLEAR_IMPROVEMENT)
            self.assertTrue(rec["headroom"]["headroom_remains"])
            rec2 = M.seed_record(root, 1685463909, cross_consistent=None)
            self.assertIsNone(rec2["headroom"]["headroom_remains"])


class TestCheckpoint(unittest.TestCase):
    SEEDS = (1685480945, 1685463909, 1685477428)

    def _build_root(self, root):
        deltas = [+0.0015, -0.0004, +0.0024]                    # seed 2 → NO_CLEAR（负），1/3 正… 见断言
        for seed, delta in zip(self.SEEDS, deltas):
            _write_run(root, f"20261005-0100-s20260929-m{seed}-short-abc1234", model_seed=seed,
                       null_expert=False, test_auc=0.8500, best_val_auc=0.8520)
            _write_run(root, f"20261005-0110-s20260929-m{seed}-short-abc1234-nullx", model_seed=seed,
                       null_expert=True, test_auc=0.8500 + delta, best_val_auc=0.8520 + delta)

    def test_checkpoint_assembles_stats_consistency_and_anchor_absent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._build_root(root)
            out = M.checkpoint(root, seeds=self.SEEDS)
            deltas = [rec["delta_test"] for rec in out["seeds"]]
            self.assertEqual(len(out["seeds"]), 3)
            self.assertAlmostEqual(out["stats"]["mean"], sum(deltas) / 3, places=15)
            self.assertAlmostEqual(out["stats"]["std"], statistics.stdev(deltas), places=15)
            self.assertEqual(out["stats"]["positive_count"], 2)                  # +0.0015、+0.0024
            self.assertEqual(out["stats"]["worst_index"], 1)                     # -0.0004
            self.assertEqual(out["stats"]["worst_seed"], 1685463909)
            for rec in out["seeds"]:
                self.assertIn("cross_seed_consistent", rec["headroom"])
                self.assertTrue(rec["headroom"]["cross_seed_consistent"])    # [+0.0015, -0.0004, +0.0024]：2/3 正、无退化
            self.assertEqual(out["anchor"]["status"], "REFERENCE_ABSENT")
            self.assertEqual(out["spec"], "docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-design.md")

    def test_cross_seed_consistent_is_majority_positive(self):
        # deltas [+0.0015, -0.0004, +0.0024] → 2/3 > 0 且无 <= -0.02 → True
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._build_root(root)
            out = M.checkpoint(root, seeds=self.SEEDS)
            self.assertTrue(out["seeds"][0]["headroom"]["cross_seed_consistent"])


class TestAnchor(unittest.TestCase):
    """预注册 §3.4 / §6.5：锚 run 不参与本轮对子选择；新旧比对 ≤1e-9 → REPRODUCED。"""

    def _write_anchor_pair(self, root, test_auc_b, test_auc_n):
        _write_run(root, M.ANCHOR_BASELINE_RUN_ID, model_seed=1685480945, null_expert=False,
                   test_auc=M.ANCHOR_BASELINE_TEST_AUC, best_val_auc=0.8527881905614896)
        _write_run(root, M.ANCHOR_NULLX_RUN_ID, model_seed=1685480945, null_expert=True,
                   test_auc=test_auc_n, best_val_auc=0.853020431042571)

    def test_anchor_reproduced_and_excluded_from_current_pair(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_anchor_pair(root, M.ANCHOR_BASELINE_TEST_AUC, M.ANCHOR_NULLX_TEST_AUC)
            _write_run(root, "20261005-0300-s20260929-m1685480945-short-beef123", model_seed=1685480945,
                       null_expert=False, test_auc=M.ANCHOR_BASELINE_TEST_AUC, best_val_auc=0.8527881905614896)
            _write_run(root, "20261005-0310-s20260929-m1685480945-short-beef123-nullx", model_seed=1685480945,
                       null_expert=True, test_auc=M.ANCHOR_NULLX_TEST_AUC, best_val_auc=0.853020431042571)
            out = M.anchor_record(root)
            self.assertEqual(out["status"], "REPRODUCED")
            self.assertTrue(out["historical_dirs_match_constants"])
            found = M.find_runs(root, 1685480945)                       # 锚被排除，只剩新对子
            self.assertEqual(found["baseline"].name, "20261005-0300-s20260929-m1685480945-short-beef123")
            self.assertEqual(found["nullx"].name, "20261005-0310-s20260929-m1685480945-short-beef123-nullx")

    def test_anchor_diverged_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_anchor_pair(root, M.ANCHOR_BASELINE_TEST_AUC, M.ANCHOR_NULLX_TEST_AUC)
            _write_run(root, "20261005-0300-s20260929-m1685480945-short-beef123", model_seed=1685480945,
                       null_expert=False, test_auc=0.8510, best_val_auc=0.85)           # 偏离 > 1e-9
            _write_run(root, "20261005-0310-s20260929-m1685480945-short-beef123-nullx", model_seed=1685480945,
                       null_expert=True, test_auc=0.8519, best_val_auc=0.85)
            out = M.anchor_record(root)
            self.assertEqual(out["status"], "DIVERGED")
            self.assertGreater(out["abs_diff_baseline"], M.REPRO_TOL)


class TestCli(unittest.TestCase):
    def test_main_writes_json_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root, out_path = Path(td) / "artifacts", Path(td) / "checkpoint.json"
            _write_run(root, "20261005-0100-s20260929-m1685463909-short-abc1234", model_seed=1685463909,
                       null_expert=False, test_auc=0.85, best_val_auc=0.853)
            _write_run(root, "20261005-0110-s20260929-m1685463909-short-abc1234-nullx", model_seed=1685463909,
                       null_expert=True, test_auc=0.8515, best_val_auc=0.8532)
            code = M.main(["--root", str(root), "--out", str(out_path), "--seeds", "1685463909"])
            self.assertEqual(code, 0)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("stats", payload)
            self.assertEqual(len(payload["seeds"]), 1)


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


class TestRealRunnerArtifactsIntegration(unittest.TestCase):
    """真实 runner 落盘格式必须可直接被 find_runs / seed_record 消费（防"仅伪造夹具自洽"盲区）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root, cls.device = Path(cls._tmp.name) / "artifacts", torch.device("cpu")
        cls.loaders, cls.stats, cls.indices = _tiny_inputs()
        cls.stage1 = run_census_benchmark.run_stage1(cls.root, epochs=2, device=cls.device, model=_tiny_model(),
                                                     loaders=cls.loaders, stats=cls.stats,
                                                     indices=cls.indices, tag="short")
        cls.rid_baseline = cls._stage2(null_expert=False)["run_id"]
        cls.rid_nullx = cls._stage2(null_expert=True)["run_id"]

    @classmethod
    def _stage2(cls, null_expert):
        return run_census_benchmark.run_stage2(cls.root, stage1_dir=Path(cls.stage1["dir"]), epochs=2,
                                               device=cls.device, model=_tiny_model(), loaders=cls.loaders,
                                               stats=cls.stats, indices=cls.indices, input_size=8, rep_dim=4,
                                               null_expert=null_expert)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_pair_found_and_delta_exact(self):
        found = M.find_runs(self.root, P.MODEL_SEED)
        self.assertEqual(found["baseline"].name, self.rid_baseline)
        self.assertEqual(found["nullx"].name, self.rid_nullx)
        rec = M.seed_record(self.root, P.MODEL_SEED)
        baseline = json.loads((found["baseline"] / "metrics.json").read_text(encoding="utf-8"))
        arm = json.loads((found["nullx"] / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["delta_test"], arm["stage2"]["test_auc"] - baseline["stage2"]["test_auc"])
        self.assertEqual(rec["classification"], M.reclassify_delta(rec["delta_test"]))
        self.assertEqual(rec["stage1_id"], self.stage1["stage1_id"])


class TestPreregDocPins(unittest.TestCase):
    def test_doc_defines_headroom_and_ci_method(self):
        text = DOC.read_text(encoding="utf-8")
        for token in ("headroom_remains", "cross_seed_consistent", "budget_right_censored",
                      "worst_seed", "positive_count"):
            self.assertIn(token, text, f"预注册文档缺少 token: {token}")
