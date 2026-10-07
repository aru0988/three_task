"""潜在跨任务诊断的单元与端到端测试（TDD：先于实现编写）。

规范唯一事实来源：docs/superpowers/specs/2026-10-08-aliccp-latent-cross-task-diagnostic-design.md。
CPU-only；除端到端外全部使用临时合成小数据；端到端复用 bench.run_stage1 产出微型 Stage-1 产物。
"""
import contextlib
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from aliccp_benchmark import bench, latent_diag, metrics, protocol, verify_latent_diag
from multitaskrec.model import NewTask


def _noop(*args, **kwargs):
    pass


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _expected_counts(rows) -> dict:
    raw = {}
    for row in rows:
        raw[row[-1]] = raw.get(row[-1], 0) + 1
    return {
        "n": len(rows),
        "click1": sum(int(r[0]) for r in rows),
        "purchase1": sum(int(r[1]) for r in rows),
        "bsi_pos": sum(0 if int(r[-1]) == 2 else 1 for r in rows),
        "bsi_raw": raw,
    }


class TestConstants(unittest.TestCase):
    def test_ranges_cover_frozen_prefix(self):
        ranges = latent_diag.split_ranges()
        self.assertEqual(
            ranges,
            {"A": (0, 1_400_000), "B": (1_400_000, 1_700_000), "C": (1_700_000, 2_000_000)},
        )
        self.assertEqual(ranges["C"][1], protocol.TRAIN_BUDGET)

    def test_frozen_thresholds(self):
        self.assertEqual(latent_diag.SHUFFLE_SEED, 20261008)
        self.assertEqual(latent_diag.CLIP_EPS, 1e-6)
        self.assertEqual(latent_diag.MIN_BSI_NEGATIVES, 100)
        self.assertEqual(latent_diag.VAR_FLOOR, 1e-8)
        self.assertEqual(
            (latent_diag.DELTA_TEST_MIN, latent_diag.DELTA_C_MIN, latent_diag.DELTA_DEV_MIN),
            (0.001, 0.001, 0.0),
        )
        self.assertEqual((latent_diag.CLASS_POS_MIN, latent_diag.CLASS_NEG_MAX), (0.001, -0.02))
        self.assertEqual((latent_diag.PROBE_C, latent_diag.PROBE_MAX_ITER), (1.0, 200))
        self.assertEqual((latent_diag.PROBE_PENALTY, latent_diag.PROBE_SOLVER), ("l2", "lbfgs"))
        self.assertEqual(
            (latent_diag.HEAD_EPOCHS, latent_diag.HEAD_PATIENCE, latent_diag.HEAD_LR),
            (protocol.STAGE2_EPOCHS, protocol.STAGE2_PATIENCE, protocol.LR),
        )
        self.assertEqual(latent_diag.ALL_SPLITS, ("B", "C", "dev", "test"))
        self.assertEqual(latent_diag.PERM_ORDER, ("B", "C", "dev", "test"))
        self.assertEqual(latent_diag.PREFIT_SPLITS, ("A", "B", "C", "dev"))  # test 不参与拟合前判定
        self.assertEqual(latent_diag.ARMS, ("base", "full", "shuffle"))
        self.assertEqual(latent_diag.FORMAL_STAGE1_ID, "s1-5c060b9c-m1688723512-e3-3a30e2c0")
        self.assertTrue(latent_diag.FORMAL_FINGERPRINT_SHA256.startswith("5c060b9c"))


class TestSplitRanges(unittest.TestCase):
    def test_scaled(self):
        self.assertEqual(
            latent_diag.split_ranges(3, 2, 1), {"A": (0, 3), "B": (3, 5), "C": (5, 6)}
        )

    def test_contiguous_non_overlapping(self):
        ranges = latent_diag.split_ranges(41, 25, 17)
        self.assertEqual(ranges["A"][1], ranges["B"][0])
        self.assertEqual(ranges["B"][1], ranges["C"][0])
        self.assertEqual(ranges["A"][0], 0)

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            latent_diag.split_ranges(0, 2, 1)


class TestPrefixTag(unittest.TestCase):
    def test_formal_tag(self):
        self.assertEqual(
            latent_diag.prefix_tag_for(2_000_000, 500_000, 1_000_000), protocol.PREFIX_TAG
        )

    def test_scaled_tag(self):
        self.assertEqual(latent_diag.prefix_tag_for(83, 13, 11), "p83-v13-t11")


class TestClipLogit(unittest.TestCase):
    def test_clips_and_reference_values(self):
        eps = latent_diag.CLIP_EPS
        z = latent_diag.clip_logit(np.array([0.0, 0.5, 1.0]))
        self.assertAlmostEqual(z[0], float(np.log(eps / (1 - eps))), places=12)
        self.assertAlmostEqual(z[1], 0.0, places=12)
        # z[2] 与 -z[0] 的数学恒等在 float64 下不精确成立：clip 上界是 fl(1-1e-6)，与实数
        # 1-1e-6 相差 ≤ ulp(1)/2 = 2^-54；logit 在 c≈1 处的敏感度 ~1/eps 把该表示误差放大到
        # ≤ 2^-54/1e-6 ≈ 5.6e-11（实测 ≈ 2.9e-11）。这是浮点表示误差而非算法差异：即便把
        # log(c/(1-c)) 改写成 log1p 等"更稳"的等价式，输入值本身的取整误差依然存在。
        # 故此处用 places=9（阈值 5e-10，约 9 倍余量）断言数学恒等。
        self.assertAlmostEqual(z[2], float(-np.log(eps / (1 - eps))), places=9)

    def test_monotone_on_clipped_range(self):
        z = latent_diag.clip_logit(np.linspace(0.0, 1.0, 101))
        self.assertTrue(bool(np.all(np.diff(z) >= 0)))


class TestStandardizer(unittest.TestCase):
    def test_stats_ddof0_and_apply(self):
        col = np.array([1.0, 2.0, 3.0])
        stats = latent_diag.standardizer_stats({"x": col})
        self.assertAlmostEqual(stats["means"]["x"], 2.0, places=12)
        self.assertAlmostEqual(stats["scales"]["x"], float(np.std(col)), places=12)
        self.assertFalse(stats["constant"]["x"])
        z = latent_diag.apply_standardizer({"x": col}, stats)["x"]
        np.testing.assert_allclose(z, (col - 2.0) / np.std(col), rtol=0, atol=1e-12)

    def test_constant_column_scale_floor(self):
        col = np.full(7, 0.25)
        stats = latent_diag.standardizer_stats({"x": col})
        self.assertTrue(stats["constant"]["x"])
        self.assertEqual(stats["scales"]["x"], 1.0)
        self.assertEqual(stats["raw_scales"]["x"], 0.0)
        z = latent_diag.apply_standardizer({"x": col}, stats)["x"]
        np.testing.assert_allclose(z, np.zeros(7), rtol=0, atol=0)


class TestPermutations(unittest.TestCase):
    def test_deterministic_and_valid(self):
        lengths = {"B": 8, "C": 8, "dev": 8, "test": 8}
        p1 = latent_diag.make_permutations(lengths)
        p2 = latent_diag.make_permutations(lengths)
        for split in latent_diag.PERM_ORDER:
            np.testing.assert_array_equal(p1[split], p2[split])
            np.testing.assert_array_equal(np.sort(p1[split]), np.arange(8))
            self.assertEqual(p1[split].dtype, np.int64)
        # 相同长度下各切分独立（固定 seed 下仍然不同）
        self.assertFalse(np.array_equal(p1["B"], p1["C"]))
        self.assertFalse(np.array_equal(p1["C"], p1["dev"]))

    def test_joint_pairing_preserved(self):
        rng = np.random.default_rng(0)
        ctr = rng.normal(size=64)
        cvr = rng.normal(size=64)
        perm = latent_diag.make_permutations({"B": 64, "C": 64, "dev": 64, "test": 64})["B"]
        before = sorted(zip(ctr.tolist(), cvr.tolist()))
        after = sorted(zip(ctr[perm].tolist(), cvr[perm].tolist()))
        self.assertEqual(before, after)

    def test_missing_split_rejected(self):
        with self.assertRaises(KeyError):
            latent_diag.make_permutations({"B": 4})

    def test_seed_sensitivity(self):
        p1 = latent_diag.make_permutations({"B": 16, "C": 16, "dev": 16, "test": 16})
        p2 = latent_diag.make_permutations({"B": 16, "C": 16, "dev": 16, "test": 16}, seed=1)
        self.assertFalse(np.array_equal(p1["B"], p2["B"]))


class TestArmColumnMapping(unittest.TestCase):
    """回归（臂语义缺陷，2026-10-08 编排审计）：预注册冻结的臂-列映射为
    base=[bsi]；full=[bsi,ctr,cvr]（行对齐原列）；shuffle=[bsi,ctr[perm],cvr[perm]]。
    历史缺陷：runner 把置换同时应用到 full 与 shuffle → 两臂输入逐位相同，正式输出
    full=shuffle 全位数一致、探针系数相同（该运行已作废）。
    本测试用手构造列 + 已知非恒等置换锁定 full 与 shuffle 的已知输入差异。"""

    def setUp(self):
        self.perm = np.array([3, 0, 4, 1, 2])  # 已知非恒等置换（长度 5）
        self.logits = {
            "B": {
                "bsi": np.array([0.11, 0.22, 0.33, 0.44, 0.55]),
                "ctr": np.array([0.91, 0.12, 0.83, 0.24, 0.75]),
                "cvr": np.array([0.15, 0.45, 0.65, 0.35, 0.85]),
            }
        }
        self.perms = {"B": self.perm}

    def _assert_frozen_mapping(self, cols_for):
        base = cols_for("base")
        self.assertEqual(set(base), {"bsi"})
        np.testing.assert_array_equal(base["bsi"], self.logits["B"]["bsi"])

        full = cols_for("full")
        self.assertEqual(set(full), {"bsi", "ctr", "cvr"})
        np.testing.assert_array_equal(full["bsi"], self.logits["B"]["bsi"])
        np.testing.assert_array_equal(full["ctr"], self.logits["B"]["ctr"])  # 对齐：不得置换
        np.testing.assert_array_equal(full["cvr"], self.logits["B"]["cvr"])

        shuffle = cols_for("shuffle")
        self.assertEqual(set(shuffle), {"bsi", "ctr", "cvr"})
        np.testing.assert_array_equal(shuffle["bsi"], self.logits["B"]["bsi"])  # BSI 不置换
        np.testing.assert_array_equal(shuffle["ctr"], self.logits["B"]["ctr"][self.perm])
        np.testing.assert_array_equal(shuffle["cvr"], self.logits["B"]["cvr"][self.perm])

        # 非恒等置换 + 互异取值：full 与 shuffle 输入必须不同（缺陷下逐位相同）
        self.assertFalse(np.array_equal(full["ctr"], shuffle["ctr"]))
        self.assertFalse(np.array_equal(full["cvr"], shuffle["cvr"]))
        # shuffle 保持联合配对（同一置换同时作用于 CTR/CVR）与边际分布
        self.assertEqual(
            sorted(zip(full["ctr"].tolist(), full["cvr"].tolist())),
            sorted(zip(shuffle["ctr"].tolist(), shuffle["cvr"].tolist())),
        )

    def test_runner_arm_columns_match_frozen_mapping(self):
        self._assert_frozen_mapping(
            lambda arm: latent_diag.arm_columns(self.logits, self.perms, arm, "B")
        )

    def test_runner_rejects_unknown_arm(self):
        with self.assertRaises(ValueError):
            latent_diag.arm_columns(self.logits, self.perms, "both", "B")


class TestHashes(unittest.TestCase):
    def test_array_sha256_deterministic_and_dtype_sensitive(self):
        a = np.array([1, 2, 3], dtype=np.int8)
        self.assertEqual(latent_diag.array_sha256(a), latent_diag.array_sha256(a.copy()))
        self.assertNotEqual(latent_diag.array_sha256(a), latent_diag.array_sha256(a.astype(np.int64)))

    def test_split_range_hashes_deterministic_and_range_sensitive(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.csv"
            latent_diag.write_aliccp_file(path, latent_diag.make_smoke_rows(8))
            ranges = {"A": (0, 4), "B": (4, 8)}
            h1 = latent_diag.split_range_hashes(path, ranges)
            self.assertEqual(h1, latent_diag.split_range_hashes(path, ranges))
            self.assertNotEqual(h1["A"], h1["B"])
            with self.assertRaises(ValueError):
                latent_diag.split_range_hashes(path, {"A": (0, 9)})


class TestRangeLabelCounts(unittest.TestCase):
    def test_matches_manual_and_full_scan(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.csv"
            rows = latent_diag.make_smoke_rows(10)
            latent_diag.write_aliccp_file(path, rows)
            ranges = {"A": (0, 5), "B": (5, 10)}
            counts = latent_diag.range_label_counts(path, ranges)
            for name, lo, hi in (("A", 0, 5), ("B", 5, 10)):
                exp = _expected_counts(rows[lo:hi])
                self.assertEqual(counts[name]["n"], exp["n"], name)
                self.assertEqual(counts[name]["click1"], exp["click1"], name)
                self.assertEqual(counts[name]["purchase1"], exp["purchase1"], name)
                self.assertEqual(counts[name]["bsi_pos"], exp["bsi_pos"], name)
                self.assertEqual(counts[name]["bsi_raw"], exp["bsi_raw"], name)
            # 与独立扫描器必须"同覆盖度"对照（0 基、尾开区间、跳表头）：
            # A=(0,5) 等价于协议前缀扫描 n=5（此前误用全量 n=10 对照，覆盖度不同必然不等）
            scan_a = protocol.scan_label_counts(path, 5)
            for key in ("n", "click1", "purchase1", "bsi_pos", "bsi_raw"):
                self.assertEqual(counts["A"][key], scan_a[key], key)
            # 覆盖整个文件时 (0,10) 必须与全量前缀扫描逐项相等
            full = latent_diag.range_label_counts(path, {"full": (0, 10)})["full"]
            scan = protocol.scan_label_counts(path, 10)
            for key in ("n", "click1", "purchase1", "bsi_pos", "bsi_raw"):
                self.assertEqual(full[key], scan[key], key)


class TestRangeView(unittest.TestCase):
    def test_slices_in_file_order(self):
        base = list(range(10))
        view = latent_diag.RangeView(base, 2, 5)
        self.assertEqual(len(view), 3)
        self.assertEqual(view[0], 2)
        self.assertEqual(view[2], 4)
        self.assertEqual(view[-1], 4)
        with self.assertRaises(IndexError):
            _ = view[3]

    def test_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            latent_diag.RangeView(list(range(4)), 2, 9)
        with self.assertRaises(ValueError):
            latent_diag.RangeView(list(range(4)), 3, 3)


class TestProbeFit(unittest.TestCase):
    def test_informative_beats_noise(self):
        rng = np.random.default_rng(7)
        n = 4000
        x_sig = rng.normal(size=n)
        x_noise = rng.normal(size=n)
        y = (x_sig + 0.5 * rng.normal(size=n) > 0).astype(int)
        base = latent_diag.fit_probe(x_noise.reshape(-1, 1), y)
        full = latent_diag.fit_probe(np.column_stack([x_noise, x_sig]), y)
        auc_base = metrics.auc_score(y, latent_diag.probe_prob(x_noise.reshape(-1, 1), base))
        auc_full = metrics.auc_score(y, latent_diag.probe_prob(np.column_stack([x_noise, x_sig]), full))
        # 该生成模型的贝叶斯 AUC 上限 ≈ Φ(2/√5) ≈ 0.814，故阈值取 0.75
        self.assertGreater(auc_full, 0.75)
        self.assertLess(auc_base, 0.65)
        self.assertGreater(full["coef"][1], 0.0)
        self.assertTrue(full["converged"])
        self.assertIsInstance(full["n_iter"], int)

    def test_extreme_imbalance_and_constant_feature(self):
        rng = np.random.default_rng(11)
        n = 20_000
        x = rng.normal(size=n)
        y = (rng.random(n) < 0.99).astype(int)  # 极端不平衡（负例 ≈ 200）
        const = np.zeros(n)
        self.assertAlmostEqual(float(y.mean()), 0.99, delta=0.01)
        probe = latent_diag.fit_probe(np.column_stack([const, x]), y)
        probs = latent_diag.probe_prob(np.column_stack([const, x]), probe)
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(bool(np.all((probs >= 0.0) & (probs <= 1.0))))
        self.assertIsInstance(probe["converged"], bool)
        auc = metrics.auc_score(y, probs)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)


class TestDeltaAndClassify(unittest.TestCase):
    def test_delta_uses_max_of_controls(self):
        aucs = {"base": 0.700, "full": 0.702, "shuffle": 0.701}
        self.assertAlmostEqual(latent_diag.delta_of(aucs), 0.001, places=12)
        aucs = {"base": 0.700, "full": 0.702, "shuffle": 0.705}
        self.assertAlmostEqual(latent_diag.delta_of(aucs), -0.003, places=12)

    def test_classification_bands(self):
        self.assertEqual(latent_diag.classify_delta(0.001), "POSITIVE")
        self.assertEqual(latent_diag.classify_delta(0.05), "POSITIVE")
        self.assertEqual(latent_diag.classify_delta(0.000999), "NO_CLEAR")
        self.assertEqual(latent_diag.classify_delta(-0.019999), "NO_CLEAR")
        self.assertEqual(latent_diag.classify_delta(-0.02), "CLEAR_DECLINE")


def gate_facts(**overrides):
    facts = {
        "identity_ok": True,
        "freeze_ok": True,
        "counts_ok": True,
        "negatives": {"A": 500, "B": 200, "C": 200, "dev": 200, "test": 200},
        "ranges_ok": True,
        "perms_ok": True,
        "variances": {
            "B": {"ctr": 1e-3, "cvr": 1e-5},
            "C": {"ctr": 1e-3, "cvr": 1e-5},
            "test": {"ctr": 1e-3, "cvr": 1e-5},
        },
        "deltas": {"test": 0.002, "C": 0.002, "dev": 0.001},
        "git_dirty": False,
    }
    facts.update(overrides)
    return facts


class TestGates(unittest.TestCase):
    def test_all_pass(self):
        out = latent_diag.evaluate_gates(gate_facts())
        self.assertTrue(out["signal_positive"])
        self.assertEqual(out["verdict"], "SIGNAL_POSITIVE")
        for gate_id, gate in out["gates"].items():
            self.assertEqual(gate["verdict"], "PASS", gate_id)

    def test_boundary_values_pass(self):
        out = latent_diag.evaluate_gates(gate_facts(deltas={"test": 0.001, "C": 0.001, "dev": 0.0}))
        self.assertTrue(out["signal_positive"])

    def test_below_boundary_fails(self):
        out = latent_diag.evaluate_gates(gate_facts(deltas={"test": 0.000999, "C": 0.001, "dev": 0.0}))
        self.assertEqual(out["gates"]["S1_delta_test"]["verdict"], "FAIL")
        self.assertFalse(out["signal_positive"])
        self.assertEqual(out["verdict"], "NO_GO_FOR_EXTENSION")
        out = latent_diag.evaluate_gates(gate_facts(deltas={"test": 0.001, "C": 0.000999, "dev": 0.0}))
        self.assertEqual(out["gates"]["S2_delta_C"]["verdict"], "FAIL")
        out = latent_diag.evaluate_gates(gate_facts(deltas={"test": 0.001, "C": 0.001, "dev": -1e-9}))
        self.assertEqual(out["gates"]["S3_delta_dev"]["verdict"], "FAIL")

    def test_nonconst_requires_b_ct_test(self):
        bad = gate_facts()
        bad["variances"] = {
            "B": {"ctr": 0.0, "cvr": 1e-5},
            "C": {"ctr": 1e-3, "cvr": 1e-5},
            "test": {"ctr": 1e-3, "cvr": 1e-5},
        }
        out = latent_diag.evaluate_gates(bad)
        self.assertEqual(out["gates"]["P2_nonconst"]["verdict"], "FAIL")
        self.assertFalse(out["signal_positive"])

    def test_integrity_gate_failures_flip_signal(self):
        for key in ("identity_ok", "freeze_ok", "counts_ok", "ranges_ok", "perms_ok"):
            out = latent_diag.evaluate_gates(gate_facts(**{key: False}))
            self.assertFalse(out["signal_positive"], key)
        out = latent_diag.evaluate_gates(gate_facts(git_dirty=True))
        self.assertEqual(out["gates"]["G1_git"]["verdict"], "FAIL")
        self.assertFalse(out["signal_positive"])


def prefit_facts(**overrides):
    # negatives/loader_parity 中刻意保留 "test" 条目：预注册要求拟合前判定忽略 test
    # （冻结前不触碰 test），这些条目存在时也必须不产生任何中止原因。
    facts = {
        "identity_ok": True,
        "identity_detail": {},
        "freeze_ok": True,
        "freeze_detail": {},
        "negatives": {"A": 500, "B": 200, "C": 200, "dev": 200, "test": 200},
        "old_task_var": {
            "B": {"ctr": 1e-3, "cvr": 1e-5},
            "C": {"ctr": 1e-3, "cvr": 1e-5},
            "dev": {"ctr": 1e-3, "cvr": 1e-5},
        },
        "loader_parity": {"B": True, "C": True, "dev": True, "test": True},
        "length_mismatches": [],
    }
    facts.update(overrides)
    return facts


class TestPrefitAbort(unittest.TestCase):
    def test_clean_facts_no_reasons(self):
        self.assertEqual(latent_diag.prefit_abort_reasons(prefit_facts()), [])

    def test_identity_and_freeze(self):
        reasons = latent_diag.prefit_abort_reasons(
            prefit_facts(identity_ok=False, identity_detail={"stage1_id_recorded": False})
        )
        self.assertEqual([r["code"] for r in reasons], ["IDENTITY_MISMATCH"])
        reasons = latent_diag.prefit_abort_reasons(prefit_facts(freeze_ok=False))
        self.assertEqual([r["code"] for r in reasons], ["FREEZE_FAIL"])

    def test_min_negatives_per_split(self):
        facts = prefit_facts()
        facts["negatives"] = dict(facts["negatives"], C=99)
        reasons = latent_diag.prefit_abort_reasons(facts)
        self.assertEqual(len(reasons), 1)
        self.assertEqual(reasons[0]["code"], "MIN_NEGATIVES")
        self.assertEqual(reasons[0]["split"], "C")
        self.assertEqual(reasons[0]["required"], latent_diag.MIN_BSI_NEGATIVES)

    def test_constant_preds_requires_both_tasks(self):
        facts = prefit_facts()
        facts["old_task_var"] = {
            "B": {"ctr": 0.0, "cvr": 1e-3},  # 只有 CTR 恒定 → 不触发（CVR 仍携带信息）
            "C": {"ctr": 1e-3, "cvr": 1e-3},
            "dev": {"ctr": 1e-3, "cvr": 1e-3},
        }
        self.assertEqual(latent_diag.prefit_abort_reasons(facts), [])
        facts["old_task_var"]["B"] = {"ctr": 0.0, "cvr": 0.0}
        reasons = latent_diag.prefit_abort_reasons(facts)
        self.assertEqual([r["code"] for r in reasons], ["CONSTANT_OLD_TASK_PREDS"])
        self.assertEqual(reasons[0]["split"], "B")

    def test_length_and_loader_parity(self):
        facts = prefit_facts(length_mismatches=[{"split": "B", "actual": 299_999, "expected": 300_000}])
        reasons = latent_diag.prefit_abort_reasons(facts)
        self.assertEqual([r["code"] for r in reasons], ["SPLIT_LENGTH_MISMATCH"])
        facts = prefit_facts(loader_parity={"B": True, "C": False, "dev": True, "test": True})
        reasons = latent_diag.prefit_abort_reasons(facts)
        self.assertEqual([r["code"] for r in reasons], ["LOADER_SCAN_MISMATCH"])
        self.assertEqual(reasons[0]["split"], "C")

    def test_test_split_never_triggers_prefit_reasons(self):
        # 预注册：三探针冻结前不触碰 test → test 的负例/长度/加载器一致性一律不参与拟合前判定；
        # test 的负例下限随最终门禁 D1_counts 在冻结后判定。
        facts = prefit_facts()
        facts["negatives"] = dict(facts["negatives"], test=0)
        facts["loader_parity"] = {"B": True, "C": True, "dev": True, "test": False}
        facts["length_mismatches"] = [{"split": "test", "actual": 10, "expected": 11}]
        self.assertEqual(latent_diag.prefit_abort_reasons(facts), [])

    def test_prefit_splits_is_the_boundary(self):
        # A/B/C/dev 四个切分内的问题必须照常触发（边界没有被放宽到只剩 B/C）
        for split in latent_diag.PREFIT_SPLITS:
            facts = prefit_facts()
            facts["negatives"] = dict(facts["negatives"], **{split: 99})
            reasons = latent_diag.prefit_abort_reasons(facts)
            self.assertEqual([r["code"] for r in reasons], ["MIN_NEGATIVES"], split)
            self.assertEqual(reasons[0]["split"], split)


def prepare_tiny_inputs(td, sizes=None, dev_start=1000, test_start=2000):
    """合成数据文件 + 微型 Stage-1（bench.run_stage1）；返回 (root, data_files, meta)。

    默认与既有测试完全一致（SMOKE_SIZES + 起点 1000/2000）；放大 sizes 时须把 dev/test 起点
    移到 train 总行数之外，避免与 train 行重复。
    """
    sizes = dict(latent_diag.SMOKE_SIZES if sizes is None else sizes)
    root = Path(td) / "artifacts"
    data_dir = Path(td) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    total = sizes["A"] + sizes["B"] + sizes["C"]
    latent_diag.write_aliccp_file(
        data_dir / "train.csv", latent_diag.make_smoke_rows(total, start=0)
    )
    latent_diag.write_aliccp_file(
        data_dir / "dev.csv", latent_diag.make_smoke_rows(sizes["dev"], start=dev_start)
    )
    latent_diag.write_aliccp_file(
        data_dir / "test.csv", latent_diag.make_smoke_rows(sizes["test"], start=test_start)
    )
    data_files = {
        "train": str(data_dir / "train.csv"),
        "val": str(data_dir / "dev.csv"),
        "test": str(data_dir / "test.csv"),
    }
    prefix_tag = latent_diag.prefix_tag_for(total, sizes["dev"], sizes["test"])
    meta = bench.run_stage1(
        root=root, data_files=data_files,
        budgets={"train": total, "val": sizes["dev"], "test": sizes["test"]},
        prefix_tag=prefix_tag, model_seed=123, env_seed=456, epochs=1, patience=1,
        device=torch.device("cpu"), log=_noop,
    )
    return root, data_files, meta


def run_tiny_pipeline(td, *, enforce=False, require_clean_git=False, tag="latdiag-smoke"):
    """端到端小流程：合成文件 → 微型 Stage-1（bench.run_stage1）→ 诊断运行。"""
    root, data_files, meta = prepare_tiny_inputs(td)
    sizes = latent_diag.SMOKE_SIZES
    result = latent_diag.run_diagnostic(
        root=root, stage1_id=meta["stage1_id"], data_files=data_files,
        a_rows=sizes["A"], b_rows=sizes["B"], c_rows=sizes["C"],
        val_budget=sizes["dev"], test_budget=sizes["test"],
        model_seed=123, device=torch.device("cpu"), enforce=enforce,
        require_clean_git=require_clean_git, tag=tag, batch_size=8, epochs=1, patience=1,
        log=_noop,
    )
    return root, data_files, meta, result


def _designed_collect_split_probs(model, head, loader, device):
    """设计型合成收集器（替换模型输出；标签一律取自 loader，不伪造）：
    - bsi_prob 恒 0.5：BSI 基线无信息（正确训练的头在该设计中得不到信号）；
    - ctr_prob 由 BSI 标签决定（y=1 → 0.9 段、y=0 → 0.1 段，两段不重叠）：旧任务分数强信息；
    - cvr_prob 为与标签无关的确定性噪声。
    y/click/purchase 与计数取自 loader，与 CSV 扫描计数保持 loader-parity。"""
    ys, clicks, purchases = [], [], []
    for click, purchase, y, _features in loader:
        ys.append(y.reshape(-1))
        clicks.append(click.reshape(-1))
        purchases.append(purchase.reshape(-1))
    y_arr = torch.cat(ys).numpy().astype(np.int8)
    click_arr = torch.cat(clicks).numpy().astype(np.int8)
    purchase_arr = torch.cat(purchases).numpy().astype(np.int8)
    n = int(len(y_arr))
    idx = np.arange(n, dtype=np.float64)
    bsi = np.full(n, 0.5, dtype=np.float64)
    ctr = np.where(y_arr == 1, 0.9, 0.1) + 0.01 * ((idx * 7.0) % 11.0) / 11.0
    cvr = 0.3 + 0.2 * ((idx * 5.0) % 13.0) / 13.0
    return {
        "y": y_arr, "click": click_arr, "purchase": purchase_arr,
        "bsi_prob": bsi.astype(np.float32),
        "ctr_prob": ctr.astype(np.float32),
        "cvr_prob": cvr.astype(np.float32),
        "n": n,
        "click1": int(click_arr.sum()),
        "purchase1": int(purchase_arr.sum()),
        "bsi_pos": int(y_arr.sum()),
    }


def run_designed_pipeline(td, *, extra_patches=()):
    """设计信息型端到端小流程：数据 / 微型 Stage-1 / 身份 / 指纹 / 门禁全部走真实路径，
    仅收集阶段以合成分数替换模型输出（旧任务 CTR 强信息、BSI 基线无信息）。
    extra_patches 为在 run_diagnostic 调用期间生效的 mock.patch 对象（验证器拒绝
    缺陷产物的回归测试使用）。"""
    root, data_files, meta = prepare_tiny_inputs(td)
    sizes = latent_diag.SMOKE_SIZES
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(latent_diag, "collect_split_probs", _designed_collect_split_probs)
        )
        for patch in extra_patches:
            stack.enter_context(patch)
        result = latent_diag.run_diagnostic(
            root=root, stage1_id=meta["stage1_id"], data_files=data_files,
            a_rows=sizes["A"], b_rows=sizes["B"], c_rows=sizes["C"],
            val_budget=sizes["dev"], test_budget=sizes["test"],
            model_seed=123, device=torch.device("cpu"), enforce=False,
            require_clean_git=False, tag="latdiag-designed",
            batch_size=8, epochs=1, patience=1, log=_noop,
        )
    return root, data_files, meta, result


# ---- enforce=True 小规模直通基座（拟合前中止为空的规模）----
# 每切分 605 行 → raw==2 的 BSI 负例 = 121 ≥ MIN_BSI_NEGATIVES(100)：A/B/C/dev 均不触发
# MIN_NEGATIVES 中止，enforce=True 可完整走到拟合与完整指纹校验；dev/test 起点移到 train
# 总行数（1815）之外，避免跨切分重复行。
NEGATIVES_SAFE_SIZES = {"A": 605, "B": 605, "C": 605, "dev": 605, "test": 605}
NEGATIVES_SAFE_TOTAL = NEGATIVES_SAFE_SIZES["A"] + NEGATIVES_SAFE_SIZES["B"] + NEGATIVES_SAFE_SIZES["C"]
NEGATIVES_SAFE_PREFIX_TAG = latent_diag.prefix_tag_for(
    NEGATIVES_SAFE_TOTAL, NEGATIVES_SAFE_SIZES["dev"], NEGATIVES_SAFE_SIZES["test"]
)


def prepare_negatives_safe_inputs(td):
    return prepare_tiny_inputs(
        td, sizes=NEGATIVES_SAFE_SIZES, dev_start=100_000, test_start=200_000
    )


def run_enforce_tiny(root, data_files, meta, fp, *, tag, patches=(), log=_noop, **kwargs):
    """enforce=True 直通运行：FORMAL_* 常量 mock 为本地微型身份；git 要求显式放开（本测试不判 git）。

    patches 为在 run_diagnostic 调用期间生效的 mock.patch 对象列表（调用方自行构造），
    与 FORMAL_* mock 一起由 ExitStack 管理。
    """
    sizes = NEGATIVES_SAFE_SIZES
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(latent_diag, "FORMAL_STAGE1_ID", meta["stage1_id"]))
        stack.enter_context(
            mock.patch.object(latent_diag, "FORMAL_FINGERPRINT_SHA256", fp["fingerprint_sha256"])
        )
        for patch in patches:
            stack.enter_context(patch)
        return latent_diag.run_diagnostic(
            root=root, stage1_id=meta["stage1_id"], data_files=data_files,
            a_rows=sizes["A"], b_rows=sizes["B"], c_rows=sizes["C"],
            val_budget=sizes["dev"], test_budget=sizes["test"],
            model_seed=123, device=torch.device("cpu"), enforce=True,
            require_clean_git=False, tag=tag, batch_size=8, epochs=1, patience=1,
            log=log, **kwargs,
        )


class TestCliSmoke(unittest.TestCase):
    def test_smoke_command_runs_and_self_verifies(self):
        import run_aliccp_latent_diag

        with tempfile.TemporaryDirectory() as td:
            code = run_aliccp_latent_diag.main(["smoke", "--root", td, "--device", "cpu"])
            self.assertEqual(code, 0)
            # smoke 在 <root>/<时间戳>/ 下自建微型产物，全部本地化
            verifications = list(Path(td).glob("*/runs/*/verification.json"))
            self.assertEqual(len(verifications), 1)
            record = load_json(verifications[0])
            self.assertEqual(record["overall"], "PASS")
            self.assertEqual(record["recomputed"]["status"], "OK")
            # smoke 不写正式 SUMMARY
            self.assertEqual(list(Path(td).glob("*/" + latent_diag.SUMMARY_FILE)), [])


class TestEndToEndTiny(unittest.TestCase):
    def test_smoke_sizes_hit_singleton_batches(self):
        # batch=8 下 A/B/C 均有余 1 行 → 覆盖 batch 单例路径
        for key in ("A", "B", "C"):
            self.assertEqual(latent_diag.SMOKE_SIZES[key] % 8, 1, key)

    def test_pipeline_writes_full_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root, _, meta, result = run_tiny_pipeline(td)
            self.assertEqual(result["status"], "OK")
            self.assertEqual(result["verdict"], "NO_GO_FOR_EXTENSION")  # 小数据必不达标
            run_dir = Path(result["run_dir"])
            for name in (
                "config.json", "reported.json", "gate_report.json", "probes.json",
                "raw.npz", "perms.npz", "probe_probs.npz", "bsi_head.pt",
            ):
                self.assertTrue((run_dir / name).exists(), name)

            raw = np.load(run_dir / "raw.npz")
            try:
                for split in latent_diag.ALL_SPLITS:
                    for field in ("y", "click", "purchase", "bsi_prob", "ctr_prob", "cvr_prob"):
                        self.assertIn(f"{split}__{field}", raw.files)
                self.assertEqual(len(raw["B__y"]), latent_diag.SMOKE_SIZES["B"])
                self.assertEqual(raw["B__bsi_prob"].dtype, np.float32)
            finally:
                raw.close()

            with np.load(run_dir / "perms.npz") as perms:
                self.assertEqual(len(perms.files), 4)

            with np.load(run_dir / "probe_probs.npz") as probs:
                self.assertEqual(len(probs.files), 12)  # 4 splits × 3 arms

            reported = load_json(run_dir / "reported.json")
            self.assertEqual(reported["stage1_id"], meta["stage1_id"])
            self.assertEqual(reported["head"]["best_epoch"], 1)
            self.assertEqual(reported["splits"]["B"]["n"], latent_diag.SMOKE_SIZES["B"])
            self.assertFalse(reported["signal_positive"])
            self.assertEqual(
                sorted(reported["aucs"].keys()), sorted(latent_diag.ALL_SPLITS)
            )
            for split in latent_diag.ALL_SPLITS:
                self.assertEqual(sorted(reported["aucs"][split].keys()), sorted(latent_diag.ARMS))
            # Δ = AUC(full) − max(AUC(base), AUC(shuffle))
            for split in latent_diag.ALL_SPLITS:
                aucs = reported["aucs"][split]
                expected = aucs["full"] - max(aucs["base"], aucs["shuffle"])
                self.assertAlmostEqual(reported["deltas"][split]["delta"], expected, places=12)

            config = load_json(run_dir / "config.json")["config"]
            self.assertEqual(config["splits"]["A"]["range"], [0, latent_diag.SMOKE_SIZES["A"]])
            self.assertEqual(config["head"]["epochs"], 1)
            self.assertFalse(config["enforce"])
            # OK 运行：test 计数在探针冻结后扫描并写入 config（而非拟合前）
            self.assertIsNotNone(config["splits"]["test"]["counts_raw"])
            self.assertEqual(
                config["splits"]["test"]["counts_raw"]["n"], latent_diag.SMOKE_SIZES["test"]
            )

    def test_verdict_fields_present_for_smoke_reasons(self):
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_tiny_pipeline(td)  # enforce=False：负例不足只记录不中止
            self.assertEqual(result["status"], "OK")
            codes = {r["code"] for r in result["recorded_reasons"]}
            self.assertIn("MIN_NEGATIVES", codes)


class TestDesignedInformativeEndToEnd(unittest.TestCase):
    """回归（臂语义缺陷的端到端判别）：设计型数据下旧任务 CTR 分数对 BSI 标签强信息、BSI 基线
    无信息 → 修复语义下 full（行对齐）与 shuffle（CTR/CVR 联合置换）必须产生不同的探针、
    不同的逐行概率（缺陷下两臂输入逐位相同 → 全部逐位相同）。

    只断言"两臂不同"与设计稠查（信息型旧任务分数在 out-of-sample C 上被 full 臂识别）；
    不主张任何性能增益：不断言 Δ、门禁或 verdict（小规模合成数据的信号不能外推）。"""

    def test_full_and_shuffle_arms_are_distinct_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_designed_pipeline(td)
            self.assertEqual(result["status"], "OK")
            run_dir = Path(result["run_dir"])

            probes = load_json(run_dir / "probes.json")["arms"]
            full_coef = np.asarray(probes["full"]["coef"], dtype=np.float64)
            shuffle_coef = np.asarray(probes["shuffle"]["coef"], dtype=np.float64)
            self.assertFalse(
                np.allclose(full_coef, shuffle_coef, rtol=0, atol=1e-9),
                "full 与 shuffle 探针系数不得相同（历史缺陷下逐位相同）",
            )

            with np.load(run_dir / "probe_probs.npz") as probs:
                for split in latent_diag.ALL_SPLITS:
                    diff = float(np.max(np.abs(
                        probs[f"{split}__full_prob"] - probs[f"{split}__shuffle_prob"]
                    )))
                    self.assertGreater(diff, 0.0, f"{split}: full/shuffle 逐行概率不得相同")

            aucs = load_json(run_dir / "reported.json")["aucs"]
            # 设计稠查：强信息旧任务分数必须在 out-of-sample C 上被 full 臂识别
            self.assertGreater(aucs["C"]["full"], 0.9)
            self.assertGreater(aucs["C"]["full"], aucs["C"]["shuffle"])

            # 独立验证器在修复语义下必须接受这次诚实运行（runner 与验证器重算一致）
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            self.assertEqual(
                verdict["overall"], "PASS", [c for c in verdict["checks"] if not c["ok"]]
            )


class TestSingletonHeadForward(unittest.TestCase):
    """runner 侧单例批次适配：核心 NewTask 内部 squeeze() 在 batch=1 时折叠 batch 维
    （model.py:926 torch.stack(dim=2) 越界），核心模型不可改 → 复制-丢弃在 runner 完成。"""

    def _head(self):
        torch.manual_seed(0)
        head = NewTask(
            input_size=protocol.INPUT_SIZE,
            rep_dim=protocol.NEWTASK_REP_DIM,
            tower_dnn_hidden_units=list(protocol.TOWER_HIDDEN),
            reg_dnn=protocol.REG_DNN,
            device=torch.device("cpu"),
        )
        head.eval()
        return head

    def _inputs(self, batch):
        torch.manual_seed(1)
        dnn_input = torch.randn(batch, protocol.INPUT_SIZE)
        gen_rep = torch.randn(batch, protocol.NEWTASK_REP_DIM)
        spec_reps = [
            torch.randn(batch, protocol.NEWTASK_REP_DIM) for _ in range(protocol.NUM_TASKS)
        ]
        env_embs = [torch.randn(protocol.NEWTASK_REP_DIM) for _ in range(protocol.NUM_TASKS)]
        return dnn_input, gen_rep, spec_reps, env_embs

    @staticmethod
    def _duplicate(inputs):
        dnn_input, gen_rep, spec_reps, env_embs = inputs
        return (
            dnn_input.repeat(2, 1),
            gen_rep.repeat(2, 1),
            [rep.repeat(2, 1) for rep in spec_reps],
            env_embs,  # env_embs 每项形状 (rep_dim,)，与 batch 无关
        )

    def test_singleton_via_duplicate_matches_duplicated_batch(self):
        head = self._head()
        x1 = self._inputs(1)
        out1 = latent_diag.head_forward_batch_safe(head, *x1)
        self.assertEqual(tuple(out1.shape), (1,))
        out2 = head(*self._duplicate(x1))
        self.assertEqual(tuple(out2.shape), (2,))
        # NewTask 行间无耦合（LayerNorm/softmax 均沿特征维）→ 复制批次两行输出逐位相同，
        # 且等于单例经复制-丢弃路径得到的输出
        self.assertTrue(torch.equal(out2[0], out2[1]))
        self.assertTrue(torch.equal(out1[0], out2[0]))

    def test_singleton_one_output_row_per_original_label(self):
        head = self._head()
        x1 = self._inputs(1)
        pred = latent_diag.head_forward_batch_safe(head, *x1)
        y = torch.tensor([1.0])
        self.assertEqual(pred.numel(), y.numel())  # 复制行输出已丢弃：每原始标签恰好一个预测
        loss_single = torch.nn.BCELoss()(pred.reshape(-1), y)
        # "复制行+复制标签"的均值损失 == 单例损失（(l+l)/2=l）→ 训练权重不会被放大
        pred_dup = head(*self._duplicate(x1))
        loss_dup = torch.nn.BCELoss()(pred_dup.reshape(-1), y.repeat(2))
        self.assertEqual(loss_single.item(), loss_dup.item())

    def test_singleton_gradients_not_doubled(self):
        head_a = self._head()
        head_b = copy.deepcopy(head_a)
        x1 = self._inputs(1)
        y = torch.tensor([1.0])
        torch.nn.BCELoss()(
            latent_diag.head_forward_batch_safe(head_a, *x1).reshape(-1), y
        ).backward()
        torch.nn.BCELoss()(
            head_b(*self._duplicate(x1)).reshape(-1), y.repeat(2)
        ).backward()
        for (name, param_a), (_, param_b) in zip(
            head_a.named_parameters(), head_b.named_parameters()
        ):
            self.assertIsNotNone(param_a.grad, name)
            self.assertTrue(
                torch.allclose(param_a.grad, param_b.grad, rtol=0, atol=1e-7),
                f"{name}: 单例路径与复制批次梯度应一致（权重不放大）",
            )

    def test_multirow_batches_pass_through_unchanged(self):
        head = self._head()
        x3 = self._inputs(3)
        self.assertTrue(
            torch.equal(latent_diag.head_forward_batch_safe(head, *x3), head(*x3))
        )


class TestProvisionalFingerprintChecks(unittest.TestCase):
    """provisional 指纹身份校验（自哈希 + train/val）：绝不触碰 test 源文件；篡改必须逐项失败。"""

    def _build(self, td):
        data_dir = Path(td) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        specs = {
            "train": ("train.csv", 20, 0),
            "val": ("dev.csv", 10, 1000),
            "test": ("test.csv", 8, 2000),
        }
        paths, files_budgets = {}, {}
        for tag, (name, n, start) in specs.items():
            path = data_dir / name
            latent_diag.write_aliccp_file(path, latent_diag.make_smoke_rows(n, start=start))
            paths[tag] = path
            files_budgets[tag] = (path, n)
        fp = protocol.build_fingerprint("p20-v10-t8", files_budgets)
        return paths, fp

    def test_train_val_only_and_no_test_read(self):
        with tempfile.TemporaryDirectory() as td:
            paths, fp = self._build(td)
            test_path = paths["test"].resolve()
            calls = {"test_reads": 0}

            def guard(real):
                def wrapper(path, *args, **kwargs):
                    if Path(path).resolve() == test_path:
                        calls["test_reads"] += 1
                    return real(path, *args, **kwargs)
                return wrapper

            with mock.patch.object(protocol, "prefix_sha256", guard(protocol.prefix_sha256)), \
                    mock.patch.object(protocol, "scan_label_counts", guard(protocol.scan_label_counts)), \
                    mock.patch.object(protocol, "_header_sha256", guard(protocol._header_sha256)):
                out = latent_diag.provisional_fingerprint_checks(fp)
            self.assertTrue(out["passed"])
            self.assertEqual(out["scope"], ["train", "val"])
            self.assertEqual(set(out["files"]), {"train", "val"})
            self.assertEqual(calls["test_reads"], 0, "provisional 校验不得读取 test 源文件")
            for detail in out["files"].values():
                self.assertTrue(
                    detail["prefix_sha256"] and detail["label_counts"] and detail["header_sha256"]
                )
                self.assertIsNone(detail["error"])

    def test_tampered_train_file_fails_provisionally(self):
        with tempfile.TemporaryDirectory() as td:
            paths, fp = self._build(td)
            lines = paths["train"].read_text(encoding="utf-8").splitlines()
            parts = lines[1].split(",")
            parts[0] = "1" if parts[0] == "0" else "0"
            lines[1] = ",".join(parts)
            paths["train"].write_text("\n".join(lines) + "\n", encoding="utf-8")
            out = latent_diag.provisional_fingerprint_checks(fp)
            self.assertFalse(out["passed"])
            self.assertTrue(out["self_hash"])  # 指纹文件本身未改动
            self.assertFalse(out["files"]["train"]["prefix_sha256"])
            self.assertFalse(out["files"]["train"]["label_counts"])

    def test_tampered_fingerprint_body_fails_self_hash(self):
        with tempfile.TemporaryDirectory() as td:
            _, fp = self._build(td)
            fp["budgets"]["train"] += 1  # 字段被改动但自哈希未重算
            out = latent_diag.provisional_fingerprint_checks(fp)
            self.assertFalse(out["self_hash"])
            self.assertFalse(out["passed"])


class TestFormalPrefitNoTestAccess(unittest.TestCase):
    """回归（独立代码审计发现的拟合前 test 读取泄漏）：旧实现在 enforce=True 拟合前调用
    protocol.verify_fingerprint(fp)，其内部对 train/val/test 全部重读前缀字节并扫描标签
    （protocol.py:249-264，含 test）→ 违反"三探针冻结前不触碰 test"（预注册）。新实现拟合前
    仅做 provisional（自哈希 + train/val），完整校验推迟到三探针冻结后、test 评估前。

    插桩直接替换 protocol 模块属性：verify_fingerprint 内部经由模块全局名调用 → 旧实现的
    泄漏读同样被记录（这正是"旧序必败、新序必过"的判别点：断言要求任何 test 源文件读取
    都必须发生在 3 次 fit_probe 之后）。"""

    def test_no_test_source_access_before_probes_frozen(self):
        with tempfile.TemporaryDirectory() as td:
            events = []  # (kind, fits_at_event)
            state = {"fits": 0}
            real_fit = latent_diag.fit_probe
            real_scan = protocol.scan_label_counts
            real_prefix = protocol.prefix_sha256
            real_header = protocol._header_sha256
            real_verify = protocol.verify_fingerprint
            real_dataset = latent_diag.AliCCPDataset
            real_loader = latent_diag.DataLoader
            test_path = {"resolved": None}

            def _is_test(path) -> bool:
                if test_path["resolved"] is None:
                    return False
                try:
                    return Path(path).resolve() == test_path["resolved"]
                except TypeError:
                    return False

            def spy_fit(x, y):
                out = real_fit(x, y)
                state["fits"] += 1
                return out

            def spy_scan(path, n):
                if _is_test(path):
                    events.append(("scan_test", state["fits"]))
                return real_scan(path, n)

            def spy_prefix(path, n):
                if _is_test(path):
                    events.append(("prefix_test", state["fits"]))
                return real_prefix(path, n)

            def spy_header(path):
                if _is_test(path):
                    events.append(("header_test", state["fits"]))
                return real_header(path)

            def spy_verify(fp):
                events.append(("verify_fingerprint", state["fits"]))
                return real_verify(fp)

            def spy_dataset(path, *args, **kwargs):
                ds = real_dataset(path, *args, **kwargs)
                if _is_test(path):
                    events.append(("dataset_test", state["fits"]))
                    ds._callorder_test_marker = True
                return ds

            def spy_loader(dataset, *args, **kwargs):
                if getattr(dataset, "_callorder_test_marker", False):
                    events.append(("loader_test", state["fits"]))
                return real_loader(dataset, *args, **kwargs)

            patches = [
                mock.patch.object(protocol, "scan_label_counts", spy_scan),
                mock.patch.object(protocol, "prefix_sha256", spy_prefix),
                mock.patch.object(protocol, "_header_sha256", spy_header),
                mock.patch.object(protocol, "verify_fingerprint", spy_verify),
                mock.patch.object(latent_diag, "fit_probe", spy_fit),
                mock.patch.object(latent_diag, "AliCCPDataset", spy_dataset),
                mock.patch.object(latent_diag, "DataLoader", spy_loader),
            ]
            # 数据准备（含 run_stage1 的指纹构建，合成数据豁免）在插桩之外完成
            root, data_files, meta = prepare_negatives_safe_inputs(td)
            test_path["resolved"] = Path(data_files["test"]).resolve()
            fp = protocol.load_fingerprint(root, NEGATIVES_SAFE_PREFIX_TAG)

            result = run_enforce_tiny(
                root, data_files, meta, fp, tag="latdiag-order", patches=patches
            )
            self.assertEqual(result["status"], "OK", result.get("abort_reasons"))

            violations = [(kind, fits) for kind, fits in events if fits < 3]
            self.assertEqual(violations, [], f"三探针冻结前触碰 test：{events}")
            kinds = [kind for kind, _ in events]
            self.assertEqual(kinds.count("verify_fingerprint"), 1, events)
            self.assertEqual(
                [fits for kind, fits in events if kind == "verify_fingerprint"], [3],
                "完整指纹校验必须在 3 次 fit_probe 之后才调用",
            )
            # 冻结后 test 触碰结构：完整校验（前缀/标签计数/表头各 1 次）＋ 诊断自身单次评估触碰
            self.assertEqual(kinds.count("prefix_test"), 1, events)
            self.assertEqual(kinds.count("header_test"), 1, events)
            self.assertEqual(kinds.count("scan_test"), 2, events)
            self.assertEqual(kinds.count("dataset_test"), 1, events)
            self.assertEqual(kinds.count("loader_test"), 1, events)

            # 身份字段区分：provisional（train/val，拟合前）与 full（train/val/test，冻结后）
            config = load_json(Path(result["run_dir"]) / "config.json")["config"]
            verification = config["fingerprint"]["verification"]
            self.assertEqual(verification["provisional"]["scope"], ["train", "val"])
            self.assertTrue(verification["provisional"]["passed"])
            self.assertEqual(verification["full"]["scope"], ["train", "val", "test"])
            self.assertEqual(verification["full"]["timing"], "post_freeze_pre_test_eval")
            self.assertTrue(verification["full"]["passed"])
            reported = load_json(Path(result["run_dir"]) / "reported.json")
            self.assertTrue(reported["fingerprint_verification"]["full"]["passed"])


class TestFullFingerprintFailureAborts(unittest.TestCase):
    """完整指纹校验（含 test）失败 → 中止且不产生任何 test 指标（不扫描 test 标签、不建 test
    数据集/DataLoader、不推理、无 reported.json/probes 产物）；且此类因源文件与预注册指纹
    不匹配的运行不得被独立验证器判 PASS（预注册：不匹配即作废）——验证器自身的
    fingerprint_files 重算必须抓住同一处不匹配。"""

    def test_tampered_test_file_aborts_without_test_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, meta = prepare_negatives_safe_inputs(td)
            fp = protocol.load_fingerprint(root, NEGATIVES_SAFE_PREFIX_TAG)
            # 指纹构建（run_stage1）之后篡改 test 首行 click 标签：train/val 完好 → provisional
            # 通过；完整校验在 test 前缀字节（与标签计数）上失败
            test_path = Path(data_files["test"])
            lines = test_path.read_text(encoding="utf-8").splitlines()
            parts = lines[1].split(",")
            parts[0] = "1" if parts[0] == "0" else "0"
            lines[1] = ",".join(parts)
            test_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            result = run_enforce_tiny(root, data_files, meta, fp, tag="latdiag-fullfail")
            self.assertEqual(result["status"], "ABORT")
            self.assertEqual(result["verdict"], "ABORT_NO_GO")
            self.assertEqual(
                [r["code"] for r in result["abort_reasons"]], ["FULL_FINGERPRINT_VERIFY_FAIL"]
            )
            self.assertIsNone(result["aucs"])
            self.assertIsNone(result["deltas"])
            self.assertIsNone(result["classification"])
            self.assertFalse(result["signal_positive"])

            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "abort.json").exists())
            self.assertFalse((run_dir / "reported.json").exists())
            self.assertFalse((run_dir / "probes.json").exists())
            with np.load(run_dir / "raw.npz") as raw:
                self.assertEqual([k for k in raw.files if k.startswith("test__")], [])
                self.assertTrue([k for k in raw.files if k.startswith("B__")])
            config = load_json(run_dir / "config.json")["config"]
            self.assertIsNone(config["splits"]["test"]["counts_raw"])
            verification = config["fingerprint"]["verification"]
            self.assertTrue(verification["provisional"]["passed"])
            self.assertFalse(verification["full"]["passed"])
            self.assertIn("test", verification["full"]["detail"])
            abort = load_json(run_dir / "abort.json")
            self.assertEqual(abort["abort_phase"], "post_freeze_pre_test_eval")

            # 数据与指纹不匹配的运行不得被独立验证器判 PASS（无 SUMMARY；不匹配即作废）
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            self.assertEqual(verdict["overall"], "FAIL")
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertIn("fingerprint_files", failed)


class TestTestSplitCallOrder(unittest.TestCase):
    """回归（独立代码审计发现的协议违规）：run_diagnostic 曾在三探针冻结前扫描 label 计数并
    构建 AliCCPDataset(test)。预注册要求：三探针全部冻结前不读 test CSV、不建 test 数据集/
    DataLoader、不扫描 test 标签。

    本测试为 smoke 路径（enforce=False，合成数据豁免）：ensure_fingerprint 允许在拟合前构建/
    校验含 test 的完整指纹（指纹内部调用真实模块函数，经 proxy 与诊断自身的触碰区分开）；
    正式路径（enforce=True）拟合前对 test 的零读取由 TestFormalPrefitNoTestAccess 覆盖。"""

    def test_test_split_untouched_until_probes_frozen(self):
        from aliccp_benchmark import protocol as protocol_mod

        with tempfile.TemporaryDirectory() as td:
            root, data_files, meta = prepare_tiny_inputs(td)
            test_path = Path(data_files["test"]).resolve()

            events = []  # (what, probes_frozen_at_event)
            state = {"fits": 0}
            real_scan = protocol_mod.scan_label_counts
            real_dataset = latent_diag.AliCCPDataset
            real_loader = latent_diag.DataLoader
            real_fit = latent_diag.fit_probe

            def _is_test(path) -> bool:
                try:
                    return Path(path).resolve() == test_path
                except TypeError:
                    return False

            def spy_fit_probe(x, y):
                out = real_fit(x, y)
                state["fits"] += 1
                return out

            def spy_scan(path, n):
                if _is_test(path):
                    events.append(("scan_test", state["fits"] >= 3))
                return real_scan(path, n)

            def spy_dataset(path, *args, **kwargs):
                ds = real_dataset(path, *args, **kwargs)
                if _is_test(path):
                    events.append(("dataset_test", state["fits"] >= 3))
                    ds._latdiag_test_marker = True
                return ds

            def spy_loader(dataset, *args, **kwargs):
                if getattr(dataset, "_latdiag_test_marker", False):
                    events.append(("loader_test", state["fits"] >= 3))
                return real_loader(dataset, *args, **kwargs)

            class _ProtocolProxy:
                """只拦截诊断自身对 protocol 属性的访问：verify/ensure_fingerprint 内部走真实
                模块全局函数，故其身份校验（既有协议允许的 test 字节读取）不会被误记为触碰。"""

                def __init__(self, module):
                    self._module = module
                    self.scan_label_counts = spy_scan

                def __getattr__(self, name):
                    return getattr(self._module, name)

            with mock.patch.object(latent_diag, "protocol", _ProtocolProxy(protocol_mod)), \
                    mock.patch.object(latent_diag, "fit_probe", spy_fit_probe), \
                    mock.patch.object(latent_diag, "AliCCPDataset", spy_dataset), \
                    mock.patch.object(latent_diag, "DataLoader", spy_loader):
                result = latent_diag.run_diagnostic(
                    root=root, stage1_id=meta["stage1_id"], data_files=data_files,
                    a_rows=latent_diag.SMOKE_SIZES["A"], b_rows=latent_diag.SMOKE_SIZES["B"],
                    c_rows=latent_diag.SMOKE_SIZES["C"],
                    val_budget=latent_diag.SMOKE_SIZES["dev"],
                    test_budget=latent_diag.SMOKE_SIZES["test"],
                    model_seed=123, device=torch.device("cpu"), enforce=False,
                    require_clean_git=False, tag="latdiag-callorder",
                    batch_size=8, epochs=1, patience=1, log=_noop,
                )

            self.assertEqual(result["status"], "OK")
            violations = [what for what, frozen in events if not frozen]
            self.assertEqual(violations, [], f"test 切分在探针冻结前被触碰: {events}")
            kinds = [what for what, _ in events]
            self.assertEqual(kinds.count("scan_test"), 1, f"test 标签扫描应恰好一次: {events}")
            self.assertEqual(kinds.count("dataset_test"), 1, f"test 数据集构建应恰好一次: {events}")
            self.assertEqual(kinds.count("loader_test"), 1, f"test DataLoader 构建应恰好一次: {events}")


if __name__ == "__main__":
    unittest.main()
