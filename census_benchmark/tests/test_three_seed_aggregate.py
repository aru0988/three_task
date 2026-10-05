"""3-seed 聚合工具单测（TDD；CPU、秒级、不读真实数据集、不加载权重）。

唯一事实来源：docs/superpowers/specs/2026-10-06-census-stage2-residual-prompt-3seed-expansion-design.md
§4.4（聚合公式）/ §4.5（S1–S8 终局判定）。

夹具全部为合成 run 目录（临时目录内的 metrics.json + gate_report.json），与真实产物的
键集同构（≥ `paired_verdict._load_run` 所读取的字段）。
"""
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from census_benchmark import three_seed_aggregate as A

REPO = Path(__file__).resolve().parents[2]
SEEDS = (1685480945, 1685463909, 1685477428)
SID = {1685480945: "s1-096f8f16-m1685480945-e2-cb2094b3",
       1685463909: "s1-096f8f16-m1685463909-e2-8b53a3bf",
       1685477428: "s1-096f8f16-m1685477428-e2-15eabcb4"}
BASE_TEST = 0.85          # 合成基线 test AUC（每种子同值即可，聚合只看差值）
INTEGRITY = ("A1", "A2", "A4", "A5", "B1", "B2", "B4")


def _make_run(dirpath, *, seed, variant, test_auc, best_val_auc, best_epoch=5, epochs=5,
              gate_overrides=None, mech_pass=True, run_seed=None, stage1_id=None):
    """写一个键集与真实 run 同构的合成 run 目录。run_seed 用于构造 run_id 种子不匹配的负例。"""
    rid_seed = seed if run_seed is None else run_seed
    suffix = "-rpg" if variant == "residual-prompt" else ""
    metrics = {
        "run_id": f"20261006-0500-s20260929-m{rid_seed}-short-c0ffee1{suffix}",
        "stage1_id": SID[seed] if stage1_id is None else stage1_id,
        "commit": "c0ffee1", "variant": variant,
        "stage2": {"epoch_records": [{"epoch": i + 1, "loss": 0.6, "auc_val_education": best_val_auc}
                                     for i in range(epochs)],
                   "best_epoch": best_epoch, "best_val_auc": best_val_auc, "test_auc": test_auc},
    }
    report = {key: {"pass": True, "detail": {}} for key in INTEGRITY}
    report["B3"] = {"pass": True, "detail": {"gate_mean": [0.9, 0.9]}}
    for key, ok in (gate_overrides or {}).items():
        report[key]["pass"] = ok
    if variant == "residual-prompt":
        metrics["rp_arm"] = {
            "E1": {"pass": False, "rule": ">= 0.8521", "observed": test_auc},
            "classification": "VALID_NEGATIVE",
            **{f"G{i}": {"pass": mech_pass, "rule": "x", "observed": {}} for i in range(1, 6)},
            "pass": False}
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
    (dirpath / "gate_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return dirpath


def build_pairs(tmp, deltas, val_deltas=None, mech=None, gate_overrides=None, run_seed=None,
                stage1_overrides=None):
    """按 seed → delta_test 构造三对合成 run；其余量取同构默认。"""
    val_deltas = val_deltas if val_deltas is not None else dict(deltas)
    mech = mech or {}
    gate_overrides = gate_overrides or {}
    run_seed = run_seed or {}
    stage1_overrides = stage1_overrides or {}
    pairs = {}
    for seed in SEEDS:
        base = _make_run(tmp / f"b{seed}", seed=seed, variant="baseline",
                         test_auc=BASE_TEST, best_val_auc=BASE_TEST,
                         run_seed=run_seed.get(seed),
                         stage1_id=stage1_overrides.get(seed))
        treat = _make_run(tmp / f"t{seed}", seed=seed, variant="residual-prompt",
                          test_auc=BASE_TEST + deltas[seed], best_val_auc=BASE_TEST + val_deltas[seed],
                          mech_pass=mech.get(seed, True), gate_overrides=gate_overrides.get(seed),
                          run_seed=run_seed.get(seed),
                          stage1_id=stage1_overrides.get(seed))
        pairs[seed] = (base, treat)
    return pairs


DELTAS_A = {1685480945: -0.0023308435, 1685463909: 0.0029909067604868556, 1685477428: 0.002}
DELTAS_B = {1685480945: 0.002, 1685463909: 0.0025, 1685477428: 0.0018}


class TestFrozenConstants(unittest.TestCase):
    def test_constants_pinned(self):
        self.assertEqual(A.CANONICAL_SEEDS, SEEDS)
        self.assertEqual(A.T_TWO_SIDED_95_DF2, 4.302652729696142)
        self.assertEqual(A.T_ONE_SIDED_95_DF2, 2.919985580355516)
        self.assertEqual(A.MEAN_MIN, 0.001)
        self.assertEqual(A.WORST_FLOOR, -0.001)
        self.assertEqual(A.EXPECTED_STAGE1_IDS, SID)

    def test_variant_strings_match_mechanism_module(self):
        from census_benchmark import residual_prompt as RP       # 仅测试期导入（工具本身不依赖 torch）
        self.assertEqual(A.VARIANT_BASELINE, RP.BASELINE_VARIANT)
        self.assertEqual(A.VARIANT_TREATMENT, RP.VARIANT)


class TestAggregateMath(unittest.TestCase):
    def test_historical_like_case_math_and_unstable_verdict(self):
        with tempfile.TemporaryDirectory() as td:
            report = A.three_seed_report(build_pairs(Path(td), DELTAS_A))
        stats = report["stats"]
        self.assertAlmostEqual(stats["mean"], 0.0008866877534956185, places=12)
        self.assertAlmostEqual(stats["sd"], 0.0028301686482276023, places=12)
        self.assertAlmostEqual(stats["ci_two_sided_95"][0], -0.006143840916092897, places=12)
        self.assertAlmostEqual(stats["ci_two_sided_95"][1], 0.007917216423084135, places=12)
        self.assertAlmostEqual(stats["lb_one_sided_95"], -0.0038845646870712736, places=12)
        self.assertEqual(stats["sign_positive_count"], 2)
        self.assertEqual(stats["positive_improvement_count"], 2)
        self.assertEqual(stats["worst_seed"], 1685480945)
        self.assertAlmostEqual(stats["worst_delta"], -0.0023308435, places=12)
        self.assertEqual(stats["mechanism_pass_count"], 3)
        self.assertEqual(stats["val_direction_consistency"], 3)
        self.assertEqual(report["S_failed"], ["S1", "S3", "S4", "S5"])
        self.assertEqual(report["verdict"], "UTILITY_SEED_UNSTABLE")
        self.assertEqual(report["decision"], "CLOSE_LINE")

    def test_tight_positive_case_is_stable_in_principle(self):
        with tempfile.TemporaryDirectory() as td:
            report = A.three_seed_report(build_pairs(Path(td), DELTAS_B))
        stats = report["stats"]
        self.assertAlmostEqual(stats["mean"], 0.0021, places=12)
        self.assertAlmostEqual(stats["sd"], 0.000360555127546399, places=12)
        self.assertAlmostEqual(stats["ci_two_sided_95"][0], 0.0012043314105081393, places=12)
        self.assertAlmostEqual(stats["ci_two_sided_95"][1], 0.0029956685894918606, places=12)
        self.assertAlmostEqual(stats["lb_one_sided_95"], 0.0014921565298442537, places=12)
        self.assertEqual(report["S_failed"], [])
        self.assertTrue(all(report["S"].values()))
        self.assertEqual(report["verdict"], "STABLE_UTILITY")
        self.assertEqual(report["decision"], "REGISTER_NEXT_VALIDATION")

    def test_row_records_full_pairing_details(self):
        with tempfile.TemporaryDirectory() as td:
            report = A.three_seed_report(build_pairs(Path(td), DELTAS_A))
        rows = {row["seed"]: row for row in report["seeds"]}
        row = rows[1685463909]
        self.assertEqual(row["stage1_id"], SID[1685463909])
        self.assertEqual(row["baseline"]["variant"], "baseline")
        self.assertEqual(row["treatment"]["variant"], "residual-prompt")
        self.assertAlmostEqual(row["delta_test"], DELTAS_A[1685463909], places=12)
        self.assertEqual(row["classification_new"], "POSITIVE_IMPROVEMENT")
        self.assertEqual(row["historical"]["classification"], "VALID_NEGATIVE")
        self.assertTrue(row["historical"]["g1_g5_all_pass"])
        self.assertTrue(row["mechanism_all_pass"])
        self.assertTrue(row["integrity_all_pass"])
        self.assertTrue(row["sign_match_test_val"])
        self.assertEqual(set(row["g1_g5_pass"]), {f"G{i}" for i in range(1, 6)})

    def test_val_direction_mismatch_lowers_consistency(self):
        val = dict(DELTAS_A)
        val[1685477428] = -0.002                       # 第三种子 val 与 test 反号（合成负例）
        with tempfile.TemporaryDirectory() as td:
            report = A.three_seed_report(build_pairs(Path(td), DELTAS_A, val_deltas=val))
        self.assertEqual(report["stats"]["val_direction_consistency"], 2)
        self.assertFalse(report["S"]["S6"])


class TestIntegrityChecks(unittest.TestCase):
    def test_requires_exactly_the_three_canonical_seeds(self):
        with tempfile.TemporaryDirectory() as td:
            pairs = build_pairs(Path(td), DELTAS_A)
            with self.assertRaises(ValueError):
                A.three_seed_report({k: v for k, v in pairs.items() if k != 1685477428})
            wrong = dict(pairs)
            wrong[12345] = wrong.pop(1685477428)
            with self.assertRaises(ValueError):
                A.three_seed_report(wrong)

    def test_stage1_id_pin_violation_raises(self):
        overrides = {1685477428: "s1-096f8f16-m9999999999-e2-00000000"}
        with tempfile.TemporaryDirectory() as td:
            pairs = build_pairs(Path(td), DELTAS_A, stage1_overrides=overrides)
            with self.assertRaises(ValueError):
                A.three_seed_report(pairs)

    def test_run_id_seed_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            pairs = build_pairs(Path(td), DELTAS_A, run_seed={1685463909: 42})
            with self.assertRaises(ValueError):
                A.three_seed_report(pairs)

    def test_legacy_baseline_without_variant_field_is_accepted(self):
        """历史基线 run（variant 字段引入前，如 904f8d0）无 variant 键 ⇒ None 视为 baseline。"""
        with tempfile.TemporaryDirectory() as td:
            pairs = build_pairs(Path(td), DELTAS_A)
            base, treat = pairs[1685480945]
            metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
            del metrics["variant"]                                   # 模拟历史 run
            (base / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
            report = A.three_seed_report(pairs)
        row = {r["seed"]: r for r in report["seeds"]}[1685480945]
        self.assertIsNone(row["baseline"]["variant"])
        self.assertEqual(row["treatment"]["variant"], "residual-prompt")

    def test_swapped_arms_raise(self):
        with tempfile.TemporaryDirectory() as td:
            pairs = build_pairs(Path(td), DELTAS_A)
            base, treat = pairs[1685477428]
            pairs[1685477428] = (treat, base)          # baseline 目录放进处理臂位置 → 无 rp_arm
            with self.assertRaises(ValueError):
                A.three_seed_report(pairs)

    def test_gate_failure_flags_s8(self):
        overrides = {1685477428: {"B2": False}}
        with tempfile.TemporaryDirectory() as td:
            report = A.three_seed_report(build_pairs(Path(td), DELTAS_B, gate_overrides=overrides))
        self.assertFalse(report["S"]["S8"])
        self.assertEqual(report["verdict"], "UTILITY_SEED_UNSTABLE")
        self.assertEqual(report["decision"], "CLOSE_LINE")

    def test_mechanism_failure_flags_s7(self):
        with tempfile.TemporaryDirectory() as td:
            report = A.three_seed_report(build_pairs(Path(td), DELTAS_B, mech={1685477428: False}))
        self.assertFalse(report["S"]["S7"])
        self.assertEqual(report["stats"]["mechanism_pass_count"], 2)


class TestCli(unittest.TestCase):
    def test_parse_pair_arg_windows_paths(self):
        seed, base, treat = A.parse_pair_arg(r"1685480945:D:\a\b\run,D:\a\c\run-rpg")
        self.assertEqual(seed, 1685480945)
        self.assertEqual(base, Path(r"D:\a\b\run"))
        self.assertEqual(treat, Path(r"D:\a\c\run-rpg"))

    def test_cli_writes_json_report(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pairs = build_pairs(td, DELTAS_A)
            argv = []
            for seed in SEEDS:
                base, treat = pairs[seed]
                argv += ["--pair", f"{seed}:{base},{treat}"]
            out = td / "agg.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = A.main(argv + ["--out", str(out)])
            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["verdict"], "UTILITY_SEED_UNSTABLE")
            self.assertEqual(len(payload["seeds"]), 3)


class TestSummaryLedgerUntouched(unittest.TestCase):
    def test_summary_ledger_not_a_migration_target(self):
        """SUMMARY.md 是各分支各自的只追加台账，不得从迁移源整文件拷贝（只在本分支追加本分支 run 行）。"""
        pinned = subprocess.run(["git", "rev-parse", "41feb29:artifacts/census_stage2/SUMMARY.md"],
                                cwd=REPO, capture_output=True, text=True)
        current = subprocess.run(["git", "hash-object", "artifacts/census_stage2/SUMMARY.md"],
                                 cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(current.stdout.strip(), pinned.stdout.strip())


if __name__ == "__main__":
    unittest.main()
