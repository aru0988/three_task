"""单测：五 seed 配对汇总与二级分类（预注册 §5.2/§6；纯函数 + CLI）。

CPU、秒级、不读真实数据集、不需要 GPU。合成记录字段与 `load_record` 产物一致。
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

from aliccp_benchmark import five_seed as FS

T = FS.T_975_DF4  # 2.7764451051977987（df=4）


def make_record(seed, delta_test, delta_val, *, classification=None, secondary=None,
                mechanism_pass=True, a_class_pass=True, alpha_final=0.05,
                generator_grad_last_epoch=0.01, geff_std=0.001,
                ratio_mean=(0.03, 0.03, 0.03), cos_mean=(0.1, 0.1, 0.1),
                pred_std=0.004, ref_pred_std=0.005, best_epoch=5, epochs_run=5,
                per_epoch_val=None, baseline_test_auc=0.6, baseline_val_auc=0.58):
    """字段与 load_record 一致；delta/分类默认由给定值机械推出（可显式覆盖）。"""
    return {
        "model_seed": seed,
        "stage1_id": f"s1-test-m{seed}",
        "baseline_run": f"run-base-{seed}",
        "arm_run": f"run-arm-{seed}",
        "baseline_test_auc": baseline_test_auc,
        "baseline_val_auc": baseline_val_auc,
        "arm_test_auc": baseline_test_auc + delta_test,
        "arm_val_auc": baseline_val_auc + delta_val,
        "delta_test": delta_test,
        "delta_val": delta_val,
        "u1": delta_test >= FS.HISTORIC_U1_MIN,
        "u2": delta_val > 0.0,
        "classification": classification or ("VALID_POSITIVE" if (delta_test >= FS.HISTORIC_U1_MIN and delta_val > 0) else "VALID_NEGATIVE"),
        "secondary": secondary or FS.secondary_classification(delta_test),
        "mechanism_pass": mechanism_pass,
        "a_class_pass": a_class_pass,
        "alpha_final": alpha_final,
        "generator_grad_last_epoch": generator_grad_last_epoch,
        "ratio_mean": list(ratio_mean),
        "ratio_std": [0.001] * 3,
        "cos_mean": list(cos_mean),
        "geff": {"geff_mean": ratio_mean[0], "geff_std": geff_std, "geff_min": 0.01, "geff_max": 0.05},
        "pred_std": pred_std,
        "ref_pred_std": ref_pred_std,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "per_epoch_val": per_epoch_val or [0.46, 0.49, 0.52, 0.56, 0.59],
    }


def five_records(deltas=(0.0054, 0.0081, 0.003, 0.007, 0.002), vals=(0.004, 0.008, 0.001, 0.006, 0.003)):
    return [make_record(FS.CANONICAL_SEEDS[i], deltas[i], vals[i]) for i in range(5)]


class TestSecondaryClassification(unittest.TestCase):
    def test_boundaries_exact(self):
        self.assertEqual(FS.secondary_classification(0.001), "POSITIVE_IMPROVEMENT")
        self.assertEqual(FS.secondary_classification(0.5), "POSITIVE_IMPROVEMENT")
        self.assertEqual(FS.secondary_classification(0.000999999), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(FS.secondary_classification(0.0), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(FS.secondary_classification(-0.019999999), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(FS.secondary_classification(-0.02), "CLEAR_DEGRADATION")
        self.assertEqual(FS.secondary_classification(-0.5), "CLEAR_DEGRADATION")

    def test_constants_pinned(self):
        self.assertEqual(FS.SECONDARY_POSITIVE_MIN, 0.001)
        self.assertEqual(FS.SECONDARY_DEGRADATION_MAX, -0.02)
        self.assertEqual(FS.T_975_DF4, 2.7764451051977987)
        self.assertEqual(FS.CANONICAL_SEEDS, (1688723512, 1688723740, 1688738016, 1688749593, 1688762746))
        self.assertEqual(FS.HISTORIC_U1_MIN, 0.0055)


class TestMeanStdTci(unittest.TestCase):
    def test_mean_std_sample_ddof1(self):
        mean, std = FS.mean_std([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(mean, 3.0)
        self.assertAlmostEqual(std, math.sqrt(2.5), places=15)

    def test_std_none_for_single(self):
        self.assertEqual(FS.mean_std([1.0]), (1.0, None))

    def test_t_ci_hand_computed(self):
        values = [0.005, 0.006, 0.007, 0.008, 0.009]
        mean, std = FS.mean_std(values)
        low, high = FS.t_ci(values)
        self.assertAlmostEqual(low, 0.007 - T * std / math.sqrt(5), places=15)
        self.assertAlmostEqual(high, 0.007 + T * std / math.sqrt(5), places=15)

    def test_t_ci_matches_known_value(self):
        # mean=0.007, s=sqrt(2.5e-6)≈0.00158113883008419; half = T*s/sqrt(5) ≈ 1.963172...
        low, high = FS.t_ci([0.005, 0.006, 0.007, 0.008, 0.009])
        self.assertAlmostEqual(low, 0.007 - 2.7764451051977987 * 0.0015811388300841898 / math.sqrt(5), places=15)


class TestHeadroom(unittest.TestCase):
    def test_active_positive_consistent_censored(self):
        rec = make_record(FS.CANONICAL_SEEDS[0], 0.0005, 0.001)
        out = FS.headroom_assessment(rec, same_sign_count=5, n_seeds=5)
        self.assertEqual(out["mechanism_activity"], "ACTIVE")
        self.assertEqual(out["validation_direction"], "POSITIVE")
        self.assertEqual(out["cross_seed_consistency"], "CONSISTENT")
        self.assertEqual(out["epoch_trajectory"], "RIGHT_CENSORED_STILL_IMPROVING")
        self.assertAlmostEqual(out["gap_to_positive_threshold"], 0.001 - 0.0005, places=15)

    def test_inactive_nonpositive_inconsistent_early_stopped(self):
        rec = make_record(FS.CANONICAL_SEEDS[0], -0.0005, -0.001, alpha_final=0.0,
                          generator_grad_last_epoch=0.0, geff_std=0.0, best_epoch=3, epochs_run=5)
        out = FS.headroom_assessment(rec, same_sign_count=1, n_seeds=5)
        self.assertEqual(out["mechanism_activity"], "INACTIVE")
        self.assertEqual(out["validation_direction"], "NON_POSITIVE")
        self.assertEqual(out["cross_seed_consistency"], "INCONSISTENT")
        self.assertEqual(out["epoch_trajectory"], "EARLY_STOPPED")

    def test_consistency_threshold_is_majority(self):
        rec = make_record(FS.CANONICAL_SEEDS[0], 0.0005, 0.001)
        self.assertEqual(FS.headroom_assessment(rec, same_sign_count=4, n_seeds=5)["cross_seed_consistency"], "CONSISTENT")
        self.assertEqual(FS.headroom_assessment(rec, same_sign_count=3, n_seeds=5)["cross_seed_consistency"], "INCONSISTENT")


class TestAggregate(unittest.TestCase):
    def test_all_fields_mechanical(self):
        recs = five_records()
        agg = FS.aggregate(recs)
        deltas = [r["delta_test"] for r in recs]
        vals = [r["delta_val"] for r in recs]
        self.assertEqual(agg["n_seeds"], 5)
        mean, std = FS.mean_std(deltas)
        self.assertEqual(agg["delta_test"]["mean"], mean)
        self.assertEqual(agg["delta_test"]["std"], std)
        low, high = FS.t_ci(deltas)
        self.assertEqual(agg["delta_test"]["ci95_low"], low)
        self.assertEqual(agg["delta_test"]["ci95_high"], high)
        self.assertEqual(agg["delta_test"]["t"], FS.T_975_DF4)
        self.assertEqual(agg["delta_test"]["min"], min(deltas))
        self.assertEqual(agg["delta_test"]["min_model_seed"], FS.CANONICAL_SEEDS[deltas.index(min(deltas))])
        self.assertEqual(agg["delta_val"]["mean"], FS.mean_std(vals)[0])
        self.assertEqual(agg["counts"]["delta_test_ge_0.001"], sum(1 for d in deltas if d >= 0.001))
        self.assertEqual(agg["counts"]["delta_test_ge_0.0055"], sum(1 for d in deltas if d >= 0.0055))
        self.assertEqual(agg["counts"]["delta_val_gt_0"], 5)
        self.assertEqual(agg["sign_consistency"]["delta_test_same_sign_count"], 5)
        self.assertTrue(agg["sign_consistency"]["delta_test_all_positive"])
        self.assertTrue(agg["sign_consistency"]["delta_val_all_positive"])
        self.assertEqual(agg["secondary_counts"]["POSITIVE_IMPROVEMENT"], 5)
        self.assertEqual(agg["secondary_counts"]["NO_CLEAR_IMPROVEMENT"], 0)
        self.assertEqual(agg["secondary_counts"]["CLEAR_DEGRADATION"], 0)
        self.assertEqual(agg["mechanism_stability"]["n_full_mechanism_pass"], 5)
        self.assertEqual(agg["mechanism_stability"]["alpha_final_signs"], {"positive": 5, "negative": 0, "zero": 0})
        self.assertEqual(len(agg["mechanism_stability"]["ratio_mean_seed_range"]), 3)
        self.assertEqual(len(agg["mechanism_stability"]["cos_mean_seed_range"]), 3)
        self.assertEqual(agg["mechanism_stability"]["geff_std_range"], [0.001, 0.001])

    def test_headroom_attached_only_for_no_clear(self):
        recs = five_records(deltas=(0.005, 0.0005, -0.01, 0.007, 0.002))
        agg = FS.aggregate(recs)
        by_secondary = {r["model_seed"]: r for r in agg["per_seed"]}
        self.assertIsNotNone(by_secondary[FS.CANONICAL_SEEDS[1]]["headroom"])
        self.assertIsNotNone(by_secondary[FS.CANONICAL_SEEDS[2]]["headroom"])
        self.assertIsNone(by_secondary[FS.CANONICAL_SEEDS[0]]["headroom"])
        self.assertIsNone(by_secondary[FS.CANONICAL_SEEDS[3]]["headroom"])

    def test_sign_consistency_mixed(self):
        recs = five_records(deltas=(0.005, -0.001, 0.003, 0.007, 0.002))
        agg = FS.aggregate(recs)
        self.assertFalse(agg["sign_consistency"]["delta_test_all_positive"])
        self.assertEqual(agg["sign_consistency"]["delta_test_signs"], [1, -1, 1, 1, 1])
        self.assertEqual(agg["sign_consistency"]["delta_test_same_sign_count"], 4)

    def test_mechanism_not_all_pass(self):
        recs = five_records()
        recs[2]["mechanism_pass"] = False
        agg = FS.aggregate(recs)
        self.assertEqual(agg["mechanism_stability"]["n_full_mechanism_pass"], 4)


class TestTwentyEpochCondition(unittest.TestCase):
    def _agg(self, deltas, vals, **kw):
        recs = five_records(deltas=deltas, vals=vals)
        for r, mech in zip(recs, kw.get("mech", [True] * 5)):
            r["mechanism_pass"] = mech
        return FS.aggregate(recs)

    def test_all_clauses_pass(self):
        agg = self._agg((0.006, 0.008, 0.0055, 0.007, 0.0056), (0.004, 0.008, 0.001, 0.006, 0.003))
        cond = FS.twenty_epoch_condition(agg)
        self.assertEqual(cond["status"], "TWENTY_EPOCH_CONDITION_SATISFIED")
        self.assertTrue(all(cond[k] for k in cond if k.startswith("C")))

    def test_c1_fails_on_mechanism(self):
        agg = self._agg((0.006, 0.008, 0.0055, 0.007, 0.0056), (0.004, 0.008, 0.001, 0.006, 0.003),
                        mech=[True, False, True, True, True])
        cond = FS.twenty_epoch_condition(agg)
        self.assertFalse(cond["C1_mechanism"])
        self.assertEqual(cond["status"], "TWENTY_EPOCH_CONDITION_NOT_SATISFIED")

    def test_c2_fails_on_degradation(self):
        agg = self._agg((0.006, 0.008, -0.03, 0.007, 0.0056), (0.004, 0.008, 0.001, 0.006, 0.003))
        cond = FS.twenty_epoch_condition(agg)
        self.assertFalse(cond["C2_no_degradation"])
        self.assertEqual(cond["status"], "TWENTY_EPOCH_CONDITION_NOT_SATISFIED")

    def test_c3_c4_boundaries_via_hand_built_aggregate(self):
        """C3/C4 的闭开语义用人工 aggregate 精确钉死（不依赖浮点均值巧合）。"""
        def agg_with(mean, ci_low, n_full=5, degradation=0, c5=4, c6=4):
            return {"n_seeds": 5,
                    "mechanism_stability": {"n_full_mechanism_pass": n_full},
                    "secondary_counts": {"CLEAR_DEGRADATION": degradation},
                    "delta_test": {"mean": mean, "ci95_low": ci_low},
                    "counts": {"delta_test_ge_0.001": c5, "delta_val_gt_0": c6}}
        cond = FS.twenty_epoch_condition(agg_with(0.0055, 1e-12))
        self.assertTrue(cond["C3_mean_at_least_0.0055"])      # == 阈值 ⇒ 满足（闭）
        self.assertTrue(cond["C4_ci_low_positive"])
        self.assertEqual(cond["status"], "TWENTY_EPOCH_CONDITION_SATISFIED")
        self.assertFalse(FS.twenty_epoch_condition(agg_with(0.0055 - 1e-12, 1e-12))["C3_mean_at_least_0.0055"])
        self.assertFalse(FS.twenty_epoch_condition(agg_with(0.006, 0.0))["C4_ci_low_positive"])   # == 0 ⇒ 不满足（严格）
        self.assertFalse(FS.twenty_epoch_condition(agg_with(0.006, -1e-12))["C4_ci_low_positive"])

    def test_c3_zero_std_ci_equals_mean(self):
        d = 0.006
        agg = self._agg((d, d, d, d, d), (d, d, d, d, d))
        cond = FS.twenty_epoch_condition(agg)
        self.assertTrue(cond["C3_mean_at_least_0.0055"])
        self.assertTrue(cond["C4_ci_low_positive"])           # std=0 ⇒ CI=[mean,mean]，mean>0 ⇒ 下界 >0

    def test_c5_and_c6_count_boundaries(self):
        agg = self._agg((0.004, 0.002, 0.0005, 0.003, 0.004), (0.004, 0.008, 0.001, 0.006, 0.003))
        cond = FS.twenty_epoch_condition(agg)
        self.assertEqual(agg["counts"]["delta_test_ge_0.001"], 4)
        self.assertTrue(cond["C5_positive_count_ge_4"])
        agg2 = self._agg((0.004, 0.002, 0.0005, 0.0005, 0.004), (0.004, 0.008, 0.001, 0.006, 0.003))
        self.assertFalse(FS.twenty_epoch_condition(agg2)["C5_positive_count_ge_4"])
        agg3 = self._agg((0.004, 0.002, 0.0005, 0.003, 0.004), (0.004, -0.001, 0.001, 0.006, -0.002))
        self.assertEqual(agg3["counts"]["delta_val_gt_0"], 3)
        self.assertFalse(FS.twenty_epoch_condition(agg3)["C6_val_positive_count_ge_4"])


class TestLoadRecord(unittest.TestCase):
    def _write_run(self, root: Path, run_id, *, model_seed, stage1_id, test_auc, val_auc,
                   best_epoch=5, per_epoch=None, variant=None, rp_arm=None):
        d = root / "runs" / run_id
        d.mkdir(parents=True)
        metrics = {"run_id": run_id, "model_seed": model_seed, "stage1_id": stage1_id,
                   "test_auc_bsi": test_auc, "best_val_auc_bsi": val_auc, "best_epoch": best_epoch,
                   "epochs": 5, "patience": 2,
                   "per_epoch": [{"epoch": i + 1, "val_auc_bsi": v} for i, v in enumerate(per_epoch or [0.46, 0.49, 0.52, 0.56, 0.59])]}
        if variant:
            metrics["variant"] = variant
        if rp_arm is not None:
            metrics["rp_arm"] = rp_arm
        (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (d / "gate_report.json").write_text(json.dumps(
            {"gates": {g: {"verdict": "SKIP" if g == "A3" else "PASS"}
                       for g in ("A1", "A2", "A3", "A4", "A5", "A6")}}), encoding="utf-8")
        if rp_arm is not None:
            prompt = {
                "grad_probe": [{"epoch": 1, "alpha": 0.0, "alpha_grad_norm": 0.1, "generator_grad_norm": 0.0},
                               {"epoch": 5, "alpha": 0.05, "alpha_grad_norm": 0.1, "generator_grad_norm": 0.02}],
                "alpha_final": 0.05,
                "val_stats": {"ratio_mean": [0.03, 0.03, 0.03], "ratio_std": [0.001] * 3,
                              "ratio_max": [0.04] * 3, "delta_norm_mean": [1.0] * 3,
                              "h_norm_mean": [30.0] * 3, "cos_mean": [0.1, 0.2, 0.3], "n_zero_rep": [0, 0, 0]},
                "gate": {"geff_mean": 0.03, "geff_std": 0.001, "geff_min": 0.01, "geff_max": 0.04},
                "dispersion": {"pred_std": 0.004},
                "reference_dispersion": {"val_auc": val_auc, "pred_dispersion": {"pred_std": 0.005}},
                "params": {"new_params_total": 2385, "head_params": 8129},
                "arm": rp_arm,
            }
            (d / "prompt_report.json").write_text(json.dumps(prompt), encoding="utf-8")
        return d

    def _rp_arm(self, base_test, base_val, test_auc, val_auc, classification="VALID_NEGATIVE"):
        return {"classification": classification, "subreason": None,
                "M0": {"pass": True}, "G1": {"pass": True}, "G2": {"pass": True}, "G3": {"pass": True},
                "G4": {"pass": True}, "G5": {"pass": True}, "G6": {"pass": True}, "G7": {"pass": True},
                "G8": {"pass": True},
                "U": {"pass": False, "U1": {"pass": False, "observed": {"auc_test": test_auc, "delta_test": test_auc - base_test, "baseline_auc_test": base_test}},
                      "U2": {"pass": True, "observed": {"auc_val": val_auc, "delta_val": val_auc - base_val, "baseline_auc_val": base_val}}}}

    def test_load_and_pair_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = FS.CANONICAL_SEEDS[2]
            rp = self._rp_arm(0.6, 0.58, 0.605, 0.585)
            self._write_run(root, "base-run", model_seed=seed, stage1_id="s1-x", test_auc=0.6, val_auc=0.58)
            self._write_run(root, "arm-run", model_seed=seed, stage1_id="s1-x", test_auc=0.605, val_auc=0.585,
                            variant="residual-prompt", rp_arm=rp)
            rec = FS.load_record(root, model_seed=seed, baseline_run="base-run", arm_run="arm-run")
            self.assertEqual(rec["model_seed"], seed)
            self.assertAlmostEqual(rec["delta_test"], 0.005, places=15)
            self.assertAlmostEqual(rec["delta_val"], 0.005, places=15)
            self.assertFalse(rec["u1"])
            self.assertTrue(rec["u2"])
            self.assertEqual(rec["secondary"], "POSITIVE_IMPROVEMENT")
            self.assertTrue(rec["mechanism_pass"])
            self.assertTrue(rec["a_class_pass"])
            self.assertAlmostEqual(rec["generator_grad_last_epoch"], 0.02, places=15)
            self.assertEqual(rec["geff"]["geff_std"], 0.001)

    def test_pair_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = FS.CANONICAL_SEEDS[2]
            self._write_run(root, "base-run", model_seed=seed, stage1_id="s1-x", test_auc=0.6, val_auc=0.58)
            bad = self._rp_arm(0.5, 0.58, 0.605, 0.585)   # 记录的 baseline_auc_test 与配对基线不符
            self._write_run(root, "arm-run", model_seed=seed, stage1_id="s1-x", test_auc=0.605, val_auc=0.585,
                            variant="residual-prompt", rp_arm=bad)
            with self.assertRaises(AssertionError):
                FS.load_record(root, model_seed=seed, baseline_run="base-run", arm_run="arm-run")

    def test_stage1_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = FS.CANONICAL_SEEDS[2]
            rp = self._rp_arm(0.6, 0.58, 0.605, 0.585)
            self._write_run(root, "base-run", model_seed=seed, stage1_id="s1-x", test_auc=0.6, val_auc=0.58)
            self._write_run(root, "arm-run", model_seed=seed, stage1_id="s1-OTHER", test_auc=0.605, val_auc=0.585,
                            variant="residual-prompt", rp_arm=rp)
            with self.assertRaises(AssertionError):
                FS.load_record(root, model_seed=seed, baseline_run="base-run", arm_run="arm-run")


class TestCliSurface(unittest.TestCase):
    def test_parser_surface(self):
        parser = FS.build_parser()
        args = parser.parse_args(["--root", "r", "--pairs", "p.json", "--out", "o.json"])
        self.assertEqual(args.root, "r")
        self.assertEqual(args.pairs, "p.json")
        self.assertEqual(args.out, "o.json")


if __name__ == "__main__":
    unittest.main()
