"""A3 同配置复跑判定器测试（CPU-only；合成 run 目录；不触 GPU/数据）。

预注册：docs/superpowers/specs/2026-10-04-aliccp-a3-repeatability-calibration-design.md（§6/§8）。
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from aliccp_benchmark import a3_repeatability as a3


def _make_run(root, run_id, *, test_auc=0.5974422649550507, val_auc=0.5809347091990792,
              backbone="e5e7e6" + "0" * 58, env="5cd198" + "0" * 58, fp="5c060b" + "0" * 58,
              epochs=5, patience=2, per_epoch=None, gate_mean=(0.789178, 0.2108221875),
              newtask=b"newtask-bytes", stage1_id="s1-5c060b9c-m1688723740-e3-4e1b5c6f",
              a_gates_pass=True, config_overrides=None, metric_overrides=None):
    run_dir = Path(root) / "runs" / run_id
    run_dir.mkdir(parents=True)
    config = {
        "run_id": run_id, "tag": "short", "stage1_id": stage1_id,
        "budgets": {"train": 2000000, "val": 500000, "test": 1000000},
        "model_seed": 1688723740, "batch_size": 2000, "lr": 1e-4, "reg_dnn": 7e-6,
        "newtask_rep_dim": 64, "expert_hidden": [128, 64], "tower_hidden": [32, 32],
        "input_size": 80, "embedding_size": 5, "enforce_b": True,
        "commit": "abc1234", "git": {"commit": "abc1234", "dirty": False},
    }
    if config_overrides:
        config.update(config_overrides)
    metrics = {
        "run_id": run_id, "epochs": epochs, "patience": patience,
        "best_val_auc_bsi": val_auc, "test_auc_bsi": test_auc, "gate_mean": list(gate_mean),
        "backbone_sha256_loaded": backbone, "backbone_sha256_before": backbone,
        "backbone_sha256_after": backbone, "env_ids_sha256": env, "fingerprint_sha256": fp,
        "per_epoch": per_epoch if per_epoch is not None else [
            {"epoch": i + 1, "train_loss": 0.1 - i * 0.001, "val_auc_bsi": 0.46 + i * 0.03}
            for i in range(epochs)
        ],
        "wall_seconds": 100.0,
    }
    if metric_overrides:
        metrics.update(metric_overrides)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
    gates = {g: {"verdict": "PASS" if a_gates_pass else "FAIL", "detail": ""}
             for g in ("A1", "A2", "A4", "A5", "A6")}
    gates["A3"] = {"verdict": "SKIP", "detail": "按需复跑"}
    (run_dir / "gate_report.json").write_text(
        json.dumps({"run_id": run_id, "gates": gates, "hard_pass": a_gates_pass}), encoding="utf-8")
    (run_dir / "newtask.pt").write_bytes(newtask)
    return run_dir


class TestLoadRun(unittest.TestCase):
    def test_load_run_reads_all_fields_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run(tmp, "r1", newtask=b"hello")
            loaded = a3.load_run(run_dir)
        self.assertEqual(loaded["config"]["run_id"], "r1")
        self.assertEqual(loaded["metrics"]["epochs"], 5)
        self.assertIn("A1", loaded["gate_report"]["gates"])
        self.assertEqual(loaded["newtask_sha256"], hashlib.sha256(b"hello").hexdigest())

    def test_load_run_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "empty"
            run_dir.mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                a3.load_run(run_dir)

    def test_analyze_requires_at_least_two_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = [a3.load_run(_make_run(tmp, "r1"))]
            with self.assertRaises(ValueError):
                a3.analyze(runs)


class TestVerdict(unittest.TestCase):
    def _load(self, tmp, specs):
        return [a3.load_run(_make_run(tmp, f"r{i}", **spec)) for i, spec in enumerate(specs)]

    def test_three_bit_equal_runs_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{}, {}, {}])
            verdict = a3.a3_verdict(runs)
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["max_abs_delta_test_bsi"], 0.0)
        self.assertTrue(verdict["backbone_sha_equal"])
        self.assertTrue(all(p["within_tol"] for p in verdict["pairs"]))
        self.assertEqual(len(verdict["pairs"]), 3)

    def test_boundary_delta_at_2pow_minus_30_passes(self):
        # 2**-30 ≈ 9.31e-10 < 1e-9，且 0.5 + 2**-30 在 float64 中精确可表示（边界含等于语义的安全构造）
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{"test_auc": 0.5}, {"test_auc": 0.5 + 2 ** -30}])
            verdict = a3.a3_verdict(runs)
        self.assertEqual(verdict["pairs"][0]["abs_delta_test_bsi"], 2 ** -30)
        self.assertTrue(verdict["pass"])

    def test_delta_above_tol_fails(self):
        # 2**-29 ≈ 1.86e-9 > 1e-9
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{"test_auc": 0.5}, {"test_auc": 0.5 + 2 ** -29}])
            verdict = a3.a3_verdict(runs)
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["pairs"][0]["within_tol"])
        self.assertAlmostEqual(verdict["max_abs_delta_test_bsi"], 2 ** -29)

    def test_backbone_mismatch_fails_though_auc_equal(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{}, {"backbone": "ff" * 32}])
            verdict = a3.a3_verdict(runs)
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["backbone_sha_equal"])

    def test_backbone_instability_within_run_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{}, {"metric_overrides": {"backbone_sha256_after": "ff" * 32}}])
            verdict = a3.a3_verdict(runs)
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["backbone_sha_stable_within_run"])


class TestIdentityAndGates(unittest.TestCase):
    def _load(self, tmp, specs):
        return [a3.load_run(_make_run(tmp, f"r{i}", **spec)) for i, spec in enumerate(specs)]

    def test_identity_key_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{}, {"config_overrides": {"stage1_id": "s1-other"}}])
            identity = a3.check_config_identity(runs)
        self.assertFalse(identity["pass"])
        self.assertTrue(any(d["field"] == "config.stage1_id" for d in identity["diffs"]))

    def test_epochs_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{}, {"epochs": 10, "patience": 3}])
            identity = a3.check_config_identity(runs)
        self.assertFalse(identity["pass"])
        fields = {d["field"] for d in identity["diffs"]}
        self.assertIn("metrics.epochs", fields)
        self.assertIn("metrics.patience", fields)

    def test_provenance_fields_do_not_break_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [
                {},
                {"metric_overrides": {"wall_seconds": 123.4}, "config_overrides": {"spec_attenuation": 1.0}},
                {"config_overrides": {"commit": "deadbee"}},
            ])
            identity = a3.check_config_identity(runs)
        self.assertTrue(identity["pass"])
        self.assertEqual(identity["diffs"], [])

    def test_analyze_invalid_on_a_gate_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{}, {"a_gates_pass": False}])
            report = a3.analyze(runs)
        self.assertEqual(report["status"], "INVALID")

    def test_analyze_invalid_on_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{}, {"config_overrides": {"model_seed": 1}}])
            report = a3.analyze(runs)
        self.assertEqual(report["status"], "INVALID")


class TestAnalyzeSupplementary(unittest.TestCase):
    def _load(self, tmp, specs):
        return [a3.load_run(_make_run(tmp, f"r{i}", **spec)) for i, spec in enumerate(specs)]

    def test_pass_report_supplementary_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{}, {}, {}])
            report = a3.analyze(runs)
        self.assertEqual(report["status"], "A3_PASS")
        sup = report["supplementary"]
        self.assertTrue(sup["trajectories_bit_equal"])
        self.assertTrue(sup["checkpoint_bit_equal"])
        self.assertTrue(sup["gate_mean_bit_equal"])
        self.assertEqual(sup["max_abs_delta_val_bsi"], 0.0)
        self.assertEqual(sup["max_abs_delta_gate_mean"], 0.0)
        self.assertEqual(sup["newtask_sha256"][0], sup["newtask_sha256"][1])
        self.assertEqual(len(sup["wall_seconds"]), 3)

    def test_trajectory_difference_is_supplementary_not_verdict(self):
        # 轨迹差异（补充指标）不影响 A3 判定（Δ 与 backbone 仍达标）——预注册 §6.3 语义
        alt = [{"epoch": i + 1, "train_loss": 0.2, "val_auc_bsi": 0.99} for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{}, {"per_epoch": alt}])
            report = a3.analyze(runs)
        self.assertEqual(report["status"], "A3_PASS")
        self.assertFalse(report["supplementary"]["trajectories_bit_equal"])

    def test_checkpoint_difference_is_supplementary_not_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._load(tmp, [{}, {"newtask": b"other-bytes"}])
            report = a3.analyze(runs)
        self.assertEqual(report["status"], "A3_PASS")
        self.assertFalse(report["supplementary"]["checkpoint_bit_equal"])


class TestCli(unittest.TestCase):
    def test_main_exit_codes_and_out_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_run(tmp, "ref")
            _make_run(tmp, "r1")
            out = Path(tmp) / "report.json"
            code = a3.main(["--root", tmp, "--runs", "ref", "r1", "--out", str(out)])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "A3_PASS")

    def test_main_exit_code_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_run(tmp, "ref")
            _make_run(tmp, "r1", test_auc=0.5 + 2 ** -29)
            self.assertEqual(a3.main(["--root", tmp, "--runs", "ref", "r1"]), 1)

    def test_main_exit_code_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_run(tmp, "ref")
            _make_run(tmp, "r1", a_gates_pass=False)
            self.assertEqual(a3.main(["--root", tmp, "--runs", "ref", "r1"]), 2)


if __name__ == "__main__":
    unittest.main()
