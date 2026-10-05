"""TDD 测试：配对判定工具（本次任务新增；见预注册 2026-10-06-…-seed-recheck-design.md §4.2/4.3）。

CPU、秒级、不读真实数据集；run 目录用最小 fixture 构造（只含工具读取的字段）。
"""
import json
import tempfile
import unittest
from pathlib import Path

from census_benchmark import paired_verdict as PV


def make_run(root: Path, name: str, *, variant, auc_test, best_val, best_epoch=5, epochs=5,
             stage1_id="s1-096f8f16-m1685463909-e2-8b53a3bf", gates=None, b3_gate_mean=None,
             rp_arm=None) -> Path:
    """写最小 run 目录：metrics.json + gate_report.json（只含工具读取的键）。"""
    run = Path(root) / name
    run.mkdir(parents=True)
    payload = {
        "run_id": name, "commit": "abc1234", "variant": variant, "stage1_id": stage1_id,
        "stage2": {"epoch_records": [{"epoch": i + 1, "loss": 0.0, "auc_val_education": best_val}
                                     for i in range(epochs)],
                   "best_epoch": best_epoch, "best_val_auc": best_val, "test_auc": auc_test},
    }
    if rp_arm is not None:
        payload["rp_arm"] = rp_arm
    (run / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    gate_report = {key: {"pass": (gates or {}).get(key, True)} for key in
                   ("A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4")}
    gate_report["B3"]["detail"] = {"gate_mean": b3_gate_mean if b3_gate_mean is not None
                                   else [0.93, 0.90]}
    (run / "gate_report.json").write_text(json.dumps(gate_report), encoding="utf-8")
    return run


def ok_rp_arm(classification="VALID_NEGATIVE", e1=False):
    return {"E1": {"pass": e1}, **{f"G{i}": {"pass": True} for i in range(1, 6)},
            "classification": classification, "pass": classification == "CONFIRMED"}


class TestClassifyDelta(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(PV.classify_delta(0.001), PV.POSITIVE_IMPROVEMENT)          # 含端点
        self.assertEqual(PV.classify_delta(0.0009999), PV.NO_CLEAR_IMPROVEMENT)
        self.assertEqual(PV.classify_delta(0.0), PV.NO_CLEAR_IMPROVEMENT)
        self.assertEqual(PV.classify_delta(-0.019999), PV.NO_CLEAR_IMPROVEMENT)
        self.assertEqual(PV.classify_delta(-0.02), PV.CLEAR_DEGRADATION)             # 含端点
        self.assertEqual(PV.classify_delta(-0.0001), PV.NO_CLEAR_IMPROVEMENT)
        self.assertEqual(PV.classify_delta(0.05), PV.POSITIVE_IMPROVEMENT)

    def test_prereg_constants_pinned(self):
        self.assertEqual(PV.DELTA_POSITIVE, 0.001)
        self.assertEqual(PV.DELTA_NEGATIVE, -0.02)
        self.assertEqual(PV.VAL_CONTRADICTION, -0.001)


class TestPairedReport(unittest.TestCase):
    def _pair(self, td, *, auc_b=0.8500, auc_t=0.8515, val_b=0.8528, val_t=0.8530,
              rp_arm=None, gates_b=None, gates_t=None, b3_b=None, b3_t=None,
              stage1_t="s1-096f8f16-m1685463909-e2-8b53a3bf"):
        base = make_run(td, "run-b", variant="baseline", auc_test=auc_b, best_val=val_b,
                        gates=gates_b, b3_gate_mean=b3_b)
        treat = make_run(td, "run-t", variant="residual-prompt", auc_test=auc_t, best_val=val_t,
                         gates=gates_t, b3_gate_mean=b3_t, rp_arm=rp_arm or ok_rp_arm(),
                         stage1_id=stage1_t)
        return base, treat

    def test_deltas_and_classification(self):
        with tempfile.TemporaryDirectory() as td:
            base, treat = self._pair(td, auc_b=0.8500, auc_t=0.8515, val_b=0.8528, val_t=0.8540)
            rep = PV.paired_report(base, treat)
        self.assertAlmostEqual(rep["paired"]["delta_test"], 0.0015, places=12)
        self.assertAlmostEqual(rep["paired"]["delta_val_best"], 0.0012, places=12)
        self.assertAlmostEqual(rep["paired"]["delta_test"], rep["treatment"]["auc_test"]
                               - rep["baseline"]["auc_test"], places=15)
        self.assertEqual(rep["paired"]["classification_new"], PV.POSITIVE_IMPROVEMENT)
        self.assertEqual(rep["decision"], "EXPAND")
        self.assertEqual(rep["contradictions"], [])

    def test_historical_verdict_reported_separately(self):
        with tempfile.TemporaryDirectory() as td:
            base, treat = self._pair(td, rp_arm=ok_rp_arm(classification="VALID_NEGATIVE", e1=False))
            rep = PV.paired_report(base, treat)
        self.assertEqual(rep["paired"]["historical_classification"], "VALID_NEGATIVE")
        self.assertFalse(rep["paired"]["historical_e1_pass"])
        self.assertTrue(rep["paired"]["historical_g1_g5_all_pass"])
        self.assertIn("classification_new", rep["paired"])       # 两者并列，互不覆盖

    def test_contradiction_a_mechanism_not_all_pass(self):
        arm = ok_rp_arm(classification="MECHANISM_SILENT")
        arm["G5"]["pass"] = False
        with tempfile.TemporaryDirectory() as td:
            base, treat = self._pair(td, rp_arm=arm)
            rep = PV.paired_report(base, treat)
        self.assertEqual(rep["decision"], "STOP")                # 即使 delta 为正
        self.assertTrue(any(c.startswith("(a)") for c in rep["contradictions"]))

    def test_contradiction_b_val_moves_against(self):
        with tempfile.TemporaryDirectory() as td:
            base, treat = self._pair(td, auc_b=0.8500, auc_t=0.8515, val_b=0.8528, val_t=0.8510)
            rep = PV.paired_report(base, treat)
        self.assertEqual(rep["paired"]["classification_new"], PV.POSITIVE_IMPROVEMENT)
        self.assertEqual(rep["decision"], "STOP")
        self.assertTrue(any(c.startswith("(b)") for c in rep["contradictions"]))

    def test_contradiction_c_gate_fail_and_b3_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            base, treat = self._pair(td, gates_t={"B2": False})
            rep = PV.paired_report(base, treat)
        self.assertEqual(rep["decision"], "STOP")
        self.assertTrue(any(c.startswith("(c)") for c in rep["contradictions"]))
        with tempfile.TemporaryDirectory() as td:
            base, treat = self._pair(td, b3_b=[0.93, 0.90], b3_t=[0.94, 0.90])
            rep = PV.paired_report(base, treat)
        self.assertEqual(rep["decision"], "STOP")
        self.assertTrue(any("B3" in c and c.startswith("(c)") for c in rep["contradictions"]))

    def test_no_clear_and_clear_degradation_decisions(self):
        with tempfile.TemporaryDirectory() as td:                # NO_CLEAR → STOP
            base, treat = self._pair(td, auc_b=0.8500, auc_t=0.8504)
            rep = PV.paired_report(base, treat)
        self.assertEqual(rep["paired"]["classification_new"], PV.NO_CLEAR_IMPROVEMENT)
        self.assertEqual(rep["decision"], "STOP")
        with tempfile.TemporaryDirectory() as td:                # CLEAR_DEGRADATION → STOP
            base, treat = self._pair(td, auc_b=0.8500, auc_t=0.8250)
            rep = PV.paired_report(base, treat)
        self.assertEqual(rep["paired"]["classification_new"], PV.CLEAR_DEGRADATION)
        self.assertEqual(rep["decision"], "STOP")

    def test_stage1_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            base, treat = self._pair(td, stage1_t="s1-other")
            with self.assertRaises(ValueError):
                PV.paired_report(base, treat)

    def test_missing_rp_arm_raises(self):
        with tempfile.TemporaryDirectory() as td:
            base = make_run(td, "run-b", variant="baseline", auc_test=0.85, best_val=0.85)
            treat = make_run(td, "run-t", variant="residual-prompt", auc_test=0.85, best_val=0.85,
                             rp_arm=None)
            # make_run 只在 rp_arm 非 None 时写入；此处手动制造"处理臂缺 rp_arm"
            payload = json.loads((treat / "metrics.json").read_text(encoding="utf-8"))
            payload.pop("rp_arm", None)
            (treat / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                PV.paired_report(base, treat)

    def test_budget_and_epoch_fields_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            base, treat = self._pair(td)
            rep = PV.paired_report(base, treat)
        self.assertEqual(rep["baseline"]["best_epoch"], 5)
        self.assertEqual(rep["baseline"]["epochs_run"], 5)
        self.assertEqual(rep["treatment"]["budget_capped"], True)   # best_epoch == epochs_run

    def test_cli_writes_json_report(self):
        with tempfile.TemporaryDirectory() as td:
            base, treat = self._pair(td)
            out = Path(td) / "paired.json"
            rc = PV.main(["--baseline-run", str(base), "--treatment-run", str(treat),
                          "--out", str(out)])
            self.assertEqual(rc, 0)
            rep = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rep["decision"], "EXPAND")


if __name__ == "__main__":
    unittest.main()
