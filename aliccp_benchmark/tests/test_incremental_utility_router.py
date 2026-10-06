"""预注册 §8 最小测试：分割 / 零标签特征 / 配对效用 / 隔离 / 校准 / 路由 / 判定 / tiny 端到端（全 CPU）。

预注册 C1=40e1b9d：docs/superpowers/specs/2026-10-06-aliccp-incremental-utility-verifier-design.md
"""
from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aliccp_benchmark import incremental_utility_router as IUR, protocol

HEADER = "click,purchase,101,121,122,124,125,126,127,128,129,205,206,207,216,508,509,702,853,301"


def write_aliccp_like(path, n: int, seed: int) -> None:
    """合成 tiny AliCCP 格式文件（特征值 0..2 均 < vocab 最小值 3；raw∈{1,2,3} 覆盖 2→0 变换）。"""
    rng = np.random.default_rng(seed)
    lines = [HEADER]
    for _ in range(n):
        click = int(rng.random() < 0.3)
        purchase = int(click and rng.random() < 0.2)
        vals = [int(rng.integers(0, 3)) for _ in range(16)]
        raw = int(rng.choice([1, 2, 3]))
        row = [click, purchase, 0] + vals + [raw]
        lines.append(",".join(str(v) for v in row))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestSplitPartition(unittest.TestCase):
    def test_short_budget_bounds(self):
        b = IUR.split_bounds(2_000_000)
        self.assertEqual(b["A"], (0, 1_400_000))
        self.assertEqual(b["B"], (1_400_000, 1_700_000))
        self.assertEqual(b["C"], (1_700_000, 2_000_000))

    def test_smoke_budget_bounds(self):
        b = IUR.split_bounds(20_000)
        self.assertEqual(b["A"], (0, 14_000))
        self.assertEqual(b["B"], (14_000, 17_000))
        self.assertEqual(b["C"], (17_000, 20_000))

    def test_contiguous_disjoint_union(self):
        b = IUR.split_bounds(123_457)
        rec = IUR.split_records(123_457)
        self.assertEqual(rec[0][1], 0)
        self.assertEqual(rec[-1][2], 123_457)
        for left, right in zip(rec, rec[1:]):
            self.assertEqual(left[2], right[1])

    def test_too_small_raises(self):
        with self.assertRaises(ValueError):
            IUR.split_bounds(3)

    def test_range_sha_deterministic_and_distinct(self):
        self.assertEqual(IUR.range_sha(0, 100), IUR.range_sha(0, 100))
        self.assertNotEqual(IUR.range_sha(0, 100), IUR.range_sha(0, 101))


class TestFeaturesLabelFree(unittest.TestCase):
    def test_signature_has_no_label_param(self):
        params = list(inspect.signature(IUR.router_features).parameters)
        self.assertEqual(params, ["p_b", "p_p", "dnn_input", "gen_rep", "spec_0", "spec_1", "env_emb"])
        self.assertFalse(any(p == "y" or "label" in p.lower() for p in params))

    def test_label_kwarg_rejected(self):
        n = 4
        p = np.full(n, 0.5)
        rep = np.zeros((n, 3))
        with self.assertRaises(TypeError):
            IUR.router_features(p, p, rep, rep, rep, rep, rep, y=np.zeros(n))

    def test_columns_and_shape(self):
        n = 5
        p_b = np.linspace(0.1, 0.9, n)
        p_p = np.linspace(0.9, 0.1, n)
        rep = np.ones((n, 4))
        X = IUR.router_features(p_b, p_p, rep, rep * 2, rep * 3, rep * 4, rep * 5)
        self.assertEqual(X.shape, (n, IUR.FEATURE_DIM))
        np.testing.assert_allclose(X[:, 0], p_b, rtol=0, atol=1e-15)
        np.testing.assert_allclose(X[:, 1], p_p, rtol=0, atol=1e-15)
        np.testing.assert_allclose(X[:, 2], p_p - p_b, rtol=0, atol=1e-15)
        np.testing.assert_allclose(X[:, 3], 2.0, rtol=0, atol=1e-12)   # ||ones(4)|| = 2
        np.testing.assert_allclose(X[:, 7], 10.0, rtol=0, atol=1e-12)  # ||ones*5|| = 10


class TestUtility(unittest.TestCase):
    def test_pairwise_bce_handcalc(self):
        p_b = np.array([0.6])
        p_p = np.array([0.4])
        u_pos = IUR.utility(p_b, p_p, np.array([1]))
        np.testing.assert_allclose(u_pos, [np.log(0.4) - np.log(0.6)], rtol=0, atol=1e-12)
        self.assertLess(u_pos[0], 0.0)
        u_neg = IUR.utility(p_b, p_p, np.array([0]))
        np.testing.assert_allclose(u_neg, [np.log(0.6) - np.log(0.4)], rtol=0, atol=1e-12)
        self.assertGreater(u_neg[0], 0.0)

    def test_beneficial_is_strictly_u_gt_0(self):
        u = np.array([1e-9, 0.0, -1e-9])
        np.testing.assert_array_equal(IUR.beneficial(u), [1, 0, 0])


class TestRidgeAndSelection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = rng.normal(size=(400, 8))
        self.w0 = rng.normal(size=8)
        self.u = self.X @ self.w0 + 0.1 * rng.normal(size=400)

    def test_fit_deterministic_and_recovered(self):
        m1 = IUR.fit_ridge(self.X, self.u)
        m2 = IUR.fit_ridge(self.X, self.u)
        self.assertEqual(m1["w"], m2["w"])
        self.assertEqual(m1["b"], m2["b"])
        corr = np.corrcoef(IUR.ridge_score(m1, self.X), self.u)[0, 1]
        self.assertGreater(corr, 0.95)

    def test_fit_reads_only_B_inputs(self):
        # 固定 B 输入 ⇒ 系数逐位确定（拟合只读 (X, target)）
        m1 = IUR.fit_ridge(self.X, self.u)
        m2 = IUR.fit_ridge(self.X, self.u)
        self.assertEqual(m1["w"], m2["w"])
        self.assertEqual(m1["b"], m2["b"])
        params = list(inspect.signature(IUR.fit_ridge).parameters)
        self.assertEqual(params, ["X", "target", "alpha"])
        self.assertFalse(any(p.endswith("_B") for p in params), params)

    def test_calibration_pure_function_of_C_inputs(self):
        # v3.1 R3 语义：校准只读 C 输入——固定 C 输入下 thr/pi 为纯函数（不随任何 B 侧活动变化）；
        # 改 C 输入会改变 thr（构造性）。注意：B 标签变化经拟合→scores→thr 自然传播属流程级现象，
        # 本测试不宣称也不依赖"流程级 thr 不变"。
        rng = np.random.default_rng(11)
        s = rng.normal(size=300)
        p_b = rng.uniform(0.1, 0.9, 300)
        p_p = rng.uniform(0.1, 0.9, 300)
        y = (rng.random(300) < 0.5).astype(int)
        t1 = IUR.select_threshold(s, p_b, p_p, y)
        _ = IUR.fit_ridge(self.X, self.u * 2.0)          # 任何 B 侧活动不影响"固定 C 输入"的校准结果
        t2 = IUR.select_threshold(s, p_b, p_p, y)
        self.assertEqual(t1["thr"], t2["thr"])
        self.assertEqual(t1["pi_hat"], t2["pi_hat"])
        t3 = IUR.select_threshold(s * 2.0 + 1.0, p_b, p_p, y)
        self.assertNotEqual(t1["thr"], t3["thr"])
        for fn in (IUR.select_threshold, IUR.select_mix_alpha):
            params = list(inspect.signature(fn).parameters)
            self.assertFalse(any(p.endswith("_B") for p in params), (fn.__name__, params))

    def test_threshold_grid_has_21_candidates(self):
        rng = np.random.default_rng(3)
        s = rng.normal(size=100)
        p = rng.uniform(0.1, 0.9, 100)
        y = (rng.random(100) < 0.5).astype(int)
        info = IUR.select_threshold(s, p, p, y)
        self.assertEqual(len(info["grid"]), 21)          # 19 分位 + 2 端点（v3.1）
        self.assertEqual(len(IUR.MIX_GRID), 21)

    def test_threshold_tiebreak_prefers_smaller_pi(self):
        s = np.full(200, 0.5)                     # 所有分位点同值 ⇒ 大量并列
        p_b = np.linspace(0.0, 1.0, 200)
        p_p = np.linspace(1.0, 0.0, 200)
        y = np.tile([0, 1], 100)
        info = IUR.select_threshold(s, p_b, p_p, y)
        self.assertAlmostEqual(info["thr"], 0.5)
        self.assertEqual(info["pi_hat"], 0.0)

    def test_mix_tiebreak_prefers_smaller_alpha(self):
        p = np.linspace(0.0, 1.0, 200)
        y = np.tile([0, 1], 100)
        info = IUR.select_mix_alpha(p, p, y)      # 两臂相同 ⇒ 全部并列
        self.assertEqual(info["alpha_hat"], 0.0)


class TestRoutingAndVerdict(unittest.TestCase):
    def test_route_predict_basic(self):
        s = np.array([0.1, 0.9])
        p_b = np.array([0.2, 0.3])
        p_p = np.array([0.8, 0.7])
        np.testing.assert_allclose(IUR.route_predict(s, p_b, p_p, 0.5), [0.2, 0.7])

    def test_oracle_uses_u_gt_0(self):
        # label-assisted BCE oracle diagnostic（非 AUC 上界；u==0 不算有益 → 选 baseline）
        p_b = np.array([0.2, 0.3])
        p_p = np.array([0.8, 0.7])
        u = np.array([0.01, -0.01])
        np.testing.assert_allclose(IUR.oracle_route(p_b, p_p, u), [0.8, 0.3])
        u0 = np.array([0.0, 0.0])                 # u==0 不算有益 → 选 baseline
        np.testing.assert_allclose(IUR.oracle_route(p_b, p_p, u0), [0.2, 0.3])

    def test_random_mask_reproducible_and_rate(self):
        m1 = IUR.random_route_mask(20_000, 0.3, 12345)
        m2 = IUR.random_route_mask(20_000, 0.3, 12345)
        np.testing.assert_array_equal(m1, m2)
        self.assertLess(abs(m1.mean() - 0.3), 0.02)
        m3 = IUR.random_route_mask(20_000, 0.3, 54321)
        self.assertFalse(np.array_equal(m1, m3))

    def test_classification_boundaries(self):
        self.assertEqual(IUR.classify_delta(0.001), "POSITIVE_IMPROVEMENT")
        self.assertEqual(IUR.classify_delta(0.0009999999), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(IUR.classify_delta(-0.02), "CLEAR_DEGRADATION")
        self.assertEqual(IUR.classify_delta(-0.0199999999), "NO_CLEAR_IMPROVEMENT")

    def test_router_value_truth_table(self):
        self.assertTrue(IUR.router_value_verdict(0.002, 0.001, 0.001)["router_value"])
        self.assertFalse(IUR.router_value_verdict(0.002, -0.001, 0.001)["router_value"])
        self.assertFalse(IUR.router_value_verdict(0.002, 0.001, -0.0001)["router_value"])
        self.assertFalse(IUR.router_value_verdict(0.0005, 0.001, 0.001)["router_value"])
        self.assertEqual(IUR.router_value_verdict(-0.021, 0.001, 0.001)["classification_by_delta_base"],
                         "CLEAR_DEGRADATION")

    def test_expand_eligible_truth_table(self):
        # v3.1：扩 seed = 三比较 router 价值 ∧ 全门禁 ∧ 非 dirty（仅 Δ_base 达标不扩展）
        self.assertTrue(IUR.expand_eligible(True, True, False))
        self.assertFalse(IUR.expand_eligible(True, True, True))
        self.assertFalse(IUR.expand_eligible(True, False, False))
        self.assertFalse(IUR.expand_eligible(False, True, False))


class TestEndToEndTiny(unittest.TestCase):
    """合成 tiny AliCCP → stage1(CPU) → 路由 runner(CPU) → 独立复核脚本，全链一次。

    注：直接调 `run_router(data_files=...)`（同 test_smoke 先例），不走 CLI——
    CLI 的 `protocol.DATA_FILES` 恒为真实数据路径，属正式运行专用。
    """

    @classmethod
    def setUpClass(cls):
        # 延迟导入 bench（其 import 链含 residual_prompt）：
        # 必须晚于 test_residual_prompt 的 run-reference 示例 env 块，保证 RP 常量按既有语义解析。
        from aliccp_benchmark import bench
        import run_incremental_utility_router as RUV

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        data_dir = root / "data"
        data_dir.mkdir()
        files = {}
        for split, n, seed in (("train", 2000, 1), ("val", 200, 2), ("test", 300, 3)):
            path = data_dir / f"tiny.{split}"
            write_aliccp_like(path, n, seed)
            files[split] = str(path)
        cls.data_files = files
        cls.root = root
        cls.meta = bench.run_stage1(
            root=root, data_files=files, budgets={"train": 2000, "val": 200, "test": 300},
            prefix_tag="p2000-v200-t300", model_seed=1688723512, env_seed=20261003,
            epochs=1, patience=1, device=torch.device("cpu"), log=lambda *a, **k: None,
            batch_size=200,
        )
        cls.result = RUV.run_router(
            root=root, stage1_id=cls.meta["stage1_id"], data_files=files,
            budgets={"train": 2000, "val": 200, "test": 300}, prefix_tag="p2000-v200-t300",
            model_seed=1688723512, epochs=2, patience=2, tag="smoke",
            device=torch.device("cpu"), log=lambda *a, **k: None, batch_size=200, enforce_b=False,
        )
        cls.run_dir = Path(cls.result["run_dir"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_full_chain_and_independent_verifier(self):
        import verify_incremental_utility_router as VIUV

        for name in ("predictions.npz", "routing_report.json", "metrics.json", "config.json",
                     "gate_report.json", "arms", "status.json"):
            self.assertTrue((self.run_dir / name).exists(), name)
        status = json.loads((self.run_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "completed")
        rc2 = VIUV.main(["--run-id", self.run_dir.name, "--root", str(self.root)])
        self.assertEqual(rc2, 0)
        summary = (self.root / "SUMMARY.md").read_text(encoding="utf-8")
        self.assertIn(self.run_dir.name, summary)

    def test_arm_budget_parity_recorded(self):
        rep = json.loads((self.run_dir / "routing_report.json").read_text(encoding="utf-8"))
        mtx = json.loads((self.run_dir / "metrics.json").read_text(encoding="utf-8"))
        # 两臂同预算（同一 arm_budget 记录）+ 同数据可见性（A 子集）
        self.assertEqual(rep["arm_budget"]["A"], [0, 1400])
        self.assertEqual(rep["arm_budget"]["epochs"], 2)
        self.assertEqual(rep["router"]["feature_dim"], 8)
        self.assertEqual(rep["diagnostics"]["n_B"], 300)
        self.assertEqual(rep["diagnostics"]["n_C"], 300)
        self.assertEqual(sorted(mtx["aucs_val"].keys()), sorted(mtx["aucs_test"].keys()))
        self.assertIsInstance(mtx["expand_eligible"], bool)


if __name__ == "__main__":
    unittest.main()
