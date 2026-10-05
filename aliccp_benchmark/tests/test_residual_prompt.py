"""TDD 测试：AliCCP 阶段 2 范数受控残差 Prompt（可学习门控）——20-epoch 持久性检验（终止判定点）。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-20epoch-design.md
CPU、秒级、不读真实数据集（tiny 端到端沿用 test_smoke 夹具口径）。

移植自 `013e105:aliccp_benchmark/tests/test_residual_prompt.py`（seed-2 复现分支适配版；其祖本
`79ddefa:aliccp_benchmark/tests/test_residual_prompt.py`，git blob
`6484f5b8c9393ce22651d0f9161d81a0ef383f17`）；相对 `013e105` 钉死版恰 4 处分支适配（预注册 §6-A3‴：
docstring / DOC_PATH / WHITELIST / TestPreregConstants 一个方法），由守卫测试
`test_residual_prompt_20epoch.py` 以"钉死文本 + 恰好替换"重建等式钉死。
"""
import ast
import copy
import hashlib
import inspect
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aliccp_benchmark import bench
from aliccp_benchmark import metrics as M
from aliccp_benchmark import protocol as P
from aliccp_benchmark import residual_prompt as RP
from multitaskrec.model import NewTask

REPO = Path(__file__).resolve().parents[2]          # 测试依赖 cwd = 仓库根

PIN_BLOB = "107221b26382da7fd44990e167d9a61da2680d92"
PIN_SHA256_LF = "8139e067a57fd3e144846ca5d39015a1d51657d55da67b9da9e4e06c007658df"
PIN_TEST_BLOB = "847097b7cb48a56c8593dd76d63a71a21a4c9e4f"
FROZEN_BASE = "8133d32"
DOC_PATH = "docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-20epoch-design.md"
WHITELIST = {
    DOC_PATH,
    "aliccp_benchmark/residual_prompt.py",
    "aliccp_benchmark/tests/test_residual_prompt.py",
    "aliccp_benchmark/tests/test_residual_prompt_20epoch.py",
    "aliccp_benchmark/rp_20epoch.py",
    "run_aliccp_benchmark.py",
    "aliccp_benchmark/bench.py",
    "verify_seed2_artifact_integrity.py",
    "verify_rp_20epoch.py",
    "artifacts/aliccp_bench/SUMMARY.md",
}


def make_variant(input_size=8, rep_dim=4, hidden=RP.PROMPT_HIDDEN, seed=0):
    """同一 seed 下构造 (variant, RNG 现场)：参照从构造前 RNG 现场出发。"""
    torch.manual_seed(seed)
    rng_before = torch.get_rng_state()
    variant = RP.ResidualPromptNewTask(input_size=input_size, rep_dim=rep_dim,
                                       tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN,
                                       device=torch.device("cpu"), hidden=hidden)
    rng_after = torch.get_rng_state()
    return variant, rng_before, rng_after


def build_ref(rng_before, input_size=8, rep_dim=4):
    """在隔离 RNG 内、从给定现场构造基线 NewTask（供逐位比较）。"""
    saved = torch.get_rng_state()
    try:
        torch.set_rng_state(rng_before)
        return NewTask(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=(4, 2),
                       reg_dnn=P.REG_DNN, device=torch.device("cpu"))
    finally:
        torch.set_rng_state(saved)


def random_inputs(b=6, input_size=8, rep_dim=4, seed=1, norms=(1.0, 10.0, 100.0)):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(b, input_size, generator=gen)
    reps = [torch.randn(b, rep_dim, generator=gen) for _ in norms]
    reps = [r / r.norm(dim=-1, keepdim=True) * n for r, n in zip(reps, norms)]
    return x, reps


class TestConstructionIdentity(unittest.TestCase):
    def test_build_newtask_types_and_unknown_variant(self):
        base = RP.build_newtask(RP.BASELINE_VARIANT, input_size=8, rep_dim=4,
                                tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN, device=torch.device("cpu"))
        self.assertIs(type(base), NewTask)
        var = RP.build_newtask(RP.VARIANT, input_size=8, rep_dim=4,
                               tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN, device=torch.device("cpu"))
        self.assertIsInstance(var, RP.ResidualPromptNewTask)
        self.assertIsInstance(var, NewTask)
        with self.assertRaises(ValueError):
            RP.build_newtask("no-such-variant", input_size=8, rep_dim=4,
                             tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN, device=torch.device("cpu"))

    def test_shared_params_bit_identical_extra_keys_and_rng_endpoint(self):
        variant, rng_before, rng_after = make_variant()
        ref = build_ref(rng_before)
        ref_sd, var_sd = ref.state_dict(), variant.state_dict()
        for key in ref_sd:                                       # 共享参数 + buffer 逐位一致
            self.assertTrue(torch.equal(ref_sd[key], var_sd[key]), key)
        self.assertEqual(sorted(set(var_sd) - set(ref_sd)), sorted(RP.EXTRA_PARAM_NAMES))
        saved = torch.get_rng_state()
        try:
            torch.set_rng_state(rng_before)
            NewTask(input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                    reg_dnn=P.REG_DNN, device=torch.device("cpu"))
            endpoint = torch.get_rng_state()
        finally:
            torch.set_rng_state(saved)
        self.assertTrue(torch.equal(endpoint, rng_after))

    def test_alpha_is_exactly_zero_at_construction(self):
        variant, _, _ = make_variant()
        self.assertEqual(float(variant.prompt_gate.detach()), 0.0)

    def test_prompt_audit_construction_report(self):
        variant, rng_before, rng_after = make_variant()
        audit = RP.PromptAudit(variant, rng_before=rng_before, rng_after=rng_after,
                               input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                               reg_dnn=P.REG_DNN, device=torch.device("cpu"))
        c = audit.construction
        self.assertTrue(c["shared_params_bit_identical"])
        self.assertTrue(c["global_rng_endpoint_identical"])
        self.assertEqual(c["extra_keys"], sorted(RP.EXTRA_PARAM_NAMES))
        self.assertEqual(c["alpha_at_construction"], 0.0)


class TestInitForwardIdentity(unittest.TestCase):
    def test_forward_bit_identical_to_baseline_at_alpha_zero(self):
        variant, rng_before, _ = make_variant()
        ref = build_ref(rng_before)
        x, (h0, h1, h2) = random_inputs()
        env = [torch.randn(4, generator=torch.Generator().manual_seed(7)) for _ in range(2)]
        with torch.no_grad():
            out_v = variant(x, h0, [h1, h2], env)
            out_b = ref(x, h0, [h1, h2], env)
        self.assertTrue(torch.equal(out_v, out_b))

    def test_nonzero_alpha_changes_output(self):
        variant, rng_before, _ = make_variant()
        ref = build_ref(rng_before)
        x, (h0, h1, h2) = random_inputs()
        env = [torch.randn(4, generator=torch.Generator().manual_seed(7)) for _ in range(2)]
        with torch.no_grad():
            out_b = ref(x, h0, [h1, h2], env)
            variant.prompt_gate.fill_(0.1)
            out_v = variant(x, h0, [h1, h2], env)
        self.assertFalse(torch.equal(out_v, out_b))


class TestNormControl(unittest.TestCase):
    def test_ratio_bounded_by_alpha_and_equal_across_streams(self):
        variant, _, _ = make_variant()
        variant.prompt_gate.data.fill_(0.3)
        x, reps = random_inputs(norms=(0.5, 5.0, 50.0))
        deltas, m = variant.prompt_deltas(x, reps)
        for h, d in zip(reps, deltas):
            ratio = d.norm(dim=-1) / h.norm(dim=-1)
            self.assertLessEqual(float(ratio.max()), 0.3 + RP.BOUND_TOL)
        r = [float((d.norm(dim=-1) / h.norm(dim=-1))[0]) for h, d in zip(reps, deltas)]
        self.assertAlmostEqual(r[0], r[1], places=5)
        self.assertAlmostEqual(r[1], r[2], places=5)
        g_eff = RP.effective_gate(m, variant.prompt_gate.detach())
        self.assertLessEqual(float(g_eff.max()), 0.3 + RP.BOUND_TOL)

    def test_prompt_is_bounded_pointwise(self):
        variant, _, _ = make_variant()
        x, _ = random_inputs()
        _, m = variant.prompt_deltas(x, [torch.zeros(6, 4)])
        self.assertLessEqual(float(m.abs().max()), 1.0)

    def test_zero_norm_stream_gives_zero_delta_and_zero_ratio(self):
        variant, _, _ = make_variant()
        variant.prompt_gate.data.fill_(0.5)
        x, _ = random_inputs()
        h = torch.zeros(6, 4)
        deltas, _ = variant.prompt_deltas(x, [h])
        self.assertTrue(torch.equal(deltas[0], torch.zeros(6, 4)))
        stats = RP.PromptStats()
        g_eff = RP.effective_gate(variant.prompt_deltas(x, [h])[1], variant.prompt_gate.detach())
        stats.update([deltas[0]] * 3, [h] * 3, g_eff)
        out = stats.result()
        self.assertEqual(out["streams"]["ratio_mean"][0], 0.0)
        self.assertEqual(out["streams"]["n_zero_rep"][0], 6)

    def test_formula_matches_hand_computation(self):
        variant = RP.ResidualPromptNewTask(input_size=4, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                                           reg_dnn=P.REG_DNN, device=torch.device("cpu"), hidden=2)
        with torch.no_grad():   # 手工权重：preact = x 的前两维经 W2 扩到 4 维
            variant.prompt_generator[0].weight.copy_(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
            variant.prompt_generator[0].bias.zero_()
            variant.prompt_generator[2].weight.copy_(torch.tensor(
                [[1.0, 0], [0, 1.0], [1.0, 1.0], [0, 0]]))
            variant.prompt_generator[2].bias.zero_()
            variant.prompt_gate.data.fill_(0.25)
        x = torch.tensor([[0.5, -0.5, 9.0, -9.0]])
        h = torch.tensor([[3.0, 4.0, 0.0, 0.0]])                 # ||h|| = 5
        preact = torch.tensor([[0.5, 0.0, 0.5, 0.0]])            # ReLU([0.5, -0.5]) = [0.5, 0]
        m_expected = torch.tanh(preact)
        delta_expected = 0.25 * (5.0 / math.sqrt(4)) * m_expected
        deltas, m = variant.prompt_deltas(x, [h])
        self.assertTrue(torch.allclose(m, m_expected, atol=1e-6))
        self.assertTrue(torch.allclose(deltas[0], delta_expected, atol=1e-6))

    def test_dimension_mismatch_refuses_to_emit_numbers(self):
        variant, _, _ = make_variant(input_size=8, rep_dim=4)
        x, _ = random_inputs(rep_dim=4)
        with self.assertRaises(ValueError):
            variant.prompt_deltas(x, [torch.zeros(6, 5)])        # 维度不一致 → 拒绝出数


class TestGateGradients(unittest.TestCase):
    def test_first_backward_alpha_grad_nonzero_generator_grad_zero(self):
        variant, _, _ = make_variant()
        x, (h0, h1, h2) = random_inputs()
        env = [torch.zeros(4), torch.zeros(4)]
        pred = variant(x, h0, [h1, h2], env)
        loss = pred.sum()
        loss.backward()
        self.assertIsNotNone(variant.prompt_gate.grad)
        self.assertNotEqual(float(variant.prompt_gate.grad), 0.0)   # 首步 α 即有非零梯度
        gen_sq = sum(float(p.grad.pow(2).sum()) for p in variant.prompt_generator.parameters()
                     if p.grad is not None)
        self.assertEqual(gen_sq, 0.0)                               # α=0 的数学后果（预期）

    def test_generator_gradients_recover_after_one_step(self):
        variant, _, _ = make_variant()
        x, (h0, h1, h2) = random_inputs()
        env = [torch.zeros(4), torch.zeros(4)]
        opt = torch.optim.Adam(variant.parameters(), lr=1e-3)
        opt.zero_grad()
        variant(x, h0, [h1, h2], env).sum().backward()
        opt.step()
        opt.zero_grad()
        variant(x, h0, [h1, h2], env).sum().backward()
        gen_sq = sum(float(p.grad.pow(2).sum()) for p in variant.prompt_generator.parameters()
                     if p.grad is not None)
        self.assertGreater(gen_sq, 0.0)


class TestPromptStats(unittest.TestCase):
    def test_stats_match_hand_computation(self):
        stats = RP.PromptStats()
        h = torch.tensor([[3.0, 4.0]])                           # ||h|| = 5
        d = torch.tensor([[0.0, 2.5]])                           # ||d|| = 2.5 → ratio 0.5
        g_eff = torch.tensor([0.4], dtype=torch.float64)
        stats.update([d, d, d], [h, h, h], g_eff)                # 三路同值：断言只看第 0 路
        out = stats.result()
        self.assertAlmostEqual(out["streams"]["ratio_mean"][0], 0.5, places=6)
        self.assertAlmostEqual(out["streams"]["ratio_max"][0], 0.5, places=6)
        self.assertAlmostEqual(out["streams"]["cos_mean"][0], 2.5 * 4.0 / (2.5 * 5.0), places=6)
        self.assertAlmostEqual(out["streams"]["h_norm_mean"][0], 5.0, places=6)
        self.assertAlmostEqual(out["streams"]["delta_norm_mean"][0], 2.5, places=6)
        self.assertAlmostEqual(out["gate"]["geff_mean"], 0.4, places=6)
        self.assertEqual(out["gate"]["geff_min"], 0.4)
        self.assertEqual(out["gate"]["geff_max"], 0.4)
        self.assertEqual(out["count"], 1)

    def test_ratio_std_is_population_std(self):
        stats = RP.PromptStats()
        h = torch.tensor([[3.0, 4.0], [3.0, 4.0]])
        d = torch.tensor([[1.5, 2.0], [0.0, 0.0]])               # ratios 0.5, 0.0
        stats.update([d, d, d], [h, h, h], torch.tensor([0.25, 0.25]))
        out = stats.result()
        self.assertAlmostEqual(out["streams"]["ratio_mean"][0], 0.25, places=6)
        self.assertAlmostEqual(out["streams"]["ratio_std"][0], 0.25, places=6)

    def test_stream_order_is_gen_then_specs(self):
        stats = RP.PromptStats()
        self.assertEqual(stats.stream_names, ("gen", "spec_0", "spec_1"))

    def test_stream_count_mismatch_raises(self):
        stats = RP.PromptStats()
        h = torch.zeros(2, 4)
        with self.assertRaises(ValueError):
            stats.update([h, h], [h, h], torch.zeros(2))         # 少一路 → 拒绝


class TestReferenceHead(unittest.TestCase):
    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            RP.reference_head_stats(Path("no/such/newtask.pt"), backbone=None, loader=None,
                                    device=torch.device("cpu"), input_size=8, rep_dim=4,
                                    tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN)

    def test_incompatible_state_dict_raises_value_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "newtask.pt"
            torch.save({"bogus.key": torch.zeros(1)}, path)
            with self.assertRaises(ValueError):
                RP.reference_head_stats(path, backbone=None, loader=None, device=torch.device("cpu"),
                                        input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                                        reg_dnn=P.REG_DNN)

    def test_module_uses_aliccp_metrics(self):
        self.assertIs(RP.metrics, M)                             # 适配 A1/A2：指标模块为 aliccp_benchmark.metrics
        self.assertTrue(callable(RP.metrics.auc_score))


class TestArmVerdict(unittest.TestCase):
    def _probe(self, *, alpha_final=0.1, first_alpha_grad=1.0, last_gen_grad=0.5, last_alpha=0.1,
               bit=True, endpoint=True, init_bit=True, extra=None, alpha0=0.0):
        return {"construction": {"shared_params_bit_identical": bit,
                                 "global_rng_endpoint_identical": endpoint,
                                 "extra_keys": extra if extra is not None else sorted(RP.EXTRA_PARAM_NAMES),
                                 "alpha_at_construction": alpha0},
                "init_forward": {"bit_identical": init_bit},
                "grad_probe": [{"epoch": 1, "alpha": 0.0, "alpha_grad_norm": first_alpha_grad,
                                "generator_grad_norm": 0.0},
                               {"epoch": 2, "alpha": last_alpha, "alpha_grad_norm": 1.0,
                                "generator_grad_norm": last_gen_grad}],
                "alpha_final": alpha_final}

    def _stats(self, ratios=(0.05, 0.05, 0.05), alpha=0.1, geff_std=0.02, geff_max=0.05,
               geff_min=0.01, pred_std=0.003):
        return {"streams": {"ratio_mean": list(ratios), "ratio_max": [min(r, alpha) for r in ratios]},
                "gate": {"geff_mean": 0.03, "geff_std": geff_std, "geff_max": geff_max, "geff_min": geff_min},
                "dispersion": {"pred_std": pred_std}}

    def _params(self, new_total=RP.EXPECTED_NEW_PARAMS_TOTAL, head=RP.EXPECTED_HEAD_PARAMS, names=None):
        return {"new_param_list": [{"name": n} for n in (names if names is not None else sorted(RP.EXTRA_PARAM_NAMES))],
                "new_params_total": new_total, "head_params": head}

    def _reference(self, val_auc=RP.BASELINE_AUC_VAL, pred_std=RP.REFERENCE_PRED_STD):
        return {"val_auc": val_auc, "pred_dispersion": {"pred_std": pred_std}}

    def _verdict(self, *, auc_test=RP.BASELINE_AUC_TEST + 0.006, auc_val=RP.BASELINE_AUC_VAL + 0.001,
                 probe=None, val_stats=None, params=None, reference=None, protocol_ok=True):
        return RP.arm_verdict(auc_test=auc_test, auc_val=auc_val,
                              probe=probe if probe is not None else self._probe(),
                              val_stats=val_stats if val_stats is not None else self._stats(),
                              params=params if params is not None else self._params(),
                              reference=reference if reference is not None else self._reference(),
                              protocol_ok=protocol_ok)

    def test_u1_threshold_boundary(self):
        arm = self._verdict(auc_test=RP.BASELINE_AUC_TEST + RP.AUC_TEST_DELTA_MIN + 1e-9)
        self.assertTrue(arm["U"]["U1"]["pass"])
        arm = self._verdict(auc_test=RP.BASELINE_AUC_TEST + RP.AUC_TEST_DELTA_MIN - 1e-9)
        self.assertFalse(arm["U"]["U1"]["pass"])
        # Δ 的口径：auc_test − BASELINE_AUC_TEST（观测值逐位钉死）
        auc = RP.BASELINE_AUC_TEST + RP.AUC_TEST_DELTA_MIN + 1e-9
        self.assertEqual(self._verdict(auc_test=auc)["U"]["U1"]["observed"]["delta_test"],
                         auc - RP.BASELINE_AUC_TEST)

    def test_u2_requires_strict_positive_direction(self):
        arm = self._verdict(auc_val=RP.BASELINE_AUC_VAL)
        self.assertFalse(arm["U"]["U2"]["pass"])                 # == 基线 ⇒ FAIL（严格 >0）
        arm = self._verdict(auc_val=RP.BASELINE_AUC_VAL + 1e-9)
        self.assertTrue(arm["U"]["U2"]["pass"])
        self.assertFalse(self._verdict(auc_val=RP.BASELINE_AUC_VAL - 0.01)["U"]["pass"])

    def test_band_boundaries_are_inclusive(self):
        arm = self._verdict(val_stats=self._stats(ratios=(0.005, 0.5, 0.05)))
        self.assertTrue(arm["G5"]["pass"])
        arm = self._verdict(val_stats=self._stats(ratios=(0.0049, 0.05, 0.05)))
        self.assertFalse(arm["G5"]["pass"])

    def test_classification_matrix(self):
        cases = [
            (dict(), "VALID_POSITIVE", None),
            (dict(auc_test=RP.BASELINE_AUC_TEST + 0.001), "VALID_NEGATIVE", None),
            (dict(probe=self._probe(last_gen_grad=0.0)), "MECHANISM_FAIL", "MECHANISM_INACTIVE"),
            (dict(probe=self._probe(first_alpha_grad=0.0)), "MECHANISM_FAIL", "MECHANISM_INACTIVE"),
            (dict(probe=self._probe(alpha_final=0.0, last_alpha=0.0),
                  val_stats=self._stats(ratios=(0.0, 0.0, 0.0), alpha=0.0, geff_std=0.0, geff_max=0.0, geff_min=0.0)),
             "MECHANISM_FAIL", "MECHANISM_INACTIVE"),
            (dict(val_stats=self._stats(ratios=(0.004, 0.05, 0.05))), "MECHANISM_FAIL", "MECHANISM_SILENT"),
            (dict(val_stats=self._stats(ratios=(0.6, 0.05, 0.05))), "MECHANISM_FAIL", "MECHANISM_OVER_PERTURB"),
            (dict(probe=self._probe(bit=False)), "MECHANISM_FAIL", "INVALID_IMPLEMENTATION"),
            (dict(probe=self._probe(init_bit=False)), "MECHANISM_FAIL", "INVALID_IMPLEMENTATION"),
            (dict(probe=self._probe(extra=["prompt_gate"])), "MECHANISM_FAIL", "INVALID_IMPLEMENTATION"),
            (dict(val_stats=self._stats(geff_std=0.0)), "MECHANISM_FAIL", "GATE_DEGENERATE"),
            (dict(val_stats=self._stats(pred_std=RP.REFERENCE_PRED_STD * 0.5 - 1e-9)),
             "MECHANISM_FAIL", "PREDICTION_COLLAPSE"),
            (dict(reference=self._reference(val_auc=0.5)), "MECHANISM_FAIL", "REFERENCE_IDENTITY"),
            (dict(params=self._params(new_total=2384)), "MECHANISM_FAIL", "INVALID_IMPLEMENTATION"),
            (dict(protocol_ok=False), "MECHANISM_FAIL", "PROTOCOL_INVALID"),
        ]
        for kwargs, expected, subreason in cases:
            arm = self._verdict(**kwargs)
            self.assertEqual(arm["classification"], expected, kwargs)
            self.assertEqual(arm["subreason"], subreason, kwargs)
        self.assertTrue(self._verdict()["pass"])
        self.assertFalse(self._verdict(auc_test=RP.BASELINE_AUC_TEST + 0.001)["pass"])
        # 参照缺失 ⇒ M0 FAIL（不允许静默绕过）
        arm = self._verdict(reference={})
        self.assertFalse(arm["M0"]["pass"])
        self.assertEqual(arm["subreason"], "REFERENCE_IDENTITY")

    def test_g7_requires_relative_no_collapse(self):
        # 与参照同量级 ⇒ PASS（BSI 预测天然集中，绝对下限不可用）
        arm = self._verdict(val_stats=self._stats(pred_std=RP.REFERENCE_PRED_STD * 0.5))
        self.assertTrue(arm["G7"]["pass"])
        arm = self._verdict(val_stats=self._stats(pred_std=0.0))
        self.assertFalse(arm["G7"]["pass"])

    def test_bound_violation_is_invalid(self):
        stats = self._stats()
        stats["streams"]["ratio_max"] = [0.05, 0.3, 0.05]        # 0.3 > |α|=0.1
        arm = self._verdict(probe=self._probe(alpha_final=0.1), val_stats=stats)
        self.assertFalse(arm["G4"]["pass"])
        self.assertEqual(arm["classification"], "MECHANISM_FAIL")

    def test_protocol_10_1_report_field(self):
        arm = self._verdict()
        self.assertTrue(arm["protocol_10_1"]["delta_test_ge_0.005"])
        self.assertTrue(arm["protocol_10_1"]["val_positive"])
        arm = self._verdict(auc_test=RP.BASELINE_AUC_TEST + 0.0049)
        self.assertFalse(arm["protocol_10_1"]["delta_test_ge_0.005"])


class TestParamReport(unittest.TestCase):
    def test_real_dimensions_match_preregistration(self):
        var = RP.build_newtask(RP.VARIANT, input_size=80, rep_dim=64,
                               tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN,
                               device=torch.device("cpu"))
        report = RP.param_report(var)
        self.assertEqual(report["new_params_total"], 2385)                 # 预注册 2.2
        self.assertEqual(report["head_params"], 8129)
        self.assertEqual(report["total_params"], 10514)
        self.assertEqual({item["name"] for item in report["new_param_list"]},
                         set(RP.EXTRA_PARAM_NAMES))
        self.assertAlmostEqual(report["new_ratio_of_head"], 2385 / 8129, places=12)


# ---- tiny 夹具（与 test_smoke 同口径）----

HEADER = "click,purchase,X,121,122,301"
TINY_VOCAB = {"121": 5, "122": 4}

TRAIN_ROWS = [
    [0, 0, 9, 1, 0, 1],
    [1, 0, 9, 2, 1, 2],
    [0, 0, 8, 0, 2, 3],
    [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3],
    [1, 0, 7, 1, 1, 2],
    [0, 0, 9, 2, 2, 1],
    [1, 1, 9, 0, 1, 3],
    [0, 0, 8, 3, 0, 2],
    [1, 0, 8, 1, 1, 1],
    [0, 0, 7, 4, 2, 3],
    [1, 0, 7, 2, 1, 2],
]
VAL_ROWS = [
    [0, 0, 9, 1, 0, 1],
    [1, 0, 9, 2, 1, 2],
    [0, 0, 8, 0, 2, 3],
    [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3],
    [1, 0, 7, 1, 1, 2],
]
TEST_ROWS = [
    [0, 0, 9, 2, 2, 1],
    [1, 0, 9, 0, 1, 3],
    [0, 0, 8, 3, 0, 2],
    [1, 1, 8, 1, 1, 1],
    [0, 0, 7, 4, 2, 3],
    [1, 0, 7, 2, 1, 2],
    [0, 0, 9, 1, 0, 3],
    [1, 0, 8, 4, 1, 2],
]


def write_aliccp_file(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


def make_tiny_root(td: str):
    root = Path(td) / "artifacts"
    data_dir = Path(td) / "data"
    data_dir.mkdir()
    write_aliccp_file(data_dir / "train.csv", TRAIN_ROWS)
    write_aliccp_file(data_dir / "val.csv", VAL_ROWS)
    write_aliccp_file(data_dir / "test.csv", TEST_ROWS)
    data_files = {
        "train": str(data_dir / "train.csv"),
        "val": str(data_dir / "val.csv"),
        "test": str(data_dir / "test.csv"),
    }
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return root, data_files, budgets


def _silent(*args, **kwargs):
    return None


def run_tiny_stage1(root, data_files, budgets):
    return bench.run_stage1(
        root=root, data_files=data_files, budgets=budgets, prefix_tag="tiny",
        model_seed=123, env_seed=456, epochs=2, patience=2,
        device=torch.device("cpu"), vocab=TINY_VOCAB,
        expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
        log=_silent,
    )


def run_tiny_stage2(root, data_files, budgets, stage1_id, **kwargs):
    return bench.run_stage2(
        root=root, stage1_id=stage1_id, data_files=data_files, budgets=budgets,
        prefix_tag="tiny", model_seed=123, epochs=2, patience=2, tag="smoke",
        device=torch.device("cpu"), vocab=TINY_VOCAB,
        expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
        enforce_b=False, log=_silent, **kwargs,
    )


class TestRunnerWiring(unittest.TestCase):
    def test_default_variant_is_baseline_and_reference_optional(self):
        sig = inspect.signature(bench.run_stage2)
        self.assertEqual(sig.parameters["variant"].default, RP.BASELINE_VARIANT)
        self.assertIsNone(sig.parameters["prompt_reference_newtask"].default)

    def test_cli_stage2_surface(self):
        import run_aliccp_benchmark as cli
        parser = cli.build_parser()
        ns = parser.parse_args(["stage2", "--stage1-id", "sid"])
        self.assertEqual(ns.variant, RP.BASELINE_VARIANT)
        self.assertIsNone(ns.prompt_reference_newtask)
        ns = parser.parse_args(["stage2", "--stage1-id", "sid", "--variant", "residual-prompt",
                                "--prompt-reference-newtask", "ref/newtask.pt"])
        self.assertEqual(ns.variant, RP.VARIANT)
        self.assertEqual(ns.prompt_reference_newtask.as_posix(), "ref/newtask.pt")

    def test_treatment_arm_without_reference_refuses_to_emit_numbers(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = make_tiny_root(td)
            with self.assertRaises(ValueError):
                run_tiny_stage2(root, data_files, budgets, stage1_id="nonexistent",
                                variant=RP.VARIANT)

    def test_baseline_purity_and_variant_arm_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = make_tiny_root(td)
            meta = run_tiny_stage1(root, data_files, budgets)
            base = run_tiny_stage2(root, data_files, budgets, meta["stage1_id"])
            base_path = P.run_dir(root, base["run_id"])
            self.assertNotIn(RP.RUN_ID_SUFFIX, base["run_id"])
            self.assertFalse((base_path / "prompt_report.json").exists())
            base_metrics = json.loads((base_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(base_metrics["variant"], RP.BASELINE_VARIANT)
            self.assertNotIn("rp_arm", base_metrics)
            self.assertNotIn("prompt_hidden", base_metrics)
            base_report = json.loads((base_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertNotIn("residual_prompt", base_report)
            base_config = json.loads((base_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(base_config["variant"], RP.BASELINE_VARIANT)
            self.assertNotIn("prompt_hidden", base_config)

            var = run_tiny_stage2(root, data_files, budgets, meta["stage1_id"], variant=RP.VARIANT,
                                  prompt_reference_newtask=base_path / "newtask.pt")
            var_path = P.run_dir(root, var["run_id"])
            self.assertIn(RP.RUN_ID_SUFFIX, var["run_id"])
            self.assertTrue((var_path / "prompt_report.json").exists())
            var_metrics = json.loads((var_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(var_metrics["variant"], RP.VARIANT)
            self.assertEqual(var_metrics["prompt_hidden"], RP.PROMPT_HIDDEN)
            arm = var_metrics["rp_arm"]
            self.assertEqual(set(arm), {"M0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8",
                                        "U", "protocol_10_1", "classification", "subreason", "pass"})
            # tiny 夹具不可能满足真实基线参照身份（0.5781533414372665）⇒ M0 必须 FAIL 且为最高优先级
            self.assertFalse(arm["M0"]["pass"])
            self.assertEqual(arm["classification"], "MECHANISM_FAIL")
            self.assertEqual(arm["subreason"], "REFERENCE_IDENTITY")
            var_report = json.loads((var_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertIn("residual_prompt", var_report)
            self.assertNotIn("residual_prompt", var_report["gates"])          # 不污染 A/B 门禁
            self.assertEqual(set(var_report) - set(base_report), {"residual_prompt"})
            var_config = json.loads((var_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(var_config["prompt_hidden"], RP.PROMPT_HIDDEN)
            prompt_report = json.loads((var_path / "prompt_report.json").read_text(encoding="utf-8"))
            for key in ("construction_identity", "init_forward", "grad_probe", "alpha_final", "gate",
                        "val_stats", "dispersion", "reference_dispersion", "source_gates", "params", "arm"):
                self.assertIn(key, prompt_report)
            self.assertTrue(prompt_report["construction_identity"]["shared_params_bit_identical"])
            self.assertTrue(prompt_report["construction_identity"]["global_rng_endpoint_identical"])
            self.assertNotEqual(prompt_report["alpha_final"], 0.0)   # 至少经过一次 Adam step
            self.assertTrue(arm["G1"]["pass"])
            self.assertTrue(arm["G2"]["pass"])
            self.assertTrue(arm["G3"]["pass"])                       # 门控在 tiny 连跑中确实被激活
            # grad_probe：epoch 1 首 batch 生成器梯度为 0（α=0 的数学后果），其后恢复
            self.assertEqual(prompt_report["grad_probe"][0]["generator_grad_norm"], 0.0)
            self.assertGreater(prompt_report["grad_probe"][-1]["generator_grad_norm"], 0.0)
            self.assertNotEqual(prompt_report["grad_probe"][0]["alpha_grad_norm"], 0.0)
            # 预测离散度与参照同口径记录
            self.assertIn("pred_std", prompt_report["dispersion"])
            self.assertIn("pred_std", prompt_report["reference_dispersion"]["pred_dispersion"])
            # 参照头只读复算 val AUC 落在 [0,1]
            ref_auc = prompt_report["reference_dispersion"]["val_auc"]
            self.assertGreaterEqual(ref_auc, 0.0)
            self.assertLessEqual(ref_auc, 1.0)
            # SUMMARY 只追加、末行含 -rpg run_id
            summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
            self.assertIn(var["run_id"], summary[-1])


class TestStaticGuards(unittest.TestCase):
    PROTECTED = ["multitaskrec/", "config.py", "aliccp_benchmark/protocol.py",
                 "aliccp_benchmark/metrics.py", "aliccp_benchmark/tests/test_protocol.py",
                 "aliccp_benchmark/tests/test_metrics.py", "aliccp_benchmark/tests/test_smoke.py",
                 "AliCCP_MPTRec.py", "AliCCP_NewTask.py", "CensusIncome_MPTRec.py",
                 "CensusIncome_NewTask.py", "baseline/", "mask/", "analysis/"]

    def test_protected_files_unchanged_vs_base(self):
        diff = subprocess.run(["git", "diff", "--name-only", FROZEN_BASE, "--", *self.PROTECTED],
                              cwd=REPO, capture_output=True, text=True).stdout.strip()
        self.assertEqual(diff, "", f"协议/模型文件不得改动: {diff}")

    def test_tracked_changes_subset_of_whitelist(self):
        changed = subprocess.run(["git", "diff", "--name-only", FROZEN_BASE],
                                 cwd=REPO, capture_output=True, text=True).stdout.split()
        extra = set(changed) - WHITELIST
        self.assertEqual(extra, set(), f"白名单外改动: {extra}")


class TestMechanismPin(unittest.TestCase):
    """移植语义守卫：机制文件除预注册 1.5 的 A1–A6 外与钉死 Census blob AST 逐字一致。"""

    def _pinned_blob(self) -> bytes:
        proc = subprocess.run(["git", "cat-file", "blob", PIN_BLOB], cwd=REPO, capture_output=True)
        if proc.returncode != 0:
            self.fail("钉死 Census 机制 blob 不可得（系谱依赖，如实响亮失败）: "
                      f"{proc.stderr.decode('utf-8', 'replace').strip()}")
        return proc.stdout

    @staticmethod
    def _dump(node) -> str:
        return ast.dump(node, include_attributes=False)

    @staticmethod
    def _normalize_auc_score(node):
        node = copy.deepcopy(node)

        class V(ast.NodeTransformer):
            def visit_Attribute(self, inner):
                inner = self.generic_visit(inner)
                if inner.attr == "auc_score":
                    inner.attr = "auc"
                return inner

        return V().visit(node)

    def test_pinned_blob_bytes_and_test_blob_available(self):
        raw = self._pinned_blob()
        self.assertEqual(hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(), PIN_SHA256_LF)
        proc = subprocess.run(["git", "cat-file", "blob", PIN_TEST_BLOB], cwd=REPO, capture_output=True)
        self.assertEqual(proc.returncode, 0, "钉死 Census 测试 blob 不可得")

    def test_ported_module_matches_pinned_ast_except_documented_adaptations(self):
        pinned = ast.parse(self._pinned_blob().decode("utf-8"))
        ported = ast.parse((REPO / "aliccp_benchmark/residual_prompt.py").read_text(encoding="utf-8"))

        def classes(tree):
            return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

        def funcs(tree):
            return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

        pin_classes, por_classes = classes(pinned), classes(ported)
        self.assertEqual(set(pin_classes), set(por_classes), "类名集合必须一致")
        for name, node in pin_classes.items():
            pin_methods = {m.name: m for m in node.body if isinstance(m, ast.FunctionDef)}
            por_methods = {m.name: m for m in por_classes[name].body if isinstance(m, ast.FunctionDef)}
            self.assertEqual(set(pin_methods), set(por_methods), f"{name} 方法集合必须一致")
            for mname, mnode in pin_methods.items():
                self.assertEqual(self._dump(mnode), self._dump(por_methods[mname]),
                                 f"{name}.{mname} 必须与钉死 blob AST 逐字一致")

        pin_funcs, por_funcs = funcs(pinned), funcs(ported)
        self.assertEqual(set(pin_funcs), set(por_funcs), "模块级函数集合必须一致")
        strict = {"isolated_cpu_rng", "effective_gate", "build_newtask",
                  "evaluate_prompt_diagnostics", "param_report"}
        for fname in strict:
            self.assertEqual(self._dump(pin_funcs[fname]), self._dump(por_funcs[fname]),
                             f"{fname} 必须与钉死 blob AST 逐字一致")
        # 适配 A2：reference_head_stats 仅允许 metrics.auc → metrics.auc_score 一处改名
        self.assertEqual(self._dump(pin_funcs["reference_head_stats"]),
                         self._dump(self._normalize_auc_score(por_funcs["reference_head_stats"])),
                         "reference_head_stats 除 auc_score 改名外必须逐字一致")
        # 适配 A4：arm_verdict 重写（语义由 TestArmVerdict 覆盖）
        self.assertIn("arm_verdict", por_funcs)
        self.assertIn("arm_verdict", pin_funcs)


class TestPreregConstants(unittest.TestCase):
    def test_thresholds_and_baseline_reference(self):
        self.assertEqual(RP.PROMPT_HIDDEN, 16)
        self.assertEqual(RP.AUC_TEST_DELTA_MIN, 0.0055)
        self.assertEqual(RP.AUC_VAL_DIRECTION_MIN, 0.0)
        self.assertEqual(RP.BASELINE_AUC_TEST, 0.5974422649550507)
        self.assertEqual(RP.BASELINE_AUC_VAL, 0.5809347091990792)
        self.assertEqual(RP.PRED_STD_MIN_RATIO, 0.5)
        self.assertEqual(RP.REFERENCE_PRED_STD, 0.005217193225189258)
        self.assertEqual(RP.REFERENCE_IDENTITY_TOL, 1e-9)
        self.assertEqual(RP.EXPECTED_NEW_PARAMS_TOTAL, 2385)
        self.assertEqual(RP.EXPECTED_HEAD_PARAMS, 8129)
        self.assertEqual(RP.RATIO_BAND, (0.005, 0.5))
        self.assertEqual(RP.BOUND_TOL, 1e-6)
        self.assertEqual(RP.RUN_ID_SUFFIX, "-rpg")
        self.assertEqual(RP.VARIANTS, ("baseline", "residual-prompt"))

    def test_prereg_doc_tokens_and_summary_ledger(self):
        doc = (REPO / DOC_PATH).read_text(encoding="utf-8")
        for token in ["013e105", "79ddefa", "bbd8a61", "8133d32", "47ecb0c", "d990cb0", "296e03a",
                      "e931cd7", "a40836e", "3e2f083", "221580a", "2b1d585", "e46e5d2", "0935257",
                      "28aa1fe", "f2ccec2", "99b9510",
                      PIN_BLOB,
                      "5dc7158ca999c9e7e6217a999c37869231411549",
                      "674213f619c5d5039811c71242a7727118348daf",
                      "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc",
                      "768a477b5c71af32c5c59ec6feb20c29f7873557",
                      "dab737a1422b88befc4c19f2bee408618061de7fd3bf0b2f83ebcdd85fd1cbc2",
                      "229c25a1c86719fa6fb6b057d6cff3f4120fa053",
                      "2966ec3985e5b91e277f429c2f236fb551de7479d114def90f767f2d813ad285",
                      "bf3647686838157bf5fd7645615faad9f2d7185d",
                      "b063a36667173b66fc8a7ac932502cc34753c91e2bfcfdafaedfce443f1db3b4",
                      "e02da3071ded554175cfdad5d7a725dd43ed7c89",
                      "b8d62c5735715440a7e23dcbc4f206ee38ec7f6f9a6e5aac1184ceb8dcb15bb0",
                      "f2aa202c426d619cd752035e19cf8803dc5afccec0ce32f923c48a20ff1d72f3",
                      "docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md",
                      "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
                      "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07",
                      "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg",
                      "20261004-0427-p2M-v500k-t1M-m1688723740-long-2b1d585",
                      "20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg",
                      "20261003-0724-p2M-v500k-t1M-m1688723740-long-bbd8a61",
                      "0.5809347091990792", "0.5974422649550507", "0.005217193225189258",
                      "0.6055453782825184", "0.5895572066556973",
                      "0.008103113327467715", "0.008622497456618139",
                      "0.0074396653390377265", "0.0048363447472773435",
                      "0.6614121223087556", "0.6401127811737995",
                      "0.6688517876477933", "0.6449491259210769",
                      "0.005910402347593546", "0.0031538117454772883",
                      "0.0019944278461222166", "0.009826376849064875", "2.7764451051977987",
                      "0.001456339669678841", "0.0013698631373653125",
                      "0.004951998671071434", "0.004212769651563031",
                      "0.009646016187411788", "0.011667457451331131",
                      "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f",
                      "cd7b033423499e0d36eea988835cea4ac4e2a8e26427a2aea627333822101e0b",
                      "4660be5aaa3c59f53dd5b4394f77114a87db69064b32c45517db049a6b9157e7",
                      "61a66d81ce3dcf6bae64f4c2b37cf12e722943def646336a588746aba932931d",
                      "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c",
                      "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0",
                      "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8",
                      "4e1b5c6ffe9b6ff49cda34691de27390669396337e64e867f0f83b6ec4da7981",
                      "0.0055", "0.001", "−0.02", "2385", "8129", "-rpg", "xlong",
                      "VALID_POSITIVE", "PERSISTS", "NOT_PERSIST", "INVALID",
                      "POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION",
                      "positive_persistence_claim", "ABLATION_ELIGIBLE", "ABLATION_NOT_ELIGIBLE",
                      "TWENTY_EPOCH_CONDITION_SATISFIED", "RIGHT_CENSORED_STILL_IMPROVING",
                      "EARLY_STOPPED", "MONOTONE_NARROWING", "不外推", "原样保留"]:
            self.assertIn(token, doc)
        summary = (REPO / "artifacts" / "aliccp_bench" / "SUMMARY.md").read_text(encoding="utf-8")
        row = [line for line in summary.splitlines() if "b2e17f9" in line and "run_id" not in line]
        self.assertEqual(len(row), 1)
        self.assertIn("0.598839", row[0])
        self.assertIn("0.578153", row[0])


if __name__ == "__main__":
    unittest.main()
