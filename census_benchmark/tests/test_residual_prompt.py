"""TDD 测试：阶段 2 范数受控残差 Prompt（可学习门控）。

唯一事实来源：docs/superpowers/specs/2026-10-01-stage2-residual-prompt-gate-design.md
CPU、秒级、不读真实数据集（tiny 端到端用 test_smoke 同款夹具口径）。
"""
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import protocol as P
from census_benchmark import residual_prompt as RP
from multitaskrec.model import NewTask
import run_census_benchmark

REPO = Path(__file__).resolve().parents[2]          # 测试依赖 cwd = 仓库根


def make_variant(input_size=8, rep_dim=4, hidden=RP.PROMPT_HIDDEN, seed=0):
    """同一 seed 下构造 (variant, baseline 参照) 两枚头：参照从构造前 RNG 现场出发。"""
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
        # RNG 端点：从同一现场构造基线头后，全局 CPU RNG 应与构造 variant 后逐位一致
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
        # S2：同一 x 下三路相对扰动一致（与流自身范数无关）
        r = [float((d.norm(dim=-1) / h.norm(dim=-1))[0]) for h, d in zip(reps, deltas)]
        self.assertAlmostEqual(r[0], r[1], places=5)
        self.assertAlmostEqual(r[1], r[2], places=5)
        # 门控上界即 alpha：g_eff = |alpha| * ||m|| / sqrt(d)
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

    def test_roundtrip_on_tiny_fixture(self):
        torch.manual_seed(3)
        head = NewTask(input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                       reg_dnn=P.REG_DNN, device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "newtask.pt"
            torch.save(head.state_dict(), path)
            loaders, _, _ = tiny_inputs()
            out = RP.reference_head_stats(path, backbone=tiny_backbone(), loader=loaders["val"],
                                          device=torch.device("cpu"), input_size=8, rep_dim=4,
                                          tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN)
        self.assertEqual(out["n_samples"], 16)
        self.assertIn("pred_std", out["pred_dispersion"])
        self.assertGreaterEqual(out["val_auc"], 0.0)
        self.assertLessEqual(out["val_auc"], 1.0)


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

    def _stats(self, ratios=(0.05, 0.05, 0.05), alpha=0.1):
        return {"streams": {"ratio_mean": list(ratios),
                            "ratio_max": [min(r, alpha) for r in ratios]},
                "gate": {"geff_mean": 0.03}}

    def test_e1_threshold_boundary(self):
        arm = RP.arm_verdict(auc_test=0.8521, probe=self._probe(), val_stats=self._stats())
        self.assertTrue(arm["E1"]["pass"])
        arm = RP.arm_verdict(auc_test=0.8520999, probe=self._probe(), val_stats=self._stats())
        self.assertFalse(arm["E1"]["pass"])

    def test_band_boundaries_are_inclusive(self):
        arm = RP.arm_verdict(auc_test=0.86, probe=self._probe(),
                             val_stats=self._stats(ratios=(0.005, 0.5, 0.05)))
        self.assertTrue(arm["G5"]["pass"])
        arm = RP.arm_verdict(auc_test=0.86, probe=self._probe(),
                             val_stats=self._stats(ratios=(0.0049, 0.05, 0.05)))
        self.assertFalse(arm["G5"]["pass"])

    def test_classification_matrix(self):
        cases = [
            (dict(auc_test=0.86), "CONFIRMED"),
            (dict(auc_test=0.84), "VALID_NEGATIVE"),
            (dict(auc_test=0.86, probe=self._probe(last_gen_grad=0.0)), "MECHANISM_INACTIVE"),
            (dict(auc_test=0.86, probe=self._probe(first_alpha_grad=0.0)), "MECHANISM_INACTIVE"),
            # α_final = 0 与"有扰动"的 stats 不自洽：ratio_max 必须随 α=0 归零
            (dict(auc_test=0.86, probe=self._probe(alpha_final=0.0, last_alpha=0.0),
                  val_stats=self._stats(ratios=(0.0, 0.0, 0.0), alpha=0.0)), "MECHANISM_INACTIVE"),
            (dict(auc_test=0.86, val_stats=self._stats(ratios=(0.004, 0.05, 0.05))), "MECHANISM_SILENT"),
            (dict(auc_test=0.86, val_stats=self._stats(ratios=(0.6, 0.05, 0.05))), "MECHANISM_OVER_PERTURB"),
            (dict(auc_test=0.86, probe=self._probe(bit=False)), "INVALID_IMPLEMENTATION"),
            (dict(auc_test=0.86, probe=self._probe(init_bit=False)), "INVALID_IMPLEMENTATION"),
            (dict(auc_test=0.86, probe=self._probe(extra=["prompt_gate"])), "INVALID_IMPLEMENTATION"),
        ]
        for kwargs, expected in cases:
            kwargs.setdefault("probe", self._probe())
            kwargs.setdefault("val_stats", self._stats())
            arm = RP.arm_verdict(**kwargs)
            self.assertEqual(arm["classification"], expected, kwargs)
        self.assertFalse(RP.arm_verdict(auc_test=0.84, probe=self._probe(),
                                        val_stats=self._stats())["pass"])

    def test_bound_violation_is_invalid(self):
        stats = {"streams": {"ratio_mean": [0.05, 0.05, 0.05], "ratio_max": [0.05, 0.3, 0.05]},
                 "gate": {"geff_mean": 0.03}}
        arm = RP.arm_verdict(auc_test=0.86, probe=self._probe(alpha_final=0.1), val_stats=stats)
        self.assertFalse(arm["G4"]["pass"])
        self.assertEqual(arm["classification"], "INVALID_IMPLEMENTATION")


class TestParamReport(unittest.TestCase):
    def test_real_dimensions_match_preregistration(self):
        var = RP.build_newtask(RP.VARIANT, input_size=123, rep_dim=128,
                               tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN,
                               device=torch.device("cpu"))
        report = RP.param_report(var)
        self.assertEqual(report["new_params_total"], 4161)                 # spec 2.2
        self.assertEqual(report["head_params"], 27063)
        self.assertEqual({item["name"] for item in report["new_param_list"]},
                         set(RP.EXTRA_PARAM_NAMES))
        self.assertAlmostEqual(report["new_ratio_of_head"], 4161 / 27063, places=12)


class TestRunnerWiring(unittest.TestCase):
    def test_default_variant_is_baseline(self):
        import inspect
        sig = inspect.signature(run_census_benchmark.run_stage2)
        self.assertEqual(sig.parameters["variant"].default, RP.BASELINE_VARIANT)

    def test_baseline_arm_purity_and_variant_arm_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            loaders, stats, indices = tiny_inputs()
            stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_backbone(),
                                                     loaders=loaders, stats=stats, indices=indices)
            base = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2,
                                                   device=device, model=tiny_backbone(), loaders=loaders,
                                                   stats=stats, indices=indices, input_size=8, rep_dim=4)
            base_path = Path(base["run_dir"])
            self.assertNotIn(RP.RUN_ID_SUFFIX, base["run_id"])
            self.assertFalse((base_path / "prompt_report.json").exists())
            base_metrics = json.loads((base_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(base_metrics["variant"], RP.BASELINE_VARIANT)
            self.assertNotIn("rp_arm", base_metrics)
            base_report = json.loads((base_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertNotIn("residual_prompt", base_report)

            var = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2,
                                                  device=device, model=tiny_backbone(), loaders=loaders,
                                                  stats=stats, indices=indices, input_size=8, rep_dim=4,
                                                  variant=RP.VARIANT)
            var_path = Path(var["run_dir"])
            self.assertIn(RP.RUN_ID_SUFFIX, var["run_id"])
            self.assertTrue((var_path / "prompt_report.json").exists())
            var_metrics = json.loads((var_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(var_metrics["variant"], RP.VARIANT)
            arm = var_metrics["rp_arm"]
            self.assertEqual(set(arm), {"E1", "G1", "G2", "G3", "G4", "G5", "classification", "pass"})
            self.assertIn("prompt", var_metrics["mechanism"])
            var_report = json.loads((var_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertIn("residual_prompt", var_report)
            self.assertNotIn("residual_prompt", [k for k, v in var_report.items()
                                                 if k in ("A1", "B1")])   # 不污染 A/B 门禁
            prompt_report = json.loads((var_path / "prompt_report.json").read_text(encoding="utf-8"))
            for key in ("construction_identity", "init_forward", "grad_probe", "gate",
                        "val_stats", "params", "dispersion"):
                self.assertIn(key, prompt_report)
            # grad_probe：epoch 1 首 batch 生成器梯度为 0（α=0 的数学后果），epoch 2 恢复
            self.assertEqual(prompt_report["grad_probe"][0]["generator_grad_norm"], 0.0)
            self.assertGreater(prompt_report["grad_probe"][-1]["generator_grad_norm"], 0.0)
            summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
            self.assertIn(var["run_id"], summary[-1])                    # 只追加、含 -rpg 后缀


class TestStaticGuards(unittest.TestCase):
    def test_protocol_files_unchanged_vs_base(self):
        files = ["multitaskrec/model.py", "config.py", "census_benchmark/protocol.py",
                 "census_benchmark/metrics.py"]
        diff = subprocess.run(["git", "diff", "--name-only", "87afe03", "--", *files],
                              cwd=REPO, capture_output=True, text=True).stdout.strip()
        self.assertEqual(diff, "", f"协议/模型文件不得改动: {diff}")


class TestPreregConstants(unittest.TestCase):
    def test_thresholds_and_baseline_reference(self):
        self.assertEqual(RP.AUC_TEST_MIN, 0.8521)
        self.assertEqual(RP.RATIO_BAND, (0.005, 0.5))
        self.assertEqual(RP.BASELINE_AUC_TEST, 0.8500685307175756)
        summary = (REPO / "artifacts" / "census_stage2" / "SUMMARY.md").read_text(encoding="utf-8")
        row = [line for line in summary.splitlines() if "904f8d0" in line and "stage1_id" not in line]
        self.assertEqual(len(row), 1)
        self.assertIn(f"{RP.BASELINE_AUC_TEST:.6f}", row[0])             # 基线值钉在 SUMMARY 台账上


# ---- tiny 夹具（与 test_smoke 同口径）----

class TinyCensus(Dataset):
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


def tiny_backbone():
    return run_census_benchmark.MPTRec(num_tasks=2, feature_vocabulary={"a": 3, "b": 2},
                                       embedding_size=4, input_size=8, expert_dnn_hidden_units=(8, 4),
                                       tower_dnn_hidden_units=(4, 2), device=torch.device("cpu"))


def tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


if __name__ == "__main__":
    unittest.main()
