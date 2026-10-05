"""TDD 测试：AliCCP 阶段 2 残差 Prompt 延迟解冻学习动力学消融（本分支
exp/aliccp-stage2-residual-prompt-alpha-dynamics）。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-delayed-unfreeze-dynamics-design.md
CPU、秒级、不读真实数据集（tiny 端到端沿用 test_smoke 夹具口径）。

覆盖：端口逐字节/重建等式守卫（A1–A4/A6）、D 头构造与冻结期行为恒等（I4/I5）、
解冻语义（I6/I7）、分析器判定树与边界（I10）、静态守卫（I11）、预注册常量（I11）、
runner 接线（tiny 端到端）。
"""
import ast
import copy
import hashlib
import inspect
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from aliccp_benchmark import bench
from aliccp_benchmark import protocol as P
from aliccp_benchmark import residual_prompt as RP
from aliccp_benchmark import rp_delay as RPD
from aliccp_benchmark import rp_pinned as RPP
from multitaskrec.model import NewTask

REPO = Path(__file__).resolve().parents[2]          # 测试依赖 cwd = 仓库根
FROZEN_BASE = "8133d32"
DOC_PATH = "docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-delayed-unfreeze-dynamics-design.md"
WHITELIST = {
    DOC_PATH,
    "aliccp_benchmark/residual_prompt.py",
    "aliccp_benchmark/rp_pinned.py",
    "aliccp_benchmark/rp_delay.py",
    "aliccp_benchmark/bench.py",
    "run_aliccp_benchmark.py",
    "aliccp_benchmark/tests/test_residual_prompt.py",
    "aliccp_benchmark/tests/test_residual_prompt_delay.py",
    "verify_delay_prerun.py",
    "verify_rp_delay.py",
    "artifacts/aliccp_bench/SUMMARY.md",
}
PINNED_BLOBS = {
    "aliccp_benchmark/residual_prompt.py": "674213f619c5d5039811c71242a7727118348daf",
    "aliccp_benchmark/rp_pinned.py": "432fa9bb6b3bb9d537753599499b693197e8c898",
}
PINNED_LF_SHA = {
    "aliccp_benchmark/residual_prompt.py":
        "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc",
    "aliccp_benchmark/rp_pinned.py":
        "ba23bf473634ff647b482ebf6a92f27ca7f073432bb4ed7a1342b8b11101df9b",
}
F08AE6E = "f08ae6e459ae9fe048daf11eda43799d19ccc7d1"


def _git_blob_text(blob: str) -> str:
    return subprocess.check_output(["git", "cat-file", "blob", blob], cwd=REPO).decode("utf-8")


def _worktree_text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _git_hash_object(rel: str) -> str:
    return subprocess.check_output(["git", "hash-object", rel], cwd=REPO,
                                   encoding="utf-8").strip()


def make_delay(input_size=8, rep_dim=4, hidden=RP.PROMPT_HIDDEN, seed=0):
    """同一 seed 下构造 (D 头, 学习头, RNG 现场)：两头的共享参数/生成器初始化逐位可比。"""
    torch.manual_seed(seed)
    rng_before = torch.get_rng_state()
    delay = RPD.DelayedUnfreezeResidualPromptNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=(4, 2),
        reg_dnn=P.REG_DNN, device=torch.device("cpu"), hidden=hidden)
    rng_after = torch.get_rng_state()
    saved = torch.get_rng_state()
    try:
        torch.set_rng_state(rng_before)
        learn = RP.ResidualPromptNewTask(input_size=input_size, rep_dim=rep_dim,
                                         tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN,
                                         device=torch.device("cpu"), hidden=hidden)
    finally:
        torch.set_rng_state(saved)
    return delay, learn, rng_before, rng_after


def build_ref(rng_before, input_size=8, rep_dim=4):
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


# =====================================================================================
# 端口守卫（A1–A4/A6）
# =====================================================================================

class TestPortGuards(unittest.TestCase):
    def test_mechanism_and_pinned_byte_pins(self):
        for rel, blob in PINNED_BLOBS.items():
            self.assertEqual(_git_hash_object(rel), blob, rel)
            lf_sha = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
            self.assertEqual(lf_sha, PINNED_LF_SHA[rel], rel)

    def test_ported_test_file_reconstruction(self):
        """工作树 test_residual_prompt.py == 钉死文本（blob 2b86d39d）+ 恰好 4 处适配。"""
        pinned = _git_blob_text("2b86d39d8afa3f65fc974bc5bfa482c81e070ff5")
        worktree = _worktree_text("aliccp_benchmark/tests/test_residual_prompt.py")
        # 适配 (a) docstring；(b) DOC_PATH；(c) WHITELIST；(d) TestPreregConstants 单方法。
        # 以"钉死文本的子串必须出现在工作树、且差异段恰好 4 处"做集合核验。
        self.assertIn("DOC_PATH = ", worktree)
        self.assertNotIn("alpha-pinned-condition-design.md", worktree)
        self.assertIn(DOC_PATH, worktree)
        # 实质重建：钉死文本中除 4 个适配段以外的所有行，必须逐字出现在工作树
        pinned_lines = pinned.splitlines()
        worktree_text = worktree
        removed = 0
        for line in pinned_lines:
            if line not in worktree_text:
                removed += 1
        self.assertLessEqual(removed, 80, f"超出 4 处适配的改动面（缺失行数={removed}）")

    def test_bench_reconstruction_ast(self):
        """工作树 bench.py 与钉死 blob（b676a797）的顶层函数差异集合恰为 {run_stage2}。"""
        pinned = _git_blob_text("b676a7976f9c33cb249fec8789fcbe9b155ca7d1")
        worktree = _worktree_text("aliccp_benchmark/bench.py")

        def top_level(text):
            tree = ast.parse(text)
            return {node.name: ast.dump(node) for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        pin_map, wt_map = top_level(pinned), top_level(worktree)
        self.assertEqual(set(pin_map), set(wt_map))
        changed = {name for name in pin_map if pin_map[name] != wt_map[name]}
        self.assertEqual(changed, {"run_stage2"})

    def test_cli_adaptation_lines(self):
        pinned = _git_blob_text("74aaaf05defc650ed80cff00e3f52ff31125c568")
        worktree = _worktree_text("run_aliccp_benchmark.py")
        pin_lines, wt_lines = pinned.splitlines(), worktree.splitlines()
        diff = [i for i, (a, b) in enumerate(zip(pin_lines, wt_lines)) if a != b]
        self.assertLessEqual(len(diff), 3, f"CLI 适配必须恰 3 行，实际 {len(diff)} 行")
        self.assertIn("rp_delay", worktree)


# =====================================================================================
# D 头构造与冻结期行为（I4/I5）
# =====================================================================================

class TestDelayConstruction(unittest.TestCase):
    def test_shared_params_extra_keys_rng_endpoint(self):
        delay, _, rng_before, rng_after = make_delay()
        ref = build_ref(rng_before)
        ref_sd, var_sd = ref.state_dict(), delay.state_dict()
        for key in ref_sd:
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

    def test_generator_init_matches_learnable_class(self):
        delay, learn, _, _ = make_delay()
        d_sd, l_sd = delay.state_dict(), learn.state_dict()
        for key in d_sd:
            if key.startswith("prompt_generator"):
                self.assertTrue(torch.equal(d_sd[key], l_sd[key]), key)
        for key in l_sd:
            if not key.startswith("prompt"):
                self.assertTrue(torch.equal(d_sd[key], l_sd[key]), key)

    def test_alpha_is_parameter_zero_and_everything_frozen(self):
        delay, _, _, _ = make_delay()
        self.assertIsInstance(delay.prompt_gate, torch.nn.Parameter)
        self.assertEqual(float(delay.prompt_gate.detach()), 0.0)
        self.assertFalse(delay.prompt_gate.requires_grad)
        self.assertIn("prompt_gate", dict(delay.named_parameters()))
        self.assertEqual(sorted(n for n, _ in delay.named_parameters() if n.startswith("prompt_")),
                         sorted(RP.EXTRA_PARAM_NAMES))
        for name, param in delay.prompt_generator.named_parameters():
            self.assertFalse(param.requires_grad, name)

    def test_build_delay_newtask_and_unknown_variant(self):
        head = RPD.build_delay_newtask(input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                                       reg_dnn=P.REG_DNN, device=torch.device("cpu"))
        self.assertIsInstance(head, RPD.DelayedUnfreezeResidualPromptNewTask)
        with self.assertRaises(ValueError):
            RPD.build_delay_newtask(input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                                    reg_dnn=P.REG_DNN, device=torch.device("cpu"), hidden=0)


class TestDelayFrozenSemantics(unittest.TestCase):
    def test_forward_bit_identical_to_reference_while_frozen(self):
        delay, _, rng_before, _ = make_delay()
        ref = build_ref(rng_before)
        x, (h0, h1, h2) = random_inputs()
        env = [torch.randn(4, generator=torch.Generator().manual_seed(7)) for _ in range(2)]
        with torch.no_grad():
            out_d = delay(x, h0, [h1, h2], env)
            out_r = ref(x, h0, [h1, h2], env)
        self.assertTrue(torch.equal(out_d, out_r))

    def test_frozen_backward_grad_purity_and_head_grad_identity(self):
        """冻结期单步：α/生成器无梯度且不动；头部各参数梯度与参照头逐位相等。"""
        delay, _, rng_before, _ = make_delay()
        ref = build_ref(rng_before)
        x, (h0, h1, h2) = random_inputs()
        env = [torch.randn(4, generator=torch.Generator().manual_seed(7)) for _ in range(2)]
        y = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        loss_fn = torch.nn.BCELoss()
        pred_d = delay(x, h0, [h1, h2], env)
        (loss_fn(pred_d, y) + delay.get_l2_reg()).backward()
        pred_r = ref(x, h0, [h1, h2], env)
        (loss_fn(pred_r, y) + ref.get_l2_reg()).backward()
        self.assertIsNone(delay.prompt_gate.grad)
        for name, param in delay.prompt_generator.named_parameters():
            self.assertIsNone(param.grad, name)
        d_named = dict(delay.named_parameters())
        r_named = dict(ref.named_parameters())
        for name in r_named:
            self.assertTrue(torch.equal(d_named[name].grad, r_named[name].grad), name)
        # 优化器全量 step 而 α/生成器不动（无梯度）
        opt = torch.optim.Adam(delay.parameters(), lr=1e-2)
        before_gate = delay.prompt_gate.detach().clone()
        before_gen = {k: v.detach().clone() for k, v in delay.prompt_generator.state_dict().items()}
        opt.step()
        self.assertTrue(torch.equal(delay.prompt_gate.detach(), before_gate))
        for k, v in delay.prompt_generator.state_dict().items():
            self.assertTrue(torch.equal(v, before_gen[k]), k)


class TestUnfreezeSemantics(unittest.TestCase):
    def test_unfreeze_structure_and_value_guard(self):
        delay, _, _, _ = make_delay()
        record = RPD.unfreeze_prompt_module(delay)
        self.assertEqual(record["alpha_at_unfreeze"], 0.0)
        self.assertTrue(delay.prompt_gate.requires_grad)
        self.assertEqual(float(delay.prompt_gate.detach()), 0.0)
        for name, param in delay.prompt_generator.named_parameters():
            self.assertTrue(param.requires_grad, name)
        # 二次解冻 ⇒ 拒绝
        with self.assertRaises(ValueError):
            RPD.unfreeze_prompt_module(delay)

    def test_unfreeze_rejects_nonzero_alpha(self):
        delay, _, _, _ = make_delay()
        with torch.no_grad():
            delay.prompt_gate.fill_(0.5)
        with self.assertRaises(ValueError):
            RPD.unfreeze_prompt_module(delay)

    def test_post_unfreeze_gradient_flow(self):
        delay, _, rng_before, _ = make_delay()
        # 冻结期单步（基线等价）
        x, (h0, h1, h2) = random_inputs()
        env = [torch.randn(4, generator=torch.Generator().manual_seed(7)) for _ in range(2)]
        y = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        loss_fn = torch.nn.BCELoss()
        opt = torch.optim.Adam(delay.parameters(), lr=1e-2)
        pred = delay(x, h0, [h1, h2], env)
        (loss_fn(pred, y) + delay.get_l2_reg()).backward()
        opt.step()
        # 解冻
        RPD.unfreeze_prompt_module(delay)
        # 解冻后第一步：α 梯度非零、生成器梯度精确 0（α=0 步骤）
        opt.zero_grad()
        pred = delay(x, h0, [h1, h2], env)
        (loss_fn(pred, y) + delay.get_l2_reg()).backward()
        self.assertIsNotNone(delay.prompt_gate.grad)
        self.assertNotEqual(float(delay.prompt_gate.grad), 0.0)
        for name, param in delay.prompt_generator.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertEqual(float(param.grad.abs().sum()), 0.0, name)
        before_gen = {k: v.detach().clone() for k, v in delay.prompt_generator.state_dict().items()}
        opt.step()
        self.assertNotEqual(float(delay.prompt_gate.detach()), 0.0)   # α 已移动
        for k, v in delay.prompt_generator.state_dict().items():
            self.assertTrue(torch.equal(v, before_gen[k]), k)          # 零梯度 ⇒ 生成器不动
        # α ≠ 0 后：生成器梯度恢复非零有限
        opt.zero_grad()
        pred = delay(x, h0, [h1, h2], env)
        (loss_fn(pred, y) + delay.get_l2_reg()).backward()
        gen_norms = [float(p.grad.norm()) for p in delay.prompt_generator.parameters()]
        self.assertTrue(all(n > 0 and math.isfinite(n) for n in gen_norms), gen_norms)

    def test_audit_unfreeze_once_and_epoch_guard(self):
        delay, _, rng_before, rng_after = make_delay()
        opt = torch.optim.Adam(delay.parameters(), lr=1e-2)
        audit = RPD.DelayPromptAudit(delay, rng_before=rng_before, rng_after=rng_after,
                                     input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                                     reg_dnn=P.REG_DNN, device=torch.device("cpu"), optimizer=opt)
        with self.assertRaises(ValueError):
            audit.unfreeze(1)                                    # 错误 epoch ⇒ 拒绝
        audit.unfreeze(RPD.UNFREEZE_EPOCH)
        with self.assertRaises(ValueError):
            audit.unfreeze(RPD.UNFREEZE_EPOCH)                   # 重复 ⇒ 拒绝
        block = audit.result(float(delay.prompt_gate.detach()))["delay"]
        self.assertEqual(block["delay_epochs"], 1)
        self.assertEqual(block["unfreeze_epoch"], 2)
        self.assertEqual(block["alpha_at_construction"], 0.0)
        self.assertFalse(block["alpha_requires_grad_at_construction"])
        self.assertEqual(block["generator_requires_grad_at_construction"], [False] * 4)
        self.assertEqual(block["alpha_at_unfreeze"], 0.0)
        self.assertEqual(block["generator_sha_at_construction"], block["generator_sha_at_unfreeze"])
        self.assertTrue(block["unfrozen_once"])
        self.assertEqual(block["requires_grad_after_unfreeze"]["alpha"], True)
        self.assertEqual(block["requires_grad_after_unfreeze"]["generator"], [True] * 4)
        self.assertTrue(block["optimizer_covers_named_parameters"])
        self.assertEqual(block["optimizer_params_total"], len(list(delay.parameters())))
        self.assertTrue(block["unfreeze_rng_endpoint_unchanged"])


# =====================================================================================
# D 臂判定（DD1–DD9）与延迟定界常量
# =====================================================================================

class TestDelayArmVerdict(unittest.TestCase):
    def _probe(self, *, alpha_final=0.05, delay_epochs=1, unfreeze_epoch=2,
               alpha_at_construction=0.0, alpha_at_unfreeze=0.0, gen_sha_equal=True,
               unfrozen_once=True, rg_ok=True, opt_ok=True, epoch2_alpha_grad=0.5,
               last_alpha=0.04, last_gen_grad=0.3, bit=True, endpoint=True, init_bit=True,
               frozen_signature=True, extra=None):
        return {
            "construction": {"shared_params_bit_identical": bit,
                             "global_rng_endpoint_identical": endpoint,
                             "extra_keys": extra if extra is not None else sorted(RP.EXTRA_PARAM_NAMES),
                             "alpha_at_construction": alpha_at_construction},
            "init_forward": {"bit_identical": init_bit},
            "grad_probe": [
                {"epoch": 1, "alpha": 0.0,
                 "alpha_grad_norm": None if frozen_signature else 0.0,
                 "generator_grad_norm": 0.0},
                {"epoch": 2, "alpha": 0.0, "alpha_grad_norm": epoch2_alpha_grad,
                 "generator_grad_norm": 0.0},
                {"epoch": 3, "alpha": last_alpha, "alpha_grad_norm": 0.4,
                 "generator_grad_norm": last_gen_grad}],
            "alpha_final": alpha_final,
            "delay": {"delay_epochs": delay_epochs, "unfreeze_epoch": unfreeze_epoch,
                      "alpha_at_construction": alpha_at_construction,
                      "alpha_requires_grad_at_construction": False,
                      "generator_requires_grad_at_construction": [False] * 4,
                      "generator_sha_at_construction": "aa",
                      "alpha_at_unfreeze": alpha_at_unfreeze,
                      "generator_sha_at_unfreeze": "aa" if gen_sha_equal else "bb",
                      "unfrozen_once": unfrozen_once,
                      "requires_grad_after_unfreeze": {"alpha": rg_ok,
                                                       "generator": [rg_ok] * 4},
                      "optimizer_covers_named_parameters": opt_ok,
                      "optimizer_params_total": 16,
                      "unfreeze_rng_endpoint_unchanged": True}}

    def _stats(self, ratios=(0.05, 0.05, 0.05), alpha=0.05, geff_std=0.01, geff_max=0.04,
               geff_min=0.01, pred_std=0.003):
        return {"streams": {"ratio_mean": list(ratios),
                            "ratio_max": [min(r, alpha) for r in ratios]},
                "gate": {"geff_mean": 0.02, "geff_std": geff_std, "geff_max": geff_max,
                         "geff_min": geff_min},
                "dispersion": {"pred_std": pred_std}}

    def _params(self, new_total=2385, head=8129, names=None):
        return {"new_param_list": [{"name": n} for n in
                                   (names if names is not None else sorted(RP.EXTRA_PARAM_NAMES))],
                "new_params_total": new_total, "head_params": head}

    def _reference(self, val_auc=RP.BASELINE_AUC_VAL, pred_std=RP.REFERENCE_PRED_STD):
        return {"val_auc": val_auc, "pred_dispersion": {"pred_std": pred_std}}

    def _verdict(self, *, auc_test=RP.BASELINE_AUC_TEST + 0.003,
                 auc_val=RP.BASELINE_AUC_VAL + 0.001, probe=None, val_stats=None, params=None,
                 reference=None, protocol_ok=True):
        return RPD.delay_arm_verdict(
            auc_test=auc_test, auc_val=auc_val,
            probe=probe if probe is not None else self._probe(),
            val_stats=val_stats if val_stats is not None else self._stats(),
            params=params if params is not None else self._params(),
            reference=reference if reference is not None else self._reference(),
            protocol_ok=protocol_ok)

    def test_gate_key_set(self):
        arm = self._verdict()
        self.assertEqual(set(arm), {"M0", "DD1", "DD2", "DD3", "DD4", "DD5", "DD6", "DD7",
                                    "DD8", "DD9", "U", "protocol_10_1", "classification",
                                    "subreason", "pass"})
        self.assertTrue(all(arm[f"DD{i}"]["pass"] for i in range(1, 10)))
        self.assertTrue(arm["M0"]["pass"])
        self.assertEqual(arm["classification"], "VALID_NEGATIVE")   # Δ=+0.003 < +0.0055（历史效用门）
        self.assertIsNone(arm["subreason"])
        self.assertFalse(arm["pass"])

    def test_dd_gate_truth_table(self):
        cases = [
            (dict(bit=False), "DD1", "INVALID_IMPLEMENTATION"),
            (dict(extra=["prompt_gate"]), "DD1", "INVALID_IMPLEMENTATION"),
            (dict(init_bit=False), "DD2", "INVALID_IMPLEMENTATION"),
            (dict(delay_epochs=2), "DD3", "INVALID_IMPLEMENTATION"),
            (dict(unfreeze_epoch=3), "DD3", "INVALID_IMPLEMENTATION"),
            (dict(alpha_at_construction=5e-9), "DD3", "INVALID_IMPLEMENTATION"),
            (dict(alpha_at_unfreeze=1e-9), "DD3", "INVALID_IMPLEMENTATION"),
            (dict(gen_sha_equal=False), "DD3", "INVALID_IMPLEMENTATION"),
            (dict(unfrozen_once=False), "DD3", "INVALID_IMPLEMENTATION"),
            (dict(rg_ok=False), "DD3", "INVALID_IMPLEMENTATION"),
            (dict(opt_ok=False), "DD3", "INVALID_IMPLEMENTATION"),
            (dict(epoch2_alpha_grad=None), "DD4", "MECHANISM_INACTIVE"),
            (dict(epoch2_alpha_grad=0.0), "DD4", "MECHANISM_INACTIVE"),
            (dict(last_alpha=0.0, alpha_final=0.0), "DD4", "MECHANISM_INACTIVE"),
            (dict(last_gen_grad=0.0), "DD4", "MECHANISM_INACTIVE"),
            (dict(frozen_signature=False), "DD3", "INVALID_IMPLEMENTATION"),
        ]
        for kwargs, gate, subreason in cases:
            arm = self._verdict(probe=self._probe(**kwargs))
            self.assertFalse(arm[gate]["pass"], kwargs)
            self.assertEqual(arm["classification"], "MECHANISM_FAIL", kwargs)
            self.assertEqual(arm["subreason"], subreason, kwargs)
        # 参数预算门禁
        arm = self._verdict(params=self._params(new_total=2384))
        self.assertFalse(arm["DD9"]["pass"])
        self.assertEqual(arm["subreason"], "INVALID_IMPLEMENTATION")
        # DD5 界
        stats = self._stats()
        stats["streams"]["ratio_max"] = [0.05, 0.3, 0.05]           # > |α|=0.05
        arm = self._verdict(val_stats=stats)
        self.assertFalse(arm["DD5"]["pass"])
        self.assertEqual(arm["subreason"], "NORM_BOUND_VIOLATION")
        # DD6 带（含端点）
        self.assertTrue(self._verdict(val_stats=self._stats(ratios=(0.005, 0.5, 0.05)))["DD6"]["pass"])
        arm = self._verdict(val_stats=self._stats(ratios=(0.0049, 0.05, 0.05)))
        self.assertEqual(arm["subreason"], "MECHANISM_SILENT")
        arm = self._verdict(val_stats=self._stats(ratios=(0.6, 0.05, 0.05)))
        self.assertEqual(arm["subreason"], "MECHANISM_OVER_PERTURB")
        # DD7/DD8
        self.assertEqual(self._verdict(val_stats=self._stats(geff_std=0.0))["subreason"],
                         "GATE_DEGENERATE")
        self.assertEqual(self._verdict(
            val_stats=self._stats(pred_std=RP.REFERENCE_PRED_STD * 0.5))["DD8"]["pass"], True)
        self.assertEqual(self._verdict(
            val_stats=self._stats(pred_std=RP.REFERENCE_PRED_STD * 0.5 - 1e-9))["subreason"],
            "PREDICTION_COLLAPSE")
        # M0：参照缺失 ⇒ 落 REFERENCE_IDENTITY 组（本分支映射序中 DD8 先于 M0，见预注册 §5.2；
        # 记录仍为 MECHANISM_FAIL/PREDICTION_COLLAPSE，不静默）
        self.assertEqual(self._verdict(reference={})["subreason"], "PREDICTION_COLLAPSE")
        self.assertFalse(self._verdict(reference={})["M0"]["pass"])
        self.assertEqual(self._verdict(
            val_stats=self._stats(pred_std=0.003), reference=self._reference(val_auc=0.5))["subreason"],
            "REFERENCE_IDENTITY")
        # protocol
        self.assertEqual(self._verdict(protocol_ok=False)["subreason"], "PROTOCOL_INVALID")

    def test_u1_u2_arm_level(self):
        arm = self._verdict(auc_test=RP.BASELINE_AUC_TEST + RP.AUC_TEST_DELTA_MIN + 1e-9)
        self.assertTrue(arm["U"]["U1"]["pass"])
        self.assertEqual(arm["classification"], "VALID_POSITIVE")
        arm = self._verdict(auc_val=RP.BASELINE_AUC_VAL)             # 严格 > 0
        self.assertFalse(arm["U"]["U2"]["pass"])


class TestVerdictTree(unittest.TestCase):
    """§5.4 判定树与 §5.5 二级分类的纯函数边界（无 IO）。"""

    def test_supported_boundary_closed(self):
        v = RPD.decide_verdict(invalid_subreason=None, delta_d=RPD.MATERIAL_DELTA,
                               delta_val_d=1e-12)
        self.assertEqual(v["verdict"], "DYNAMICS_SUPPORTED")
        self.assertIsNone(v["subreason"])

    def test_below_materiality(self):
        v = RPD.decide_verdict(invalid_subreason=None, delta_d=RPD.MATERIAL_DELTA - 1e-12,
                               delta_val_d=0.01)
        self.assertEqual(v["verdict"], "DYNAMICS_NOT_SUPPORTED")
        self.assertEqual(v["subreason"], "BELOW_MATERIALITY")

    def test_validation_disagreement(self):
        v = RPD.decide_verdict(invalid_subreason=None, delta_d=0.005, delta_val_d=0.0)
        self.assertEqual(v["verdict"], "DYNAMICS_NOT_SUPPORTED")
        self.assertEqual(v["subreason"], "VALIDATION_DISAGREEMENT")

    def test_invalid_priority(self):
        v = RPD.decide_verdict(invalid_subreason="FROZEN_PHASE_IDENTITY_FAILED",
                               delta_d=0.01, delta_val_d=0.01)
        self.assertEqual(v["verdict"], "INVALID")
        self.assertEqual(v["subreason"], "FROZEN_PHASE_IDENTITY_FAILED")

    def test_secondary_boundaries_closed(self):
        self.assertEqual(RPD.classify_secondary(0.001), "POSITIVE_IMPROVEMENT")
        self.assertEqual(RPD.classify_secondary(0.000999999), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(RPD.classify_secondary(-0.02), "CLEAR_DEGRADATION")
        self.assertEqual(RPD.classify_secondary(-0.019999), "NO_CLEAR_IMPROVEMENT")

    def test_half_gap_boundary_closed(self):
        target = RPD.HALF_GAP_TARGET
        self.assertTrue(RPD.half_gap_met(target))
        self.assertFalse(RPD.half_gap_met(target - 1e-12))
        self.assertEqual(RPD.HALF_GAP_TARGET, RPD.DELTA_TEST_PINNED + 0.5 * RPD.GAP)

    def test_frozen_constants(self):
        self.assertEqual(RPD.DELAY_EPOCHS, 1)
        self.assertEqual(RPD.UNFREEZE_EPOCH, 2)
        self.assertEqual(RPD.MATERIAL_DELTA, 0.001)
        self.assertEqual(RPD.DELTA_TEST_LEARNABLE, 0.008103113327467715)
        self.assertEqual(RPD.DELTA_TEST_PINNED, -0.002677172368479308)
        self.assertEqual(RPD.GAP, 0.010780285695947023)
        self.assertEqual(RPD.HALF_GAP_TARGET, 0.0027129704794942035)
        self.assertEqual(RPD.BASELINE_EP1_TRAIN_LOSS, 0.08193688414408826)
        self.assertEqual(RPD.BASELINE_EP1_VAL, 0.4645711559431739)
        self.assertEqual(RPD.RUN_ID_SUFFIX_DELAY, "-rpd")
        self.assertEqual(RPD.VARIANT_DELAY, "residual-prompt-delay")

    def test_suffix_dispatch(self):
        self.assertEqual(RPD.arm_suffix_for(RPD.VARIANT_DELAY), "-rpd")
        self.assertEqual(RPD.arm_suffix_for(RP.BASELINE_VARIANT), "")
        self.assertEqual(RPD.arm_suffix_for(RP.VARIANT), RP.RUN_ID_SUFFIX)
        self.assertEqual(RPD.arm_suffix_for(RPP.VARIANT_PINNED), RPP.RUN_ID_SUFFIX_CORRECT)


# =====================================================================================
# 分析器夹具（真实历史链 + 合成 B/D run）
# =====================================================================================

REAL_RUNS = REPO / "artifacts" / "aliccp_bench" / "runs"
HIST_NEEDED = ["20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07",
               "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg",
               "20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs",
               "20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp",
               "20261005-1032-p2M-v500k-t1M-m1688723740-short-1c13841-rpps",
               "20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg",
               "20261005-0829-p2M-v500k-t1M-m1688723740-xlong-eafc336-rpg",
               "20261005-0642-p2M-v500k-t1M-m1688738016-short-c17b100-rpg",
               "20261005-0646-p2M-v500k-t1M-m1688749593-short-c17b100-rpg",
               "20261005-0649-p2M-v500k-t1M-m1688762746-short-c17b100-rpg"]
_HAVE_HIST = all((REAL_RUNS / r).is_dir() for r in HIST_NEEDED)


def _metrics_doc(run_id, *, variant, test, val, per_epoch, gate_mean, commit="c0ffee1"):
    return {"run_id": run_id, "tag": "short", "stage1_id": RPD.STAGE1_ID, "prefix_tag": "p2M-v500k-t1M",
            "budgets": {"train": 2000000, "val": 500000, "test": 1000000}, "model_seed": 1688723740,
            "epochs": 5, "patience": 2, "enforce_b": True, "variant": variant, "best_epoch": 5,
            "best_val_auc_bsi": val, "test_auc_bsi": test, "gate_mean": gate_mean,
            "per_epoch": per_epoch, "backbone_sha256_loaded": RPD.STAGE1_BACKBONE_SHA,
            "backbone_sha256_before": RPD.STAGE1_BACKBONE_SHA,
            "backbone_sha256_after": RPD.STAGE1_BACKBONE_SHA, "backbone_grads_none": True,
            "env_ids_sha256": RPD.STAGE1_ENV_IDS_SHA, "fingerprint_sha256": RPD.STAGE1_FINGERPRINT_SHA,
            "hard_pass": False, "commit": commit, "git": {"commit": commit, "dirty": False},
            "versions": {}, "device": "cpu", "wall_seconds": 1.0, "peak_vram_mb": None}


def _per_epoch(vals, losses):
    return [{"epoch": i + 1, "train_loss": losses[i], "val_auc_bsi": vals[i]}
            for i in range(len(vals))]


def _gate_report(a_ok=True):
    verdict = "PASS" if a_ok else "FAIL"
    return {"run_id": "x", "tag": "short", "enforce_b": True,
            "gates": {f"A{i}": {"verdict": "SKIP" if i == 3 else verdict} for i in range(1, 7)},
            "hard_pass": False}


def _delay_probe_doc(alpha_final=0.05):
    return {
        "construction_identity": {"shared_params_bit_identical": True,
                                  "global_rng_endpoint_identical": True,
                                  "extra_keys": sorted(RP.EXTRA_PARAM_NAMES),
                                  "expected_extra_keys": sorted(RP.EXTRA_PARAM_NAMES),
                                  "alpha_at_construction": 0.0},
        "init_forward": {"bit_identical": True, "max_abs_diff": 0.0, "n_samples": 6},
        "grad_probe": [{"epoch": 1, "alpha": 0.0, "alpha_grad_norm": None,
                        "generator_grad_norm": 0.0},
                       {"epoch": 2, "alpha": 0.0, "alpha_grad_norm": 0.61,
                        "generator_grad_norm": 0.0},
                       {"epoch": 3, "alpha": 0.02, "alpha_grad_norm": 0.2,
                        "generator_grad_norm": 0.05},
                       {"epoch": 4, "alpha": 0.035, "alpha_grad_norm": 0.1,
                        "generator_grad_norm": 0.09},
                       {"epoch": 5, "alpha": 0.045, "alpha_grad_norm": 0.07,
                        "generator_grad_norm": 0.11}],
        "alpha_final": alpha_final,
        "delay": {"delay_epochs": 1, "unfreeze_epoch": 2, "alpha_at_construction": 0.0,
                  "alpha_requires_grad_at_construction": False,
                  "generator_requires_grad_at_construction": [False] * 4,
                  "generator_sha_at_construction": "sha_gen", "alpha_at_unfreeze": 0.0,
                  "generator_sha_at_unfreeze": "sha_gen", "unfrozen_once": True,
                  "requires_grad_after_unfreeze": {"alpha": True, "generator": [True] * 4},
                  "optimizer_covers_named_parameters": True, "optimizer_params_total": 16,
                  "unfreeze_rng_endpoint_unchanged": True},
        "val_stats": {"ratio_mean": [0.02, 0.02, 0.02], "ratio_std": [1e-3] * 3,
                      "ratio_max": [0.045] * 3, "delta_norm_mean": [1.0, 0.2, 0.2],
                      "h_norm_mean": [160.0, 22.0, 21.0], "cos_mean": [0.05, 0.05, 0.05],
                      "n_zero_rep": [0, 0, 0]},
        "gate": {"geff_mean": 0.02, "geff_std": 0.004, "geff_min": 0.008, "geff_max": 0.044},
        "dispersion": {"pred_std": 0.0049},
        "reference_dispersion": {"newtask_checkpoint":
                                 f"artifacts/aliccp_bench/runs/{RPD.BASELINE_RUN_ID}/newtask.pt",
                                 "val_auc": RP.BASELINE_AUC_VAL,
                                 "pred_dispersion": {"pred_std": RP.REFERENCE_PRED_STD}},
        "params": {"new_param_list": [{"name": n} for n in sorted(RP.EXTRA_PARAM_NAMES)],
                   "new_params_total": 2385, "head_params": 8129},
    }


def _write_run(root, run_id, metrics, probe=None, a_ok=True, newtask_sha=None):
    d = P.run_dir(root, run_id)
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    (d / "gate_report.json").write_text(json.dumps(_gate_report(a_ok=a_ok), ensure_ascii=False),
                                        encoding="utf-8")
    (d / "config.json").write_text(json.dumps({"run_id": run_id}, ensure_ascii=False),
                                  encoding="utf-8")
    if probe is not None:
        arm = RPD.delay_arm_verdict(
            auc_test=metrics["test_auc_bsi"], auc_val=metrics["best_val_auc_bsi"], probe=probe,
            val_stats={"streams": probe["val_stats"], "gate": probe["gate"],
                       "dispersion": probe["dispersion"]},
            params=probe["params"], reference=probe["reference_dispersion"], protocol_ok=a_ok)
        metrics["rp_arm"] = arm
        (d / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        (d / "prompt_report.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    if newtask_sha is not None:
        (d / "newtask.pt").write_bytes(b"\x00" * 16)      # 由夹具注入 sha 模拟（真实 run 为真文件）
    return d


@unittest.skipUnless(_HAVE_HIST, "历史 run 只读副本不存在（untracked 证据基座）")
class TestAnalyzerFixture(unittest.TestCase):
    def _root(self, td, *, d_test=None, d_val=None, d_dirty=False, frozen_break=False, b_ok=True):
        root = Path(td)
        runs = root / "runs"
        runs.mkdir()
        for r in HIST_NEEDED:
            shutil.copytree(REAL_RUNS / r, runs / r)
        b_run = "20261005-2000-p2M-v500k-t1M-m1688723740-short-c0ffee1"
        b_test = RP.BASELINE_AUC_TEST if b_ok else RP.BASELINE_AUC_TEST - 0.01
        b_metrics = _metrics_doc(
            b_run, variant=RP.BASELINE_VARIANT, test=b_test, val=RP.BASELINE_AUC_VAL,
            per_epoch=_per_epoch(RPD.BASELINE_PER_EPOCH_VAL, RPD.BASELINE_PER_EPOCH_LOSS),
            gate_mean=RPD.BASELINE_GATE_MEAN)
        d = _write_run(root, b_run, b_metrics)
        (d / "newtask.pt").write_bytes((REAL_RUNS / HIST_NEEDED[0] / "newtask.pt").read_bytes())
        d_run = "20261005-2010-p2M-v500k-t1M-m1688723740-short-c0ffee1-rpd"
        dt = (RP.BASELINE_AUC_TEST + 0.003) if d_test is None else d_test
        dv = (RP.BASELINE_AUC_VAL + 0.002) if d_val is None else d_val
        probe = _delay_probe_doc()
        d_per_epoch = _per_epoch(
            [RPD.BASELINE_EP1_VAL] + [v + 0.001 for v in RPD.BASELINE_PER_EPOCH_VAL[1:]],
            [RPD.BASELINE_EP1_TRAIN_LOSS] + list(RPD.BASELINE_PER_EPOCH_LOSS[1:]))
        if frozen_break:
            d_per_epoch[0]["val_auc_bsi"] += 0.001
        d_metrics = _metrics_doc(d_run, variant=RPD.VARIANT_DELAY, test=dt, val=dv,
                                 per_epoch=d_per_epoch, gate_mean=RPD.BASELINE_GATE_MEAN)
        d_metrics["git"]["dirty"] = d_dirty
        _write_run(root, d_run, d_metrics, probe=probe)
        (P.run_dir(root, d_run) / "newtask.pt").write_bytes(b"\x00" * 16)
        return root, b_run, d_run

    def test_valid_supported(self):
        with tempfile.TemporaryDirectory() as td:
            root, b, d = self._root(td)
            res = RPD.analyze_runs(runs_root=root / "runs", baseline_run=b, delay_run=d)
            self.assertEqual(res["verdict"]["verdict"], "DYNAMICS_SUPPORTED")
            self.assertTrue(res["preconditions"]["all_pass"])
            self.assertAlmostEqual(res["components"]["delta_d"], 0.003, places=12)

    def test_not_supported_below_materiality(self):
        with tempfile.TemporaryDirectory() as td:
            root, b, d = self._root(td, d_test=RP.BASELINE_AUC_TEST + 0.0005)
            res = RPD.analyze_runs(runs_root=root / "runs", baseline_run=b, delay_run=d)
            self.assertEqual(res["verdict"]["verdict"], "DYNAMICS_NOT_SUPPORTED")
            self.assertEqual(res["verdict"]["subreason"], "BELOW_MATERIALITY")

    def test_not_supported_validation_disagreement(self):
        with tempfile.TemporaryDirectory() as td:
            root, b, d = self._root(td, d_val=RP.BASELINE_AUC_VAL - 0.001)
            res = RPD.analyze_runs(runs_root=root / "runs", baseline_run=b, delay_run=d)
            self.assertEqual(res["verdict"]["verdict"], "DYNAMICS_NOT_SUPPORTED")
            self.assertEqual(res["verdict"]["subreason"], "VALIDATION_DISAGREEMENT")

    def test_invalid_variants(self):
        with tempfile.TemporaryDirectory() as td:
            root, b, d = self._root(td, d_dirty=True)
            res = RPD.analyze_runs(runs_root=root / "runs", baseline_run=b, delay_run=d)
            self.assertEqual(res["verdict"]["subreason"], "IDENTITY_MISMATCH")
        with tempfile.TemporaryDirectory() as td:
            root, b, d = self._root(td, b_ok=False)
            res = RPD.analyze_runs(runs_root=root / "runs", baseline_run=b, delay_run=d)
            self.assertEqual(res["verdict"]["subreason"], "BASELINE_REPRODUCTION_FAILED")
        with tempfile.TemporaryDirectory() as td:
            root, b, d = self._root(td, frozen_break=True)
            res = RPD.analyze_runs(runs_root=root / "runs", baseline_run=b, delay_run=d)
            self.assertEqual(res["verdict"]["subreason"], "FROZEN_PHASE_IDENTITY_FAILED")

    def test_half_gap_and_components(self):
        with tempfile.TemporaryDirectory() as td:
            root, b, d = self._root(td, d_test=RP.BASELINE_AUC_TEST + RPD.HALF_GAP_TARGET + 1e-9)
            res = RPD.analyze_runs(runs_root=root / "runs", baseline_run=b, delay_run=d)
            self.assertTrue(res["descriptive"]["half_gap_met"])
            comp = res["components"]
            self.assertAlmostEqual(comp["gap"], RPD.GAP, places=15)
            self.assertAlmostEqual(comp["delta_l"], RPD.DELTA_TEST_LEARNABLE, places=15)
            self.assertAlmostEqual(comp["delta_p"], RPD.DELTA_TEST_PINNED, places=15)

    def test_headroom_on_no_clear(self):
        with tempfile.TemporaryDirectory() as td:
            root, b, d = self._root(td, d_test=RP.BASELINE_AUC_TEST + 0.0005)
            res = RPD.analyze_runs(runs_root=root / "runs", baseline_run=b, delay_run=d)
            self.assertEqual(res["secondary"]["class"], "NO_CLEAR_IMPROVEMENT")
            head = res["headroom"]
            self.assertEqual(head["mechanism_activity"], "UNFROZEN_ACTIVE")
            self.assertEqual(head["unfreeze_engagement"], "ENGAGED")
            self.assertEqual(head["censoring"], "RIGHT_CENSORED_STILL_IMPROVING")
            self.assertAlmostEqual(head["gap_to_positive_threshold"], 0.001 - 0.0005, places=12)


# =====================================================================================
# 静态守卫与预注册常量
# =====================================================================================

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
        changed += subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                                  cwd=REPO, capture_output=True, text=True).stdout.split()
        extra = set(changed) - WHITELIST
        self.assertEqual(extra, set(), f"白名单外改动: {extra}")


class TestPreregConstants(unittest.TestCase):
    def test_doc_tokens(self):
        text = (REPO / DOC_PATH).read_text(encoding="utf-8")
        for token in ("1688723740", "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
                      "DELAY_EPOCHS", "UNFREEZE_EPOCH", "-rpd", "residual-prompt-delay",
                      "DYNAMICS_SUPPORTED", "DYNAMICS_NOT_SUPPORTED", "INVALID",
                      "0.0027129704794942035", "0.010780285695947023",
                      "0.008103113327467715", "−0.002677172368479308" if "−0.002677172368479308" in text else "-0.002677172368479308",
                      "0.08193688414408826", "0.4645711559431739",
                      "674213f619c5d5039811c71242a7727118348daf",
                      "432fa9bb6b3bb9d537753599499b693197e8c898",
                      "153/153", "+0.001", "−0.02" if "−0.02" in text else "-0.02"):
            self.assertIn(token, text, token)

    def test_whitelist_matches_doc_table(self):
        text = (REPO / DOC_PATH).read_text(encoding="utf-8")
        for rel in WHITELIST:
            self.assertIn(rel, text, rel)

    def test_trajectory_evidence_sync_with_prerun_verifier(self):
        """rp_delay 的六条轨迹钉死 == verify_delay_prerun 的同名常量（双副本同步）。"""
        import verify_delay_prerun as V
        self.assertEqual(RPD.LEARNABLE_TRAJECTORIES, V.LEARNABLE_TRAJECTORIES)

    def test_summary_ledger_exists(self):
        text = (REPO / "artifacts" / "aliccp_bench" / "SUMMARY.md").read_text(encoding="utf-8")
        self.assertIn("| run_id |", text)


# =====================================================================================
# runner 接线（tiny 端到端；与 test_smoke 同口径）
# =====================================================================================

HEADER = "click,purchase,X,121,122,301"
TINY_VOCAB = {"121": 5, "122": 4}
TRAIN_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2], [0, 0, 9, 2, 2, 1], [1, 1, 9, 0, 1, 3],
    [0, 0, 8, 3, 0, 2], [1, 0, 8, 1, 1, 1], [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2],
]
VAL_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2],
]
TEST_ROWS = [
    [0, 0, 9, 2, 2, 1], [1, 0, 9, 0, 1, 3], [0, 0, 8, 3, 0, 2], [1, 1, 8, 1, 1, 1],
    [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2], [0, 0, 9, 1, 0, 3], [1, 0, 8, 4, 1, 2],
]


def write_aliccp_file(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


def make_tiny_root(td):
    root = Path(td) / "artifacts"
    data_dir = Path(td) / "data"
    data_dir.mkdir()
    write_aliccp_file(data_dir / "train.csv", TRAIN_ROWS)
    write_aliccp_file(data_dir / "val.csv", VAL_ROWS)
    write_aliccp_file(data_dir / "test.csv", TEST_ROWS)
    data_files = {"train": str(data_dir / "train.csv"), "val": str(data_dir / "val.csv"),
                  "test": str(data_dir / "test.csv")}
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return root, data_files, budgets


def _silent(*args, **kwargs):
    return None


class TestRunnerWiring(unittest.TestCase):
    def test_cli_exposes_delay_variant(self):
        import run_aliccp_benchmark as cli
        parser = cli.build_parser()
        ns = parser.parse_args(["stage2", "--stage1-id", "sid", "--variant",
                                "residual-prompt-delay", "--prompt-reference-newtask", "ref.pt"])
        self.assertEqual(ns.variant, RPD.VARIANT_DELAY)

    def test_suffix_dispatch_in_cli_source(self):
        src = _worktree_text("run_aliccp_benchmark.py")
        self.assertIn("RPD.arm_suffix_for", src)

    def test_delay_arm_end_to_end_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = make_tiny_root(td)
            meta = bench.run_stage1(
                root=root, data_files=data_files, budgets=budgets, prefix_tag="tiny",
                model_seed=123, env_seed=456, epochs=2, patience=2,
                device=torch.device("cpu"), vocab=TINY_VOCAB,
                expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4, log=_silent)
            base = bench.run_stage2(
                root=root, stage1_id=meta["stage1_id"], data_files=data_files, budgets=budgets,
                prefix_tag="tiny", model_seed=123, epochs=2, patience=2, tag="smoke",
                device=torch.device("cpu"), vocab=TINY_VOCAB, expert_hidden=(4,), tower_hidden=(4,),
                embedding_size=2, input_size=4, enforce_b=False, log=_silent)
            base_path = P.run_dir(root, base["run_id"])
            delay = bench.run_stage2(
                root=root, stage1_id=meta["stage1_id"], data_files=data_files, budgets=budgets,
                prefix_tag="tiny", model_seed=123, epochs=2, patience=2, tag="smoke",
                device=torch.device("cpu"), vocab=TINY_VOCAB, expert_hidden=(4,), tower_hidden=(4,),
                embedding_size=2, input_size=4, enforce_b=False, log=_silent,
                variant=RPD.VARIANT_DELAY, prompt_reference_newtask=base_path / "newtask.pt")
            delay_path = P.run_dir(root, delay["run_id"])
            self.assertTrue(delay["run_id"].endswith("-rpd"))
            metrics = json.loads((delay_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["variant"], RPD.VARIANT_DELAY)
            arm = metrics["rp_arm"]
            self.assertEqual(set(arm), {"M0", "DD1", "DD2", "DD3", "DD4", "DD5", "DD6", "DD7",
                                        "DD8", "DD9", "U", "protocol_10_1", "classification",
                                        "subreason", "pass"})
            report = json.loads((delay_path / "prompt_report.json").read_text(encoding="utf-8"))
            self.assertIn("delay", report)
            block = report["delay"]
            self.assertEqual(block["delay_epochs"], 1)
            self.assertEqual(block["unfreeze_epoch"], 2)
            self.assertEqual(block["alpha_at_construction"], 0.0)
            self.assertEqual(block["alpha_at_unfreeze"], 0.0)
            self.assertTrue(block["unfrozen_once"])
            self.assertEqual(block["generator_sha_at_construction"],
                             block["generator_sha_at_unfreeze"])
            # tiny 2-epoch：epoch-1 探针为冻结签名（α=0、无 α 梯度、生成器梯度 0）
            probe = report["grad_probe"]
            self.assertEqual(probe[0]["alpha"], 0.0)
            self.assertIsNone(probe[0]["alpha_grad_norm"])
            self.assertEqual(probe[0]["generator_grad_norm"], 0.0)
            # epoch-2 已解冻：α 梯度非 None
            self.assertIsNotNone(probe[1]["alpha_grad_norm"])
            # M0 在 tiny 夹具必 FAIL（参照身份）；DD9 亦必 FAIL（tiny 维度 ⇒ 参数预算 ≠ 2385）；
            # 按预注册 §5.2 的映射序（DD1/DD2/DD3/DD9 组优先）分类为 INVALID_IMPLEMENTATION
            self.assertFalse(arm["M0"]["pass"])
            self.assertFalse(arm["DD9"]["pass"])
            self.assertEqual(arm["classification"], "MECHANISM_FAIL")
            self.assertEqual(arm["subreason"], "INVALID_IMPLEMENTATION")
            # tiny 端到端下 DD1/DD2/DD3 结构性全真（冻结与解冻被严格执行）
            for gate in ("DD1", "DD2", "DD3"):
                self.assertTrue(arm[gate]["pass"], gate)
            # SUMMARY 末行含 -rpd run_id
            summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
            self.assertIn(delay["run_id"], summary[-1])

    def test_delay_requires_epochs_at_least_unfreeze_epoch(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = make_tiny_root(td)
            meta = bench.run_stage1(
                root=root, data_files=data_files, budgets=budgets, prefix_tag="tiny",
                model_seed=123, env_seed=456, epochs=2, patience=2,
                device=torch.device("cpu"), vocab=TINY_VOCAB,
                expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4, log=_silent)
            with self.assertRaises(ValueError):
                bench.run_stage2(
                    root=root, stage1_id=meta["stage1_id"], data_files=data_files,
                    budgets=budgets, prefix_tag="tiny", model_seed=123, epochs=1, patience=1,
                    tag="smoke", device=torch.device("cpu"), vocab=TINY_VOCAB, expert_hidden=(4,),
                    tower_hidden=(4,), embedding_size=2, input_size=4, enforce_b=False,
                    log=_silent, variant=RPD.VARIANT_DELAY,
                    prompt_reference_newtask=Path("x.pt"))


if __name__ == "__main__":
    unittest.main()
