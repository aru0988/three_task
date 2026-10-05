"""α 钉死条件对应消融分支守卫测试（预注册 §3/§4/§5：逐字节钉死 / 重建等式 / PA 门禁真值表 / 判定树边界 / 分析器夹具）。

TDD：本文件先于实现提交编写（先红后绿）。被消融对象 = `exp/aliccp-stage2-residual-prompt-seed-replication`
@ `013e105`（机制 blob `674213f6…`，逐字节移植零适配）；预注册 §3 的适配 A1–A3/A6 之外任何改动都会红：
  * A1：`residual_prompt.py` == 钉死 blob **逐字节**（零文本适配）；
  * A2a：`bench.py` == 钉死 blob 经"import 段 + `evaluate_newtask` 段 + `run_stage2` 段"三段替换的重建等式；
  * A2b：`run_aliccp_benchmark.py` == 钉死 blob 经"import 行 + `--variant` choices 行 + 臂后缀行"三行替换；
  * A3：`test_residual_prompt.py` == 钉死 blob 经"docstring + DOC_PATH + WHITELIST + 1 方法"替换。
CPU 极小、秒级；被钉 commit 不可得时**响亮失败**（不静默跳过）。
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from aliccp_benchmark import bench
from aliccp_benchmark import protocol as P
from aliccp_benchmark import residual_prompt as RP
from aliccp_benchmark import rp_pinned as RPP
from multitaskrec.model import NewTask

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_BASE = "8133d32"
SEED2_IMPL = "013e105"

PREREG_DOC_REL = "docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-alpha-pinned-condition-design.md"
GUARD_TEST_REL = "aliccp_benchmark/tests/test_residual_prompt_pinned.py"
MECHANISM_REL = "aliccp_benchmark/residual_prompt.py"
PORTED_TEST_REL = "aliccp_benchmark/tests/test_residual_prompt.py"
BENCH_REL = "aliccp_benchmark/bench.py"
CLI_REL = "run_aliccp_benchmark.py"
PINNED_MODULE_REL = "aliccp_benchmark/rp_pinned.py"
PRERUN_VERIFIER_REL = "verify_pinned_prerun.py"
POSTRUN_VERIFIER_REL = "verify_rp_pinned.py"

# ---- §3 钉死（被消融对象 @013e105；blob + LF-normalized sha256）----
PIN_MECHANISM_BLOB = "674213f619c5d5039811c71242a7727118348daf"
PIN_MECHANISM_LF_SHA = "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"
PIN_BENCH_BLOB = "bf3647686838157bf5fd7645615faad9f2d7185d"
PIN_CLI_BLOB = "229c25a1c86719fa6fb6b057d6cff3f4120fa053"
PIN_PORTED_TEST_BLOB = "768a477b5c71af32c5c59ec6feb20c29f7873557"

# ---- 新文件 LF sha256 钉死（C2 提交时计算并写死；见预注册 §3）----
# 哈希链无环：本守卫文件钉 module + 前置核验脚本；运行后复核脚本（verify_rp_pinned.py）钉本文件
# （防自指：文件不得钉自身 sha）。
PINNED_MODULE_LF_SHA = "ba23bf473634ff647b482ebf6a92f27ca7f073432bb4ed7a1342b8b11101df9b"
PRERUN_VERIFIER_LF_SHA = "13371ca418f284aedcb27439a9f36415bcf95d630fe431c020bc8e8d66844d2f"

# ---- §5 适配面钉死 ----
CLI_ADAPTED_LINE_PREFIXES = (
    "from aliccp_benchmark import residual_prompt",
    '    p2.add_argument("--variant"',
    "    arm_suffix = ",
)
ADAPTED_BENCH_FUNCTIONS = ("evaluate_newtask", "run_stage2")
ADAPTED_TEST_METHODS = {("TestPreregConstants", "test_prereg_doc_tokens_and_summary_ledger")}
EXPECTED_WHITELIST = {
    PREREG_DOC_REL,
    "aliccp_benchmark/residual_prompt.py",
    "aliccp_benchmark/rp_pinned.py",
    "aliccp_benchmark/bench.py",
    "run_aliccp_benchmark.py",
    "aliccp_benchmark/tests/test_residual_prompt.py",
    "aliccp_benchmark/tests/test_residual_prompt_pinned.py",
    "verify_pinned_prerun.py",
    "verify_rp_pinned.py",
    "artifacts/aliccp_bench/SUMMARY.md",
}


def _read_lf(rel: str) -> str:
    return (REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def _lf_sha(rel: str) -> str:
    return hashlib.sha256((REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_show_text(spec: str) -> str:
    try:
        raw = subprocess.check_output(["git", "show", spec], cwd=REPO_ROOT, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise AssertionError(f"钉死对象不可得（系谱依赖，响亮失败）: {spec}: {exc.stderr.decode()}") from exc
    return raw.replace(b"\r\n", b"\n").decode("utf-8")


def _replace_once(test: unittest.TestCase, text: str, old: str, new: str, what: str) -> str:
    test.assertEqual(text.count(old), 1, f"{what}: 旧文本应恰出现 1 次（实际 {text.count(old)}）")
    return text.replace(old, new)


def _line_starting_with(text: str, prefix: str) -> str:
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise AssertionError(f"期望恰 1 行以 {prefix!r} 开头，实际 {len(lines)} 行")
    return lines[0]


def _module_doc_segment(treesrc: str) -> str:
    tree = ast.parse(treesrc)
    first = tree.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        raise AssertionError("首语句不是 docstring")
    return ast.get_source_segment(treesrc, first)


def _module_function_segment(treesrc: str, name: str) -> str:
    for node in ast.parse(treesrc).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(treesrc, node)
    raise AssertionError(f"未找到模块级函数 {name}")


def _module_import_segment(treesrc: str, imported_name: str) -> str:
    for node in ast.parse(treesrc).body:
        if isinstance(node, ast.ImportFrom) and any(a.name == imported_name for a in node.names):
            return ast.get_source_segment(treesrc, node)
    raise AssertionError(f"未找到含 {imported_name} 的 import 语句")


def _class_method(treesrc: str, cls_name: str, method: str) -> str:
    for node in ast.parse(treesrc).body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method:
                    return ast.get_source_segment(treesrc, member)
    raise AssertionError(f"未找到 {cls_name}.{method}")


def _module_assign_node(treesrc: str, name: str):
    for node in ast.parse(treesrc).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node
    raise AssertionError(f"未找到模块级赋值 {name}")


def _module_assign_literal(treesrc: str, name: str):
    return ast.literal_eval(_module_assign_node(treesrc, name).value)


def _module_assign_set_resolved(treesrc: str, name: str) -> set:
    module_literals = {}
    for node in ast.parse(treesrc).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None):
                    try:
                        module_literals[target.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        pass
    node = _module_assign_node(treesrc, name)
    if not isinstance(node.value, ast.Set):
        raise AssertionError(f"{name} 不是集合字面量")
    resolved = set()
    for element in node.value.elts:
        if isinstance(element, ast.Constant):
            resolved.add(element.value)
        elif isinstance(element, ast.Name):
            resolved.add(module_literals[element.id])
        else:
            raise AssertionError(f"{name} 含不支持的元素: {ast.dump(element)}")
    return resolved


# =====================================================================================
# I2：适配逐字守卫（字节 / 重建等式）
# =====================================================================================

class TestMechanismByteIdentical(unittest.TestCase):
    """A1：机制文件 == 013e105 钉死 blob 逐字节（零文本适配）。"""

    def test_mechanism_byte_identical_and_pinned(self):
        self.assertEqual(_read_lf(MECHANISM_REL), _git_show_text(f"{SEED2_IMPL}:{MECHANISM_REL}"))
        self.assertEqual(_lf_sha(MECHANISM_REL), PIN_MECHANISM_LF_SHA)
        blob = subprocess.check_output(["git", "hash-object", MECHANISM_REL], cwd=REPO_ROOT, text=True).strip()
        self.assertEqual(blob, PIN_MECHANISM_BLOB)
        pinned_blob = subprocess.check_output(["git", "rev-parse", f"{SEED2_IMPL}:{MECHANISM_REL}"],
                                              cwd=REPO_ROOT, text=True).strip()
        self.assertEqual(pinned_blob, PIN_MECHANISM_BLOB)


class TestWiringReconstructEquality(unittest.TestCase):
    """A2a/A2b：接线文件 == 钉死 blob 经文档化段/行替换（逐字节重建等式）。"""

    def test_bench_equals_pinned_with_exactly_documented_segments(self):
        pinned = _git_show_text(f"{SEED2_IMPL}:{BENCH_REL}")
        working = _read_lf(BENCH_REL)
        modified = pinned
        modified = _replace_once(self, modified,
                                 _module_import_segment(modified, "residual_prompt"),
                                 _module_import_segment(working, "residual_prompt"),
                                 "bench import 段替换")
        for fname in ADAPTED_BENCH_FUNCTIONS:
            modified = _replace_once(self, modified,
                                     _module_function_segment(modified, fname),
                                     _module_function_segment(working, fname),
                                     f"bench {fname} 段替换")
        self.assertEqual(working, modified, "bench.py 相对钉死 blob 除文档化 3 段外必须逐字节一致")
        pin_tree, work_tree = ast.parse(pinned), ast.parse(working)
        pin_funcs = {n.name: n for n in pin_tree.body if isinstance(n, ast.FunctionDef)}
        work_funcs = {n.name: n for n in work_tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(pin_funcs), set(work_funcs))
        for name, node in pin_funcs.items():
            if name in ADAPTED_BENCH_FUNCTIONS:
                continue
            self.assertEqual(ast.dump(node, include_attributes=False),
                             ast.dump(work_funcs[name], include_attributes=False),
                             f"bench 模块级函数 {name} 不得改动")

    def test_cli_equals_pinned_with_exactly_documented_lines(self):
        pinned = _git_show_text(f"{SEED2_IMPL}:{CLI_REL}")
        working = _read_lf(CLI_REL)
        modified = pinned
        for prefix in CLI_ADAPTED_LINE_PREFIXES:
            old_line = _line_starting_with(modified, prefix)
            new_line = _line_starting_with(working, prefix)
            self.assertNotEqual(old_line, new_line, f"{prefix!r} 行必须发生文档化适配")
            modified = _replace_once(self, modified, old_line, new_line, f"CLI 行替换 {prefix!r}")
        self.assertEqual(working, modified, "CLI 相对钉死 blob 除文档化 3 行外必须逐字节一致")

    def test_pinned_blobs_available_and_match(self):
        for rel, blob_pin in ((BENCH_REL, PIN_BENCH_BLOB), (CLI_REL, PIN_CLI_BLOB)):
            blob = subprocess.check_output(["git", "rev-parse", f"{SEED2_IMPL}:{rel}"],
                                           cwd=REPO_ROOT, text=True).strip()
            self.assertEqual(blob, blob_pin, rel)


class TestPortedTestReconstructEquality(unittest.TestCase):
    """A3：移植测试文件 == 钉死 blob 经 4 处文档化适配（逐字节重建等式）。"""

    def _rebuild(self, working: str) -> str:
        pinned = _git_show_text(f"{SEED2_IMPL}:{PORTED_TEST_REL}")
        modified = _replace_once(self, pinned, _module_doc_segment(pinned),
                                 _module_doc_segment(working), "模块 docstring 替换")
        for name in ("DOC_PATH", "WHITELIST"):
            old_seg = ast.get_source_segment(modified, _module_assign_node(modified, name))
            new_seg = ast.get_source_segment(working, _module_assign_node(working, name))
            modified = _replace_once(self, modified, old_seg, new_seg, f"{name} 赋值替换")
        for cls_name, method in sorted(ADAPTED_TEST_METHODS):
            old_seg = _class_method(modified, cls_name, method)
            new_seg = _class_method(working, cls_name, method)
            self.assertNotEqual(old_seg, new_seg, f"{cls_name}.{method} 必须发生文档化适配")
            modified = _replace_once(self, modified, old_seg, new_seg, f"{cls_name}.{method} 替换")
        return modified

    def test_ported_test_equals_pinned_blob_with_exactly_documented_edits(self):
        working = _read_lf(PORTED_TEST_REL)
        self.assertEqual(working, self._rebuild(working),
                         "移植测试文件相对钉死 blob 除 4 处文档化适配外必须逐字节一致")

    def test_adapted_module_values_exact(self):
        working = _read_lf(PORTED_TEST_REL)
        self.assertEqual(_module_assign_literal(working, "DOC_PATH"), PREREG_DOC_REL)
        self.assertEqual(_module_assign_set_resolved(working, "WHITELIST"), EXPECTED_WHITELIST)
        pinned = _git_show_text(f"{SEED2_IMPL}:{PORTED_TEST_REL}")
        for name in ("PIN_BLOB", "PIN_SHA256_LF", "PIN_TEST_BLOB", "FROZEN_BASE"):
            self.assertEqual(_module_assign_literal(working, name),
                             _module_assign_literal(pinned, name), name)

    def test_adapted_methods_are_exactly_the_documented_set(self):
        working = _read_lf(PORTED_TEST_REL)
        pinned = _git_show_text(f"{SEED2_IMPL}:{PORTED_TEST_REL}")
        tree_old, tree_new = ast.parse(pinned), ast.parse(working)

        def classes(tree):
            return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

        def funcs(tree):
            return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

        old_classes, new_classes = classes(tree_old), classes(tree_new)
        self.assertEqual(set(old_classes), set(new_classes), "类名集合")
        differences = set()
        for cls_name, old_cls in old_classes.items():
            old_methods = {m.name: m for m in old_cls.body if isinstance(m, ast.FunctionDef)}
            new_methods = {m.name: m for m in new_classes[cls_name].body if isinstance(m, ast.FunctionDef)}
            self.assertEqual(set(old_methods), set(new_methods), f"{cls_name} 方法集合")
            for mname, old_node in old_methods.items():
                if ast.dump(old_node, include_attributes=False) != ast.dump(
                        new_methods[mname], include_attributes=False):
                    differences.add((cls_name, mname))
        self.assertEqual(differences, ADAPTED_TEST_METHODS, "AST 差异集合必须恰为文档化的 1 个方法")
        old_funcs, new_funcs = funcs(tree_old), funcs(tree_new)
        self.assertEqual(set(old_funcs), set(new_funcs), "模块级函数集合")
        for fname, old_node in old_funcs.items():
            self.assertEqual(ast.dump(old_node, include_attributes=False),
                             ast.dump(new_funcs[fname], include_attributes=False),
                             f"模块级函数 {fname} 不得改动")

    def test_pinned_ported_test_blob_available_and_matches(self):
        blob = subprocess.check_output(["git", "rev-parse", f"{SEED2_IMPL}:{PORTED_TEST_REL}"],
                                       cwd=REPO_ROOT, text=True).strip()
        self.assertEqual(blob, PIN_PORTED_TEST_BLOB)


class TestNewFileShaPins(unittest.TestCase):
    def test_pinned_module_lf_sha_pinned(self):
        self.assertNotEqual(PINNED_MODULE_LF_SHA, "TBD_PINNED_MODULE_LF_SHA", "pin 未填写（C2 计算）")
        self.assertEqual(_lf_sha(PINNED_MODULE_REL), PINNED_MODULE_LF_SHA)

    def test_prerun_verifier_lf_sha_pinned(self):
        self.assertNotEqual(PRERUN_VERIFIER_LF_SHA, "TBD_PRERUN_VERIFIER_LF_SHA", "pin 未填写（C2 计算）")
        self.assertEqual(_lf_sha(PRERUN_VERIFIER_REL), PRERUN_VERIFIER_LF_SHA)

    def test_postrun_verifier_pins_guard_and_module(self):
        """哈希链无环核对：运行后复核脚本必须钉住本守卫文件与机制/模块文件（反向不成立）。"""
        text = _read_lf(POSTRUN_VERIFIER_REL)
        self.assertIn("GUARD_TEST_LF_SHA", text)
        self.assertIn("PINNED_MODULE_LF_SHA", text)
        self.assertIn('"aliccp_benchmark/rp_pinned.py": PINNED_MODULE_LF_SHA', text)
        self.assertIn('"aliccp_benchmark/tests/test_residual_prompt_pinned.py": GUARD_TEST_LF_SHA', text)


# =====================================================================================
# §4-I4/I5/I6/I7：错排构造（含 batch-size-1）
# =====================================================================================

class TestDerangementConstruction(unittest.TestCase):
    def test_bijection_and_no_fixed_points(self):
        for n in (2, 3, 7, 100, 2000):
            for batch_index in range(4):
                perm = RPP.derange(n, "train", batch_index)
                self.assertEqual(sorted(perm), list(range(n)), f"n={n} 非双射")
                self.assertEqual(sum(1 for i, v in enumerate(perm) if i == v), 0,
                                 f"n={n} bk={batch_index} 存在不动点")

    def test_degenerate_sizes_are_identity(self):
        self.assertEqual(RPP.derange(0, "val", 0), [])
        self.assertEqual(RPP.derange(1, "val", 0), [0])
        self.assertEqual(RPP.derange(1, "test", 3), [0])

    def test_same_key_same_perm_across_instances(self):
        a = RPP.ConditioningShuffler()
        b = RPP.ConditioningShuffler()
        self.assertEqual(a.perm("train", 0, 100).tolist(), b.perm("train", 0, 100).tolist())

    def test_different_keys_differ_and_seed_frozen(self):
        self.assertEqual(RPP.COND_PERM_SEED, 20261005)
        p00 = RPP.derange(100, "train", 0)
        p01 = RPP.derange(100, "train", 1)
        pv = RPP.derange(100, "val", 0)
        self.assertNotEqual(p00, p01)
        self.assertNotEqual(p00, pv)

    def test_key_conflict_refused(self):
        shuffler = RPP.ConditioningShuffler()
        shuffler.perm("train", 0, 8)
        with self.assertRaises(ValueError):
            shuffler.perm("train", 0, 9)
        self.assertEqual(shuffler.report_counters()["key_conflicts"], 1)

    def test_label_blind_signature(self):
        sig = inspect.signature(RPP.perm_seed_material)
        self.assertEqual(list(sig.parameters), ["split", "batch_index", "n", "base_seed"])


class TestShufflerIsolationAndReport(unittest.TestCase):
    def test_rng_isolation(self):
        torch_state = torch.get_rng_state()
        py_state = random.getstate()
        shuffler = RPP.ConditioningShuffler()
        shuffler.perm("train", 0, 64)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), py_state)

    def test_report_digest_stable_and_coverage(self):
        shuffler = RPP.ConditioningShuffler()
        for split, n_rows in (("train", 12), ("val", 6), ("test", 8)):
            shuffler.perm(split, 0, n_rows)
        report = shuffler.finalize()
        self.assertEqual(report["fixed_points_total"], 0)
        self.assertEqual(report["bijection_failures"], 0)
        self.assertEqual(report["degenerate_identity_batches"], 0)
        self.assertEqual(report["n_perms_total"], 3)
        coverage = RPP.shuffle_coverage_ok(report, budgets={"train": 12, "val": 6, "test": 8},
                                           batch_size=16)
        self.assertTrue(all(v["n_rows_ok"] and v["n_batches_ok"] for v in coverage.values()))

    def test_deck_check_discriminates(self):
        """PG7/DD：真实小 deck ≠ 历史钉死值（判别力）；钉死值本身可被常量引用。"""
        shuffler = RPP.ConditioningShuffler()
        shuffler.perm("train", 0, 12)
        self.assertFalse(RPP.deck_check(shuffler.finalize())["pass"])
        self.assertEqual(RPP.DECK_PIN["total_digest"],
                         "dcf302c1a80ae7e8023345105e559d7a6daaa1ad492b016c39c22705886599d8")
        self.assertEqual(RPP.DECK_PIN["base_seed"], 20261005)


# =====================================================================================
# §4-I4/I8/I9/I10：钉死头语义（α buffer / 等价 / 优化器排除 / batch-size-1）
# =====================================================================================

def _random_inputs(b=6, input_size=8, rep_dim=4, seed=1, norms=(1.0, 10.0, 100.0)):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(b, input_size, generator=gen)
    reps = [torch.randn(b, rep_dim, generator=gen) for _ in norms]
    reps = [r / r.norm(dim=-1, keepdim=True) * n for r, n in zip(reps, norms)]
    env = [torch.randn(rep_dim, generator=gen), torch.randn(rep_dim, generator=gen)]
    return x, reps[0], reps[1:], env


def _make_pinned(seed=0, shuffled=False, input_size=8, rep_dim=4, alpha=RPP.PINNED_ALPHA):
    torch.manual_seed(seed)
    rng_before = torch.get_rng_state()
    builder = RPP.build_pinned_shuffled_newtask if shuffled else RPP.build_pinned_newtask
    head = builder(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=(4, 2),
                   reg_dnn=P.REG_DNN, device=torch.device("cpu"), pinned_alpha=alpha)
    return head, rng_before


def _make_reference(rng_before, input_size=8, rep_dim=4, head_state=None):
    saved = torch.get_rng_state()
    try:
        torch.set_rng_state(rng_before)
        ref = NewTask(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=(4, 2),
                      reg_dnn=P.REG_DNN, device=torch.device("cpu"))
    finally:
        torch.set_rng_state(saved)
    if head_state is not None:
        ref.load_state_dict({k: v for k, v in head_state.items() if k in ref.state_dict()})
    return ref


class TestPinnedHeadSemantics(unittest.TestCase):
    def test_alpha_is_non_parameter_buffer_exact(self):
        head, _ = _make_pinned()
        self.assertFalse(isinstance(head.prompt_gate, torch.nn.Parameter))
        self.assertFalse(head.prompt_gate.requires_grad)
        self.assertEqual(float(head.prompt_gate.detach()), RPP.PINNED_ALPHA)
        self.assertEqual(float(head.prompt_gate.detach()), 0.07101669907569885)

    def test_named_parameters_exclude_gate_and_keys_exact(self):
        head, _ = _make_pinned()
        names = [n for n, _ in head.named_parameters()]
        self.assertNotIn("prompt_gate", names)
        for key in RPP.EXPECTED_TRAINABLE_PROMPT_KEYS:
            self.assertIn(key, names)
        prompt_names = sorted(n for n in names if n.startswith("prompt_"))
        self.assertEqual(prompt_names, sorted(RPP.EXPECTED_TRAINABLE_PROMPT_KEYS))

    def test_state_dict_extra_keys_exact_and_construction_identity(self):
        head, rng_before = _make_pinned(seed=3)
        ref = _make_reference(rng_before)
        ref_sd, var_sd = ref.state_dict(), head.state_dict()
        self.assertEqual(sorted(set(var_sd) - set(ref_sd)), sorted(RPP.EXPECTED_STATE_EXTRA_KEYS))
        self.assertTrue(all(torch.equal(ref_sd[k].cpu(), var_sd[k].cpu()) for k in ref_sd))

    def test_param_report_budget(self):
        """PA8 预算核算必须在真实维度下成立（80/64/[32,32]）——tiny 维度数字不同（212≠2384）。"""
        torch.manual_seed(0)
        head = RPP.build_pinned_newtask(input_size=80, rep_dim=64, tower_dnn_hidden_units=(32, 32),
                                        reg_dnn=P.REG_DNN, device=torch.device("cpu"))
        params = RP.param_report(head)
        self.assertEqual(params["new_params_total"], RPP.EXPECTED_NEW_PARAMS_TOTAL)
        self.assertEqual(params["head_params"], RPP.EXPECTED_HEAD_PARAMS)
        self.assertEqual(RPP.EXPECTED_NEW_PARAMS_TOTAL, 80 * 16 + 16 + 16 * 64 + 64)

    def test_optimizer_excludes_alpha_and_covers_parameters(self):
        head, _ = _make_pinned()
        optimizer = torch.optim.Adam(params=head.parameters(), lr=1e-3)
        opt_params = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertFalse(any(p is head.prompt_gate for p in opt_params))
        self.assertEqual({id(p) for p in opt_params},
                         {id(p) for _, p in head.named_parameters()})

    def test_alpha_unchanged_under_optimizer_steps(self):
        head, _ = _make_pinned(seed=5)
        optimizer = torch.optim.Adam(params=head.parameters(), lr=1e-2)
        x, gen, specs, env = _random_inputs()
        for _ in range(3):
            out = head(x, gen, specs, env)
            loss = ((out - 0.5) ** 2).mean() + head.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertEqual(float(head.prompt_gate.detach()), RPP.PINNED_ALPHA)
        self.assertIsNone(head.prompt_gate.grad)

    def test_pinned_equivalent_to_learnable_head_at_same_alpha(self):
        """等价证明：钉死头 ≡ 学习头（其参数置为同 α、同生成器权重）——逐位。"""
        x, gen, specs, env = _random_inputs()
        pinned, rng_before = _make_pinned(seed=7)
        saved = torch.get_rng_state()
        try:
            torch.set_rng_state(rng_before)
            learnable = RP.ResidualPromptNewTask(input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                                                 reg_dnn=P.REG_DNN, device=torch.device("cpu"))
        finally:
            torch.set_rng_state(saved)
        learnable.load_state_dict(pinned.state_dict(), strict=False)
        learnable.prompt_gate.data.fill_(RPP.PINNED_ALPHA)
        pinned.eval()
        learnable.eval()
        with torch.no_grad():
            out_p = pinned(x, gen, specs, env)
            out_l = learnable(x, gen, specs, env)
        self.assertTrue(torch.equal(out_p, out_l))
        self.assertNotEqual(float(out_p.std()), 0.0)

    def test_zero_alpha_identity_with_reference(self):
        """PA4-i 单元版：α 临时置 0 ⇒ 与同共享权重参照头逐位相等；恢复后逐位不变。"""
        x, gen, specs, env = _random_inputs()
        head, rng_before = _make_pinned(seed=11)
        ref = _make_reference(rng_before, head_state=head.state_dict())
        head.eval()
        ref.eval()
        with torch.no_grad():
            out_pin = head(x, gen, specs, env)
            saved = head.prompt_gate.clone()
            head.prompt_gate.zero_()
            out_zero = head(x, gen, specs, env)
            out_ref = ref(x, gen, specs, env)
            head.prompt_gate.copy_(saved)
        self.assertTrue(torch.equal(out_zero, out_ref))
        self.assertFalse(torch.equal(out_pin, out_ref))       # α≠0 ⇒ 与参照有差（激活）
        self.assertTrue(torch.equal(head.prompt_gate, saved))
        self.assertEqual(float(head.prompt_gate), RPP.PINNED_ALPHA)

    def test_checkpoint_roundtrip_keeps_alpha(self):
        head, _ = _make_pinned(seed=13)
        state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        fresh, _ = _make_pinned(seed=99)
        fresh.load_state_dict(state)
        self.assertEqual(float(fresh.prompt_gate), RPP.PINNED_ALPHA)
        self.assertIn("prompt_gate", state)


class TestPinnedShuffledHeadSemantics(unittest.TestCase):
    def test_missing_cond_perm_refuses(self):
        head, _ = _make_pinned(shuffled=True)
        x, gen, specs, env = _random_inputs()
        with self.assertRaises(ValueError):
            head(x, gen, specs, env)

    def test_identity_perm_equivalent_to_pinned_correct_head(self):
        x, gen, specs, env = _random_inputs()
        shuf, rng_before = _make_pinned(shuffled=True, seed=17)
        saved = torch.get_rng_state()
        try:
            torch.set_rng_state(rng_before)
            correct = RPP.PinnedAlphaResidualPromptNewTask(
                input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN,
                device=torch.device("cpu"))
        finally:
            torch.set_rng_state(saved)
        correct.load_state_dict(shuf.state_dict(), strict=False)
        shuf.eval()
        correct.eval()
        identity = torch.arange(x.shape[0])
        with torch.no_grad():
            out_s = shuf(x, gen, specs, env, cond_perm=identity)
            out_c = correct(x, gen, specs, env)
        self.assertTrue(torch.equal(out_s, out_c))

    def test_non_identity_perm_changes_output_and_bound_holds(self):
        x, gen, specs, env = _random_inputs(b=6)
        head, _ = _make_pinned(shuffled=True, seed=19)
        perm_vals = RPP.derange(6, "train", 0)
        perm = torch.tensor(perm_vals, dtype=torch.long)
        head.eval()
        with torch.no_grad():
            out_s = head(x, gen, specs, env, cond_perm=perm)
            out_c = head(x, gen, specs, env, cond_perm=torch.arange(6))   # 恒等置换 = correct 语义
            deltas, m = head.prompt_deltas(x[perm], [gen, *specs])
        self.assertFalse(torch.equal(out_s, out_c))
        for delta, h in zip(deltas, [gen, *specs]):
            ratio = delta.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-12)
            self.assertLessEqual(float(ratio.max()), abs(RPP.PINNED_ALPHA) + 1e-6)

    def test_batch_size_one_preserved(self):
        """batch-size-1：π 退化恒等（I5）；前向行为与钉死基类逐位同型（squeeze 塌维为继承语义）。

        钉死 `NewTask.forward` 在 B=1 处因 `.squeeze()` 塌维而 IndexError——这是基类冻结语义，三臂
        （baseline/correct/shuffled）一致；本测试锁定"钉死臂不引入任何 B=1 差异"（同型异常 + B=2 正常）。
        """
        self.assertEqual(RPP.derange(1, "train", 0), [0])
        x1, gen1, specs1, env1 = _random_inputs(b=1)
        shuf, _ = _make_pinned(shuffled=True)
        shuf.eval()
        with self.assertRaises(IndexError):                   # 与钉死基类同型（非本分支引入）
            shuf(x1, gen1, specs1, env1, cond_perm=torch.tensor([0], dtype=torch.long))
        x2, gen2, specs2, env2 = _random_inputs(b=2)
        with torch.no_grad():
            out_shuf = shuf(x2, gen2, specs2, env2, cond_perm=torch.tensor([1, 0], dtype=torch.long))
        self.assertEqual(tuple(out_shuf.shape), (2,))
        correct, _ = _make_pinned()
        self.assertEqual(float(correct.prompt_gate), RPP.PINNED_ALPHA)


class TestPinnedAudit(unittest.TestCase):
    def _audit(self, shuffled=False):
        head, rng_before = _make_pinned(shuffled=shuffled, seed=23)
        optimizer = torch.optim.Adam(params=head.parameters(), lr=1e-3)
        shuffler = RPP.ConditioningShuffler() if shuffled else None
        audit = RPP.PinnedPromptAudit(head, rng_before=rng_before, rng_after=torch.get_rng_state(),
                                      input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                                      reg_dnn=P.REG_DNN, device=torch.device("cpu"),
                                      shuffler=shuffler, optimizer=optimizer)
        return head, audit

    def test_pin_record_construction(self):
        head, audit = self._audit()
        pin = audit.pin
        self.assertFalse(pin["is_parameter"])
        self.assertFalse(pin["requires_grad"])
        self.assertFalse(pin["in_named_parameters"])
        self.assertEqual(pin["alpha_at_construction"], RPP.PINNED_ALPHA)
        self.assertTrue(pin["optimizer_excludes_prompt_gate"])
        self.assertTrue(pin["optimizer_covers_named_parameters"])

    def test_init_forward_check_zero_alpha_and_active(self):
        head, audit = self._audit(shuffled=True)
        x, gen, specs, env = _random_inputs()
        audit.init_forward_check(x, gen, specs, env)
        block = audit.init_forward
        self.assertTrue(block["zero_alpha_bit_identical"])
        self.assertEqual(block["zero_alpha_max_abs_diff"], 0.0)
        self.assertTrue(block["alpha_restored_exact"])
        self.assertTrue(block["pinned_differs_from_reference"])
        self.assertGreater(block["pinned_max_abs_diff"], 0.0)
        self.assertEqual(block["cond_perm"], {"split": "train", "batch_index": 0, "n": 6})
        self.assertEqual(float(head.prompt_gate), RPP.PINNED_ALPHA)
        audit.init_forward_check(x, gen, specs, env)          # 幂等
        self.assertEqual(audit.init_forward, block)

    def test_after_backward_records_none_grad_and_pinned_alpha(self):
        head, audit = self._audit()
        x, gen, specs, env = _random_inputs()
        out = head(x, gen, specs, env)
        loss = ((out - 0.5) ** 2).mean()
        loss.backward()
        audit.after_backward(epoch=1, step=0)
        rec = audit.grad_records[-1]
        self.assertIsNone(rec["alpha_grad_norm"])
        self.assertEqual(rec["alpha"], RPP.PINNED_ALPHA)
        self.assertGreater(rec["generator_grad_norm"], 0.0)   # 钉死 α≠0 ⇒ 首步即有生成器梯度
        result = audit.result(float(head.prompt_gate.detach()))
        self.assertEqual(result["pin"]["alpha_after_training"], RPP.PINNED_ALPHA)
        self.assertTrue(result["pin"]["pin_unchanged_after_training"])


# =====================================================================================
# §5.3 I11：PA 门禁真值表（夹具）
# =====================================================================================

def _fixture_probe(**overrides):
    probe = {
        "construction": {"shared_params_bit_identical": True, "global_rng_endpoint_identical": True,
                         "extra_keys": sorted(RPP.EXPECTED_STATE_EXTRA_KEYS),
                         "alpha_at_construction": RPP.PINNED_ALPHA},
        "init_forward": {"zero_alpha_bit_identical": True, "zero_alpha_max_abs_diff": 0.0,
                         "alpha_restored_exact": True, "pinned_max_abs_diff": 0.5,
                         "pinned_differs_from_reference": True},
        "grad_probe": [{"epoch": 1, "alpha": RPP.PINNED_ALPHA, "alpha_grad_norm": None,
                        "generator_grad_norm": 0.1},
                       {"epoch": 2, "alpha": RPP.PINNED_ALPHA, "alpha_grad_norm": None,
                        "generator_grad_norm": 0.05}],
        "alpha_final": RPP.PINNED_ALPHA,
        "pin": {"pinned_alpha": RPP.PINNED_ALPHA, "is_parameter": False, "requires_grad": False,
                "in_named_parameters": False, "alpha_at_construction": RPP.PINNED_ALPHA,
                "alpha_after_training": RPP.PINNED_ALPHA, "pin_unchanged_after_training": True,
                "optimizer_excludes_prompt_gate": True, "optimizer_covers_named_parameters": True,
                "optimizer_params_total": 10513},
    }
    probe.update(overrides)
    return probe


def _fixture_val_stats(**overrides):
    val_stats = {"streams": {"ratio_max": [0.05, 0.05, 0.05], "ratio_mean": [0.02, 0.02, 0.02]},
                 "gate": {"geff_std": 0.001, "geff_max": 0.05, "geff_min": 0.01},
                 "dispersion": {"pred_std": 0.004}}
    val_stats.update(overrides)
    return val_stats


def _fixture_params(**overrides):
    params = {"new_param_list": [{"name": n, "shape": [1], "numel": 1}
                                 for n in RPP.EXPECTED_TRAINABLE_PROMPT_KEYS],
              "new_params_total": RPP.EXPECTED_NEW_PARAMS_TOTAL,
              "head_params": RPP.EXPECTED_HEAD_PARAMS}
    params.update(overrides)
    return params


def _fixture_reference(val_auc=RP.BASELINE_AUC_VAL, pred_std=RP.REFERENCE_PRED_STD):
    return {"val_auc": val_auc, "pred_dispersion": {"pred_std": pred_std}}


class TestPinnedMechanismGates(unittest.TestCase):
    def _gates(self, **kwargs):
        return RPP.pinned_mechanism_gates(
            probe=kwargs.pop("probe", _fixture_probe()),
            val_stats=kwargs.pop("val_stats", _fixture_val_stats()),
            params=kwargs.pop("params", _fixture_params()),
            reference=kwargs.pop("reference", _fixture_reference()), **kwargs)

    def test_all_pass_fixture(self):
        gates = self._gates()
        for name in ("M0", "PA1", "PA2", "PA3", "PA4", "PA5", "PA6", "PA7", "PA8"):
            self.assertTrue(gates[name]["pass"], name)

    def test_each_gate_fail_reachable(self):
        cases = {
            "M0": dict(reference=_fixture_reference(val_auc=0.5)),
            "PA1": dict(probe=_fixture_probe(alpha_final=RPP.PINNED_ALPHA + 1e-9)),
            "PA2": dict(probe=_fixture_probe(pin={**_fixture_probe()["pin"],
                                                 "optimizer_excludes_prompt_gate": False})),
            "PA3": dict(probe=_fixture_probe(construction={**_fixture_probe()["construction"],
                                                           "extra_keys": sorted(RPP.EXPECTED_STATE_EXTRA_KEYS)[:-1]})),
            "PA4": dict(probe=_fixture_probe(init_forward={**_fixture_probe()["init_forward"],
                                                           "zero_alpha_bit_identical": False})),
            "PA5": dict(val_stats=_fixture_val_stats(streams={"ratio_max": [0.09, 0.05, 0.05],
                                                              "ratio_mean": [0.02, 0.02, 0.02]})),
            "PA6": dict(val_stats=_fixture_val_stats(streams={"ratio_max": [0.05, 0.05, 0.05],
                                                              "ratio_mean": [0.02, 0.03, 0.02]})),
            "PA7": dict(val_stats=_fixture_val_stats(dispersion={"pred_std": 0.0001})),
            "PA8": dict(params=_fixture_params(new_params_total=2385)),
        }
        for gate, kwargs in cases.items():
            gates = self._gates(**kwargs)
            self.assertFalse(gates[gate]["pass"], f"{gate} 应可 FAIL")
            self.assertTrue(all(gates[g]["pass"] for g in gates if g != gate),
                            f"{gate} 单独翻转不得连带其它门")

    def test_bound_and_spread_tolerances(self):
        gates = self._gates(val_stats=_fixture_val_stats(
            streams={"ratio_max": [RPP.PINNED_ALPHA + 1e-6, RPP.PINNED_ALPHA, RPP.PINNED_ALPHA],
                     "ratio_mean": [0.02, 0.02, 0.02]}))
        self.assertTrue(gates["PA5"]["pass"])                 # 恰好 +1e-6 为界内（≤）
        gates = self._gates(val_stats=_fixture_val_stats(
            streams={"ratio_max": [RPP.PINNED_ALPHA + 2e-6, RPP.PINNED_ALPHA, RPP.PINNED_ALPHA],
                     "ratio_mean": [0.02, 0.02, 0.02]}))
        self.assertFalse(gates["PA5"]["pass"])


class TestPinnedArmVerdict(unittest.TestCase):
    def _verdict(self, **kwargs):
        args = dict(auc_test=RP.BASELINE_AUC_TEST + 0.01, auc_val=RP.BASELINE_AUC_VAL + 0.01,
                    probe=_fixture_probe(), val_stats=_fixture_val_stats(),
                    params=_fixture_params(), reference=_fixture_reference(),
                    protocol_ok=True, variant=RPP.VARIANT_PINNED)
        args.update(kwargs)
        return RPP.pinned_arm_verdict(**args)

    def test_valid_positive_when_u_pass(self):
        arm = self._verdict()
        self.assertEqual(arm["classification"], "VALID_POSITIVE")
        self.assertTrue(arm["U"]["U1"]["pass"])
        self.assertEqual(arm["variant"], RPP.VARIANT_PINNED)
        self.assertNotIn("shuffle", arm)

    def test_valid_negative_when_u1_fails_history_threshold(self):
        arm = self._verdict(auc_test=RP.BASELINE_AUC_TEST + 0.001)
        self.assertEqual(arm["classification"], "VALID_NEGATIVE")
        self.assertIsNone(arm["subreason"])

    def test_subreason_priority_pin_before_implementation(self):
        probe = _fixture_probe(construction={**_fixture_probe()["construction"],
                                             "shared_params_bit_identical": False})
        arm = self._verdict(probe=probe)
        self.assertEqual(arm["classification"], "MECHANISM_FAIL")
        self.assertEqual(arm["subreason"], "INVALID_IMPLEMENTATION")
        bad_pin = _fixture_probe(pin={**_fixture_probe()["pin"], "is_parameter": True})
        arm = self._verdict(probe=bad_pin, val_stats=_fixture_val_stats(
            streams={"ratio_max": [0.5, 0.5, 0.5], "ratio_mean": [0.4, 0.4, 0.4]}))
        self.assertEqual(arm["subreason"], "PINNED_ALPHA_INVALID")

    def test_norm_cross_stream_collapse_subreasons(self):
        arm = self._verdict(val_stats=_fixture_val_stats(
            streams={"ratio_max": [0.09, 0.05, 0.05], "ratio_mean": [0.02, 0.02, 0.02]}))
        self.assertEqual(arm["subreason"], "NORM_BOUND_VIOLATION")
        arm = self._verdict(val_stats=_fixture_val_stats(
            streams={"ratio_max": [0.05, 0.05, 0.05], "ratio_mean": [0.02, 0.04, 0.02]}))
        self.assertEqual(arm["subreason"], "CROSS_STREAM_INCONSISTENT")
        arm = self._verdict(val_stats=_fixture_val_stats(dispersion={"pred_std": 1e-5}))
        self.assertEqual(arm["subreason"], "PREDICTION_COLLAPSE")

    def test_shuffled_variant_attaches_pg_block_and_overwrites_on_fail(self):
        shuffler = RPP.ConditioningShuffler()
        for split, n_rows in (("train", 12), ("val", 6), ("test", 8)):
            shuffler.perm(split, 0, n_rows)
        report = shuffler.finalize()
        arm = self._verdict(variant=RPP.VARIANT_PINNED_SHUFFLED, shuffle_report=report,
                            budgets={"train": 12, "val": 6, "test": 8}, batch_size=16)
        self.assertIn("shuffle", arm)
        self.assertFalse(arm["shuffle"]["gates"]["PG7"]["pass"])      # 小 deck ≠ 钉死 deck
        self.assertEqual(arm["classification"], "MECHANISM_FAIL")
        self.assertEqual(arm["subreason"], "PERMUTATION_INVALID")
        report2 = dict(report, fixed_points_total=0, total_digest=RPP.DECK_PIN["total_digest"])
        report2["per_split"] = {"train": dict(RPP.DECK_PIN["per_split"]["train"]),
                                "val": dict(RPP.DECK_PIN["per_split"]["val"]),
                                "test": dict(RPP.DECK_PIN["per_split"]["test"])}
        report2["n_perms_total"] = RPP.DECK_PIN["n_perms_total"]
        report2["base_seed"] = RPP.DECK_PIN["base_seed"]
        arm2 = self._verdict(variant=RPP.VARIANT_PINNED_SHUFFLED, shuffle_report=report2,
                             budgets={"train": 2_000_000, "val": 500_000, "test": 1_000_000},
                             batch_size=2000)
        self.assertTrue(arm2["shuffle"]["gates"]["PG7"]["pass"])
        self.assertEqual(arm2["classification"], "VALID_POSITIVE")


# =====================================================================================
# §5.4/§5.5/§5.6 I11：判定树 / 二级分类 / headroom 边界
# =====================================================================================

class TestCausalVerdictBoundaries(unittest.TestCase):
    def test_supported_boundary_closed(self):
        out = RPP.causal_verdict(gap=0.001, delta_val_gap=1e-9)
        self.assertEqual(out["verdict"], "SAMPLE_CONDITION_SUPPORTED")
        out = RPP.causal_verdict(gap=0.001 - 1e-12, delta_val_gap=1e-9)
        self.assertEqual(out["verdict"], "CONDITION_ALIGNMENT_NOT_SUPPORTED")
        self.assertEqual(out["subreason"], "GAP_BELOW_MATERIALITY")

    def test_validation_agreement_required(self):
        out = RPP.causal_verdict(gap=0.01, delta_val_gap=0.0)
        self.assertEqual(out["verdict"], "CONDITION_ALIGNMENT_NOT_SUPPORTED")
        self.assertEqual(out["subreason"], "VALIDATION_DISAGREEMENT")
        self.assertTrue(out["components"]["gap_material_positive"])
        self.assertFalse(out["components"]["validation_agreement"])

    def test_preconditions_priority(self):
        pre = {"IDENTITY_MISMATCH": True, "BASELINE_REPRODUCTION_FAILED": True}
        out = RPP.causal_verdict(gap=0.01, delta_val_gap=0.01, preconditions=pre)
        self.assertEqual(out["verdict"], "INVALID")
        self.assertEqual(out["subreason"], "IDENTITY_MISMATCH")
        out = RPP.causal_verdict(gap=0.01, delta_val_gap=0.01,
                                 preconditions={"DECK_MISMATCH": True, "PERMUTATION_INVALID": True})
        self.assertEqual(out["subreason"], "DECK_MISMATCH")
        for name in RPP.PRECONDITION_PRIORITY:
            out = RPP.causal_verdict(gap=0.01, delta_val_gap=0.01, preconditions={name: True})
            self.assertEqual(out["subreason"], name)

    def test_shuffled_matches_or_exceeds_flag(self):
        out = RPP.causal_verdict(gap=-0.002, delta_val_gap=0.01)
        self.assertTrue(out["components"]["shuffled_matches_or_exceeds_correct"])


class TestSecondaryClassificationAndHeadroom(unittest.TestCase):
    def test_boundaries_closed(self):
        self.assertEqual(RPP.secondary_classification(0.001), "POSITIVE_IMPROVEMENT")
        self.assertEqual(RPP.secondary_classification(0.001 - 1e-12), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(RPP.secondary_classification(-0.02 + 1e-12), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(RPP.secondary_classification(-0.02), "CLEAR_DEGRADATION")

    def test_headroom_not_applicable_outside_no_clear(self):
        out = RPP.headroom_assessment(delta_test=0.01, delta_val=0.01, arm_record={}, prompt_doc={})
        self.assertEqual(out, {"applicable": False})

    def test_headroom_factors(self):
        record = {"best_epoch": 5, "epochs": 5, "per_epoch": [1, 2, 3, 4, 5]}
        doc = {"alpha_final": RPP.PINNED_ALPHA,
               "gate": {"geff_std": 0.001},
               "val_stats": {"ratio_mean": [0.02, 0.02, 0.02]}}
        out = RPP.headroom_assessment(delta_test=0.0005, delta_val=0.0002,
                                      arm_record=record, prompt_doc=doc)
        self.assertTrue(out["applicable"])
        self.assertEqual(out["mechanism_activity"], "PIN_ACTIVE")
        self.assertEqual(out["validation_direction"], "POSITIVE")
        self.assertEqual(out["epoch_trajectory"], "RIGHT_CENSORED_STILL_IMPROVING")
        self.assertEqual(out["gap_to_positive_threshold"], 0.001 - 0.0005)
        self.assertEqual(out["retained_ratio_vs_historical_correct"],
                         0.0005 / RPP.DELTA_TEST_CORRECT)
        early = dict(record, best_epoch=3)
        out = RPP.headroom_assessment(delta_test=0.0005, delta_val=-0.001,
                                      arm_record=early, prompt_doc=doc)
        self.assertEqual(out["epoch_trajectory"], "EARLY_STOPPED")
        self.assertEqual(out["validation_direction"], "NON_POSITIVE")


# =====================================================================================
# §5 I12：分析器夹具（纯 JSON；patch 钉死常量）
# =====================================================================================

def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


FIXTURE_HEAD_BYTES = b"fixture-head-bytes"

BASE_BUDGETS = {"train": 2_000_000, "val": 500_000, "test": 1_000_000}


def _fixture_metrics(*, run_id, variant, test, val, per_epoch_val, gate_mean, stage1_id,
                     rp_arm=None, commit="deadbee"):
    doc = {
        "run_id": run_id, "tag": "short", "stage1_id": stage1_id, "variant": variant,
        "best_epoch": len(per_epoch_val), "best_val_auc_bsi": val, "test_auc_bsi": test,
        "gate_mean": gate_mean, "epochs": 5, "patience": 2, "model_seed": 1688723740,
        "per_epoch": [{"epoch": i + 1, "train_loss": 0.05, "val_auc_bsi": v}
                      for i, v in enumerate(per_epoch_val)],
        "hard_pass": False, "commit": commit, "git": {"commit": commit, "dirty": False},
        "backbone_sha256_loaded": RPP.STAGE1_BACKBONE_SHA,
        "env_ids_sha256": RPP.STAGE1_ENV_IDS_SHA,
        "fingerprint_sha256": RPP.STAGE1_FINGERPRINT_SHA,
    }
    if rp_arm is not None:
        doc["rp_arm"] = rp_arm
        doc["prompt_hidden"] = 16
    return doc


def _fixture_gates():
    gates = {g: {"verdict": "PASS"} for g in ("A1", "A2", "A4", "A5", "A6", "B1", "B2", "B3")}
    gates["A3"] = {"verdict": "SKIP"}
    gates["B4"] = {"verdict": "FAIL"}
    return {"run_id": "x", "tag": "short", "enforce_b": True, "hard_pass": False, "gates": gates}


def _fixture_pinned_probe():
    return _fixture_probe()


def _fixture_pinned_arm(variant=RPP.VARIANT_PINNED, shuffle_report=None, **overrides):
    """真实形态的钉死臂判定（经 `RPP.pinned_arm_verdict` 生成；含 observed 块供独立重推）。"""
    args = dict(auc_test=RP.BASELINE_AUC_TEST + 0.006, auc_val=RP.BASELINE_AUC_VAL + 0.006,
                probe=_fixture_pinned_probe(), val_stats=_fixture_val_stats(),
                params=_fixture_params(), reference=_fixture_reference(),
                protocol_ok=True, variant=variant, shuffle_report=shuffle_report)
    if shuffle_report is not None:
        args.update(budgets=BASE_BUDGETS, batch_size=2000)
    arm = RPP.pinned_arm_verdict(**args)
    arm.update(overrides)
    return arm


def _deck_shaped_shuffle_block():
    """与历史 deck 逐位一致的 shuffle 报告（供夹具走通 PG1–PG7；不训练）。"""
    report = {"base_seed": RPP.DECK_PIN["base_seed"],
              "fixed_points_total": 0, "degenerate_identity_batches": 0,
              "bijection_failures": 0, "regeneration_mismatches": 0,
              "rng_isolation_violations": 0, "key_conflicts": 0,
              "n_perms_total": RPP.DECK_PIN["n_perms_total"],
              "per_split": {k: dict(v) for k, v in RPP.DECK_PIN["per_split"].items()},
              "total_digest": RPP.DECK_PIN["total_digest"], "samples": {}}
    return report


class AnalyzerFixtureBase(unittest.TestCase):
    """构造 B/F_c/F_s + 对照 run 的极小夹具，并 patch 钉死常量；全部 CPU、纯 JSON。"""

    BASE_PE = [0.46, 0.47, 0.48, 0.485, 0.49]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "artifacts"
        self.root.mkdir(parents=True)
        self.baseline_commit = "aaa1111"
        self.fc_commit = "bbb2222"
        self.fs_commit = "ccc3333"
        self.baseline_run = f"20261005-0001-fixture-baseline-{self.baseline_commit}"
        self.fc_run = f"20261005-0002-fixture-pinned-{self.fc_commit}-rpp"
        self.fs_run = f"20261005-0003-fixture-pinnedshuf-{self.fs_commit}-rpps"
        self.context_baseline_run = "ctx-baseline"
        self.context_correct_run = "ctx-correct"
        self.context_shuf_run = "ctx-shuf"
        self.sid = "s1-fixture-seed2"
        self.base_test, self.base_val = 0.5000, 0.4900
        self.fc_test, self.fc_val = self.base_test + 0.0080, self.base_val + 0.0080
        self.fs_test, self.fs_val = self.base_test + 0.0050, self.base_val + 0.0060
        self.correct_test, self.correct_val = self.base_test + 0.0080, self.base_val + 0.0080

    def tearDown(self):
        self._tmp.cleanup()

    def _write_context(self):
        corr_pe = [v + 0.001 for v in self.BASE_PE]
        shuf_pe = [v + 0.0015 for v in self.BASE_PE]
        _write_json(self.root / "runs" / self.context_baseline_run / "metrics.json",
                    _fixture_metrics(run_id=self.context_baseline_run, variant="baseline",
                                     test=self.base_test, val=self.base_val, per_epoch_val=self.BASE_PE,
                                     gate_mean=[0.5, 0.5], stage1_id=self.sid))
        (self.root / "runs" / self.context_baseline_run / "newtask.pt").write_bytes(FIXTURE_HEAD_BYTES)
        _write_json(self.root / "runs" / self.context_correct_run / "metrics.json",
                    _fixture_metrics(run_id=self.context_correct_run, variant="residual-prompt",
                                     test=self.correct_test, val=self.correct_val, per_epoch_val=corr_pe,
                                     gate_mean=[0.4, 0.6], stage1_id=self.sid,
                                     rp_arm={"classification": "VALID_POSITIVE"}))
        _write_json(self.root / "runs" / self.context_correct_run / "prompt_report.json",
                    {"alpha_final": RPP.PINNED_ALPHA})
        _write_json(self.root / "runs" / self.context_shuf_run / "metrics.json",
                    _fixture_metrics(run_id=self.context_shuf_run, variant="residual-prompt-shuffled",
                                     test=self.fs_test,
                                     val=self.base_val + 0.0011, per_epoch_val=shuf_pe,
                                     gate_mean=[0.4, 0.6], stage1_id=self.sid,
                                     rp_arm={"classification": "MECHANISM_FAIL"}))

    def _write_pair(self, *, fc_test=None, fs_test=None, fc_arm=None, fs_arm=None,
                    fs_shuffle=None, fc_pe=None, fs_pe=None):
        fc_test = self.fc_test if fc_test is None else fc_test
        fs_test = self.fs_test if fs_test is None else fs_test
        fc_pe = fc_pe if fc_pe is not None else [v + 0.001 for v in self.BASE_PE]
        fs_pe = fs_pe if fs_pe is not None else [v + 0.0008 for v in self.BASE_PE]
        base = self.root / "runs" / self.baseline_run
        _write_json(base / "metrics.json",
                    _fixture_metrics(run_id=self.baseline_run, variant="baseline",
                                     test=self.base_test, val=self.base_val, per_epoch_val=self.BASE_PE,
                                     gate_mean=[0.5, 0.5], stage1_id=self.sid,
                                     commit=self.baseline_commit))
        _write_json(base / "gate_report.json", _fixture_gates())
        _write_json(base / "config.json", {"budgets": dict(BASE_BUDGETS), "batch_size": 2000,
                                           "model_seed": 1688723740})
        (base / "newtask.pt").write_bytes(FIXTURE_HEAD_BYTES)
        ref_path = self.root / "runs" / self.context_baseline_run / "newtask.pt"
        fc = self.root / "runs" / self.fc_run
        fc_arm = fc_arm or _fixture_pinned_arm()
        _write_json(fc / "metrics.json",
                    _fixture_metrics(run_id=self.fc_run, variant=RPP.VARIANT_PINNED,
                                     test=fc_test, val=self.fc_val, per_epoch_val=fc_pe,
                                     gate_mean=[0.45, 0.55], stage1_id=self.sid,
                                     rp_arm=fc_arm, commit=self.fc_commit))
        fc_gate_doc = _fixture_gates()
        fc_gate_doc["residual_prompt"] = fc_arm
        _write_json(fc / "gate_report.json", fc_gate_doc)
        _write_json(fc / "prompt_report.json",
                    {"alpha_final": RPP.PINNED_ALPHA,
                     "reference_dispersion": {"newtask_checkpoint": str(ref_path)},
                     "val_stats": {"ratio_mean": [0.028, 0.028, 0.028]},
                     "arm": fc_arm, "pin": fc_arm.get("PA1", {}).get("observed", {})})
        fs = self.root / "runs" / self.fs_run
        shuffle = fs_shuffle or _deck_shaped_shuffle_block()
        fs_arm = fs_arm or _fixture_pinned_arm(variant=RPP.VARIANT_PINNED_SHUFFLED,
                                               shuffle_report=shuffle)
        _write_json(fs / "metrics.json",
                    _fixture_metrics(run_id=self.fs_run, variant=RPP.VARIANT_PINNED_SHUFFLED,
                                     test=fs_test, val=self.fs_val, per_epoch_val=fs_pe,
                                     gate_mean=[0.46, 0.54], stage1_id=self.sid,
                                     rp_arm=fs_arm, commit=self.fs_commit))
        fs_gate_doc = _fixture_gates()
        fs_gate_doc["residual_prompt"] = fs_arm
        _write_json(fs / "gate_report.json", fs_gate_doc)
        _write_json(fs / "config.json", {"budgets": dict(BASE_BUDGETS), "batch_size": 2000,
                                         "model_seed": 1688723740, "prompt_hidden": 16})
        _write_json(fs / "prompt_report.json",
                    {"alpha_final": RPP.PINNED_ALPHA,
                     "reference_dispersion": {"newtask_checkpoint": str(ref_path)},
                     "val_stats": {"ratio_mean": [0.028, 0.028, 0.028]},
                     "arm": fs_arm, "pin": fs_arm.get("PA1", {}).get("observed", {}),
                     "shuffle": {"pass": fs_arm.get("shuffle", {}).get("pass"),
                                 "gates": fs_arm.get("shuffle", {}).get("gates"),
                                 "report": shuffle}})
        with (self.root / "SUMMARY.md").open("a", encoding="utf-8") as handle:
            for run in (self.baseline_run, self.fc_run, self.fs_run):
                handle.write(f"| {run} | fixture | short | fixture |\n")

    def _patches(self):
        return mock.patch.multiple(
            RPP,
            STAGE1_ID=self.sid,
            BASELINE_RUN_ID=self.context_baseline_run,
            CORRECT_RUN_ID=self.context_correct_run,
            SHUF_HIST_RUN_ID=self.context_shuf_run,
            BASELINE_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
                             "best_val_auc_bsi": self.base_val, "test_auc_bsi": self.base_test,
                             "gate_mean": [0.5, 0.5], "per_epoch_val": list(self.BASE_PE)},
            CORRECT_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
                            "best_val_auc_bsi": self.correct_val, "test_auc_bsi": self.correct_test,
                            "gate_mean": [0.4, 0.6],
                            "per_epoch_val": [v + 0.001 for v in self.BASE_PE],
                            "variant": "residual-prompt", "classification": "VALID_POSITIVE",
                            "alpha_final": RPP.PINNED_ALPHA},
            SHUF_HIST_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
                              "best_val_auc_bsi": self.base_val + 0.0011,
                              "test_auc_bsi": self.fs_test,
                              "gate_mean": [0.4, 0.6],
                              "per_epoch_val": [v + 0.0015 for v in self.BASE_PE],
                              "variant": "residual-prompt-shuffled",
                              "classification": "MECHANISM_FAIL",
                              "alpha_final": 0.010569889098405838},
            DELTA_TEST_CORRECT=self.correct_test - self.base_test,
            DELTA_VAL_CORRECT=self.correct_val - self.base_val,
            DELTA_TEST_SHUFFLED_HIST=self.fs_test - self.base_test,
            REF_HEAD_SHA=_sha256_file(self.root / "runs" / self.context_baseline_run / "newtask.pt"),
        )

    def _analyze(self):
        with self._patches():
            return RPP.analyze_runs(root=self.root, baseline_run=self.baseline_run,
                                    arm_correct_run=self.fc_run, arm_shuffled_run=self.fs_run)

    def _scenario(self, *, gap, dval_gap, **pair_kwargs):
        self.fs_test = self.fc_test - gap
        self.fs_val = self.fc_val - dval_gap
        self._write_context()
        self._write_pair(**pair_kwargs)
        return self._analyze()


class TestAnalyzerFixtures(AnalyzerFixtureBase):
    def test_supported_scenario(self):
        result = self._scenario(gap=0.003, dval_gap=0.002)
        self.assertEqual(result["verdict"], "SAMPLE_CONDITION_SUPPORTED")
        comp = result["components"]
        self.assertAlmostEqual(comp["gap"], 0.003, places=12)
        self.assertTrue(comp["validation_agreement"])
        self.assertEqual(result["preconditions"], {k: False for k in result["preconditions"]})
        self.assertIn(result["secondary"]["arm_correct"],
                      ("POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION"))
        self.assertIn(result["deck"]["pass"], (True, False))

    def test_not_supported_scenario(self):
        result = self._scenario(gap=0.0005, dval_gap=0.0005)
        self.assertEqual(result["verdict"], "CONDITION_ALIGNMENT_NOT_SUPPORTED")
        self.assertEqual(result["subreason"], "GAP_BELOW_MATERIALITY")

    def test_validation_disagreement_scenario(self):
        result = self._scenario(gap=0.003, dval_gap=-0.001)
        self.assertEqual(result["verdict"], "CONDITION_ALIGNMENT_NOT_SUPPORTED")
        self.assertEqual(result["subreason"], "VALIDATION_DISAGREEMENT")

    def test_deck_mismatch_tamper(self):
        shuffler = RPP.ConditioningShuffler()
        for split, n_rows in (("train", 12), ("val", 6), ("test", 8)):
            shuffler.perm(split, 0, n_rows)
        real_report = shuffler.finalize()
        self.fs_test = self.fc_test - 0.003
        self.fs_val = self.fc_val - 0.002
        self._write_context()
        self._write_pair(fs_shuffle=real_report)
        result = self._analyze()
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "DECK_MISMATCH")

    def test_permutation_counter_tamper(self):
        report = _deck_shaped_shuffle_block()
        report["fixed_points_total"] = 1
        self.fs_test = self.fc_test - 0.003
        self.fs_val = self.fc_val - 0.002
        self._write_context()
        self._write_pair(fs_shuffle=report)
        result = self._analyze()
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "PERMUTATION_INVALID")

    def test_identity_tamper_dirty(self):
        self.fs_test = self.fc_test - 0.003
        self.fs_val = self.fc_val - 0.002
        self._write_context()
        self._write_pair()
        fs_metrics = json.loads((self.root / "runs" / self.fs_run / "metrics.json").read_text(encoding="utf-8"))
        fs_metrics["git"]["dirty"] = True
        _write_json(self.root / "runs" / self.fs_run / "metrics.json", fs_metrics)
        result = self._analyze()
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "IDENTITY_MISMATCH")

    def test_reproduction_tamper(self):
        self.fs_test = self.fc_test - 0.003
        self.fs_val = self.fc_val - 0.002
        self._write_context()
        self._write_pair()
        base_metrics = json.loads((self.root / "runs" / self.baseline_run / "metrics.json").read_text(encoding="utf-8"))
        base_metrics["best_val_auc_bsi"] += 1e-9
        _write_json(self.root / "runs" / self.baseline_run / "metrics.json", base_metrics)
        result = self._analyze()
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "BASELINE_REPRODUCTION_FAILED")

    def test_pa_fail_tamper(self):
        arm = _fixture_pinned_arm()
        arm = json.loads(json.dumps(arm))
        arm["PA1"]["pass"] = False
        self.fs_test = self.fc_test - 0.003
        self.fs_val = self.fc_val - 0.002
        self._write_context()
        self._write_pair(fc_arm=arm)
        result = self._analyze()
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "PINNED_ALPHA_INVALID")

    def test_comparator_tamper_pin_source(self):
        self.fs_test = self.fc_test - 0.003
        self.fs_val = self.fc_val - 0.002
        self._write_context()
        self._write_pair()
        wrong = {"alpha_final": RPP.PINNED_ALPHA + 1e-12}
        _write_json(self.root / "runs" / self.context_correct_run / "prompt_report.json", wrong)
        result = self._analyze()
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "COMPARATOR_MISMATCH")

    def test_secondary_and_headroom_in_no_clear_scenario(self):
        self.fc_test = self.base_test + 0.0005
        self.fc_val = self.base_val + 0.0005
        result = self._scenario(gap=0.0002, dval_gap=0.0002)
        self.assertEqual(result["secondary"]["arm_correct"], "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(result["secondary"]["arm_shuffled"], "NO_CLEAR_IMPROVEMENT")
        self.assertTrue(result["headroom"]["arm_correct"]["applicable"])
        self.assertIn(result["headroom"]["arm_correct"]["mechanism_activity"],
                      ("PIN_ACTIVE", "INACTIVE"))


# =====================================================================================
# 端到端（tiny CPU；含错排分派与 pin 段落盘）
# =====================================================================================

HEADER = "click,purchase,X,121,122,301"
TINY_VOCAB = {"121": 5, "122": 4}

TRAIN_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2], [0, 0, 9, 2, 2, 1], [1, 1, 9, 0, 1, 3],
]
VAL_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
]
TEST_ROWS = [
    [0, 0, 9, 2, 2, 1], [1, 0, 9, 0, 1, 3], [0, 0, 8, 3, 0, 2], [1, 1, 8, 1, 1, 1],
]


def _write_aliccp_file(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


def _make_tiny_root(td: str):
    root = Path(td) / "artifacts"
    data_dir = Path(td) / "data"
    data_dir.mkdir()
    _write_aliccp_file(data_dir / "train.csv", TRAIN_ROWS)
    _write_aliccp_file(data_dir / "val.csv", VAL_ROWS)
    _write_aliccp_file(data_dir / "test.csv", TEST_ROWS)
    data_files = {"train": str(data_dir / "train.csv"), "val": str(data_dir / "val.csv"),
                  "test": str(data_dir / "test.csv")}
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return root, data_files, budgets


def _silent(*args, **kwargs):
    return None


def _run_tiny_stage2(root, data_files, budgets, stage1_id, **kwargs):
    return bench.run_stage2(
        root=root, stage1_id=stage1_id, data_files=data_files, budgets=budgets,
        prefix_tag="tiny", model_seed=123, epochs=2, patience=2, tag="smoke",
        device=torch.device("cpu"), vocab=TINY_VOCAB,
        expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
        enforce_b=False, log=_silent, **kwargs)


class TestPinnedArmEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root, cls.data_files, cls.budgets = _make_tiny_root(cls._tmp.name)
        meta = bench.run_stage1(
            root=cls.root, data_files=cls.data_files, budgets=cls.budgets, prefix_tag="tiny",
            model_seed=123, env_seed=456, epochs=2, patience=2, device=torch.device("cpu"),
            vocab=TINY_VOCAB, expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
            log=_silent)
        cls.stage1_id = meta["stage1_id"]
        base = _run_tiny_stage2(cls.root, cls.data_files, cls.budgets, cls.stage1_id)
        cls.base_path = P.run_dir(cls.root, base["run_id"])
        cls.fc = _run_tiny_stage2(cls.root, cls.data_files, cls.budgets, cls.stage1_id,
                                  variant=RPP.VARIANT_PINNED,
                                  prompt_reference_newtask=cls.base_path / "newtask.pt")
        cls.fc_path = P.run_dir(cls.root, cls.fc["run_id"])
        cls.fs = _run_tiny_stage2(cls.root, cls.data_files, cls.budgets, cls.stage1_id,
                                  variant=RPP.VARIANT_PINNED_SHUFFLED,
                                  prompt_reference_newtask=cls.base_path / "newtask.pt")
        cls.fs_path = P.run_dir(cls.root, cls.fs["run_id"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _report(self, path):
        return json.loads((path / "prompt_report.json").read_text(encoding="utf-8"))

    def test_run_ids_and_variants_recorded(self):
        self.assertTrue(self.fc["run_id"].endswith("-rpp"))
        self.assertTrue(self.fs["run_id"].endswith("-rpps"))
        self.assertEqual(RPP.RUN_ID_SUFFIX_CORRECT, "-rpp")
        self.assertEqual(RPP.RUN_ID_SUFFIX_SHUFFLED, "-rpps")
        for path, variant in ((self.fc_path, RPP.VARIANT_PINNED),
                              (self.fs_path, RPP.VARIANT_PINNED_SHUFFLED)):
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["variant"], variant)
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], variant)

    def test_pin_block_and_alpha_frozen(self):
        for path in (self.fc_path, self.fs_path):
            report = self._report(path)
            pin = report["pin"]
            self.assertEqual(pin["pinned_alpha"], RPP.PINNED_ALPHA)
            self.assertFalse(pin["is_parameter"])
            self.assertFalse(pin["in_named_parameters"])
            self.assertTrue(pin["optimizer_excludes_prompt_gate"])
            self.assertEqual(report["alpha_final"], RPP.PINNED_ALPHA)
            state = torch.load(path / "newtask.pt", map_location="cpu")
            self.assertEqual(float(state["prompt_gate"]), RPP.PINNED_ALPHA)
            arm = report["arm"]
            self.assertEqual(arm["subreason"], "REFERENCE_IDENTITY")   # tiny 夹具 M0 不可达（同先例）
            self.assertEqual(arm["classification"], "MECHANISM_FAIL")
            for gate in ("PA1", "PA2", "PA3", "PA4", "PA5", "PA6", "PA7"):
                self.assertTrue(arm[gate]["pass"], (path.name, gate))
            # PA8 在 tiny 维度下如实失败（预算常量按真实维度 2384/8129 钉死；真实维度单元版见
            # TestPinnedHeadSemantics.test_param_report_budget）
            self.assertFalse(arm["PA8"]["pass"])
            self.assertEqual(arm["PA8"]["observed"]["new_params_total"], 4 * 16 + 16 + 16 * 4 + 4)

    def test_generator_grad_active_from_first_epoch(self):
        report = self._report(self.fc_path)
        self.assertGreater(report["grad_probe"][0]["generator_grad_norm"], 0.0)  # 钉死 α≠0 ⇒ 首步即有
        self.assertIsNone(report["grad_probe"][0]["alpha_grad_norm"])

    def test_shuffled_arm_pg_gates_and_deck_pin(self):
        report = self._report(self.fs_path)
        shuffle = report["shuffle"]
        for gate in ("PG1", "PG2", "PG3", "PG4", "PG5", "PG6"):
            self.assertTrue(shuffle["gates"][gate]["pass"], gate)
        self.assertFalse(shuffle["gates"]["PG7"]["pass"])       # tiny deck ≠ 历史钉死 deck
        self.assertEqual(shuffle["report"]["fixed_points_total"], 0)
        self.assertEqual(shuffle["report"]["rng_isolation_violations"], 0)
        self.assertEqual(report["init_forward"]["cond_perm"], {"split": "train", "batch_index": 0,
                                                               "n": len(TRAIN_ROWS)})
        self.assertTrue(report["init_forward"]["zero_alpha_bit_identical"])

    def test_correct_arm_prompt_report_has_no_shuffle_section(self):
        report = self._report(self.fc_path)
        self.assertNotIn("shuffle", report)
        self.assertIn("pin", report)
        self.assertIn("val_stats", report)
        self.assertIn("cos_mean", report["val_stats"])
        self.assertIn("ratio_max", report["val_stats"])

    def test_summary_rows_appended(self):
        summary = (self.root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
        self.assertIn(self.fs["run_id"], summary[-1])


class TestCliSurface(unittest.TestCase):
    def test_variant_choices_include_pinned(self):
        import run_aliccp_benchmark as cli
        parser = cli.build_parser()
        for variant in (RPP.VARIANT_PINNED, RPP.VARIANT_PINNED_SHUFFLED):
            ns = parser.parse_args(["stage2", "--stage1-id", "sid", "--variant", variant,
                                    "--prompt-reference-newtask", "ref/newtask.pt"])
            self.assertEqual(ns.variant, variant)
        ns = parser.parse_args(["stage2", "--stage1-id", "sid"])
        self.assertEqual(ns.variant, RP.BASELINE_VARIANT)

    def test_suffix_dispatch(self):
        self.assertEqual(RPP.run_id_suffix_for(RP.BASELINE_VARIANT), "")
        self.assertEqual(RPP.run_id_suffix_for(RP.VARIANT), "-rpg")
        self.assertEqual(RPP.run_id_suffix_for(RPP.VARIANT_PINNED), "-rpp")
        self.assertEqual(RPP.run_id_suffix_for(RPP.VARIANT_PINNED_SHUFFLED), "-rpps")

    def test_pinned_requires_reference(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = _make_tiny_root(td)
            with self.assertRaises(ValueError):
                _run_tiny_stage2(root, data_files, budgets, stage1_id="nonexistent",
                                 variant=RPP.VARIANT_PINNED)


# =====================================================================================
# 静态守卫 + 预注册常量
# =====================================================================================

class TestStaticGuards(unittest.TestCase):
    PROTECTED = ["multitaskrec/", "config.py", "aliccp_benchmark/protocol.py",
                 "aliccp_benchmark/metrics.py", "aliccp_benchmark/tests/test_protocol.py",
                 "aliccp_benchmark/tests/test_metrics.py", "aliccp_benchmark/tests/test_smoke.py",
                 "AliCCP_MPTRec.py", "AliCCP_NewTask.py", "CensusIncome_MPTRec.py",
                 "CensusIncome_NewTask.py", "baseline/", "mask/", "analysis/"]

    def test_protected_files_unchanged_vs_base(self):
        diff = subprocess.run(["git", "diff", "--name-only", FROZEN_BASE, "--", *self.PROTECTED],
                              cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        self.assertEqual(diff, "", f"协议/模型文件不得改动: {diff}")

    def test_tracked_changes_subset_of_whitelist(self):
        changed = subprocess.run(["git", "diff", "--name-only", FROZEN_BASE],
                                 cwd=REPO_ROOT, capture_output=True, text=True).stdout.split()
        extra = set(changed) - EXPECTED_WHITELIST
        self.assertEqual(extra, set(), f"白名单外改动: {extra}")


class TestPreregConstants(unittest.TestCase):
    def test_thresholds_and_pins(self):
        self.assertEqual(RPP.PINNED_ALPHA, 0.07101669907569885)
        self.assertEqual(RPP.COND_PERM_SEED, 20261005)
        self.assertEqual(RPP.MATERIAL_DELTA, 0.001)
        self.assertEqual(RPP.SECONDARY_POSITIVE_MIN, 0.001)
        self.assertEqual(RPP.SECONDARY_NEGATIVE_MAX, -0.02)
        self.assertEqual(RPP.EXPECTED_NEW_PARAMS_TOTAL, 2384)
        self.assertEqual(RPP.EXPECTED_HEAD_PARAMS, 8129)
        self.assertEqual(RPP.RATIO_SPREAD_TOL, 1e-6)
        self.assertEqual(RPP.HISTORICAL_U1_THRESHOLD, 0.0055)
        self.assertEqual(RPP.ALL_VARIANTS, ("baseline", "residual-prompt",
                                            "residual-prompt-pinned",
                                            "residual-prompt-pinned-shuffled"))
        self.assertEqual(RPP.DELTA_TEST_CORRECT, 0.008103113327467715)
        self.assertEqual(RPP.DELTA_TEST_SHUFFLED_HIST, 0.0014787611000699474)
        self.assertEqual(RPP.SHUF_HIST_RECORD["alpha_final"], 0.010569889098405838)
        self.assertEqual(RPP.BASELINE_RUN_ID,
                         "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07")
        self.assertEqual(RPP.CORRECT_RUN_ID,
                         "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg")
        self.assertEqual(RPP.SHUF_HIST_RUN_ID,
                         "20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs")

    def test_prereg_doc_tokens_and_summary_ledger(self):
        doc = (REPO_ROOT / PREREG_DOC_REL).read_text(encoding="utf-8")
        for token in ["fbfff09", "013e105", "a40836e", "e931cd7", "10e86dc", "0935257", "28aa1fe",
                      "674213f619c5d5039811c71242a7727118348daf",
                      "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc",
                      "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
                      "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07",
                      "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg",
                      "20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs",
                      "0.07101669907569885", "0.010569889098405838",
                      "6.718774285570287",
                      "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f",
                      "0.0055", "0.001", "2385", "2384", "8129", "-rpp", "-rpps", "20261005",
                      "PA1", "PA8", "PG1", "PG7", "INVALID",
                      "SAMPLE_CONDITION_SUPPORTED", "CONDITION_ALIGNMENT_NOT_SUPPORTED",
                      "GAP_BELOW_MATERIALITY", "VALIDATION_DISAGREEMENT",
                      "PINNED_ALPHA_INVALID", "PERMUTATION_INVALID", "DECK_MISMATCH",
                      "POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION",
                      "MATERIAL_DELTA"]:
            self.assertIn(token, doc)
        summary = (REPO_ROOT / "artifacts" / "aliccp_bench" / "SUMMARY.md").read_text(encoding="utf-8")
        row = [line for line in summary.splitlines() if "b2e17f9" in line and "run_id" not in line]
        self.assertEqual(len(row), 1)
        self.assertIn("0.598839", row[0])
        self.assertIn("0.578153", row[0])


if __name__ == "__main__":
    unittest.main()
