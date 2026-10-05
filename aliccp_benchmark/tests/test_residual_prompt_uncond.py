"""无条件固定条件对照分支守卫测试（预注册 §3/§4/§5：逐字节钉死 / 重建等式 / UA 门禁真值表 / 恒定向量与样本不变性 / 判定树边界 / 分析器夹具）。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-unconditional-control-design.md

TDD：本文件先于实现提交编写（先红后绿）。被消融对象 = `f08ae6e`（机制 blob `674213f6…` 与
rp_pinned blob `432fa9bb…`，逐字节移植零适配）；预注册 §3 的适配 A1–A4/A6 之外任何改动都会红：
  * A1/A2：`residual_prompt.py` / `rp_pinned.py` == 钉死 blob **逐字节**（零文本适配）；
  * A3：`bench.py` == 钉死 blob 经"import 段 + `run_stage2` 段"两段替换的重建等式；
  * A4：`run_aliccp_benchmark.py` == 钉死 blob 经"import 行 + `--variant` choices 行 + 臂后缀行"三行替换；
  * A6：`test_residual_prompt.py` == 钉死 blob 经"docstring + DOC_PATH + WHITELIST + 1 方法"替换。
CPU 极小、秒级；被钉 commit 不可得时**响亮失败**（不静默跳过）。
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
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
from aliccp_benchmark import rp_uncond as RPU
from multitaskrec.model import NewTask

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_BASE = "8133d32"
PREDECESSOR = "f08ae6e"

PREREG_DOC_REL = "docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-unconditional-control-design.md"
GUARD_TEST_REL = "aliccp_benchmark/tests/test_residual_prompt_uncond.py"
MECHANISM_REL = "aliccp_benchmark/residual_prompt.py"
RP_PINNED_REL = "aliccp_benchmark/rp_pinned.py"
UNCOND_MODULE_REL = "aliccp_benchmark/rp_uncond.py"
PORTED_TEST_REL = "aliccp_benchmark/tests/test_residual_prompt.py"
BENCH_REL = "aliccp_benchmark/bench.py"
CLI_REL = "run_aliccp_benchmark.py"
PRERUN_VERIFIER_REL = "verify_uncond_prerun.py"
POSTRUN_VERIFIER_REL = "verify_rp_uncond.py"

# ---- §3 钉死（被消融对象 @f08ae6e；blob + LF-normalized sha256）----
PIN_MECHANISM_BLOB = "674213f619c5d5039811c71242a7727118348daf"
PIN_MECHANISM_LF_SHA = "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"
PIN_RP_PINNED_BLOB = "432fa9bb6b3bb9d537753599499b693197e8c898"
PIN_RP_PINNED_LF_SHA = "ba23bf473634ff647b482ebf6a92f27ca7f073432bb4ed7a1342b8b11101df9b"
PIN_BENCH_BLOB = "b676a7976f9c33cb249fec8789fcbe9b155ca7d1"
PIN_BENCH_LF_SHA = "a02b8ea67d61ffdb5f6df0513c51d534e23653d647bc2663f13b2a1381f786d0"
PIN_CLI_BLOB = "74aaaf05defc650ed80cff00e3f52ff31125c568"
PIN_CLI_LF_SHA = "98c0e5d373508c2c1b4b40cab7c0c400792bf70fd4151bc62f40592efad18fd8"
PIN_PORTED_TEST_BLOB = "2b86d39d8afa3f65fc974bc5bfa482c81e070ff5"
PIN_PORTED_TEST_LF_SHA = "55cef9e58df3443d238eca2464debf8b514a03a5c1edd7abaf17d87023234a51"

# ---- 新文件 LF sha256 钉死（C2 提交时计算并写死；见预注册 §3）----
# 哈希链无环：本守卫文件钉 module + 前置核验脚本；运行后复核脚本（verify_rp_uncond.py）钉本文件
# （防自指：文件不得钉自身 sha）。
UNCOND_MODULE_LF_SHA = "79ab481ad8420634c6bade5b22de78afb6255864ea75b9bc1039209a74e2cf42"
PRERUN_VERIFIER_LF_SHA = "479b12a24e360de94fc2fc53175ca7d35f2eae995e6201b239eb6731b893d9f2"

# ---- §5 适配面钉死 ----
CLI_ADAPTED_LINE_PREFIXES = (
    "from aliccp_benchmark import residual_prompt",
    '    p2.add_argument("--variant"',
    "    arm_suffix = ",
)
ADAPTED_BENCH_FUNCTIONS = ("run_stage2",)
ADAPTED_TEST_METHODS = {("TestPreregConstants", "test_prereg_doc_tokens_and_summary_ledger")}
EXPECTED_WHITELIST = {
    PREREG_DOC_REL,
    "aliccp_benchmark/residual_prompt.py",
    "aliccp_benchmark/rp_pinned.py",
    "aliccp_benchmark/rp_uncond.py",
    "aliccp_benchmark/bench.py",
    "run_aliccp_benchmark.py",
    "aliccp_benchmark/tests/test_residual_prompt.py",
    "aliccp_benchmark/tests/test_residual_prompt_uncond.py",
    "verify_uncond_prerun.py",
    "verify_rp_uncond.py",
    "artifacts/aliccp_bench/SUMMARY.md",
}


def _read_lf(rel: str) -> str:
    return (REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def _lf_sha(rel: str) -> str:
    return hashlib.sha256((REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@contextlib.contextmanager
def single_thread_cpu():
    """单线程 CPU 上下文（预注册 §12 C1b-3）：多线程 GEMM 行分块在个别 batch 大小产生 ≤1 ulp 伪影；
    逐位断言在单线程上下文中执行（运行路径 = CUDA，逐位由运行期 UA5 门禁背书）。"""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


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
# I2/I12：适配逐字守卫（字节 / 重建等式）
# =====================================================================================

class TestMechanismByteIdentical(unittest.TestCase):
    """A1/A2：机制与 rp_pinned 文件 == f08ae6e 钉死 blob 逐字节（零文本适配）。"""

    def test_mechanism_byte_identical_and_pinned(self):
        self.assertEqual(_read_lf(MECHANISM_REL), _git_show_text(f"{PREDECESSOR}:{MECHANISM_REL}"))
        self.assertEqual(_lf_sha(MECHANISM_REL), PIN_MECHANISM_LF_SHA)
        blob = subprocess.check_output(["git", "hash-object", MECHANISM_REL], cwd=REPO_ROOT, text=True).strip()
        self.assertEqual(blob, PIN_MECHANISM_BLOB)

    def test_rp_pinned_byte_identical_and_pinned(self):
        self.assertEqual(_read_lf(RP_PINNED_REL), _git_show_text(f"{PREDECESSOR}:{RP_PINNED_REL}"))
        self.assertEqual(_lf_sha(RP_PINNED_REL), PIN_RP_PINNED_LF_SHA)
        blob = subprocess.check_output(["git", "hash-object", RP_PINNED_REL], cwd=REPO_ROOT, text=True).strip()
        self.assertEqual(blob, PIN_RP_PINNED_BLOB)


class TestWiringReconstructEquality(unittest.TestCase):
    """A3/A4：接线文件 == 钉死 blob 经文档化段/行替换（逐字节重建等式）。"""

    def test_bench_equals_pinned_with_exactly_documented_segments(self):
        pinned = _git_show_text(f"{PREDECESSOR}:{BENCH_REL}")
        working = _read_lf(BENCH_REL)
        modified = pinned
        modified = _replace_once(self, modified,
                                 _module_import_segment(modified, "rp_pinned"),
                                 _module_import_segment(working, "rp_pinned"),
                                 "bench import 段替换")
        for fname in ADAPTED_BENCH_FUNCTIONS:
            modified = _replace_once(self, modified,
                                     _module_function_segment(modified, fname),
                                     _module_function_segment(working, fname),
                                     f"bench {fname} 段替换")
        self.assertEqual(working, modified, "bench.py 相对钉死 blob 除文档化 2 段外必须逐字节一致")
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
        pinned = _git_show_text(f"{PREDECESSOR}:{CLI_REL}")
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
            blob = subprocess.check_output(["git", "rev-parse", f"{PREDECESSOR}:{rel}"],
                                           cwd=REPO_ROOT, text=True).strip()
            self.assertEqual(blob, blob_pin, rel)


class TestPortedTestReconstructEquality(unittest.TestCase):
    """A6：移植测试文件 == 钉死 blob 经 4 处文档化适配（逐字节重建等式）。"""

    def _rebuild(self, working: str) -> str:
        pinned = _git_show_text(f"{PREDECESSOR}:{PORTED_TEST_REL}")
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
        pinned = _git_show_text(f"{PREDECESSOR}:{PORTED_TEST_REL}")
        for name in ("PIN_BLOB", "PIN_SHA256_LF", "PIN_TEST_BLOB", "FROZEN_BASE"):
            self.assertEqual(_module_assign_literal(working, name),
                             _module_assign_literal(pinned, name), name)

    def test_adapted_methods_are_exactly_the_documented_set(self):
        working = _read_lf(PORTED_TEST_REL)
        pinned = _git_show_text(f"{PREDECESSOR}:{PORTED_TEST_REL}")
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

    def test_two_whitelist_copies_synced(self):
        from aliccp_benchmark.tests import test_residual_prompt as ported
        self.assertEqual(set(ported.WHITELIST), EXPECTED_WHITELIST, "两处 WHITELIST 副本必须同步")


class TestNewFileShaPins(unittest.TestCase):
    def test_uncond_module_lf_sha_pinned(self):
        self.assertNotEqual(UNCOND_MODULE_LF_SHA, "TBD_UNCOND_MODULE_LF_SHA", "pin 未填写（C2 计算）")
        self.assertEqual(_lf_sha(UNCOND_MODULE_REL), UNCOND_MODULE_LF_SHA)

    def test_prerun_verifier_lf_sha_pinned(self):
        self.assertNotEqual(PRERUN_VERIFIER_LF_SHA, "TBD_PRERUN_VERIFIER_LF_SHA", "pin 未填写（C2 计算）")
        self.assertEqual(_lf_sha(PRERUN_VERIFIER_REL), PRERUN_VERIFIER_LF_SHA)

    def test_postrun_verifier_pins_guard_and_module(self):
        """哈希链无环核对：运行后复核脚本必须钉住本守卫文件与模块/机制文件（反向不成立）。"""
        text = _read_lf(POSTRUN_VERIFIER_REL)
        self.assertIn("GUARD_TEST_LF_SHA", text)
        self.assertIn("UNCOND_MODULE_LF_SHA", text)
        self.assertIn('"aliccp_benchmark/rp_uncond.py": UNCOND_MODULE_LF_SHA', text)
        self.assertIn('"aliccp_benchmark/tests/test_residual_prompt_uncond.py": GUARD_TEST_LF_SHA', text)


# =====================================================================================
# §2.1 恒定向量构造（I5）
# =====================================================================================

class TestConstVectorConstruction(unittest.TestCase):
    def test_seed_and_hash_frozen(self):
        self.assertEqual(RPU.UNCOND_CONST_SEED, 20261006)
        self.assertEqual(RPU.COND_VECTOR_SHA256,
                         "0d45cc4611cebde9675858f1afc633a91b3e9114d6ddc02899a5d97f34babf58")
        vector = RPU.draw_uncond_condition(80, RPU.UNCOND_CONST_SEED)
        self.assertEqual(RPU.cond_sha256(vector), RPU.COND_VECTOR_SHA256)

    def test_redraw_bit_identical_and_shape(self):
        a = RPU.draw_uncond_condition(80)
        b = RPU.draw_uncond_condition(80)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(tuple(a.shape), (80,))
        self.assertEqual(a.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(a).all()))
        self.assertGreater(float(a.double().norm()), 0.0)

    def test_rng_isolation(self):
        torch_state = torch.get_rng_state()
        py_state = random.getstate()
        RPU.draw_uncond_condition(80)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state), "全局 torch RNG 端点不得改变")
        self.assertEqual(random.getstate(), py_state, "全局 python RNG 状态不得改变")

    def test_different_seed_or_shape_differs(self):
        base = RPU.draw_uncond_condition(80)
        self.assertFalse(torch.equal(base, RPU.draw_uncond_condition(80, RPU.UNCOND_CONST_SEED + 1)))
        self.assertEqual(RPU.draw_uncond_condition(8).shape, (8,))


# =====================================================================================
# §2.2 无条件头语义（I4/I6/I7/I8/I9）
# =====================================================================================

def make_uncond(input_size=8, rep_dim=4, hidden=RP.PROMPT_HIDDEN, seed=0):
    torch.manual_seed(seed)
    rng_before = torch.get_rng_state()
    head = RPU.UnconditionalResidualPromptNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=(4, 2),
        reg_dnn=P.REG_DNN, device=torch.device("cpu"), hidden=hidden)
    rng_after = torch.get_rng_state()
    return head, rng_before, rng_after


def make_pinned(input_size=8, rep_dim=4, hidden=RP.PROMPT_HIDDEN, seed=0):
    torch.manual_seed(seed)
    rng_before = torch.get_rng_state()
    head = RPP.PinnedAlphaResidualPromptNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=(4, 2),
        reg_dnn=P.REG_DNN, device=torch.device("cpu"), hidden=hidden)
    rng_after = torch.get_rng_state()
    return head, rng_before, rng_after


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


class TestUncondHeadSemantics(unittest.TestCase):
    def test_construction_identity_and_extra_keys(self):
        head, rng_before, rng_after = make_uncond()
        ref = build_ref(rng_before)
        ref_sd, var_sd = ref.state_dict(), head.state_dict()
        for key in ref_sd:
            self.assertTrue(torch.equal(ref_sd[key], var_sd[key]), key)
        self.assertEqual(sorted(set(var_sd) - set(ref_sd)), sorted(RPU.EXPECTED_STATE_EXTRA_KEYS))
        saved = torch.get_rng_state()
        try:
            torch.set_rng_state(rng_before)
            NewTask(input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2),
                    reg_dnn=P.REG_DNN, device=torch.device("cpu"))
            endpoint = torch.get_rng_state()
        finally:
            torch.set_rng_state(saved)
        self.assertTrue(torch.equal(endpoint, rng_after))

    def test_alpha_is_non_parameter_buffer_exact(self):
        head, _, _ = make_uncond()
        self.assertFalse(isinstance(head.prompt_gate, torch.nn.Parameter))
        self.assertFalse(bool(head.prompt_gate.requires_grad))
        self.assertEqual(float(head.prompt_gate), RPU.PINNED_ALPHA)
        self.assertNotIn("prompt_gate", dict(head.named_parameters()))
        self.assertNotIn("uncond_condition", dict(head.named_parameters()))

    def test_cond_buffer_matches_frozen_draw(self):
        head, _, _ = make_uncond(input_size=8)
        expected = RPU.draw_uncond_condition(8, RPU.UNCOND_CONST_SEED)
        self.assertTrue(torch.equal(head.uncond_condition.detach().cpu(), expected))
        self.assertEqual(int(head.uncond_condition.numel()), 8)

    def test_direction_invariant_across_inputs_and_batches(self):
        head, _, _ = make_uncond()
        head.eval()
        with single_thread_cpu(), torch.no_grad():
            _, m_ref = head.prompt_deltas(*random_inputs(b=64, seed=5))
            m_ref = m_ref[0]
            for b, seed in ((1, 3), (7, 4), (64, 5)):
                x, reps = random_inputs(b=b, seed=seed)
                _, m = head.prompt_deltas(x, reps)
                self.assertTrue(torch.equal(m, m[0:1].expand_as(m)), "行间必须逐位相同")
                if b == 1:
                    # C1b-3：B≤2 与 B≥6 的 GEMM 分块不同 ⇒ 跨 B 行首为 ≤1 ulp；单线程下 B≥6 逐位
                    self.assertLessEqual(float((m[0] - m_ref).abs().max()), 1e-6)
                else:
                    self.assertTrue(torch.equal(m[0], m_ref), f"B={b} 行首必须与 B=64 逐位相同")
            x, reps = random_inputs(b=6, seed=9)
            _, m = head.prompt_deltas(torch.zeros_like(x), reps)
            self.assertTrue(torch.equal(m[0], m_ref), "零输入下方向必须逐位相同（同 B）")
            _, m_huge = head.prompt_deltas(x * 1e6, reps)
            self.assertTrue(torch.equal(m_huge[0], m_ref), "极端输入下方向必须逐位相同（同 B）")

    def test_delta_rows_share_direction_and_ratio_is_constant(self):
        head, _, _ = make_uncond()
        with single_thread_cpu():
            x, reps = random_inputs(b=5, seed=11, norms=(1.0, 10.0, 100.0))
            deltas, _ = head.prompt_deltas(x, reps)
        alpha = abs(float(head.prompt_gate))
        for delta, h in zip(deltas, reps):
            unit = delta / delta.norm(dim=-1, keepdim=True)
            for row in unit[1:]:
                self.assertLessEqual(float((row - unit[0]).abs().max()), 1e-6, "注入方向逐行一致")
            ratio = delta.norm(dim=-1) / h.norm(dim=-1)
            self.assertLessEqual(float(ratio.max() - ratio.min()), RPU.UNCOND_STD_TOL, "逐行 ratio 恒定")
            self.assertLessEqual(float(ratio.max()), alpha + RP.BOUND_TOL)

    def test_equivalence_to_pinned_head_with_constant_condition(self):
        """I8：把 batch 全行置为常量条件时，U 的 deltas 与同权重钉死 correct 头逐位相等。"""
        uncond, rng_before, _ = make_uncond(seed=7)
        pinned, _, _ = make_pinned(seed=7)            # 同 seed ⇒ 同构造 RNG ⇒ 同权重
        sd_u, sd_p = uncond.state_dict(), pinned.state_dict()
        for key in sd_p:
            self.assertTrue(torch.equal(sd_u[key], sd_p[key]), key)
        x, reps = random_inputs(b=6, seed=13)
        cond_batch = uncond.uncond_condition.detach().unsqueeze(0).expand(x.shape[0], -1)
        deltas_p, _ = pinned.prompt_deltas(cond_batch, reps)
        deltas_u, _ = uncond.prompt_deltas(x, reps)
        for du, dp in zip(deltas_u, deltas_p):
            self.assertTrue(torch.equal(du, dp), "常量条件极限下两臂 deltas 必须逐位相等")

    def test_generator_gradients_valid_under_pinned_alpha(self):
        head, _, _ = make_uncond(seed=5)
        x, reps = random_inputs(b=6, seed=17)
        env = [torch.randn(4, generator=torch.Generator().manual_seed(7)) for _ in range(2)]
        out = head(x, reps[0], list(reps[1:]), env)
        loss = out.sum()
        loss.backward()
        for name, param in head.prompt_generator.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(bool(torch.isfinite(param.grad).all()), name)
            self.assertGreater(float(param.grad.detach().norm()), 0.0, f"{name} 梯度必须非零")

    def test_zero_alpha_identity_with_reference(self):
        head, rng_before, _ = make_uncond()
        ref = build_ref(rng_before)
        ref.load_state_dict({k: v for k, v in head.state_dict().items() if k in ref.state_dict()})
        x, reps = random_inputs(b=6, seed=19)
        env = [torch.randn(4, generator=torch.Generator().manual_seed(7)) for _ in range(2)]
        saved = head.prompt_gate.detach().clone()
        try:
            head.prompt_gate.zero_()
            out_zero = head(x, reps[0], list(reps[1:]), env)
            out_ref = ref(x, reps[0], list(reps[1:]), env)
            self.assertTrue(torch.equal(out_zero, out_ref), "零 α 下必须与参照头逐位相等")
        finally:
            head.prompt_gate.copy_(saved)
        self.assertEqual(float(head.prompt_gate), RPU.PINNED_ALPHA)
        out_pinned = head(x, reps[0], list(reps[1:]), env)
        self.assertFalse(torch.equal(out_pinned, out_ref), "钉死 α 下必须与参照不同（激活）")

    def test_optimizer_excludes_buffers_and_covers_parameters(self):
        head, _, _ = make_uncond()
        optimizer = torch.optim.Adam(params=head.parameters(), lr=1e-4)
        opt_params = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertFalse(any(p is head.prompt_gate for p in opt_params))
        self.assertFalse(any(p is head.uncond_condition for p in opt_params))
        self.assertEqual({id(p) for p in opt_params}, {id(p) for _, p in head.named_parameters()})

    def test_param_report_budget_real_dimensions(self):
        torch.manual_seed(0)
        head = RPU.UnconditionalResidualPromptNewTask(
            input_size=80, rep_dim=64, tower_dnn_hidden_units=(32, 32),
            reg_dnn=P.REG_DNN, device=torch.device("cpu"))
        report = RP.param_report(head)
        names = sorted(item["name"] for item in report["new_param_list"])
        self.assertEqual(names, sorted(RPU.EXPECTED_TRAINABLE_PROMPT_KEYS))
        self.assertEqual(report["new_params_total"], RPU.EXPECTED_NEW_PARAMS_TOTAL)
        self.assertEqual(report["head_params"], RPU.EXPECTED_HEAD_PARAMS)
        self.assertEqual(int(head.uncond_condition.numel()), RPU.EXPECTED_COND_NUMEL)

    def test_checkpoint_roundtrip_keeps_alpha_and_cond(self):
        head, _, _ = make_uncond()
        state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        other, _, _ = make_uncond(seed=99)
        other.load_state_dict(state)
        self.assertEqual(float(other.prompt_gate), RPU.PINNED_ALPHA)
        self.assertTrue(torch.equal(other.uncond_condition, head.uncond_condition))

    def test_dimension_mismatch_refuses_to_emit_numbers(self):
        head, _, _ = make_uncond()
        x, reps = random_inputs(b=4)
        bad = [torch.randn(4, 7)]
        with self.assertRaises(ValueError):
            head.prompt_deltas(x, bad)


# =====================================================================================
# §5.2 UA1–UA11 门禁（运行期形态夹具 + 判定真值表）
# =====================================================================================

def _fixture_probe(*, regen=True, cond_sha=None, rows_ok=True, zero_ok=True,
                   ratio_spread=0.0, grads=(0.1, 0.2), uncond_keys=True):
    cond_sha = cond_sha or RPU.COND_VECTOR_SHA256
    extra_keys = sorted(RPU.EXPECTED_STATE_EXTRA_KEYS if uncond_keys
                        else RPP.EXPECTED_STATE_EXTRA_KEYS)
    probe = {
        "construction": {
            "shared_params_bit_identical": True,
            "global_rng_endpoint_identical": True,
            "extra_keys": extra_keys,
            "expected_extra_keys": extra_keys,
            "alpha_at_construction": RPU.PINNED_ALPHA,
        },
        "init_forward": {
            "zero_alpha_bit_identical": True, "zero_alpha_max_abs_diff": 0.0,
            "pinned_max_abs_diff": 0.001, "pinned_differs_from_reference": True,
            "alpha_restored_exact": True, "alpha_after_check": RPU.PINNED_ALPHA,
            "n_samples": 12,
        },
        "grad_probe": [{"epoch": e + 1, "alpha": RPU.PINNED_ALPHA, "alpha_grad_norm": None,
                        "generator_grad_norm": g} for e, g in enumerate(grads)],
        "alpha_final": RPU.PINNED_ALPHA,
        "pin": {
            "pinned_alpha": RPU.PINNED_ALPHA, "is_parameter": False, "requires_grad": False,
            "in_named_parameters": False, "alpha_at_construction": RPU.PINNED_ALPHA,
            "alpha_after_training": RPU.PINNED_ALPHA, "pin_unchanged_after_training": True,
            "optimizer_excludes_prompt_gate": True,
            "optimizer_covers_named_parameters": True, "optimizer_params_total": 16,
            "alpha_numel": 1,
        },
        "uncond": {
            "const_seed": RPU.UNCOND_CONST_SEED, "cond_sha256": cond_sha, "cond_numel": 80,
            "cond_dim": 80, "cond_stats": {"mean": 0.0, "std": 1.0, "min": -2.0, "max": 2.0,
                                           "norm": 8.0},
            "regen_bit_identical": regen, "rng_isolation_ok": True,
            "invariance": {
                "direction_bit_identical_across_batch_rows": rows_ok,
                "direction_max_abs_diff_across_rows": 0.0 if rows_ok else 1e-6,
                "direction_bit_identical_under_zero_input": zero_ok,
                "direction_max_abs_diff_zero_input": 0.0 if zero_ok else 1e-6,
                "ratio_rows_max_minus_min": ratio_spread,
                "n_rows": 2000,
            },
        },
    }
    return probe


def _fixture_val_stats(*, geff_std=1e-12, ratio_std=1e-12, ratio_mean=(0.02, 0.02, 0.02),
                       ratio_max=(0.03, 0.03, 0.03), pred_std=0.005):
    return {
        "streams": {"ratio_mean": list(ratio_mean), "ratio_std": [ratio_std] * 3,
                    "ratio_max": list(ratio_max), "delta_norm_mean": [0.001] * 3,
                    "h_norm_mean": [0.05] * 3, "cos_mean": [0.0] * 3, "n_zero_rep": [0, 0, 0]},
        "gate": {"geff_mean": 0.02, "geff_std": geff_std, "geff_min": 0.02, "geff_max": 0.03},
        "count": 500000,
        "dispersion": {"pred_std": pred_std},
    }


def _fixture_params():
    return {"new_param_list": [{"name": n, "shape": [1], "numel": 1}
                               for n in RPU.EXPECTED_TRAINABLE_PROMPT_KEYS],
            "new_params_total": RPU.EXPECTED_NEW_PARAMS_TOTAL,
            "head_params": RPU.EXPECTED_HEAD_PARAMS, "total_params": 10513,
            "new_ratio_of_head": RPU.EXPECTED_NEW_PARAMS_TOTAL / RPU.EXPECTED_HEAD_PARAMS}


def _fixture_reference(*, val_auc=RP.BASELINE_AUC_VAL, pred_std=RP.REFERENCE_PRED_STD):
    return {"val_auc": val_auc, "pred_dispersion": {"pred_std": pred_std}}


class TestUncondMechanismGates(unittest.TestCase):
    def test_all_pass_fixture(self):
        gates = RPU.uncond_mechanism_gates(probe=_fixture_probe(), val_stats=_fixture_val_stats(),
                                           params=_fixture_params(), reference=_fixture_reference())
        for name in ("M0",) + tuple(f"UA{i}" for i in range(1, 12)):
            self.assertTrue(gates[name]["pass"], name)

    def test_each_gate_fail_reachable(self):
        cases = {
            "UA1": dict(probe_mod=lambda p: p["pin"].update({"is_parameter": True})),
            "UA2": dict(probe_mod=lambda p: p["pin"].update({"optimizer_excludes_prompt_gate": False})),
            "UA3": dict(probe_mod=lambda p: p["construction"].update({"extra_keys": ["x"]})),
            "UA4": dict(probe_mod=lambda p: p["init_forward"].update({"zero_alpha_bit_identical": False})),
            "UA5": dict(probe_mod=lambda p: p["uncond"]["invariance"].update(
                {"direction_bit_identical_across_batch_rows": False})),
            "UA6": dict(val_mod=lambda v: v["streams"].update({"ratio_max": [0.09, 0.03, 0.03]})),
            "UA7": dict(val_mod=lambda v: v["streams"].update({"ratio_mean": [0.02, 0.03, 0.02]})),
            "UA8": dict(val_mod=lambda v: v["gate"].update({"geff_std": 1e-5})),
            "UA9": dict(val_mod=lambda v: v["dispersion"].update({"pred_std": 0.0001})),
            "UA10": dict(params_mod=lambda p: p.update({"new_params_total": 1})),
            "UA11": dict(probe_mod=lambda p: p.update(
                {"grad_probe": [{"epoch": 1, "alpha": RPU.PINNED_ALPHA, "alpha_grad_norm": None,
                                 "generator_grad_norm": 0.0}]})),
        }
        for gate, mods in cases.items():
            probe = _fixture_probe()
            val = _fixture_val_stats()
            params = _fixture_params()
            if "probe_mod" in mods:
                mods["probe_mod"](probe)
            if "val_mod" in mods:
                mods["val_mod"](val)
            if "params_mod" in mods:
                mods["params_mod"](params)
            gates = RPU.uncond_mechanism_gates(probe=probe, val_stats=val, params=params,
                                               reference=_fixture_reference())
            self.assertFalse(gates[gate]["pass"], gate)

    def test_m0_and_boundaries(self):
        gates = RPU.uncond_mechanism_gates(probe=_fixture_probe(), val_stats=_fixture_val_stats(),
                                           params=_fixture_params(),
                                           reference=_fixture_reference(val_auc=RP.BASELINE_AUC_VAL + 1e-6))
        self.assertFalse(gates["M0"]["pass"])
        ok = RPU.uncond_mechanism_gates(probe=_fixture_probe(),
                                        val_stats=_fixture_val_stats(geff_std=RPU.UNCOND_STD_TOL),
                                        params=_fixture_params(), reference=_fixture_reference())
        self.assertTrue(ok["UA8"]["pass"], "边界（≤）为闭")

    def test_arm_verdict_mapping_and_labels(self):
        probe = _fixture_probe()
        arm = RPU.uncond_arm_verdict(auc_test=RP.BASELINE_AUC_TEST + 0.006,
                                     auc_val=RP.BASELINE_AUC_VAL + 0.006, probe=probe,
                                     val_stats=_fixture_val_stats(), params=_fixture_params(),
                                     reference=_fixture_reference(), protocol_ok=True,
                                     variant=RPU.VARIANT_UNCOND)
        self.assertEqual(arm["classification"], "VALID_POSITIVE")
        probe = _fixture_probe()
        probe["pin"]["is_parameter"] = True
        arm = RPU.uncond_arm_verdict(auc_test=RP.BASELINE_AUC_TEST, auc_val=RP.BASELINE_AUC_VAL,
                                     probe=probe, val_stats=_fixture_val_stats(),
                                     params=_fixture_params(), reference=_fixture_reference(),
                                     protocol_ok=True, variant=RPU.VARIANT_UNCOND)
        self.assertEqual((arm["classification"], arm["subreason"]),
                         ("MECHANISM_FAIL", "UNCONDITION_PIN_INVALID"))
        probe = _fixture_probe(rows_ok=False)
        arm = RPU.uncond_arm_verdict(auc_test=RP.BASELINE_AUC_TEST, auc_val=RP.BASELINE_AUC_VAL,
                                     probe=probe, val_stats=_fixture_val_stats(),
                                     params=_fixture_params(), reference=_fixture_reference(),
                                     protocol_ok=True, variant=RPU.VARIANT_UNCOND)
        self.assertEqual((arm["classification"], arm["subreason"]),
                         ("MECHANISM_FAIL", "NOT_TRULY_UNCONDITIONAL"))
        probe = _fixture_probe()
        arm = RPU.uncond_arm_verdict(auc_test=RP.BASELINE_AUC_TEST, auc_val=RP.BASELINE_AUC_VAL,
                                     probe=probe, val_stats=_fixture_val_stats(geff_std=1e-5),
                                     params=_fixture_params(), reference=_fixture_reference(),
                                     protocol_ok=True, variant=RPU.VARIANT_UNCOND)
        self.assertEqual((arm["classification"], arm["subreason"]),
                         ("MECHANISM_FAIL", "UNCONDITIONAL_GATE_NOT_CONSTANT"))
        probe = _fixture_probe()
        arm = RPU.uncond_arm_verdict(auc_test=RP.BASELINE_AUC_TEST, auc_val=RP.BASELINE_AUC_VAL,
                                     probe=probe, val_stats=_fixture_val_stats(),
                                     params=_fixture_params(), reference=_fixture_reference(),
                                     protocol_ok=True, variant=RPU.VARIANT_UNCOND)
        self.assertEqual((arm["classification"], arm["subreason"]), ("VALID_NEGATIVE", None))


# =====================================================================================
# §5.4/§5.5 判定树与二级分类边界（I10）
# =====================================================================================

class TestCausalVerdictBoundaries(unittest.TestCase):
    def test_supported_boundary_closed(self):
        out = RPU.causal_verdict(gap_u=RPU.MATERIAL_DELTA, delta_val_gap_u=1e-9, g_u=0.0005)
        self.assertEqual(out["verdict"], "SAMPLE_CONDITIONING_SUPPORTED")
        self.assertEqual(out["subreason"], "CONDITIONING_ESSENTIAL")
        out = RPU.causal_verdict(gap_u=0.002, delta_val_gap_u=1e-9, g_u=0.006)
        self.assertEqual(out["subreason"], "PARTIAL_CONDITIONING_CONTRIBUTION")

    def test_below_materiality(self):
        out = RPU.causal_verdict(gap_u=RPU.MATERIAL_DELTA - 1e-12, delta_val_gap_u=0.001, g_u=0.0)
        self.assertEqual(out["verdict"], "SAMPLE_CONDITIONING_NOT_SUPPORTED")
        self.assertEqual(out["subreason"], "GAP_BELOW_MATERIALITY")
        out_neg = RPU.causal_verdict(gap_u=-1e-4, delta_val_gap_u=0.001, g_u=0.0)
        self.assertTrue(out_neg["components"]["unconditional_matches_or_exceeds_conditioned"])

    def test_validation_agreement_required(self):
        out = RPU.causal_verdict(gap_u=0.003, delta_val_gap_u=0.0, g_u=0.006)
        self.assertEqual(out["verdict"], "SAMPLE_CONDITIONING_NOT_SUPPORTED")
        self.assertEqual(out["subreason"], "VALIDATION_DISAGREEMENT")

    def test_preconditions_priority(self):
        order = ["IDENTITY_MISMATCH", "BASELINE_REPRODUCTION_FAILED", "CROSS_EXPERIMENT_REPLAY_FAILED",
                 "PROTOCOL_INVALID", "PINNED_ALPHA_INVALID", "UNCONDITION_INVALID",
                 "MECHANISM_FAIL", "COMPARATOR_MISMATCH", "CONSTANT_VECTOR_MISMATCH"]
        pre = {name: False for name in order}
        pre["COMPARATOR_MISMATCH"] = True
        pre["IDENTITY_MISMATCH"] = True
        out = RPU.causal_verdict(gap_u=0.003, delta_val_gap_u=0.002, g_u=0.006, preconditions=pre)
        self.assertEqual((out["verdict"], out["subreason"]), ("INVALID", "IDENTITY_MISMATCH"))
        pre = {name: False for name in order}
        pre["CROSS_EXPERIMENT_REPLAY_FAILED"] = True
        out = RPU.causal_verdict(gap_u=0.003, delta_val_gap_u=0.002, g_u=0.006, preconditions=pre)
        self.assertEqual((out["verdict"], out["subreason"]), ("INVALID", "CROSS_EXPERIMENT_REPLAY_FAILED"))


class TestSecondaryClassificationAndHeadroom(unittest.TestCase):
    def test_boundaries_closed(self):
        self.assertEqual(RPU.secondary_classification(0.001), "POSITIVE_IMPROVEMENT")
        self.assertEqual(RPU.secondary_classification(-0.02), "CLEAR_DEGRADATION")
        self.assertEqual(RPU.secondary_classification(0.000999), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(RPU.secondary_classification(-0.019999), "NO_CLEAR_IMPROVEMENT")

    def test_headroom_not_applicable_outside_no_clear(self):
        out = RPU.headroom_assessment(delta_test=0.002, delta_val=0.001,
                                      arm_record={"best_epoch": 5, "epochs": 5, "per_epoch": [1] * 5},
                                      prompt_doc={})
        self.assertFalse(out["applicable"])

    def test_headroom_factors(self):
        record = {"best_epoch": 5, "epochs": 5, "per_epoch": [1, 2, 3, 4, 5]}
        doc = {"alpha_final": RPU.PINNED_ALPHA,
               "gate": {"geff_std": 1e-12, "geff_max": 0.03},
               "val_stats": {"ratio_mean": [0.02, 0.02, 0.02]}}
        out = RPU.headroom_assessment(delta_test=0.0005, delta_val=0.0002,
                                      arm_record=record, prompt_doc=doc)
        self.assertTrue(out["applicable"])
        self.assertEqual(out["mechanism_activity"], "PIN_ACTIVE")
        self.assertEqual(out["validation_direction"], "POSITIVE")
        self.assertEqual(out["epoch_trajectory"], "RIGHT_CENSORED_STILL_IMPROVING")
        self.assertEqual(out["gap_to_positive_threshold"], 0.001 - 0.0005)
        self.assertEqual(out["retained_ratio_vs_historical_correct"],
                         0.0005 / RPU.DELTA_TEST_CORRECT)
        early = dict(record, best_epoch=3)
        out = RPU.headroom_assessment(delta_test=0.0005, delta_val=-0.001,
                                      arm_record=early, prompt_doc=doc)
        self.assertEqual(out["epoch_trajectory"], "EARLY_STOPPED")
        self.assertEqual(out["validation_direction"], "NON_POSITIVE")


# =====================================================================================
# §5 I11：分析器夹具（纯 JSON；patch 钉死常量）
# =====================================================================================

def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


FIXTURE_HEAD_BYTES = b"fixture-head-bytes"


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


def _fixture_uncond_arm(**overrides):
    """真实形态的无条件臂判定（经 `RPU.uncond_arm_verdict` 生成；含 observed 块供独立重推）。"""
    args = dict(auc_test=RP.BASELINE_AUC_TEST + 0.006, auc_val=RP.BASELINE_AUC_VAL + 0.006,
                probe=_fixture_probe(uncond_keys=True), val_stats=_fixture_val_stats(),
                params=_fixture_params(), reference=_fixture_reference(), protocol_ok=True,
                variant=RPU.VARIANT_UNCOND)
    arm = RPU.uncond_arm_verdict(**args)
    arm.update(overrides)
    return arm


def _fixture_pinned_arm(**overrides):
    args = dict(auc_test=RP.BASELINE_AUC_TEST + 0.006, auc_val=RP.BASELINE_AUC_VAL + 0.006,
                probe=_fixture_probe(uncond_keys=False), val_stats=_fixture_val_stats(),
                params=_fixture_params(), reference=_fixture_reference(), protocol_ok=True,
                variant=RPP.VARIANT_PINNED)
    arm = RPP.pinned_arm_verdict(**args)
    arm.update(overrides)
    return arm


def _fixture_prompt_doc(*, arm, ref_path, uncond_keys=False, regen=True, cond_sha=None,
                        rows_ok=True):
    probe = _fixture_probe(uncond_keys=uncond_keys, regen=regen, cond_sha=cond_sha, rows_ok=rows_ok)
    doc = {"alpha_final": RPU.PINNED_ALPHA,
           "reference_dispersion": {"newtask_checkpoint": str(ref_path),
                                    "val_auc": RP.BASELINE_AUC_VAL,
                                    "pred_dispersion": {"pred_std": RP.REFERENCE_PRED_STD}},
           "val_stats": {"ratio_mean": [0.028, 0.028, 0.028], "ratio_std": [1e-12] * 3,
                         "ratio_max": [0.046, 0.046, 0.046]},
           "gate": {"geff_mean": 0.028, "geff_std": 1e-12, "geff_min": 0.01, "geff_max": 0.046},
           "dispersion": {"pred_std": 0.0045},
           "construction_identity": probe["construction"],
           "init_forward": probe["init_forward"],
           "grad_probe": probe["grad_probe"],
           "params": _fixture_params(),
           "arm": arm, "pin": dict(probe["pin"])}
    return doc


class AnalyzerFixtureBase(unittest.TestCase):
    """构造 B/C_p/U + 五个对照 run 的极小夹具，并 patch 钉死常量；全部 CPU、纯 JSON。"""

    BASE_PE = [0.46, 0.47, 0.48, 0.485, 0.49]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "artifacts"
        self.root.mkdir(parents=True)
        self.baseline_commit = "aaa1111"
        self.cp_commit = "bbb2222"
        self.up_commit = "ccc3333"
        self.baseline_run = f"20261005-0001-fixture-baseline-{self.baseline_commit}"
        self.cp_run = f"20261005-0002-fixture-pinned-{self.cp_commit}-rpp"
        self.up_run = f"20261005-0003-fixture-uncond-{self.up_commit}-rpu"
        self.context_baseline_run = "ctx-baseline"
        self.context_correct_run = "ctx-correct"
        self.context_shuf_run = "ctx-shuf"
        self.context_pinned_correct_run = "ctx-pinned-correct"
        self.context_pinned_shuffled_run = "ctx-pinned-shuffled"
        self.sid = "s1-fixture-seed2"
        self.base_test, self.base_val = 0.5000, 0.4900
        self.cp_test, self.cp_val = self.base_test + 0.0050, self.base_val + 0.0050
        self.up_test, self.up_val = self.base_test + 0.0030, self.base_val + 0.0040
        self.correct_test, self.correct_val = self.base_test + 0.0080, self.base_val + 0.0080
        self.pc_test, self.pc_val = self.cp_test, self.cp_val
        self.ps_test, self.ps_val = self.base_test + 0.0035, self.base_val + 0.0045
        self.cond_sha = RPU.cond_sha256(RPU.draw_uncond_condition(80))

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
                    {"alpha_final": RPU.PINNED_ALPHA})
        _write_json(self.root / "runs" / self.context_shuf_run / "metrics.json",
                    _fixture_metrics(run_id=self.context_shuf_run, variant="residual-prompt-shuffled",
                                     test=self.ps_test, val=self.ps_val, per_epoch_val=shuf_pe,
                                     gate_mean=[0.4, 0.6], stage1_id=self.sid,
                                     rp_arm={"classification": "MECHANISM_FAIL"}))
        pc_pe = [v + 0.0004 for v in self.BASE_PE]
        _write_json(self.root / "runs" / self.context_pinned_correct_run / "metrics.json",
                    _fixture_metrics(run_id=self.context_pinned_correct_run,
                                     variant=RPP.VARIANT_PINNED, test=self.pc_test, val=self.pc_val,
                                     per_epoch_val=pc_pe, gate_mean=[0.45, 0.55], stage1_id=self.sid,
                                     rp_arm={"classification": "VALID_NEGATIVE"}))
        (self.root / "runs" / self.context_pinned_correct_run / "newtask.pt").write_bytes(FIXTURE_HEAD_BYTES)
        _write_json(self.root / "runs" / self.context_pinned_correct_run / "prompt_report.json",
                    {"alpha_final": RPU.PINNED_ALPHA})
        _write_json(self.root / "runs" / self.context_pinned_shuffled_run / "metrics.json",
                    _fixture_metrics(run_id=self.context_pinned_shuffled_run,
                                     variant=RPP.VARIANT_PINNED_SHUFFLED, test=self.ps_test,
                                     val=self.ps_val, per_epoch_val=shuf_pe,
                                     gate_mean=[0.46, 0.54], stage1_id=self.sid,
                                     rp_arm={"classification": "VALID_NEGATIVE"}))
        _write_json(self.root / "runs" / self.context_pinned_shuffled_run / "prompt_report.json",
                    {"alpha_final": RPU.PINNED_ALPHA})

    def _write_triple(self, *, cp_test=None, up_test=None, cp_arm=None, up_arm=None,
                      cp_pe=None, up_pe=None, cond_sha=None, regen=True, rows_ok=True):
        cp_test = self.cp_test if cp_test is None else cp_test
        up_test = self.up_test if up_test is None else up_test
        cp_pe = cp_pe if cp_pe is not None else [v + 0.0004 for v in self.BASE_PE]
        up_pe = up_pe if up_pe is not None else [v + 0.0004 for v in self.BASE_PE]
        base = self.root / "runs" / self.baseline_run
        _write_json(base / "metrics.json",
                    _fixture_metrics(run_id=self.baseline_run, variant="baseline",
                                     test=self.base_test, val=self.base_val, per_epoch_val=self.BASE_PE,
                                     gate_mean=[0.5, 0.5], stage1_id=self.sid,
                                     commit=self.baseline_commit))
        _write_json(base / "gate_report.json", _fixture_gates())
        (base / "newtask.pt").write_bytes(FIXTURE_HEAD_BYTES)
        ref_path = self.root / "runs" / self.context_baseline_run / "newtask.pt"
        cp = self.root / "runs" / self.cp_run
        cp_arm = cp_arm or _fixture_pinned_arm()
        _write_json(cp / "metrics.json",
                    _fixture_metrics(run_id=self.cp_run, variant=RPP.VARIANT_PINNED,
                                     test=cp_test, val=self.cp_val, per_epoch_val=cp_pe,
                                     gate_mean=[0.45, 0.55], stage1_id=self.sid,
                                     rp_arm=cp_arm, commit=self.cp_commit))
        _write_json(cp / "gate_report.json", _fixture_gates())
        cp_doc = _fixture_prompt_doc(arm=cp_arm, ref_path=ref_path, uncond_keys=False)
        _write_json(cp / "prompt_report.json", cp_doc)
        (cp / "newtask.pt").write_bytes(FIXTURE_HEAD_BYTES)
        up = self.root / "runs" / self.up_run
        probe = _fixture_probe(regen=regen, cond_sha=cond_sha or self.cond_sha, rows_ok=rows_ok)
        up_arm = up_arm or RPU.uncond_arm_verdict(
            auc_test=up_test, auc_val=self.up_val, probe=probe, val_stats=_fixture_val_stats(),
            params=_fixture_params(), reference=_fixture_reference(), protocol_ok=True,
            variant=RPU.VARIANT_UNCOND)
        _write_json(up / "metrics.json",
                    _fixture_metrics(run_id=self.up_run, variant=RPU.VARIANT_UNCOND,
                                     test=up_test, val=self.up_val, per_epoch_val=up_pe,
                                     gate_mean=[0.47, 0.53], stage1_id=self.sid,
                                     rp_arm=up_arm, commit=self.up_commit))
        _write_json(up / "gate_report.json", _fixture_gates())
        up_doc = _fixture_prompt_doc(arm=up_arm, ref_path=ref_path, uncond_keys=True,
                                     regen=regen, cond_sha=cond_sha or self.cond_sha,
                                     rows_ok=rows_ok)
        up_doc["uncond"] = probe["uncond"]
        _write_json(up / "prompt_report.json", up_doc)
        (up / "newtask.pt").write_bytes(FIXTURE_HEAD_BYTES)
        with (self.root / "SUMMARY.md").open("a", encoding="utf-8") as handle:
            for run in (self.baseline_run, self.cp_run, self.up_run):
                handle.write(f"| {run} | fixture | short | fixture |\n")

    def _patches(self):
        return mock.patch.multiple(
            RPU,
            STAGE1_ID=self.sid,
            BASELINE_RUN_ID=self.context_baseline_run,
            CORRECT_RUN_ID=self.context_correct_run,
            SHUF_HIST_RUN_ID=self.context_shuf_run,
            PINNED_CORRECT_RUN_ID=self.context_pinned_correct_run,
            PINNED_SHUFFLED_RUN_ID=self.context_pinned_shuffled_run,
            BASELINE_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
                             "best_val_auc_bsi": self.base_val, "test_auc_bsi": self.base_test,
                             "gate_mean": [0.5, 0.5], "per_epoch_val": list(self.BASE_PE)},
            CORRECT_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
                            "best_val_auc_bsi": self.correct_val, "test_auc_bsi": self.correct_test,
                            "gate_mean": [0.4, 0.6],
                            "per_epoch_val": [v + 0.001 for v in self.BASE_PE],
                            "variant": "residual-prompt", "classification": "VALID_POSITIVE",
                            "alpha_final": RPU.PINNED_ALPHA},
            SHUF_HIST_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
                              "best_val_auc_bsi": self.ps_val,
                              "test_auc_bsi": self.ps_test,
                              "gate_mean": [0.4, 0.6],
                              "per_epoch_val": [v + 0.0015 for v in self.BASE_PE],
                              "variant": "residual-prompt-shuffled",
                              "classification": "MECHANISM_FAIL",
                              "alpha_final": 0.010569889098405838},
            PINNED_CORRECT_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2,
                                   "model_seed": 1688723740,
                                   "best_val_auc_bsi": self.pc_val, "test_auc_bsi": self.pc_test,
                                   "gate_mean": [0.45, 0.55],
                                   "per_epoch_val": [v + 0.0004 for v in self.BASE_PE],
                                   "variant": RPP.VARIANT_PINNED, "classification": "VALID_NEGATIVE",
                                   "alpha_final": RPU.PINNED_ALPHA},
            PINNED_SHUFFLED_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2,
                                    "model_seed": 1688723740,
                                    "best_val_auc_bsi": self.ps_val, "test_auc_bsi": self.ps_test,
                                    "gate_mean": [0.46, 0.54],
                                    "per_epoch_val": [v + 0.0015 for v in self.BASE_PE],
                                    "variant": RPP.VARIANT_PINNED_SHUFFLED,
                                    "classification": "VALID_NEGATIVE",
                                    "alpha_final": RPU.PINNED_ALPHA},
            PINNED_CORRECT_NEWTASK_SHA=_sha256_file(
                self.root / "runs" / self.context_pinned_correct_run / "newtask.pt"),
            COND_VECTOR_SHA256=self.cond_sha,
            REF_HEAD_SHA=_sha256_file(self.root / "runs" / self.context_baseline_run / "newtask.pt"),
        )

    def _analyze(self):
        with self._patches():
            return RPU.analyze_runs(root=self.root, baseline_run=self.baseline_run,
                                    arm_cond_run=self.cp_run, arm_uncond_run=self.up_run)

    def _scenario(self, *, gap_u, dval_gap_u, **triple_kwargs):
        self.up_test = self.cp_test - gap_u
        self.up_val = self.cp_val - dval_gap_u
        self._write_context()
        self._write_triple(**triple_kwargs)
        return self._analyze()


class TestAnalyzerFixtures(AnalyzerFixtureBase):
    def test_supported_scenario(self):
        result = self._scenario(gap_u=0.003, dval_gap_u=0.002)
        self.assertEqual(result["verdict"], "SAMPLE_CONDITIONING_SUPPORTED")
        comp = result["components"]
        self.assertAlmostEqual(comp["gap_u"], 0.003, places=12)
        self.assertTrue(comp["validation_agreement"])
        self.assertEqual(result["preconditions"], {k: False for k in result["preconditions"]})
        self.assertIn(result["secondary"]["arm_cond"],
                      ("POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION"))

    def test_not_supported_scenario(self):
        result = self._scenario(gap_u=0.0005, dval_gap_u=0.0005)
        self.assertEqual(result["verdict"], "SAMPLE_CONDITIONING_NOT_SUPPORTED")
        self.assertEqual(result["subreason"], "GAP_BELOW_MATERIALITY")

    def test_validation_disagreement_scenario(self):
        result = self._scenario(gap_u=0.003, dval_gap_u=-0.001)
        self.assertEqual(result["verdict"], "SAMPLE_CONDITIONING_NOT_SUPPORTED")
        self.assertEqual(result["subreason"], "VALIDATION_DISAGREEMENT")

    def test_identity_tamper_dirty(self):
        self._scenario(gap_u=0.003, dval_gap_u=0.002)
        up_metrics = json.loads((self.root / "runs" / self.up_run / "metrics.json").read_text(encoding="utf-8"))
        up_metrics["git"]["dirty"] = True
        _write_json(self.root / "runs" / self.up_run / "metrics.json", up_metrics)
        result = self._analyze()
        self.assertEqual((result["verdict"], result["subreason"]), ("INVALID", "IDENTITY_MISMATCH"))

    def test_reproduction_tamper(self):
        self._scenario(gap_u=0.003, dval_gap_u=0.002)
        base_metrics = json.loads((self.root / "runs" / self.baseline_run / "metrics.json").read_text(encoding="utf-8"))
        base_metrics["best_val_auc_bsi"] += 1e-9
        _write_json(self.root / "runs" / self.baseline_run / "metrics.json", base_metrics)
        result = self._analyze()
        self.assertEqual((result["verdict"], result["subreason"]),
                         ("INVALID", "BASELINE_REPRODUCTION_FAILED"))

    def test_replay_cp_tamper(self):
        self._scenario(gap_u=0.003, dval_gap_u=0.002)
        cp_metrics = json.loads((self.root / "runs" / self.cp_run / "metrics.json").read_text(encoding="utf-8"))
        cp_metrics["per_epoch"][0]["val_auc_bsi"] += 1e-9
        _write_json(self.root / "runs" / self.cp_run / "metrics.json", cp_metrics)
        result = self._analyze()
        self.assertEqual((result["verdict"], result["subreason"]),
                         ("INVALID", "CROSS_EXPERIMENT_REPLAY_FAILED"))

    def test_uncond_gate_tamper(self):
        arm = _fixture_uncond_arm()
        arm = json.loads(json.dumps(arm))
        arm["UA1"]["pass"] = False
        arm["UA1"]["observed"]["is_parameter"] = True
        self._scenario(gap_u=0.003, dval_gap_u=0.002, up_arm=arm)
        result = self._analyze()
        self.assertEqual((result["verdict"], result["subreason"]), ("INVALID", "UNCONDITION_INVALID"))

    def test_cond_vector_tamper(self):
        self._scenario(gap_u=0.003, dval_gap_u=0.002)     # 先写正常 fixture
        up_doc = json.loads((self.root / "runs" / self.up_run / "prompt_report.json").read_text(encoding="utf-8"))
        up_doc["uncond"]["cond_sha256"] = "0" * 64
        _write_json(self.root / "runs" / self.up_run / "prompt_report.json", up_doc)
        result = self._analyze()
        self.assertEqual((result["verdict"], result["subreason"]),
                         ("INVALID", "CONSTANT_VECTOR_MISMATCH"))

    def test_comparator_tamper_pin_source(self):
        self._scenario(gap_u=0.003, dval_gap_u=0.002)
        wrong = {"alpha_final": RPU.PINNED_ALPHA + 1e-12}
        _write_json(self.root / "runs" / self.context_correct_run / "prompt_report.json", wrong)
        result = self._analyze()
        self.assertEqual((result["verdict"], result["subreason"]), ("INVALID", "COMPARATOR_MISMATCH"))

    def test_replayed_g_c_check(self):
        result = self._scenario(gap_u=0.003, dval_gap_u=0.002)
        checks = result["comparator"]["checks"]
        self.assertTrue(checks["replayed_g_c"])

    def test_secondary_and_headroom_in_no_clear_scenario(self):
        self.cp_test = self.base_test + 0.0005
        self.cp_val = self.base_val + 0.0005
        self.pc_test, self.pc_val = self.cp_test, self.cp_val   # 重放靶随 cp 同步（保持 REP 一致）
        self.up_test = self.cp_test - 0.0004
        self.up_val = self.cp_val - 0.0004
        self._write_context()
        self._write_triple()
        result = self._analyze()
        self.assertEqual(result["secondary"]["arm_cond"], "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(result["secondary"]["arm_uncond"], "NO_CLEAR_IMPROVEMENT")
        self.assertTrue(result["headroom"]["arm_cond"]["applicable"])
        self.assertTrue(result["headroom"]["arm_uncond"]["applicable"])
        self.assertEqual(result["headroom"]["arm_uncond"]["mechanism_activity"], "PIN_ACTIVE")


# =====================================================================================
# 端到端（tiny 夹具；不构成性能证据）
# =====================================================================================

TINY_HEADER = "click,purchase,X,121,122,301"
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


def _write_aliccp_file(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(TINY_HEADER + "\n")
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


def _silent(_message: str) -> None:
    return None


def _run_tiny_stage2(root, data_files, budgets, stage1_id, **kwargs):
    return bench.run_stage2(
        root=root, stage1_id=stage1_id, data_files=data_files, budgets=budgets,
        prefix_tag="tiny", model_seed=123, epochs=2, patience=2, tag="smoke",
        device=torch.device("cpu"), vocab=TINY_VOCAB,
        expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
        enforce_b=False, log=_silent, **kwargs)


class TestUncondArmEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)                      # C1b-3：单线程上下文（I6 逐位断言）
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
        cls.cp = _run_tiny_stage2(cls.root, cls.data_files, cls.budgets, cls.stage1_id,
                                  variant=RPP.VARIANT_PINNED,
                                  prompt_reference_newtask=cls.base_path / "newtask.pt")
        cls.cp_path = P.run_dir(cls.root, cls.cp["run_id"])
        cls.up = _run_tiny_stage2(cls.root, cls.data_files, cls.budgets, cls.stage1_id,
                                  variant=RPU.VARIANT_UNCOND,
                                  prompt_reference_newtask=cls.base_path / "newtask.pt")
        cls.up_path = P.run_dir(cls.root, cls.up["run_id"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        torch.set_num_threads(cls._prev_threads)

    def _report(self, path):
        return json.loads((path / "prompt_report.json").read_text(encoding="utf-8"))

    def test_run_ids_and_variants_recorded(self):
        self.assertTrue(self.cp["run_id"].endswith("-rpp"))
        self.assertTrue(self.up["run_id"].endswith("-rpu"))
        self.assertEqual(RPU.RUN_ID_SUFFIX_UNCOND, "-rpu")
        for path, variant in ((self.cp_path, RPP.VARIANT_PINNED),
                              (self.up_path, RPU.VARIANT_UNCOND)):
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["variant"], variant)
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], variant)

    def test_pin_block_and_alpha_frozen(self):
        for path in (self.cp_path, self.up_path):
            report = self._report(path)
            pin = report["pin"]
            self.assertEqual(pin["pinned_alpha"], RPU.PINNED_ALPHA)
            self.assertFalse(pin["is_parameter"])
            self.assertFalse(pin["in_named_parameters"])
            self.assertTrue(pin["optimizer_excludes_prompt_gate"])
            self.assertEqual(report["alpha_final"], RPU.PINNED_ALPHA)
            state = torch.load(path / "newtask.pt", map_location="cpu")
            self.assertEqual(float(state["prompt_gate"]), RPU.PINNED_ALPHA)

    def test_uncond_block_and_invariance(self):
        report = self._report(self.up_path)
        uncond = report["uncond"]
        self.assertEqual(uncond["const_seed"], RPU.UNCOND_CONST_SEED)
        self.assertTrue(uncond["regen_bit_identical"])
        self.assertTrue(uncond["rng_isolation_ok"])
        invariance = uncond["invariance"]
        self.assertTrue(invariance["direction_bit_identical_across_batch_rows"])
        self.assertTrue(invariance["direction_bit_identical_under_zero_input"])
        self.assertEqual(invariance["direction_max_abs_diff_across_rows"], 0.0)
        self.assertEqual(invariance["direction_max_abs_diff_zero_input"], 0.0)
        self.assertLessEqual(invariance["ratio_rows_max_minus_min"], RPU.UNCOND_STD_TOL)
        state = torch.load(self.up_path / "newtask.pt", map_location="cpu")
        self.assertTrue(torch.equal(state["uncond_condition"].cpu(),
                                    RPU.draw_uncond_condition(4)))

    def test_arm_gates_tiny(self):
        report = self._report(self.up_path)
        arm = report["arm"]
        for gate in ("UA1", "UA2", "UA3", "UA4", "UA5", "UA6", "UA9", "UA11"):
            self.assertTrue(arm[gate]["pass"], gate)
        # tiny 维度下 UA10/M0 如实失败（预算常量按真实维度 2384/8129 钉死；真实维度单元版见
        # TestUncondHeadSemantics.test_param_report_budget_real_dimensions）
        self.assertFalse(arm["UA10"]["pass"])
        self.assertEqual(arm["subreason"], "REFERENCE_IDENTITY")   # tiny 夹具 M0 不可达（同先例）
        # UA7/UA8：tiny val（6 行）含 1 个零范数行（机制约定 δ=0/ratio 记 0）⇒ ratio_mean 稀释
        # 1/6、ratio_std 抬升，按字面判据如实 FAIL（§12 C1b-5：val 表征与头无关，真实 500k val
        # 的 n_zero_rep=[0,0,0]（f08ae6e C_p 实测、结构性同源）⇒ 真实路径不受此影响）
        self.assertGreaterEqual(report["val_stats"]["n_zero_rep"][0], 1)
        self.assertFalse(arm["UA7"]["pass"])
        self.assertFalse(arm["UA8"]["pass"])

    def test_generator_grad_active_from_first_epoch(self):
        report = self._report(self.up_path)
        self.assertGreater(report["grad_probe"][0]["generator_grad_norm"], 0.0)
        self.assertIsNone(report["grad_probe"][0]["alpha_grad_norm"])

    def test_correct_arm_report_has_no_uncond_section(self):
        report = self._report(self.cp_path)
        self.assertNotIn("uncond", report)
        self.assertIn("pin", report)

    def test_summary_rows_appended(self):
        summary = (self.root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
        self.assertIn(self.up["run_id"], summary[-1])


class TestCliSurface(unittest.TestCase):
    def test_variant_choices_include_uncond_without_pinned_shuffled(self):
        import run_aliccp_benchmark as cli
        parser = cli.build_parser()
        for variant in (RPP.VARIANT_PINNED, RPU.VARIANT_UNCOND):
            ns = parser.parse_args(["stage2", "--stage1-id", "sid", "--variant", variant,
                                    "--prompt-reference-newtask", "ref/newtask.pt"])
            self.assertEqual(ns.variant, variant)
        with self.assertRaises(SystemExit):
            parser.parse_args(["stage2", "--stage1-id", "sid", "--variant",
                               RPP.VARIANT_PINNED_SHUFFLED])
        ns = parser.parse_args(["stage2", "--stage1-id", "sid"])
        self.assertEqual(ns.variant, RP.BASELINE_VARIANT)

    def test_suffix_dispatch(self):
        self.assertEqual(RPU.arm_suffix_for(RP.BASELINE_VARIANT), "")
        self.assertEqual(RPU.arm_suffix_for(RP.VARIANT), "-rpg")
        self.assertEqual(RPU.arm_suffix_for(RPP.VARIANT_PINNED), "-rpp")
        self.assertEqual(RPU.arm_suffix_for(RPP.VARIANT_PINNED_SHUFFLED), "-rpps")
        self.assertEqual(RPU.arm_suffix_for(RPU.VARIANT_UNCOND), "-rpu")

    def test_uncond_requires_reference(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = _make_tiny_root(td)
            with self.assertRaises(ValueError):
                _run_tiny_stage2(root, data_files, budgets, stage1_id="nonexistent",
                                 variant=RPU.VARIANT_UNCOND)

    def test_import_rp_uncond_has_no_rng_side_effects(self):
        import sys
        code = ("import torch; s = torch.get_rng_state(); import aliccp_benchmark.rp_uncond; "
                "print(bool(torch.equal(torch.get_rng_state(), s)))")
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(proc.stdout.strip().splitlines()[-1], "True", proc.stderr)


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
        self.assertEqual(RPU.PINNED_ALPHA, 0.07101669907569885)
        self.assertEqual(RPU.UNCOND_CONST_SEED, 20261006)
        self.assertEqual(RPU.MATERIAL_DELTA, 0.001)
        self.assertEqual(RPU.SECONDARY_POSITIVE_MIN, 0.001)
        self.assertEqual(RPU.SECONDARY_NEGATIVE_MAX, -0.02)
        self.assertEqual(RPU.EXPECTED_NEW_PARAMS_TOTAL, 2384)
        self.assertEqual(RPU.EXPECTED_HEAD_PARAMS, 8129)
        self.assertEqual(RPU.EXPECTED_COND_NUMEL, 80)
        self.assertEqual(RPU.UNCOND_STD_TOL, 1e-6)                 # §12 C1b-2 校正后
        self.assertEqual(RPU.RATIO_SPREAD_TOL, 1e-6)
        self.assertEqual(RPU.RATIO_REL_DIFF_DISCLOSURE_TOL, 0.05)
        self.assertEqual(RPU.HISTORICAL_U1_THRESHOLD, 0.0055)
        self.assertEqual(RPU.VARIANT_UNCOND, "residual-prompt-uncond")
        self.assertEqual(RPU.RUN_ID_SUFFIX_UNCOND, "-rpu")
        self.assertEqual(RPU.DELTA_TEST_CORRECT, 0.008103113327467715)
        self.assertEqual(RPU.BASELINE_RUN_ID,
                         "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07")
        self.assertEqual(RPU.PINNED_CORRECT_RUN_ID,
                         "20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp")
        self.assertEqual(RPU.PINNED_SHUFFLED_RUN_ID,
                         "20261005-1032-p2M-v500k-t1M-m1688723740-short-1c13841-rpps")
        self.assertEqual(RPU.PINNED_CORRECT_NEWTASK_SHA,
                         "264aedbb9f3f605e5a8e5dd9b2aee8c1945d2dcf0388be722541d2a84f4062b8")
        g_c_hist = (RPU.PINNED_CORRECT_RECORD["test_auc_bsi"] - RPU.BASELINE_RECORD["test_auc_bsi"])
        self.assertEqual(g_c_hist, -0.002677172368479308)
        self.assertEqual(RPU.PINNED_CORRECT_RECORD["alpha_final"], RPU.PINNED_ALPHA)

    def test_prereg_doc_tokens_and_summary_ledger(self):
        doc = (REPO_ROOT / PREREG_DOC_REL).read_text(encoding="utf-8")
        for token in ["79ddefa", "3e2f083", "221580a", "013e105", "e931cd7", "a40836e",
                      "fbfff09", "10e86dc", "0935257", "28aa1fe", "f08ae6e",
                      "674213f619c5d5039811c71242a7727118348daf",
                      "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc",
                      "432fa9bb6b3bb9d537753599499b693197e8c898",
                      "ba23bf473634ff647b482ebf6a92f27ca7f073432bb4ed7a1342b8b11101df9b",
                      "5dc7158ca999c9e7e6217a999c37869231411549",
                      "0d45cc4611cebde9675858f1afc633a91b3e9114d6ddc02899a5d97f34babf58",
                      "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
                      "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07",
                      "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg",
                      "20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs",
                      "20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp",
                      "20261005-1032-p2M-v500k-t1M-m1688723740-short-1c13841-rpps",
                      "0.5809347091990792", "0.5974422649550507", "0.005217193225189258",
                      "0.6055453782825184", "0.5895572066556973",
                      "0.008103113327467715", "0.008622497456618139",
                      "0.07101669907569885",
                      "0.5798886656441761", "0.5947650925865714",
                      "0.5817369266195509", "0.5966263705980125",
                      "0.5989210260551207", "0.0014787611000699474",
                      "0.002677172368479308", "0.000815894357038216", "0.001861278011441092",
                      "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f",
                      "264aedbb9f3f605e5a8e5dd9b2aee8c1945d2dcf0388be722541d2a84f4062b8",
                      "77f02899",
                      "185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e",
                      "daa9e78a000f49e6e1785d4b3cc47e4e3fbe2baa4331332c1ae47afb530c0d07",
                      "20261006", "0.0055", "0.001", "2384", "8129", "-rpp", "-rpu", "20261005",
                      "VALID_POSITIVE", "VALID_NEGATIVE", "MECHANISM_FAIL",
                      "SAMPLE_CONDITIONING_SUPPORTED", "SAMPLE_CONDITIONING_NOT_SUPPORTED", "INVALID",
                      "CONDITIONING_ESSENTIAL", "PARTIAL_CONDITIONING_CONTRIBUTION",
                      "GAP_BELOW_MATERIALITY", "VALIDATION_DISAGREEMENT",
                      "UNCONDITION_PIN_INVALID", "NOT_TRULY_UNCONDITIONAL",
                      "UNCONDITIONAL_GATE_NOT_CONSTANT", "GRADIENT_INVALID",
                      "CONSTANT_VECTOR_MISMATCH", "CROSS_EXPERIMENT_REPLAY_FAILED",
                      "POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION",
                      "MATERIAL_DELTA", "UA1", "UA11", "CV"]:
            self.assertIn(token, doc)
        summary = (REPO_ROOT / "artifacts" / "aliccp_bench" / "SUMMARY.md").read_text(encoding="utf-8")
        row = [line for line in summary.splitlines() if "b2e17f9" in line and "run_id" not in line]
        self.assertEqual(len(row), 1)
        self.assertIn("0.598839", row[0])
        self.assertIn("0.578153", row[0])


if __name__ == "__main__":
    unittest.main()
