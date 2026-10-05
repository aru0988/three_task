"""shuffled-condition 因果消融分支守卫测试（预注册 §3/§4/§5：逐字节钉死 / 重建等式 / 判定树边界 / 分析器夹具）。

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
from aliccp_benchmark import rp_shuffled as RPS
from multitaskrec.model import NewTask

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_BASE = "8133d32"
SEED2_IMPL = "013e105"

PREREG_DOC_REL = "docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-shuffled-condition-design.md"
GUARD_TEST_REL = "aliccp_benchmark/tests/test_residual_prompt_shuffled.py"
MECHANISM_REL = "aliccp_benchmark/residual_prompt.py"
PORTED_TEST_REL = "aliccp_benchmark/tests/test_residual_prompt.py"
BENCH_REL = "aliccp_benchmark/bench.py"
CLI_REL = "run_aliccp_benchmark.py"
SHUFFLED_MODULE_REL = "aliccp_benchmark/rp_shuffled.py"
PRERUN_VERIFIER_REL = "verify_shuffled_prerun.py"
POSTRUN_VERIFIER_REL = "verify_rp_shuffled.py"

# ---- §3 钉死（被消融对象 @013e105；blob + LF-normalized sha256）----
PIN_MECHANISM_BLOB = "674213f619c5d5039811c71242a7727118348daf"
PIN_MECHANISM_LF_SHA = "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"
PIN_BENCH_BLOB = "bf3647686838157bf5fd7645615faad9f2d7185d"
PIN_BENCH_LF_SHA = "b063a36667173b66fc8a7ac932502cc34753c91e2bfcfdafaedfce443f1db3b4"
PIN_CLI_BLOB = "229c25a1c86719fa6fb6b057d6cff3f4120fa053"
PIN_CLI_LF_SHA = "2966ec3985e5b91e277f429c2f236fb551de7479d114def90f767f2d813ad285"
PIN_PORTED_TEST_BLOB = "768a477b5c71af32c5c59ec6feb20c29f7873557"
PIN_PORTED_TEST_LF_SHA = "dab737a1422b88befc4c19f2bee408618061de7fd3bf0b2f83ebcdd85fd1cbc2"

# ---- 新文件 LF sha256 钉死（C2 提交时计算并写死；见预注册 §3）----
# 哈希链无环：本守卫文件钉 module + 前置核验脚本；运行后复核脚本（verify_rp_shuffled.py）钉本文件
# （防自指：文件不得钉自身 sha）。
SHUFFLED_MODULE_LF_SHA = "81fb9ba9bec9864cfdcd71256fb44a2b10939466e3a01f39726e21b813a2e6bf"
PRERUN_VERIFIER_LF_SHA = "871e59a5915c1889773f14740925c49435687033f13efb783e4958f17f189b80"

# ---- §3-A3：移植测试文件恰好适配（差异集合精确钉死）----
ADAPTED_TEST_METHODS = {("TestPreregConstants", "test_prereg_doc_tokens_and_summary_ledger")}
EXPECTED_WHITELIST = {
    PREREG_DOC_REL,
    MECHANISM_REL,
    SHUFFLED_MODULE_REL,
    BENCH_REL,
    CLI_REL,
    PORTED_TEST_REL,
    GUARD_TEST_REL,
    PRERUN_VERIFIER_REL,
    POSTRUN_VERIFIER_REL,
    "artifacts/aliccp_bench/SUMMARY.md",
}
# ---- §3-A2a/A2b：接线适配段/行（重建等式）----
ADAPTED_BENCH_SEGMENTS = ("import", "evaluate_newtask", "run_stage2")
CLI_ADAPTED_LINE_PREFIXES = ("from aliccp_benchmark import residual_prompt as RP",
                             '    p2.add_argument("--variant", choices=',
                             "    arm_suffix = ")

# ---- §4-I11：预注册文档 token 钉死 ----
DOC_TOKENS = [
    "013e105", "e931cd7", "a40836e", "79ddefa", "3e2f083", "221580a", "10e86dc", "0935257", "28aa1fe",
    "674213f619c5d5039811c71242a7727118348daf", PIN_MECHANISM_LF_SHA, PIN_BENCH_BLOB, PIN_CLI_BLOB,
    PIN_PORTED_TEST_BLOB,
    "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
    "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07",
    "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg",
    "0.5809347091990792", "0.5974422649550507", "0.005217193225189258",
    "0.6055453782825184", "0.5895572066556973",
    "0.008103113327467715", "0.008622497456618139",
    "0.0074396653390377265", "0.0048363447472773435",
    "0.004155967377889924", "0.0023134736258212385", "0.005910402347593546",
    "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f",
    "20261005", "-rpgs", "2385", "8129", "0.0055", "0.001", "−0.02",
    "SAMPLE_CONDITION_SUPPORTED", "CONDITION_ALIGNMENT_NOT_SUPPORTED", "INVALID",
    "FULL_GAIN_REQUIRES_ALIGNMENT", "PARTIAL_ALIGNMENT_CONTRIBUTION", "SHUFFLED_RETAINS_GAIN",
    "POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION",
    "PG1", "PG2", "PG3", "PG4", "PG5", "PG6",
    "IDENTITY_MISMATCH", "BASELINE_REPRODUCTION_FAILED", "PROTOCOL_INVALID", "MECHANISM_FAIL",
    "COMPARATOR_MISMATCH", "PERMUTATION_INVALID", "COMPARATOR_GAIN_NOT_MATERIAL",
    "rp_shuffled_compare.json", "MATERIAL_DELTA", "Sattolo",
]


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
    tree = ast.parse(treesrc)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(treesrc, node)
    raise AssertionError(f"未找到模块级函数 {name}")


def _module_import_segment(treesrc: str, imported_name: str) -> str:
    tree = ast.parse(treesrc)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(a.name == imported_name for a in node.names):
            return ast.get_source_segment(treesrc, node)
    raise AssertionError(f"未找到含 {imported_name} 的 import 语句")


def _class_method(treesrc: str, cls_name: str, method: str) -> str:
    tree = ast.parse(treesrc)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method:
                    return ast.get_source_segment(treesrc, member)
    raise AssertionError(f"未找到 {cls_name}.{method}")


def _module_assign_node(treesrc: str, name: str):
    tree = ast.parse(treesrc)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node
    raise AssertionError(f"未找到模块级赋值 {name}")


def _module_assign_literal(treesrc: str, name: str):
    return ast.literal_eval(_module_assign_node(treesrc, name).value)


def _module_assign_set_resolved(treesrc: str, name: str) -> set:
    tree = ast.parse(treesrc)
    module_literals = {}
    for node in tree.body:
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
        for fname in ("evaluate_newtask", "run_stage2"):
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
            if name in ("evaluate_newtask", "run_stage2"):
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
        for rel, blob_pin, sha_pin in ((BENCH_REL, PIN_BENCH_BLOB, PIN_BENCH_LF_SHA),
                                       (CLI_REL, PIN_CLI_BLOB, PIN_CLI_LF_SHA)):
            blob = subprocess.check_output(["git", "rev-parse", f"{SEED2_IMPL}:{rel}"],
                                           cwd=REPO_ROOT, text=True).strip()
            self.assertEqual(blob, blob_pin, rel)
            pinned = _git_show_text(f"{SEED2_IMPL}:{rel}")
            self.assertEqual(hashlib.sha256(pinned.encode("utf-8")).hexdigest(), sha_pin, rel)


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
        pinned = _git_show_text(f"{SEED2_IMPL}:{PORTED_TEST_REL}")
        self.assertEqual(hashlib.sha256(pinned.encode("utf-8")).hexdigest(), PIN_PORTED_TEST_LF_SHA)


class TestNewFileShaPins(unittest.TestCase):
    def test_shuffled_module_lf_sha_pinned(self):
        self.assertNotEqual(SHUFFLED_MODULE_LF_SHA, "TBD_SHUFFLED_MODULE_LF_SHA", "pin 未填写（C2 计算）")
        self.assertEqual(_lf_sha(SHUFFLED_MODULE_REL), SHUFFLED_MODULE_LF_SHA)

    def test_prerun_verifier_lf_sha_pinned(self):
        self.assertNotEqual(PRERUN_VERIFIER_LF_SHA, "TBD_PRERUN_VERIFIER_LF_SHA", "pin 未填写（C2 计算）")
        self.assertEqual(_lf_sha(PRERUN_VERIFIER_REL), PRERUN_VERIFIER_LF_SHA)

    def test_postrun_verifier_pins_guard_and_module(self):
        """哈希链无环核对：运行后复核脚本必须钉住本守卫文件与机制/模块文件（反向不成立）。"""
        text = _read_lf(POSTRUN_VERIFIER_REL)
        self.assertIn("GUARD_TEST_LF_SHA", text)
        self.assertIn("SHUFFLED_MODULE_LF_SHA", text)
        self.assertIn('"aliccp_benchmark/rp_shuffled.py": SHUFFLED_MODULE_LF_SHA', text)
        self.assertIn('"aliccp_benchmark/tests/test_residual_prompt_shuffled.py": GUARD_TEST_LF_SHA', text)


# =====================================================================================
# §4-I4/I5/I6：错排构造
# =====================================================================================

class TestDerangementConstruction(unittest.TestCase):
    def test_bijection_and_no_fixed_points(self):
        for n in (2, 3, 7, 100, 2000):
            for batch_index in range(4):
                perm = RPS.derange(n, "train", batch_index)
                self.assertEqual(sorted(perm), list(range(n)), f"n={n} 双射失败")
                fixed = sum(1 for i, v in enumerate(perm) if i == v)
                self.assertEqual(fixed, 0, f"n={n} batch={batch_index} 出现不动点")

    def test_small_n_degenerate_identity(self):
        self.assertEqual(RPS.derange(0, "val", 0), [])
        self.assertEqual(RPS.derange(1, "val", 0), [0])

    def test_deterministic_same_key_across_instances(self):
        a = RPS.ConditioningShuffler().perm("train", 3, 64)
        b = RPS.ConditioningShuffler().perm("train", 3, 64)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.dtype, torch.long)
        self.assertEqual(a.shape, (64,))

    def test_different_keys_differ(self):
        p0 = RPS.derange(64, "train", 0)
        p1 = RPS.derange(64, "train", 1)
        pv = RPS.derange(64, "val", 0)
        self.assertNotEqual(p0, p1)
        self.assertNotEqual(p0, pv)

    def test_split_is_part_of_key(self):
        self.assertNotEqual(RPS.perm_seed_material("train", 0, 64), RPS.perm_seed_material("val", 0, 64))
        self.assertNotEqual(RPS.perm_seed_material("train", 0, 64), RPS.perm_seed_material("train", 0, 63))

    def test_key_conflict_different_n_raises(self):
        shuffler = RPS.ConditioningShuffler()
        shuffler.perm("train", 0, 8)
        with self.assertRaises(ValueError):
            shuffler.perm("train", 0, 9)

    def test_base_seed_frozen(self):
        self.assertEqual(RPS.COND_PERM_SEED, 20261005)
        self.assertNotEqual(RPS.derange(64, "train", 0),
                            RPS.derange(64, "train", 0, base_seed=RPS.COND_PERM_SEED + 1))


class TestShufflerIsolationAndBlindness(unittest.TestCase):
    def test_rng_isolation_torch_and_python(self):
        shuffler = RPS.ConditioningShuffler()
        torch_state = torch.get_rng_state()
        py_state = random.getstate()
        shuffler.perm("train", 0, 128)
        shuffler.perm("val", 5, 128)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state), "torch 全局 RNG 被扰动")
        self.assertEqual(random.getstate(), py_state, "python 全局 RNG 被扰动")
        self.assertEqual(shuffler.report_counters()["rng_isolation_violations"], 0)

    def test_label_blind_signature(self):
        sig = inspect.signature(RPS.derange)
        self.assertEqual(list(sig.parameters), ["n", "split", "batch_index", "base_seed"])
        sig_mat = inspect.signature(RPS.perm_seed_material)
        self.assertEqual(list(sig_mat.parameters), ["split", "batch_index", "n", "base_seed"])

    def test_same_key_same_perm_under_interleaving(self):
        s1 = RPS.ConditioningShuffler()
        p_a = s1.perm("train", 0, 32)
        s1.perm("val", 0, 32)
        s1.perm("test", 0, 32)
        p_a2 = s1.perm("train", 0, 32)
        self.assertTrue(torch.equal(p_a, p_a2))


class TestShufflerReport(unittest.TestCase):
    def test_finalize_report_structure_and_integrity(self):
        shuffler = RPS.ConditioningShuffler()
        for step in range(3):
            shuffler.perm("train", step, 16)
        for step in range(2):
            shuffler.perm("val", step, 16)
        report = shuffler.finalize()
        self.assertEqual(report["base_seed"], RPS.COND_PERM_SEED)
        for key in ("fixed_points_total", "degenerate_identity_batches", "bijection_failures",
                    "regeneration_mismatches", "rng_isolation_violations", "key_conflicts",
                    "n_perms_total", "per_split", "total_digest", "samples"):
            self.assertIn(key, report)
        self.assertEqual(report["fixed_points_total"], 0)
        self.assertEqual(report["bijection_failures"], 0)
        self.assertEqual(report["regeneration_mismatches"], 0)
        self.assertEqual(report["rng_isolation_violations"], 0)
        self.assertEqual(report["n_perms_total"], 5)
        self.assertEqual(report["per_split"]["train"]["n_batches"], 3)
        self.assertEqual(report["per_split"]["train"]["n_rows"], 48)
        self.assertEqual(report["per_split"]["val"]["n_rows"], 32)
        self.assertRegex(report["total_digest"], r"^[0-9a-f]{64}$")
        self.assertRegex(report["per_split"]["train"]["digest"], r"^[0-9a-f]{64}$")
        self.assertTrue(len(report["samples"]["train"]) >= 1)
        s2 = RPS.ConditioningShuffler()
        for step in range(3):
            s2.perm("train", step, 16)
        for step in range(2):
            s2.perm("val", step, 16)
        self.assertEqual(s2.finalize()["total_digest"], report["total_digest"])


# =====================================================================================
# §4-I7：shuffled 头语义
# =====================================================================================

def _make_heads(input_size=8, rep_dim=4, seed=0):
    """同一 RNG 现场构造 correct/shuffled 两头（G1 构造恒等口径：错排头不得多消耗任何随机流）。"""
    torch.manual_seed(seed)
    rng_before = torch.get_rng_state()
    correct = RP.ResidualPromptNewTask(input_size=input_size, rep_dim=rep_dim,
                                       tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN,
                                       device=torch.device("cpu"))
    state_after_correct = torch.get_rng_state()
    torch.set_rng_state(rng_before)
    shuffled = RPS.ShuffledConditionResidualPromptNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=(4, 2),
        reg_dnn=P.REG_DNN, device=torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), state_after_correct), "构造 RNG 端点必须一致"
    return correct, shuffled


def _random_inputs(b=6, input_size=8, rep_dim=4, seed=1, norms=(1.0, 10.0, 100.0)):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(b, input_size, generator=gen)
    reps = [torch.randn(b, rep_dim, generator=gen) for _ in norms]
    reps = [r / r.norm(dim=-1, keepdim=True) * n for r, n in zip(reps, norms)]
    return x, reps


def _env(rep_dim=4, seed=3):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(rep_dim, generator=gen), torch.randn(rep_dim, generator=gen)]


class TestShuffledHeadSemantics(unittest.TestCase):
    def test_construction_identity_with_correct_head(self):
        correct, shuffled = _make_heads()
        self.assertIsInstance(shuffled, RP.ResidualPromptNewTask)
        self.assertIsInstance(shuffled, NewTask)
        csd, ssd = correct.state_dict(), shuffled.state_dict()
        self.assertEqual(set(csd), set(ssd), "state_dict 键集必须一致（错排器不入 state_dict）")
        for key in csd:
            self.assertTrue(torch.equal(csd[key], ssd[key]), key)
        self.assertEqual(float(shuffled.prompt_gate.detach()), 0.0)

    def test_required_cond_perm_sentinel_raises(self):
        _, shuffled = _make_heads()
        x, reps = _random_inputs()
        with self.assertRaises(ValueError):
            shuffled(x, reps[0], reps[1:], _env())

    def test_alpha_zero_bit_identical_with_real_perm(self):
        correct, shuffled = _make_heads()
        x, reps = _random_inputs()
        env = _env()
        perm = RPS.ConditioningShuffler().perm("train", 0, x.shape[0])
        out_ref = correct(x, reps[0], reps[1:], env)
        out_shuf = shuffled(x, reps[0], reps[1:], env, cond_perm=perm)
        self.assertTrue(torch.equal(out_ref, out_shuf), "α=0 ⇒ 置换不应改变输出（G2 结构性）")

    def test_identity_perm_equivalent_to_correct_head_at_nonzero_alpha(self):
        correct, shuffled = _make_heads()
        with torch.no_grad():
            shuffled.load_state_dict(correct.state_dict())
            shuffled.prompt_gate.fill_(0.05)
            correct.prompt_gate.fill_(0.05)
        x, reps = _random_inputs()
        env = _env()
        identity = torch.arange(x.shape[0])
        out_ref = correct(x, reps[0], reps[1:], env)
        out_shuf = shuffled(x, reps[0], reps[1:], env, cond_perm=identity)
        self.assertTrue(torch.equal(out_ref, out_shuf), "恒等置换 ⇒ 与 correct 头逐位相等（PG6-c）")

    def test_nonidentity_perm_changes_output_and_correspondence(self):
        correct, shuffled = _make_heads()
        with torch.no_grad():
            shuffled.load_state_dict(correct.state_dict())
            shuffled.prompt_gate.fill_(0.1)
        x, reps = _random_inputs()
        env = _env()
        perm = torch.tensor([1, 2, 3, 4, 5, 0])           # 6-轮换
        out_id = shuffled(x, reps[0], reps[1:], env, cond_perm=torch.arange(6))
        out_sh = shuffled(x, reps[0], reps[1:], env, cond_perm=perm)
        self.assertFalse(torch.equal(out_id, out_sh))
        # 手算对应：delta_i 由 x[perm[i]] 生成
        deltas_sh, m_sh = shuffled.prompt_deltas(x[perm], [reps[0], *reps[1:]])
        _, m_manual = shuffled.prompt_deltas(x[perm], [reps[0]])
        deltas_id, _ = shuffled.prompt_deltas(x, [reps[0], *reps[1:]])
        self.assertTrue(torch.equal(m_sh, m_manual))
        self.assertFalse(torch.equal(deltas_sh[0], deltas_id[0]))

    def test_norm_bound_holds_under_permutation(self):
        correct, shuffled = _make_heads()
        with torch.no_grad():
            shuffled.load_state_dict(correct.state_dict())
            shuffled.prompt_gate.fill_(0.3)
        x, reps = _random_inputs()
        perm = RPS.ConditioningShuffler().perm("val", 7, x.shape[0])
        deltas, m = shuffled.prompt_deltas(x[perm], reps)
        alpha = float(shuffled.prompt_gate.detach())
        for h, delta in zip(reps, deltas):
            ratio = delta.norm(dim=-1) / h.norm(dim=-1)
            self.assertLessEqual(float(ratio.max()), abs(alpha) + 1e-6)


class TestForwardDispatch(unittest.TestCase):
    def test_none_shuffler_is_verbatim_call(self):
        correct, _ = _make_heads()
        x, reps = _random_inputs()
        env = _env()
        with torch.no_grad():
            out_direct = correct(x, reps[0], reps[1:], env)
            out_dispatch = RPS.forward_with_conditioning(correct, x, reps[0], reps[1:], env, None, None, 0)
        self.assertTrue(torch.equal(out_direct, out_dispatch))

    def test_shuffler_applies_perm(self):
        _, shuffled = _make_heads()
        x, reps = _random_inputs()
        env = _env()
        shuffler = RPS.ConditioningShuffler()
        with torch.no_grad():
            shuffled.prompt_gate.fill_(0.2)
            out = RPS.forward_with_conditioning(shuffled, x, reps[0], reps[1:], env, shuffler, "train", 0)
            perm = shuffler.perm("train", 0, x.shape[0])
            manual = torch.tensor(RPS.derange(x.shape[0], "train", 0))
            self.assertTrue(torch.equal(perm, manual))
            manual_out = shuffled(x, reps[0], reps[1:], env, cond_perm=manual)
        self.assertTrue(torch.equal(out, manual_out))
        self.assertEqual(out.shape, (x.shape[0],))


# =====================================================================================
# §5.5/§5.6：判定树边界（纯函数）与分析器（夹具）
# =====================================================================================

class TestVerdictBoundaries(unittest.TestCase):
    """预注册 §5.5/§5.6 的精确边界（纯函数层；浮点恰值构造）。"""

    def test_material_loss_boundary_closed(self):
        self.assertTrue(0.008 - 0.007 == RPS.MATERIAL_DELTA)          # 恰在边界（浮点精确）
        ok = RPS.causal_verdict(g_c=0.008, g_s=0.007)
        self.assertEqual(ok["verdict"], "SAMPLE_CONDITION_SUPPORTED")
        below = RPS.causal_verdict(g_c=0.008, g_s=math.nextafter(0.007, 1))
        self.assertEqual(below["verdict"], "CONDITION_ALIGNMENT_NOT_SUPPORTED")

    def test_partial_boundary_closed(self):
        self.assertEqual(RPS.causal_verdict(g_c=0.008, g_s=0.001)["subreason"],
                         "PARTIAL_ALIGNMENT_CONTRIBUTION")
        self.assertEqual(RPS.causal_verdict(g_c=0.008, g_s=math.nextafter(0.001, 0))["subreason"],
                         "FULL_GAIN_REQUIRES_ALIGNMENT")

    def test_comparator_gain_not_material(self):
        result = RPS.causal_verdict(g_c=0.0005, g_s=0.0004)
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "COMPARATOR_GAIN_NOT_MATERIAL")

    def test_precondition_priority(self):
        result = RPS.causal_verdict(g_c=0.008, g_s=0.002,
                                    preconditions={"IDENTITY_MISMATCH": False,
                                                   "COMPARATOR_MISMATCH": True})
        self.assertEqual(result["subreason"], "COMPARATOR_MISMATCH")
        result = RPS.causal_verdict(g_c=0.008, g_s=0.002,
                                    preconditions={"IDENTITY_MISMATCH": True,
                                                   "BASELINE_REPRODUCTION_FAILED": True,
                                                   "PERMUTATION_INVALID": True})
        self.assertEqual(result["subreason"], "IDENTITY_MISMATCH")

    def test_shuffled_exceeds_correct(self):
        result = RPS.causal_verdict(g_c=0.008, g_s=0.010)
        self.assertEqual(result["verdict"], "CONDITION_ALIGNMENT_NOT_SUPPORTED")
        self.assertTrue(result["components"]["shuffled_exceeds_correct"])

    def test_secondary_boundaries(self):
        self.assertEqual(RPS.secondary_classification(0.001), "POSITIVE_IMPROVEMENT")
        self.assertEqual(RPS.secondary_classification(math.nextafter(0.001, 0)), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(RPS.secondary_classification(-0.02), "CLEAR_DEGRADATION")
        self.assertEqual(RPS.secondary_classification(math.nextafter(-0.02, 0)), "NO_CLEAR_IMPROVEMENT")


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
        "backbone_sha256_loaded": RPS.STAGE1_BACKBONE_SHA,
        "env_ids_sha256": RPS.STAGE1_ENV_IDS_SHA,
        "fingerprint_sha256": RPS.STAGE1_FINGERPRINT_SHA,
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


def _fixture_shuffle(per_split=None, **overrides):
    report = {"base_seed": RPS.COND_PERM_SEED, "fixed_points_total": 0,
              "degenerate_identity_batches": 0, "bijection_failures": 0,
              "regeneration_mismatches": 0, "rng_isolation_violations": 0, "key_conflicts": 0,
              "n_perms_total": 3, "per_split": per_split or {}, "total_digest": "0" * 64,
              "samples": {}}
    block = {"pass": True,
             "gates": {g: {"pass": True} for g in ("PG1", "PG2", "PG3", "PG4", "PG5", "PG6")},
             "report": report}
    block.update(overrides)
    return block


def _fixture_rp_arm(**overrides):
    arm = {"M0": {"pass": True}, "G1": {"pass": True}, "G2": {"pass": True}, "G3": {"pass": True},
           "G4": {"pass": True}, "G5": {"pass": True}, "G6": {"pass": True}, "G7": {"pass": True},
           "G8": {"pass": True}, "U": {"pass": True}, "protocol_10_1": {},
           "classification": "VALID_POSITIVE", "subreason": None, "pass": True,
           "shuffle": _fixture_shuffle()}
    arm.update(overrides)
    return arm


class AnalyzerFixtureBase(unittest.TestCase):
    """构造 P0/P1/对照 run 的极小夹具，并 patch 钉死常量；全部 CPU、纯 JSON。"""

    BASE_PE = [0.46, 0.47, 0.48, 0.485, 0.49]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "artifacts"
        self.root.mkdir(parents=True)
        self.baseline_commit = "aaa1111"
        self.arm_commit = "bbb2222"
        self.baseline_run = f"20261005-0001-fixture-baseline-{self.baseline_commit}"
        self.arm_run = f"20261005-0002-fixture-arm-{self.arm_commit}-rpgs"
        self.context_baseline_run = "ctx-baseline"
        self.context_correct_run = "ctx-correct"
        self.sid = "s1-fixture-seed2"
        self.base_test, self.base_val = 0.5000, 0.4900
        self.g_c = 0.0080
        self.correct_test = self.base_test + self.g_c
        self.correct_val = self.base_val + 0.0080
        self.arm_test = self.base_test + 0.003
        self.arm_val = self.base_val + 0.0040

    def tearDown(self):
        self._tmp.cleanup()

    # ---- 写 run ----
    def _write_context(self):
        corr_pe = [v + 0.001 for v in self.BASE_PE]
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
                    {"alpha_final": 0.07101669907569885})

    def _real_shuffle_block(self, splits=(("train", 12), ("val", 6), ("test", 8))):
        """用真实 ConditioningShuffler 为该夹具生成完整 shuffle 块（digest 可被独立复核重推）。"""
        shuffler = RPS.ConditioningShuffler()
        for split, n_rows in splits:
            shuffler.perm(split, 0, n_rows)
        report = shuffler.finalize()
        return {"pass": True,
                "gates": {g: {"pass": True} for g in ("PG1", "PG2", "PG3", "PG4", "PG5", "PG6")},
                "report": report}

    def _write_pair(self, *, baseline_pe=None, arm_pe=None, arm_rp_arm=None, arm_shuffle=None):
        baseline_pe = baseline_pe if baseline_pe is not None else self.BASE_PE
        arm_pe = arm_pe if arm_pe is not None else [v + 0.001 for v in baseline_pe]
        p0 = self.root / "runs" / self.baseline_run
        _write_json(p0 / "metrics.json",
                    _fixture_metrics(run_id=self.baseline_run, variant="baseline",
                                     test=self.base_test, val=self.base_val, per_epoch_val=baseline_pe,
                                     gate_mean=[0.5, 0.5], stage1_id=self.sid,
                                     commit=self.baseline_commit))
        _write_json(p0 / "gate_report.json", _fixture_gates())
        _write_json(p0 / "config.json", {"budgets": {"train": 12, "val": 6, "test": 8},
                                         "batch_size": 16, "model_seed": 1688723740})
        (p0 / "newtask.pt").write_bytes(FIXTURE_HEAD_BYTES)
        arm = self.root / "runs" / self.arm_run
        arm_arm = arm_rp_arm or _fixture_rp_arm()
        _write_json(arm / "metrics.json",
                    _fixture_metrics(run_id=self.arm_run, variant=RPS.VARIANT_SHUFFLED,
                                     test=self.arm_test, val=self.arm_val, per_epoch_val=arm_pe,
                                     gate_mean=[0.4, 0.6], stage1_id=self.sid,
                                     rp_arm=arm_arm, commit=self.arm_commit))
        arm_gate_doc = _fixture_gates()
        arm_gate_doc["residual_prompt"] = arm_arm
        _write_json(arm / "gate_report.json", arm_gate_doc)
        _write_json(arm / "config.json", {"budgets": {"train": 12, "val": 6, "test": 8},
                                          "batch_size": 16, "model_seed": 1688723740,
                                          "prompt_hidden": 16})
        ref_path = self.root / "runs" / self.context_baseline_run / "newtask.pt"
        shuffle = arm_shuffle or self._real_shuffle_block()
        _write_json(arm / "prompt_report.json",
                    {"alpha_final": 0.05,
                     "reference_dispersion": {"newtask_checkpoint": str(ref_path)},
                     "arm": arm_arm, "shuffle": shuffle})
        with (self.root / "SUMMARY.md").open("a", encoding="utf-8") as handle:
            handle.write(f"| {self.baseline_run} | {self.baseline_commit} | short | fixture |\n")
            handle.write(f"| {self.arm_run} | {self.arm_commit} | short | fixture |\n")

    def _patches(self):
        return mock.patch.multiple(
            RPS,
            STAGE1_ID=self.sid,
            BASELINE_RUN_ID=self.context_baseline_run,
            CORRECT_RUN_ID=self.context_correct_run,
            BASELINE_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
                             "best_val_auc_bsi": self.base_val, "test_auc_bsi": self.base_test,
                             "gate_mean": [0.5, 0.5], "per_epoch_val": list(self.BASE_PE)},
            CORRECT_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
                            "best_val_auc_bsi": self.correct_val, "test_auc_bsi": self.correct_test,
                            "gate_mean": [0.4, 0.6],
                            "per_epoch_val": [v + 0.001 for v in self.BASE_PE],
                            "variant": "residual-prompt", "classification": "VALID_POSITIVE",
                            "alpha_final": 0.07101669907569885},
            DELTA_TEST_CORRECT=self.correct_test - self.base_test,
            DELTA_VAL_CORRECT=self.correct_val - self.base_val,
            REF_HEAD_SHA=_sha256_file(self.root / "runs" / self.context_baseline_run / "newtask.pt"),
        )

    def _analyze(self):
        with self._patches():
            return RPS.analyze_runs(root=self.root, baseline_run=self.baseline_run, arm_run=self.arm_run)

    def _scenario(self, g_s, **pair_kwargs):
        self.arm_test = self.base_test + g_s
        self._write_context()
        self._write_pair(**pair_kwargs)
        return self._analyze()


class TestVerdictTreeFixtures(AnalyzerFixtureBase):
    def test_supported_full_gain_requires_alignment(self):
        result = self._scenario(-0.002)
        self.assertEqual(result["verdict"], "SAMPLE_CONDITION_SUPPORTED")
        self.assertEqual(result["subreason"], "FULL_GAIN_REQUIRES_ALIGNMENT")
        self.assertFalse(result["components"]["shuffled_above_baseline"])

    def test_supported_partial_alignment_contribution(self):
        result = self._scenario(0.003)
        self.assertEqual(result["verdict"], "SAMPLE_CONDITION_SUPPORTED")
        self.assertEqual(result["subreason"], "PARTIAL_ALIGNMENT_CONTRIBUTION")
        self.assertTrue(result["components"]["shuffled_material_positive"])

    def test_not_supported_when_retained(self):
        result = self._scenario(0.0075)
        self.assertEqual(result["verdict"], "CONDITION_ALIGNMENT_NOT_SUPPORTED")
        self.assertEqual(result["subreason"], "SHUFFLED_RETAINS_GAIN")
        self.assertAlmostEqual(result["components"]["retained_ratio"], 0.0075 / 0.0080, places=9)

    def test_material_loss_near_boundary(self):
        result = self._scenario(0.0070 - 1e-6)        # L = 0.001 + 1e-6 → SUPPORTED
        self.assertEqual(result["verdict"], "SAMPLE_CONDITION_SUPPORTED")
        result = self._scenario(0.0070 + 1e-6)        # L = 0.001 − 1e-6 → NOT_SUPPORTED
        self.assertEqual(result["verdict"], "CONDITION_ALIGNMENT_NOT_SUPPORTED")

    def test_validation_agreement_recorded(self):
        result = self._scenario(0.003)
        self.assertTrue(result["components"]["validation_agreement"])
        self.assertTrue(result["components"]["val_sign_matches_correct"])

    def test_invalid_on_reproduction_failure(self):
        result = self._scenario(0.002, baseline_pe=[0.1, 0.2, 0.3, 0.4, 0.45])
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "BASELINE_REPRODUCTION_FAILED")

    def test_invalid_on_permutation_gate_failure(self):
        bad = _fixture_shuffle(per_split={})
        bad["report"]["fixed_points_total"] = 3          # 分析器从 report 计数机械重算 PG（不信任记录布尔）
        result = self._scenario(-0.002, arm_shuffle=bad)
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "PERMUTATION_INVALID")
        self.assertFalse(result["permutation_gates"]["PG1"]["pass"])

    def test_invalid_on_comparator_mismatch(self):
        self.arm_test = self.base_test + 0.002
        self._write_context()
        _write_json(self.root / "runs" / self.context_correct_run / "metrics.json",
                    _fixture_metrics(run_id=self.context_correct_run, variant="residual-prompt",
                                     test=self.correct_test + 0.001, val=self.correct_val,
                                     per_epoch_val=[v + 0.001 for v in self.BASE_PE],
                                     gate_mean=[0.4, 0.6], stage1_id=self.sid,
                                     rp_arm={"classification": "VALID_POSITIVE"}))
        self._write_pair()
        result = self._analyze()
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "COMPARATOR_MISMATCH")

    def test_invalid_on_protocol_failure(self):
        self.arm_test = self.base_test + 0.002
        self._write_context()
        self._write_pair()
        gates = _fixture_gates()
        gates["gates"]["A1"] = {"verdict": "FAIL"}
        _write_json(self.root / "runs" / self.arm_run / "gate_report.json", gates)
        result = self._analyze()
        self.assertEqual(result["verdict"], "INVALID")
        self.assertEqual(result["subreason"], "PROTOCOL_INVALID")


class TestSecondaryClassificationFixtures(AnalyzerFixtureBase):
    def test_positive(self):
        self.assertEqual(self._scenario(0.002)["secondary"], "POSITIVE_IMPROVEMENT")

    def test_clear_degradation(self):
        self.assertEqual(self._scenario(-0.05)["secondary"], "CLEAR_DEGRADATION")

    def test_no_clear_headroom_applicable(self):
        result = self._scenario(-0.0005)
        self.assertEqual(result["secondary"], "NO_CLEAR_IMPROVEMENT")
        self.assertTrue(result["headroom"]["applicable"])
        self.assertAlmostEqual(result["headroom"]["gap_to_positive_threshold"], 0.0015, places=6)
        self.assertIn(result["headroom"]["mechanism_activity"], ("ACTIVE", "INACTIVE"))
        self.assertIn(result["headroom"]["validation_direction"], ("POSITIVE", "NON_POSITIVE"))

    def test_headroom_not_applicable_for_positive(self):
        result = self._scenario(0.002)
        self.assertFalse(result["headroom"]["applicable"])


class TestVerifierFixture(AnalyzerFixtureBase):
    """端到端夹具复核 `verify_rp_shuffled.main`：独立重推路径全过（文件 pin 依赖仓库内真实文件）。"""

    def test_verifier_all_pass_on_fixture(self):
        import contextlib
        import io
        import verify_rp_shuffled as V
        self._scenario(0.003)
        v_patches = mock.patch.multiple(
            V,
            STAGE1_ID=self.sid,
            BASELINE_RUN_ID=self.context_baseline_run,
            CORRECT_RUN_ID=self.context_correct_run,
            REF_HEAD_SHA=_sha256_file(self.root / "runs" / self.context_baseline_run / "newtask.pt"),
            BASELINE_RECORD={"best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
                             "best_val_auc_bsi": self.base_val, "test_auc_bsi": self.base_test,
                             "gate_mean": [0.5, 0.5], "per_epoch_val": list(self.BASE_PE)},
            CORRECT_RECORD={"best_val_auc_bsi": self.correct_val, "test_auc_bsi": self.correct_test,
                            "variant": "residual-prompt", "classification": "VALID_POSITIVE",
                            "alpha_final": 0.07101669907569885},
            DELTA_TEST_CORRECT=self.correct_test - self.base_test,
            DELTA_VAL_CORRECT=self.correct_val - self.base_val,
        )
        with v_patches, contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = V.main(["--root", str(self.root), "--baseline-run", self.baseline_run,
                         "--arm-run", self.arm_run])
        self.assertEqual(rc, 0, buf.getvalue())
        report = json.loads((self.root / "runs" / self.arm_run / "verify_report.json")
                            .read_text(encoding="utf-8"))
        self.assertTrue(report["all_pass"])
        self.assertGreater(report["n_checks"], 40)
        self.assertEqual(report["recomputed"]["verdict"], "SAMPLE_CONDITION_SUPPORTED")


# =====================================================================================
# tiny 端到端（接线）
# =====================================================================================

HEADER = "click,purchase,bsi,121,122,301"
TRAIN_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2], [0, 0, 9, 2, 2, 1], [1, 1, 9, 0, 1, 3],
    [0, 0, 8, 3, 0, 2], [1, 0, 8, 1, 1, 1], [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2],
]
VAL_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3],
    [1, 1, 8, 3, 1, 1], [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2],
]
TEST_ROWS = [
    [0, 0, 9, 2, 2, 1], [1, 0, 9, 0, 1, 3], [0, 0, 8, 3, 0, 2], [1, 1, 8, 1, 1, 1],
    [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2], [0, 0, 9, 1, 0, 3], [1, 0, 8, 4, 1, 2],
]
TINY_VOCAB = {"121": 5, "122": 4}
TINY_NEW_PARAMS_TOTAL = 4 * 16 + 16 + 16 * 4 + 4 + 1       # Linear(4,16) + Linear(16,4) + α


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


class TestShuffledArmEndToEnd(unittest.TestCase):
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
        cls.shuffled = _run_tiny_stage2(cls.root, cls.data_files, cls.budgets, cls.stage1_id,
                                        variant=RPS.VARIANT_SHUFFLED,
                                        prompt_reference_newtask=cls.base_path / "newtask.pt")
        cls.shuffled_path = P.run_dir(cls.root, cls.shuffled["run_id"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _report(self):
        return json.loads((self.shuffled_path / "prompt_report.json").read_text(encoding="utf-8"))

    def test_run_id_suffix_and_variant_recorded(self):
        self.assertTrue(self.shuffled["run_id"].endswith(RPS.RUN_ID_SUFFIX))
        self.assertEqual(RPS.RUN_ID_SUFFIX, "-rpgs")
        metrics = json.loads((self.shuffled_path / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["variant"], RPS.VARIANT_SHUFFLED)
        config = json.loads((self.shuffled_path / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["variant"], RPS.VARIANT_SHUFFLED)

    def test_prompt_report_shuffle_section_and_gates(self):
        report = self._report()
        for key in ("construction_identity", "init_forward", "grad_probe", "alpha_final", "gate",
                    "val_stats", "dispersion", "reference_dispersion", "source_gates", "params",
                    "arm", "shuffle"):
            self.assertIn(key, report)
        shuffle = report["shuffle"]
        self.assertEqual(shuffle["report"]["fixed_points_total"], 0)
        self.assertEqual(shuffle["report"]["bijection_failures"], 0)
        self.assertEqual(shuffle["report"]["rng_isolation_violations"], 0)
        self.assertEqual(shuffle["report"]["regeneration_mismatches"], 0)
        for gate in ("PG1", "PG2", "PG3", "PG4", "PG5", "PG6"):
            self.assertTrue(shuffle["gates"][gate]["pass"], gate)
        self.assertTrue(shuffle["pass"])
        per_split = shuffle["report"]["per_split"]
        self.assertEqual(per_split["train"]["n_rows"], len(TRAIN_ROWS))
        self.assertEqual(per_split["val"]["n_rows"], len(VAL_ROWS))
        self.assertEqual(per_split["test"]["n_rows"], len(TEST_ROWS))
        self.assertEqual(shuffle["report"]["degenerate_identity_batches"], 0)

    def test_g1_g2_hold_and_perm_used_in_init_forward(self):
        report = self._report()
        arm = report["arm"]
        self.assertTrue(arm["G1"]["pass"])
        self.assertTrue(arm["G2"]["pass"])
        self.assertTrue(report["construction_identity"]["shared_params_bit_identical"])
        self.assertTrue(report["init_forward"]["bit_identical"])
        self.assertFalse(arm["M0"]["pass"])            # tiny 夹具参照身份不可达（同 correct 臂先例）
        self.assertEqual(arm["classification"], "MECHANISM_FAIL")
        self.assertEqual(arm["subreason"], "REFERENCE_IDENTITY")

    def test_alpha_and_grad_probe_active(self):
        report = self._report()
        self.assertNotEqual(report["alpha_final"], 0.0)
        self.assertEqual(report["grad_probe"][0]["generator_grad_norm"], 0.0)      # α=0 首步已知
        self.assertGreater(report["grad_probe"][-1]["generator_grad_norm"], 0.0)
        self.assertTrue(report["arm"]["G3"]["pass"])

    def test_parameters_unchanged_vs_correct_arm(self):
        report = self._report()
        params = report["params"]
        self.assertEqual(sorted(item["name"] for item in params["new_param_list"]),
                         sorted(RP.EXTRA_PARAM_NAMES))
        self.assertEqual(params["new_params_total"], TINY_NEW_PARAMS_TOTAL)
        self.assertEqual(params["head_params"], sum(int(p.numel())
                                                    for name, p in RPS.build_shuffled_newtask(
                                                        input_size=4, rep_dim=4,
                                                        tower_dnn_hidden_units=(4,),
                                                        reg_dnn=P.REG_DNN,
                                                        device=torch.device("cpu")
                                                    ).named_parameters()
                                                    if not name.startswith("prompt_")))

    def test_summary_row_appended(self):
        summary = (self.root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
        self.assertIn(self.shuffled["run_id"], summary[-1])


class TestCliSurface(unittest.TestCase):
    def test_variant_choices_include_shuffled(self):
        import run_aliccp_benchmark as cli
        parser = cli.build_parser()
        ns = parser.parse_args(["stage2", "--stage1-id", "sid", "--variant", RPS.VARIANT_SHUFFLED,
                                "--prompt-reference-newtask", "ref/newtask.pt"])
        self.assertEqual(ns.variant, RPS.VARIANT_SHUFFLED)
        ns = parser.parse_args(["stage2", "--stage1-id", "sid"])
        self.assertEqual(ns.variant, RP.BASELINE_VARIANT)

    def test_shuffled_requires_reference(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = _make_tiny_root(td)
            with self.assertRaises(ValueError):
                _run_tiny_stage2(root, data_files, budgets, stage1_id="nonexistent",
                                 variant=RPS.VARIANT_SHUFFLED)


class TestShuffledArmVerdict(unittest.TestCase):
    def _probe(self):
        return {"construction": {"shared_params_bit_identical": True,
                                 "global_rng_endpoint_identical": True,
                                 "extra_keys": sorted(RP.EXTRA_PARAM_NAMES),
                                 "alpha_at_construction": 0.0},
                "init_forward": {"bit_identical": True},
                "grad_probe": [{"alpha_grad_norm": 0.1, "alpha": 0.0, "generator_grad_norm": 0.0},
                               {"alpha_grad_norm": 0.1, "alpha": 0.05, "generator_grad_norm": 0.1}],
                "alpha_final": 0.05}

    def _val_stats(self):
        return {"streams": {"ratio_max": [0.04, 0.04, 0.04], "ratio_mean": [0.03, 0.03, 0.03]},
                "gate": {"geff_std": 0.001, "geff_max": 0.04, "geff_min": 0.01},
                "dispersion": {"pred_std": 0.004}}

    def _params(self):
        return {"new_param_list": [{"name": n} for n in RP.EXTRA_PARAM_NAMES],
                "new_params_total": RP.EXPECTED_NEW_PARAMS_TOTAL,
                "head_params": RP.EXPECTED_HEAD_PARAMS}

    def _reference(self):
        return {"val_auc": RP.BASELINE_AUC_VAL,
                "pred_dispersion": {"pred_std": RP.REFERENCE_PRED_STD}}

    def _report(self, **overrides):
        report = {"base_seed": RPS.COND_PERM_SEED, "fixed_points_total": 0,
                  "degenerate_identity_batches": 0, "bijection_failures": 0,
                  "regeneration_mismatches": 0, "rng_isolation_violations": 0, "key_conflicts": 0,
                  "n_perms_total": 4,
                  "per_split": {"train": {"n_batches": 1, "n_rows": 8, "digest": "d" * 64},
                                "val": {"n_batches": 1, "n_rows": 8, "digest": "d" * 64},
                                "test": {"n_batches": 1, "n_rows": 8, "digest": "d" * 64}},
                  "total_digest": "x", "samples": {}}
        report.update(overrides)
        return report

    def _verdict(self, report):
        return RPS.shuffled_arm_verdict(auc_test=RP.BASELINE_AUC_TEST + 0.01,
                                        auc_val=RP.BASELINE_AUC_VAL + 0.01, probe=self._probe(),
                                        val_stats=self._val_stats(), params=self._params(),
                                        reference=self._reference(), protocol_ok=True,
                                        shuffle_report=report)

    def test_all_gates_pass_and_classification_kept(self):
        arm = self._verdict(self._report())
        self.assertEqual(arm["classification"], "VALID_POSITIVE")
        self.assertTrue(arm["shuffle"]["pass"])
        for gate in ("PG1", "PG2", "PG3", "PG4", "PG5", "PG6"):
            self.assertTrue(arm["shuffle"]["gates"][gate]["pass"], gate)

    def test_permutation_failure_overrides_classification(self):
        arm = self._verdict(self._report(fixed_points_total=3))
        self.assertEqual(arm["classification"], "MECHANISM_FAIL")
        self.assertEqual(arm["subreason"], "PERMUTATION_INVALID")
        self.assertFalse(arm["shuffle"]["gates"]["PG1"]["pass"])

    def test_pg2_pg4_failures(self):
        arm = self._verdict(self._report(bijection_failures=1, rng_isolation_violations=2))
        self.assertFalse(arm["shuffle"]["gates"]["PG2"]["pass"])
        self.assertFalse(arm["shuffle"]["gates"]["PG4"]["pass"])
        self.assertEqual(arm["subreason"], "PERMUTATION_INVALID")


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

    def test_whitelist_copies_synced(self):
        ported = _module_assign_set_resolved(_read_lf(PORTED_TEST_REL), "WHITELIST")
        self.assertEqual(ported, EXPECTED_WHITELIST, "两处 WHITELIST 副本必须逐项一致")


class TestPreregConstants(unittest.TestCase):
    def test_frozen_constants(self):
        self.assertEqual(RPS.VARIANT_SHUFFLED, "residual-prompt-shuffled")
        self.assertEqual(RPS.RUN_ID_SUFFIX, "-rpgs")
        self.assertEqual(RPS.COND_PERM_SEED, 20261005)
        self.assertEqual(RPS.MATERIAL_DELTA, 0.001)
        self.assertEqual(RPS.SECONDARY_POSITIVE_MIN, 0.001)
        self.assertEqual(RPS.SECONDARY_NEGATIVE_MAX, -0.02)
        self.assertEqual(RPS.STAGE1_ID, "s1-5c060b9c-m1688723740-e3-4e1b5c6f")
        self.assertEqual(RPS.REF_HEAD_SHA,
                         "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f")

    def test_prereg_doc_tokens_and_summary_ledger(self):
        doc = _read_lf(PREREG_DOC_REL)
        missing = [token for token in DOC_TOKENS if token not in doc]
        self.assertEqual(missing, [], f"预注册文档缺失 token: {missing}")
        summary = _read_lf("artifacts/aliccp_bench/SUMMARY.md")
        row = [line for line in summary.splitlines() if "b2e17f9" in line and "run_id" not in line]
        self.assertEqual(len(row), 1)
        self.assertIn("0.598839", row[0])
        self.assertIn("0.578153", row[0])


if __name__ == "__main__":
    unittest.main()
