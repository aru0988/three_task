"""残差 prompt 20-epoch 持久性检验（终止判定点）：分支守卫 + CLI override + 分析器测试
（预注册：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-20epoch-design.md §6/§7）。

TDD：本文件先于移植与分析器实现编写（先红后绿）。守卫对象（§6 适配清单）：
  * A1‴：`aliccp_benchmark/residual_prompt.py` == `013e105` 逐字节（零适配）；
  * A2‴a：`aliccp_benchmark/bench.py` == `013e105` 逐字节；
  * A2‴b：`run_aliccp_benchmark.py` == `013e105` 恰 1 行适配（`--tag` choices 增加 `"xlong"`）；
  * A3‴：`aliccp_benchmark/tests/test_residual_prompt.py` == `013e105` 恰 4 处适配（docstring /
    DOC_PATH / WHITELIST / `TestPreregConstants.test_prereg_doc_tokens_and_summary_ledger`）；
  * A4‴：`aliccp_benchmark/rp_20epoch.py` 的 `persistence_verdict` / `_read_json` / `_a_class_ok`
    自 `f2ccec2:aliccp_benchmark/rp_longer_budget.py` 移植（verdict 恰 1 类适配：阈值常量名 ×2 处）。
另含 I4（epoch/patience/CLI override）、I5（历史判定边界 + 二级分类/claim 真值表 + 预算趋势）、
I6（分析器：只读/identity/机制门禁/夹具端到端/篡改 fixture）、I7（白名单 + 受保护文件零 diff）、
I8（文档 token 钉死）。
CPU 极小、纯静态 + 夹具；被钉 commit 不可得时**响亮失败**（不静默跳过）。
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from aliccp_benchmark import bench, protocol as P, residual_prompt as RP
from aliccp_benchmark import rp_20epoch as T20

import run_aliccp_benchmark

REPO = Path(__file__).resolve().parents[2]          # 测试依赖 cwd = 仓库根（git 调用）

# ---- 钉死 commit / 路径 ----
SEED_REPL_COMMIT = "013e105"           # 机制与接线逐字来源（seed-2 复现实现提交）
PRED_IMPL_COMMIT = "f2ccec2"           # 直接前驱实现提交（10-epoch；分析器移植来源）
FROZEN_BASE = "8133d32"                # 本分支基点（infra/aliccp-fair-benchmark 顶端）
PREREG_DOC_REL = "docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-20epoch-design.md"
GUARD_TEST_REL = "aliccp_benchmark/tests/test_residual_prompt_20epoch.py"
MECHANISM_REL = "aliccp_benchmark/residual_prompt.py"
PORTED_TEST_REL = "aliccp_benchmark/tests/test_residual_prompt.py"
CLI_REL = "run_aliccp_benchmark.py"
BENCH_REL = "aliccp_benchmark/bench.py"
ANALYZER_REL = "aliccp_benchmark/rp_20epoch.py"
INTEGRITY_SCRIPT_REL = "verify_seed2_artifact_integrity.py"
VERIFIER_SCRIPT_REL = "verify_rp_20epoch.py"
SEED_REPL_DOC_REL = "docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md"

# ---- 钉死 blob / LF sha256（§6-A1‴/A2‴/A4‴；2026-10-05 审计）----
PIN_MECHANISM_BLOB = "674213f619c5d5039811c71242a7727118348daf"
PIN_MECHANISM_LF_SHA = "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"
PIN_BENCH_BLOB = "bf3647686838157bf5fd7645615faad9f2d7185d"
PIN_BENCH_LF_SHA = "b063a36667173b66fc8a7ac932502cc34753c91e2bfcfdafaedfce443f1db3b4"
PIN_CLI_BLOB = "229c25a1c86719fa6fb6b057d6cff3f4120fa053"
PIN_CLI_LF_SHA_013E105 = "2966ec3985e5b91e277f429c2f236fb551de7479d114def90f767f2d813ad285"
PIN_PORTED_TEST_BLOB = "768a477b5c71af32c5c59ec6feb20c29f7873557"
PIN_PORTED_TEST_LF_SHA_013E105 = "dab737a1422b88befc4c19f2bee408618061de7fd3bf0b2f83ebcdd85fd1cbc2"
PIN_ANALYZER_PRED_BLOB = "e02da3071ded554175cfdad5d7a725dd43ed7c89"
PIN_ANALYZER_PRED_LF_SHA = "b8d62c5735715440a7e23dcbc4f206ee38ec7f6f9a6e5aac1184ceb8dcb15bb0"
BASE_CLI_LF_SHA = "3940de3eb9a4d0f9d42d9d1359ccca52a55bc5e2814b4cdc4b8f8224ae5e10d0"
BASE_BENCH_LF_SHA = "95123344e1c7f107e91aa0e26ee7b316725f2b48d70e9425b2ed923e15be0dc0"

# ---- 工作树文件 LF sha256 钉死（C2 提交时计算并写死；§6）----
PORTED_CLI_LF_SHA = "ed95e9693c4438de1cca677c251f955225c087499551c93d7afa5b6dd9d3873f"
PORTED_TEST_WORKING_LF_SHA = "4e1d50028e388939782448fc50b7b413a8a1854054b572f17f44ca38bf95c149"
ANALYZER_LF_SHA = "0f427ce8f2d594d58be7a953d0e4091c8aac91d79338949acc82b205f042116a"

# ---- §6-A2‴b：CLI 恰 1 行适配（差异集合精确钉死）----
PINNED_TAG_LINE = '    parser.add_argument("--tag", type=str, default="short", choices=["short", "smoke"])'
ADAPTED_TAG_LINE = '    parser.add_argument("--tag", type=str, default="short", choices=["short", "smoke", "xlong"])'

# ---- §6-A3‴：移植测试文件恰好适配（差异集合精确钉死）----
ADAPTED_TEST_METHODS = {("TestPreregConstants", "test_prereg_doc_tokens_and_summary_ledger")}
EXPECTED_WHITELIST = {
    PREREG_DOC_REL,
    "aliccp_benchmark/residual_prompt.py",
    "aliccp_benchmark/tests/test_residual_prompt.py",
    GUARD_TEST_REL,
    "aliccp_benchmark/rp_20epoch.py",
    "run_aliccp_benchmark.py",
    "aliccp_benchmark/bench.py",
    INTEGRITY_SCRIPT_REL,
    VERIFIER_SCRIPT_REL,
    "artifacts/aliccp_bench/SUMMARY.md",
}

# ---- §6-A4‴：persistence_verdict 恰 1 类适配（阈值常量名 ×2 处）----
PINNED_VERDICT_CONST = "LONGER_BUDGET_DELTA_TEST_MIN"
ADAPTED_VERDICT_CONST = "DELTA_TEST_MIN"
PINNED_VERDICT_CONST_OCCURRENCES = 2

SEED2 = 1688723740
STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"

# ---- §7-I8 文档 token 钉死（与移植测试同一 token 表）----
DOC_TOKENS = [
    "013e105", "79ddefa", "bbd8a61", "8133d32", "47ecb0c", "d990cb0", "296e03a", "e931cd7",
    "a40836e", "3e2f083", "221580a", "2b1d585", "e46e5d2", "0935257", "28aa1fe", "f2ccec2",
    "99b9510",
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
    "107221b26382da7fd44990e167d9a61da2680d92",
    "f2aa202c426d619cd752035e19cf8803dc5afccec0ce32f923c48a20ff1d72f3",
    SEED_REPL_DOC_REL,
    "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
    "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07",
    "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg",
    "20261004-0427-p2M-v500k-t1M-m1688723740-long-2b1d585",
    "20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg",
    "20261003-0724-p2M-v500k-t1M-m1688723740-long-bbd8a61",
    "0.5809347091990792", "0.5974422649550507", "0.005217193225189258",
    "0.6055453782825184", "0.5895572066556973",
    "0.008103113327467715", "0.008622497456618139",
    "0.0074396653390377265", "0.0048363447472773435", "0.008916659527500093",
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
    "TWENTY_EPOCH_CONDITION_SATISFIED", "RIGHT_CENSORED_STILL_IMPROVING", "EARLY_STOPPED",
    "MONOTONE_NARROWING", "不外推", "原样保留",
]


# ---- 静态守卫工具 ----
def _read_lf(rel: str) -> str:
    return (REPO / rel).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def _lf_sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_show_text(spec: str) -> str:
    try:
        raw = subprocess.check_output(["git", "show", spec], cwd=REPO, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise AssertionError(f"钉死对象不可得（系谱依赖，响亮失败）: {spec}: {exc.stderr.decode()}") from exc
    return raw.replace(b"\r\n", b"\n").decode("utf-8")


def _git_blob(rel: str) -> str:
    return subprocess.check_output(["git", "hash-object", rel], cwd=REPO, text=True).strip()


def _replace_once(test: unittest.TestCase, text: str, old: str, new: str, what: str) -> str:
    test.assertEqual(text.count(old), 1, f"{what}: 旧文本应恰出现 1 次（实际 {text.count(old)}）")
    return text.replace(old, new)


def _module_doc_segment(treesrc: str) -> str:
    tree = ast.parse(treesrc)
    first = tree.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        raise AssertionError("首语句不是 docstring")
    segment = ast.get_source_segment(treesrc, first)
    if segment is None:
        raise AssertionError("docstring 源段不可得")
    return segment


def _module_function_segment(treesrc: str, name: str) -> str:
    tree = ast.parse(treesrc)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(treesrc, node)
            if segment is None:
                raise AssertionError(f"{name} 源段不可得")
            return segment
    raise AssertionError(f"未找到模块级函数 {name}")


def _class_method(treesrc: str, cls_name: str, method: str) -> str:
    tree = ast.parse(treesrc)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method:
                    segment = ast.get_source_segment(treesrc, member)
                    if segment is None:
                        raise AssertionError(f"{cls_name}.{method} 源段不可得")
                    return segment
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
    """集合字面量求值：元素为 Constant 或用其它模块级赋值名（如 DOC_PATH）解析。"""
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


def _line_starting_with(text: str, prefix: str) -> str:
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise AssertionError(f"期望恰 1 行以 {prefix!r} 开头，实际 {len(lines)} 行")
    return lines[0]


class TestMechanismByteIdentical(unittest.TestCase):
    """A1‴：机制文件 == 013e105 逐字节（零适配）。"""

    def test_mechanism_byte_identical_to_pinned_blob(self):
        self.assertEqual(_read_lf(MECHANISM_REL),
                         _git_show_text(f"{SEED_REPL_COMMIT}:{MECHANISM_REL}"),
                         "机制文件必须与 013e105 逐字节一致（零适配）")

    def test_mechanism_blob_and_lf_sha_pinned(self):
        blob = subprocess.check_output(["git", "rev-parse", f"{SEED_REPL_COMMIT}:{MECHANISM_REL}"],
                                       cwd=REPO, text=True).strip()
        self.assertEqual(blob, PIN_MECHANISM_BLOB)
        self.assertEqual(hashlib.sha256(_git_show_text(f"{SEED_REPL_COMMIT}:{MECHANISM_REL}").encode("utf-8")).hexdigest(),
                         PIN_MECHANISM_LF_SHA)
        self.assertEqual(_lf_sha(MECHANISM_REL), PIN_MECHANISM_LF_SHA)
        self.assertEqual(_git_blob(MECHANISM_REL), PIN_MECHANISM_BLOB)

    def test_mechanism_docstring_lineage_tokens(self):
        doc = ast.get_docstring(ast.parse(_read_lf(MECHANISM_REL)), clean=False) or ""
        for token in ("79ddefa", "5dc7158ca999c9e7e6217a999c37869231411549",
                      SEED_REPL_DOC_REL, "1688723740"):
            self.assertIn(token, doc, token)


class TestWiringGuards(unittest.TestCase):
    """A2‴a/A2‴b：bench.py 逐字节 == 013e105；CLI == 013e105 恰 1 行适配。"""

    def test_bench_byte_identical_to_pinned_blob(self):
        self.assertEqual(_read_lf(BENCH_REL), _git_show_text(f"{SEED_REPL_COMMIT}:{BENCH_REL}"))
        self.assertEqual(_lf_sha(BENCH_REL), PIN_BENCH_LF_SHA)
        self.assertEqual(_git_blob(BENCH_REL), PIN_BENCH_BLOB)

    def test_bench_diff_vs_base_equals_seed1_diff(self):
        self.assertEqual(hashlib.sha256(_git_show_text(f"{FROZEN_BASE}:{BENCH_REL}").encode("utf-8")).hexdigest(),
                         BASE_BENCH_LF_SHA)
        ours = subprocess.check_output(["git", "diff", FROZEN_BASE, "--", BENCH_REL], cwd=REPO)
        seed1 = subprocess.check_output(["git", "diff", FROZEN_BASE, SEED_REPL_COMMIT, "--", BENCH_REL], cwd=REPO)
        self.assertEqual(ours.replace(b"\r\n", b"\n"), seed1.replace(b"\r\n", b"\n"),
                         "bench.py 相对基线的 diff 必须与 seed-2 复现（013e105）相同")

    def test_cli_equals_pinned_with_exactly_one_line_adaptation(self):
        working = _read_lf(CLI_REL)
        modified = _replace_once(self, working, ADAPTED_TAG_LINE, PINNED_TAG_LINE, "--tag 行还原")
        # 还原后必须逐字节等于钉死 blob ⇒ 工作树 == 钉死 + 恰该 1 行适配
        self.assertEqual(modified, _git_show_text(f"{SEED_REPL_COMMIT}:{CLI_REL}"),
                         "CLI 相对 013e105 除 --tag choices 增加 xlong 外必须逐字节一致")

    def test_cli_tag_lines_exact(self):
        pinned = _git_show_text(f"{SEED_REPL_COMMIT}:{CLI_REL}")
        self.assertEqual(_line_starting_with(pinned, '    parser.add_argument("--tag"'), PINNED_TAG_LINE)
        self.assertNotIn("xlong", PINNED_TAG_LINE)
        working = _read_lf(CLI_REL)
        self.assertEqual(_line_starting_with(working, '    parser.add_argument("--tag"'), ADAPTED_TAG_LINE)

    def test_cli_lf_sha_pinned(self):
        self.assertNotEqual(PORTED_CLI_LF_SHA, "TBD_PORTED_CLI_LF_SHA", "pin 未填写（C2 时计算）")
        self.assertEqual(_lf_sha(CLI_REL), PORTED_CLI_LF_SHA)
        # 基点与 seed-1 版 sha 照旧可查（系谱完整性）
        self.assertEqual(hashlib.sha256(_git_show_text(f"{FROZEN_BASE}:{CLI_REL}").encode("utf-8")).hexdigest(),
                         BASE_CLI_LF_SHA)
        self.assertEqual(hashlib.sha256(_git_show_text(f"{SEED_REPL_COMMIT}:{CLI_REL}").encode("utf-8")).hexdigest(),
                         PIN_CLI_LF_SHA_013E105)


class TestPortedTestReconstructEquality(unittest.TestCase):
    """A3‴：移植测试文件 == 钉死 blob + docstring + DOC_PATH + WHITELIST + 1 方法（逐字节重建等式）。"""

    def _rebuild(self, working: str) -> str:
        pinned = _git_show_text(f"{SEED_REPL_COMMIT}:{PORTED_TEST_REL}")
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
                         "移植测试文件相对 013e105 除 4 处文档化适配外必须逐字节一致")

    def test_ported_test_lf_sha_pinned(self):
        self.assertNotEqual(PORTED_TEST_WORKING_LF_SHA, "TBD_PORTED_TEST_WORKING_LF_SHA", "pin 未填写")
        self.assertEqual(_lf_sha(PORTED_TEST_REL), PORTED_TEST_WORKING_LF_SHA)
        self.assertEqual(hashlib.sha256(_git_show_text(f"{SEED_REPL_COMMIT}:{PORTED_TEST_REL}").encode("utf-8")).hexdigest(),
                         PIN_PORTED_TEST_LF_SHA_013E105)

    def test_adapted_module_values_exact(self):
        working = _read_lf(PORTED_TEST_REL)
        self.assertEqual(_module_assign_literal(working, "DOC_PATH"), PREREG_DOC_REL)
        self.assertEqual(_module_assign_set_resolved(working, "WHITELIST"), EXPECTED_WHITELIST)
        pinned = _git_show_text(f"{SEED_REPL_COMMIT}:{PORTED_TEST_REL}")
        for name in ("PIN_BLOB", "PIN_SHA256_LF", "PIN_TEST_BLOB", "FROZEN_BASE"):
            self.assertEqual(_module_assign_literal(working, name),
                             _module_assign_literal(pinned, name), name)

    def test_adapted_methods_are_exactly_the_documented_set(self):
        working = _read_lf(PORTED_TEST_REL)
        pinned = _git_show_text(f"{SEED_REPL_COMMIT}:{PORTED_TEST_REL}")
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
        old_funcs, new_funcs = funcs(tree_old), funcs(tree_new)
        self.assertEqual(set(old_funcs), set(new_funcs), "模块级函数集合")
        for fname, old_node in old_funcs.items():
            self.assertEqual(ast.dump(old_node, include_attributes=False),
                             ast.dump(new_funcs[fname], include_attributes=False),
                             f"模块级函数 {fname} 不得改动")
        self.assertEqual(differences, ADAPTED_TEST_METHODS, "AST 差异集合必须恰为文档化的 1 个方法")

    def test_pinned_ported_test_blob_available_and_matches(self):
        blob = subprocess.check_output(["git", "rev-parse", f"{SEED_REPL_COMMIT}:{PORTED_TEST_REL}"],
                                       cwd=REPO, text=True).strip()
        self.assertEqual(blob, PIN_PORTED_TEST_BLOB)


class TestAnalyzerPortGuards(unittest.TestCase):
    """A4‴：分析器移植项与 f2ccec2 的逐字/重建等式守卫。"""

    def test_read_json_and_a_class_ok_verbatim(self):
        pinned = _git_show_text(f"{PRED_IMPL_COMMIT}:aliccp_benchmark/rp_longer_budget.py")
        working = _read_lf(ANALYZER_REL)
        for name in ("_read_json", "_a_class_ok"):
            self.assertEqual(_module_function_segment(working, name),
                             _module_function_segment(pinned, name), f"{name} 必须逐字移植")

    def test_persistence_verdict_reconstructs_with_exactly_documented_adaptations(self):
        pinned_seg = _module_function_segment(
            _git_show_text(f"{PRED_IMPL_COMMIT}:aliccp_benchmark/rp_longer_budget.py"), "persistence_verdict")
        working_seg = _module_function_segment(_read_lf(ANALYZER_REL), "persistence_verdict")
        self.assertIn(PINNED_VERDICT_CONST, pinned_seg)
        self.assertEqual(pinned_seg.count(PINNED_VERDICT_CONST), PINNED_VERDICT_CONST_OCCURRENCES)
        self.assertNotIn(PINNED_VERDICT_CONST, working_seg, "适配后不得残留前驱常量名")
        rebuilt = pinned_seg.replace(PINNED_VERDICT_CONST, ADAPTED_VERDICT_CONST)
        self.assertEqual(rebuilt, working_seg,
                         "persistence_verdict 相对 f2ccec2 除阈值常量名（×2 处）外必须逐字一致")

    def test_analyzer_lf_sha_pinned(self):
        self.assertNotEqual(ANALYZER_LF_SHA, "TBD_ANALYZER_LF_SHA", "pin 未填写（C2 时计算）")
        self.assertEqual(_lf_sha(ANALYZER_REL), ANALYZER_LF_SHA)
        blob = subprocess.check_output(["git", "rev-parse", f"{PRED_IMPL_COMMIT}:aliccp_benchmark/rp_longer_budget.py"],
                                       cwd=REPO, text=True).strip()
        self.assertEqual(blob, PIN_ANALYZER_PRED_BLOB)
        self.assertEqual(hashlib.sha256(
            _git_show_text(f"{PRED_IMPL_COMMIT}:aliccp_benchmark/rp_longer_budget.py").encode("utf-8")).hexdigest(),
            PIN_ANALYZER_PRED_LF_SHA)


class TestEpochPatienceOverride(unittest.TestCase):
    """I4：CLI 默认逐位不变；override 解析与 main() 线程化；--tag xlong 可解析。"""

    def test_cli_defaults_bit_identical(self):
        parser = run_aliccp_benchmark.build_parser()
        s2 = parser.parse_args(["stage2", "--stage1-id", "x"])
        self.assertEqual(s2.epochs, P.STAGE2_EPOCHS)
        self.assertEqual(s2.epochs, 5)
        self.assertEqual(s2.patience, P.STAGE2_PATIENCE)
        self.assertEqual(s2.patience, 2)
        self.assertEqual(s2.tag, "short")
        self.assertEqual(s2.model_seed, P.MODEL_SEED)
        self.assertEqual(s2.variant, RP.BASELINE_VARIANT)
        self.assertIsNone(s2.prompt_reference_newtask)
        s1 = parser.parse_args(["stage1"])
        self.assertEqual(s1.epochs, P.STAGE1_EPOCHS)
        self.assertEqual(s1.patience, P.STAGE1_PATIENCE)

    def test_tag_xlong_accepted_default_unchanged(self):
        parser = run_aliccp_benchmark.build_parser()
        self.assertEqual(parser.parse_args(["stage2", "--stage1-id", "x", "--tag", "xlong"]).tag, "xlong")
        self.assertEqual(parser.parse_args(["stage1", "--tag", "xlong"]).tag, "xlong")
        self.assertEqual(parser.parse_args(["stage1"]).tag, "short")       # 默认不变

    def test_override_parses(self):
        args = run_aliccp_benchmark.build_parser().parse_args(
            ["stage2", "--stage1-id", "x", "--epochs", "20", "--patience", "3", "--tag", "xlong",
             "--variant", "residual-prompt", "--prompt-reference-newtask", "ref/newtask.pt"])
        self.assertEqual((args.epochs, args.patience, args.tag), (20, 3, "xlong"))
        self.assertEqual(args.variant, RP.VARIANT)
        self.assertEqual(args.prompt_reference_newtask.as_posix(), "ref/newtask.pt")

    def test_main_threads_override_kwargs(self):
        """预注册配置经 main() 线程化到 run_stage2 调用参数（捕获 kwargs，不训练）。"""
        captured = {}
        fake = {"gates": {g: {"verdict": "PASS"} for g in ("A1", "A2", "A3", "A4", "A5", "A6",
                                                           "B1", "B2", "B3", "B4")},
                "run_id": "fake-run", "hard_pass": True, "metrics": {}}

        def fake_stage2(**kwargs):
            captured.update(kwargs)
            return fake

        with tempfile.TemporaryDirectory() as td, mock.patch.object(bench, "run_stage2", fake_stage2):
            rc = run_aliccp_benchmark.main(["stage2", "--stage1-id", "sid", "--epochs", "20", "--patience", "3",
                                            "--tag", "xlong", "--model-seed", str(SEED2), "--root", td])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["epochs"], 20)
        self.assertEqual(captured["patience"], 3)
        self.assertEqual(captured["tag"], "xlong")
        self.assertEqual(captured["model_seed"], SEED2)
        self.assertEqual(captured["stage1_id"], "sid")
        self.assertEqual(captured["variant"], RP.BASELINE_VARIANT)
        self.assertIsNone(captured["prompt_reference_newtask"])
        self.assertTrue(captured["enforce_b"])                             # tag != smoke → B 类判定
        self.assertNotIn("-rpg", captured["run_id"])

    def test_main_threads_arm_kwargs_and_run_id_suffix(self):
        captured = {}
        fake = {"gates": {g: {"verdict": "PASS"} for g in ("A1", "A2", "A3", "A4", "A5", "A6",
                                                           "B1", "B2", "B3", "B4")},
                "run_id": "fake-run-rpg", "hard_pass": True,
                "metrics": {"rp_arm": {"classification": "VALID_NEGATIVE", "subreason": None,
                                       "U": {"U1": {"pass": True}, "U2": {"pass": True}}}}}

        def fake_stage2(**kwargs):
            captured.update(kwargs)
            return fake

        with tempfile.TemporaryDirectory() as td, mock.patch.object(bench, "run_stage2", fake_stage2):
            rc = run_aliccp_benchmark.main(["stage2", "--stage1-id", "sid", "--epochs", "20", "--patience", "3",
                                            "--tag", "xlong", "--model-seed", str(SEED2), "--root", td,
                                            "--variant", "residual-prompt",
                                            "--prompt-reference-newtask", "ref/newtask.pt"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["variant"], RP.VARIANT)
        self.assertEqual(captured["prompt_reference_newtask"].as_posix(), "ref/newtask.pt")
        self.assertTrue(captured["run_id"].endswith("-rpg"))


class TestPreregConstants(unittest.TestCase):
    """I5/I8：预注册常量与判定边界钉死；文档 token 一致。"""

    def test_pinned_constants(self):
        self.assertEqual(T20.TWENTY_EPOCH_EPOCHS, 20)
        self.assertEqual(T20.TWENTY_EPOCH_PATIENCE, 3)
        self.assertEqual(T20.TWENTY_EPOCH_TAG, "xlong")
        self.assertEqual(T20.DELTA_TEST_MIN, 0.0055)
        self.assertEqual(T20.SECONDARY_POSITIVE_MIN, 0.001)
        self.assertEqual(T20.SECONDARY_DEGRADATION_MAX, -0.02)
        self.assertEqual(T20.CLAIM_DELTA_TEST_MIN, 0.001)
        self.assertEqual(T20.SEED2_STAGE1_ID, STAGE1_ID)
        self.assertEqual(T20.SEED2_MODEL_SEED, SEED2)
        self.assertEqual(T20.BASELINE_VARIANT, "baseline")
        self.assertEqual(T20.ARM_VARIANT, "residual-prompt")
        self.assertEqual(T20.ARM_RUN_ID_SUFFIX, "-rpg")
        self.assertEqual(T20.REFERENCE_CHECKPOINT_REL,
                         "artifacts/aliccp_bench/runs/"
                         "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt")
        self.assertEqual(T20.REFERENCE_CHECKPOINT_SHA256,
                         "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f")
        # 5-epoch context 钉死值（残差 prompt canonical 先例记录，审计逐位复核）
        self.assertEqual(T20.SEED2_5EPOCH_BASELINE_RUN_ID, "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07")
        self.assertEqual(T20.SEED2_5EPOCH_ARM_RUN_ID, "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg")
        self.assertEqual(T20.SEED2_5EPOCH_BASELINE_TEST_AUC, 0.5974422649550507)
        self.assertEqual(T20.SEED2_5EPOCH_BASELINE_VAL_AUC, 0.5809347091990792)
        self.assertEqual(T20.SEED2_5EPOCH_ARM_TEST_AUC, 0.6055453782825184)
        self.assertEqual(T20.SEED2_5EPOCH_ARM_VAL_AUC, 0.5895572066556973)
        self.assertEqual(T20.SEED2_5EPOCH_DELTA_TEST, 0.008103113327467715)
        self.assertEqual(T20.SEED2_5EPOCH_DELTA_VAL, 0.008622497456618139)
        self.assertAlmostEqual(T20.SEED2_5EPOCH_ARM_TEST_AUC - T20.SEED2_5EPOCH_BASELINE_TEST_AUC,
                               T20.SEED2_5EPOCH_DELTA_TEST, places=15)
        self.assertAlmostEqual(T20.SEED2_5EPOCH_ARM_VAL_AUC - T20.SEED2_5EPOCH_BASELINE_VAL_AUC,
                               T20.SEED2_5EPOCH_DELTA_VAL, places=15)
        # 10-epoch context 钉死值（直接前驱记录，审计逐位复核）
        self.assertEqual(T20.SEED2_10EPOCH_BASELINE_RUN_ID, "20261004-0427-p2M-v500k-t1M-m1688723740-long-2b1d585")
        self.assertEqual(T20.SEED2_10EPOCH_ARM_RUN_ID, "20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg")
        self.assertEqual(T20.SEED2_10EPOCH_BASELINE_TEST_AUC, 0.6614121223087556)
        self.assertEqual(T20.SEED2_10EPOCH_BASELINE_VAL_AUC, 0.6401127811737995)
        self.assertEqual(T20.SEED2_10EPOCH_ARM_TEST_AUC, 0.6688517876477933)
        self.assertEqual(T20.SEED2_10EPOCH_ARM_VAL_AUC, 0.6449491259210769)
        self.assertEqual(T20.SEED2_10EPOCH_DELTA_TEST, 0.0074396653390377265)
        self.assertEqual(T20.SEED2_10EPOCH_DELTA_VAL, 0.0048363447472773435)
        self.assertAlmostEqual(T20.SEED2_10EPOCH_ARM_TEST_AUC - T20.SEED2_10EPOCH_BASELINE_TEST_AUC,
                               T20.SEED2_10EPOCH_DELTA_TEST, places=15)
        self.assertAlmostEqual(T20.SEED2_10EPOCH_ARM_VAL_AUC - T20.SEED2_10EPOCH_BASELINE_VAL_AUC,
                               T20.SEED2_10EPOCH_DELTA_VAL, places=15)

    def test_analyze_defaults_match_prereg(self):
        sig = inspect.signature(T20.analyze_runs)
        self.assertEqual(sig.parameters["expected_stage1_id"].default, T20.SEED2_STAGE1_ID)
        self.assertEqual(sig.parameters["expected_model_seed"].default, T20.SEED2_MODEL_SEED)
        self.assertEqual(sig.parameters["expected_epochs"].default, T20.TWENTY_EPOCH_EPOCHS)
        self.assertEqual(sig.parameters["expected_patience"].default, T20.TWENTY_EPOCH_PATIENCE)
        self.assertEqual(sig.parameters["expected_tag"].default, T20.TWENTY_EPOCH_TAG)
        self.assertEqual(sig.parameters["expected_reference_checkpoint"].default, T20.REFERENCE_CHECKPOINT_REL)
        self.assertEqual(sig.parameters["expected_reference_sha256"].default, T20.REFERENCE_CHECKPOINT_SHA256)

    def test_verdict_boundary_delta_test_min(self):
        # 用 0.0 基线使 Δ 在 float64 下精确可表示（0.60+0.0055−0.60 ≠ 0.0055，不能用作边界构造）
        ok = T20.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0, arm_test_auc=0.0055,
                                     arm_val_auc=1e-6, a_class_pass=True, mechanism_gates_pass=True)
        self.assertTrue(ok["checks"]["delta_test_ge_min"])                       # 恰好 0.0055 → 通过（闭）
        self.assertEqual(ok["classification"], "PERSISTS")
        below = T20.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0,
                                        arm_test_auc=0.0055 - 1e-12,
                                        arm_val_auc=1e-6, a_class_pass=True, mechanism_gates_pass=True)
        self.assertFalse(below["checks"]["delta_test_ge_min"])
        self.assertEqual(below["classification"], "NOT_PERSIST")

    def test_verdict_boundary_val_direction_is_strict(self):
        flat = T20.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0, arm_test_auc=0.01,
                                       arm_val_auc=0.0, a_class_pass=True, mechanism_gates_pass=True)
        self.assertFalse(flat["checks"]["delta_val_positive"])                   # 恰好 0 → 不通过（严格）
        self.assertEqual(flat["classification"], "NOT_PERSIST")
        above = T20.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0, arm_test_auc=0.01,
                                        arm_val_auc=1e-12, a_class_pass=True, mechanism_gates_pass=True)
        self.assertTrue(above["checks"]["delta_val_positive"])
        self.assertEqual(above["classification"], "PERSISTS")

    def test_verdict_requires_a_class_and_mechanism_gates(self):
        v = T20.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0, arm_test_auc=0.01,
                                    arm_val_auc=0.01, a_class_pass=False, mechanism_gates_pass=True)
        self.assertTrue(v["checks"]["delta_test_ge_min"])
        self.assertTrue(v["checks"]["delta_val_positive"])
        self.assertFalse(v["checks"]["a_class_pass"])
        self.assertEqual(v["classification"], "NOT_PERSIST")
        self.assertEqual(v["delta_test_min"], 0.0055)
        # 机制门禁失败 ⇒ NOT_PERSIST（即使 Δ 全达标）
        m = T20.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0, arm_test_auc=0.01,
                                    arm_val_auc=0.01, a_class_pass=True, mechanism_gates_pass=False)
        self.assertFalse(m["checks"]["mechanism_gates_pass"])
        self.assertEqual(m["classification"], "NOT_PERSIST")

    def test_secondary_classification_boundaries(self):
        # 闭开边界：+0.001 恰过（闭端）；−0.02 恰落 DEGRADATION（闭端）
        self.assertEqual(T20.secondary_classification(0.001), "POSITIVE_IMPROVEMENT")
        self.assertEqual(T20.secondary_classification(0.001 - 1e-12), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(T20.secondary_classification(-0.02), "CLEAR_DEGRADATION")
        self.assertEqual(T20.secondary_classification(-0.02 + 1e-12), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(T20.secondary_classification(0.05), "POSITIVE_IMPROVEMENT")
        self.assertEqual(T20.secondary_classification(0.0), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(T20.secondary_classification(-0.5), "CLEAR_DEGRADATION")

    def test_claim_truth_table(self):
        base = dict(identity_all_pass=True, a_class_pass=True, mechanism_gates_pass=True,
                    delta_test=0.002, delta_val=0.001, delta_val_final=0.001)
        ok = T20.positive_persistence_claim(**base)
        self.assertTrue(ok["claim"])
        self.assertTrue(all(ok["checks"].values()))
        self.assertEqual(ok["delta_test_min"], 0.001)
        # Δtest 恰 +0.001 → 过（闭）；低 1e-12 → 不过
        self.assertTrue(T20.positive_persistence_claim(**{**base, "delta_test": 0.001})["claim"])
        self.assertFalse(T20.positive_persistence_claim(**{**base, "delta_test": 0.001 - 1e-12})["claim"])
        # Δval 恰 0 → 不过（严格）
        self.assertFalse(T20.positive_persistence_claim(**{**base, "delta_val": 0.0})["claim"])
        # P6：末共同 epoch Δval 恰 0 → 不过（严格）
        self.assertFalse(T20.positive_persistence_claim(**{**base, "delta_val_final": 0.0})["claim"])
        self.assertTrue(T20.positive_persistence_claim(**{**base, "delta_val_final": 1e-12})["claim"])
        # None（无共同 epoch）→ 不过
        self.assertFalse(T20.positive_persistence_claim(**{**base, "delta_val_final": None})["claim"])
        # 身份/门禁任一 false ⇒ claim false
        for key in ("identity_all_pass", "a_class_pass", "mechanism_gates_pass"):
            self.assertFalse(T20.positive_persistence_claim(**{**base, key: False})["claim"], key)

    def test_budget_trend_labels(self):
        tr = T20.budget_trend(0.008, 0.007, 0.006)
        self.assertEqual(tr["trend"], "MONOTONE_NARROWING")
        self.assertFalse(tr["sign_flip"])
        self.assertEqual(T20.budget_trend(0.001, 0.002, 0.003)["trend"], "MONOTONE_WIDENING")
        self.assertEqual(T20.budget_trend(0.008, 0.004, 0.006)["trend"], "TROUGH_AT_10")
        self.assertEqual(T20.budget_trend(0.004, 0.008, 0.006)["trend"], "PEAK_AT_10")
        self.assertEqual(T20.budget_trend(0.005, 0.005, 0.005)["trend"], "FLAT")
        self.assertEqual(T20.budget_trend(0.005, 0.005, 0.004)["trend"], "NON_MONOTONE_MIXED")
        mixed = T20.budget_trend(-0.001, 0.002, 0.003)
        self.assertEqual(mixed["trend"], "MONOTONE_WIDENING")
        self.assertTrue(mixed["sign_flip"])
        self.assertEqual(mixed["delta_test_5epoch"], -0.001)
        self.assertEqual(mixed["e1_10_minus_5"], 0.003)

    def test_headroom_applicability_and_factors(self):
        arm_rec = {"best_epoch": 20, "epochs": 20, "per_epoch": [{"val_auc_bsi": 0.5}] * 20}
        prompt = {"alpha_final": 0.1, "grad_probe": [{"epoch": 20, "generator_grad_norm": 0.05}],
                  "gate": {"geff_std": 0.01}}
        h = T20.headroom_assessment(secondary="NO_CLEAR_IMPROVEMENT", delta_test=0.0007,
                                    delta_val=0.001, delta_val_traj=[0.002, 0.001],
                                    arm_rec=arm_rec, arm_prompt=prompt)
        self.assertTrue(h["applicable"])
        self.assertEqual(h["mechanism_activity"], "ACTIVE")
        self.assertEqual(h["delta_val_direction"], "POSITIVE")
        self.assertEqual(h["epoch_trajectory"], "RIGHT_CENSORED_STILL_IMPROVING")
        self.assertAlmostEqual(h["gap_to_positive_threshold"], 0.001 - 0.0007, places=15)
        self.assertEqual(h["budget_trend"]["trend"], T20.budget_trend(
            T20.SEED2_5EPOCH_DELTA_TEST, T20.SEED2_10EPOCH_DELTA_TEST, 0.0007)["trend"])
        inactive = T20.headroom_assessment(secondary="NO_CLEAR_IMPROVEMENT", delta_test=0.0007,
                                           delta_val=-0.001, delta_val_traj=[0.002, -0.001],
                                           arm_rec=arm_rec, arm_prompt={**prompt, "alpha_final": 0.0})
        self.assertEqual(inactive["mechanism_activity"], "INACTIVE")
        self.assertEqual(inactive["delta_val_direction"], "NON_POSITIVE")
        not_applicable = T20.headroom_assessment(secondary="POSITIVE_IMPROVEMENT", delta_test=0.008,
                                                 delta_val=0.008, delta_val_traj=[0.01, 0.008],
                                                 arm_rec=arm_rec, arm_prompt=prompt)
        self.assertFalse(not_applicable["applicable"])
        early = T20.headroom_assessment(secondary="NO_CLEAR_IMPROVEMENT", delta_test=0.0007,
                                        delta_val=0.001, delta_val_traj=[0.002, 0.001],
                                        arm_rec={**arm_rec, "best_epoch": 17}, arm_prompt=prompt)
        self.assertEqual(early["epoch_trajectory"], "EARLY_STOPPED")
        none_prompt = T20.headroom_assessment(secondary="NO_CLEAR_IMPROVEMENT", delta_test=0.0007,
                                              delta_val=0.001, delta_val_traj=[0.001],
                                              arm_rec=arm_rec, arm_prompt=None)
        self.assertEqual(none_prompt["mechanism_activity"], "INACTIVE")

    def test_doc_preregisters_tokens(self):
        text = (REPO / PREREG_DOC_REL).read_text(encoding="utf-8")
        missing = [token for token in DOC_TOKENS if token not in text]
        self.assertEqual(missing, [], f"预注册文档缺失 token: {missing}")

    def test_summary_ledger_lineage_row(self):
        summary = (REPO / "artifacts" / "aliccp_bench" / "SUMMARY.md").read_text(encoding="utf-8")
        row = [line for line in summary.splitlines() if "b2e17f9" in line and "run_id" not in line]
        self.assertEqual(len(row), 1)
        self.assertIn("0.598839", row[0])
        self.assertIn("0.578153", row[0])


# ---- 极小 AliCCP 夹具（与 seed-2 复现测试同构，自包含）----
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
TINY_DIMS = dict(expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4)


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
    data_files = {
        "train": str(data_dir / "train.csv"),
        "val": str(data_dir / "val.csv"),
        "test": str(data_dir / "test.csv"),
    }
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return root, data_files, budgets


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestAnalyzerFixture(unittest.TestCase):
    """I6：分析器（只读、判定、二级分类/claim、identity、机制门禁、context、篡改 fixture、CLI）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root, cls.data_files, cls.budgets = _make_tiny_root(cls._tmp.name)
        cls.device = torch.device("cpu")
        cls.noop = staticmethod(lambda *a, **k: None)
        cls.meta = bench.run_stage1(
            root=cls.root, data_files=cls.data_files, budgets=cls.budgets, prefix_tag="tiny",
            model_seed=SEED2, env_seed=456, epochs=2, patience=2, device=cls.device, vocab=TINY_VOCAB,
            **TINY_DIMS, log=cls.noop,
        )
        common = dict(root=cls.root, stage1_id=cls.meta["stage1_id"], data_files=cls.data_files,
                      budgets=cls.budgets, prefix_tag="tiny", model_seed=SEED2, epochs=1, patience=1,
                      tag="smoke", device=cls.device, vocab=TINY_VOCAB, **TINY_DIMS,
                      enforce_b=False, log=cls.noop)
        cls.baseline = bench.run_stage2(**common)
        cls.baseline_path = P.run_dir(cls.root, cls.baseline["run_id"])
        cls.arm = bench.run_stage2(**common, variant=RP.VARIANT,
                                   prompt_reference_newtask=cls.baseline_path / "newtask.pt")
        cls.arm_path = P.run_dir(cls.root, cls.arm["run_id"])
        cls.expect_kwargs = dict(expected_stage1_id=cls.meta["stage1_id"], expected_model_seed=SEED2,
                                 expected_epochs=1, expected_patience=1, expected_tag="smoke",
                                 expected_reference_checkpoint=str(cls.baseline_path / "newtask.pt"),
                                 expected_reference_sha256=_sha256_file(cls.baseline_path / "newtask.pt"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_analyze_end_to_end_and_output(self):
        out_path = Path(self.arm["run_dir"]) / T20.OUTPUT_NAME
        result = T20.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                                  arm_run_id=self.arm["run_id"], output_path=out_path, **self.expect_kwargs)
        self.assertTrue(all(result["identity_checks"].values()), result["identity_checks"])
        # 夹具参照头不满足真实 M0 身份 ⇒ 机制门禁必须 FAIL ⇒ NOT_PERSIST + claim false（不允许静默通过）
        self.assertFalse(result["mechanism"]["all_pass"])
        self.assertIn("M0", result["mechanism"]["failures"])
        self.assertEqual(result["classification"], "NOT_PERSIST")
        self.assertEqual(result["classification"], result["verdict"]["classification"])
        self.assertFalse(result["verdict"]["checks"]["mechanism_gates_pass"])
        self.assertFalse(result["positive_persistence_claim"]["claim"])
        self.assertFalse(result["ablation_eligibility"]["eligible"])
        self.assertEqual(set(result["mechanism"]["gates"]), set(T20.MECHANISM_GATES))
        # 判定取自记录值（float64 重算）
        base_rec = json.loads((self.baseline_path / "metrics.json").read_text("utf-8"))
        arm_rec = json.loads((self.arm_path / "metrics.json").read_text("utf-8"))
        self.assertAlmostEqual(result["verdict"]["delta_test_auc"],
                               arm_rec["test_auc_bsi"] - base_rec["test_auc_bsi"], places=15)
        self.assertAlmostEqual(result["verdict"]["delta_val_auc"],
                               arm_rec["best_val_auc_bsi"] - base_rec["best_val_auc_bsi"], places=15)
        # 二级分类 = 机械边界函数结果
        self.assertEqual(result["secondary"]["classification"],
                         T20.secondary_classification(result["verdict"]["delta_test_auc"]))
        # 描述量：best epoch / 早停 / 轨迹 / rp_arm 摘要 / 机制诊断 / 末段 Δval
        for arm_name in ("baseline", "arm"):
            desc = result["description"][arm_name]
            for key in ("best_epoch", "epochs_run", "stopped_early", "best_is_last_epoch",
                        "val_traj", "wall_seconds", "gate_mean", "variant"):
                self.assertIn(key, desc, key)
            self.assertEqual(desc["epochs_run"], len(desc["val_traj"]))
            self.assertEqual(desc["best_epoch"], desc["val_traj"].index(max(desc["val_traj"])) + 1)
        self.assertIn("rp_arm", result["description"]["arm"])
        diagnostics = result["description"]["arm"]["mechanism_diagnostics"]
        for key in ("construction_identity", "grad_probe", "alpha_final", "gate", "val_stats",
                    "dispersion", "reference_dispersion", "source_gates", "params"):
            self.assertIn(key, diagnostics, key)
        self.assertNotIn("mechanism_diagnostics", result["description"]["baseline"])
        common = min(len(result["description"]["baseline"]["val_traj"]),
                     len(result["description"]["arm"]["val_traj"]))
        self.assertEqual(result["description"]["delta_traj_common_epochs"], common)
        self.assertEqual(len(result["description"]["delta_traj"]), common)
        if common:
            self.assertEqual(result["description"]["delta_val_final"],
                             result["description"]["delta_traj"][-1])
            self.assertEqual(result["description"]["late_epoch_contradiction"],
                             result["description"]["delta_traj"][-1] <= 0.0)
        # headroom：机械因子恒落盘；applicable 由二级分类决定
        self.assertEqual(result["headroom"]["applicable"],
                         result["secondary"]["classification"] == "NO_CLEAR_IMPROVEMENT")
        self.assertIn(result["headroom"]["mechanism_activity"], ("ACTIVE", "INACTIVE"))
        self.assertIn(result["headroom"]["epoch_trajectory"],
                      ("RIGHT_CENSORED_STILL_IMPROVING", "EARLY_STOPPED"))
        bt = result["headroom"]["budget_trend"]
        self.assertEqual(bt["delta_test_5epoch"], T20.SEED2_5EPOCH_DELTA_TEST)
        self.assertEqual(bt["delta_test_10epoch"], T20.SEED2_10EPOCH_DELTA_TEST)
        self.assertEqual(bt["delta_test_20epoch"], result["verdict"]["delta_test_auc"])
        # context：5-/10-epoch 钉死值，只作对照
        ctx = result["context"]
        self.assertEqual(ctx["five_epoch"]["delta_test_auc"], T20.SEED2_5EPOCH_DELTA_TEST)
        self.assertEqual(ctx["five_epoch"]["delta_val_auc"], T20.SEED2_5EPOCH_DELTA_VAL)
        self.assertEqual(ctx["five_epoch"]["baseline_run_id"], T20.SEED2_5EPOCH_BASELINE_RUN_ID)
        self.assertEqual(ctx["five_epoch"]["arm_run_id"], T20.SEED2_5EPOCH_ARM_RUN_ID)
        self.assertEqual(ctx["ten_epoch"]["delta_test_auc"], T20.SEED2_10EPOCH_DELTA_TEST)
        self.assertEqual(ctx["ten_epoch"]["delta_val_auc"], T20.SEED2_10EPOCH_DELTA_VAL)
        self.assertEqual(ctx["ten_epoch"]["baseline_run_id"], T20.SEED2_10EPOCH_BASELINE_RUN_ID)
        self.assertEqual(ctx["ten_epoch"]["arm_run_id"], T20.SEED2_10EPOCH_ARM_RUN_ID)
        self.assertAlmostEqual(ctx["delta_test_change_vs_ten_epoch"],
                               result["verdict"]["delta_test_auc"] - T20.SEED2_10EPOCH_DELTA_TEST, places=15)
        self.assertTrue(out_path.exists())
        on_disk = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["classification"], result["classification"])
        self.assertEqual(on_disk["positive_persistence_claim"], result["positive_persistence_claim"])

    def _tamper(self) -> str:
        """篡改 fixture：机制门禁全过 + Δ 达标 + 末段 Δval 为正 ⇒ PERSISTS ∧ claim ∧ eligible。"""
        tampered_id = self.arm["run_id"] + "-tampered"
        tampered_path = P.run_dir(self.root, tampered_id)
        if tampered_path.exists():
            shutil.rmtree(tampered_path)
        shutil.copytree(self.arm_path, tampered_path)
        metrics = json.loads((tampered_path / "metrics.json").read_text(encoding="utf-8"))
        for gate in T20.MECHANISM_GATES:
            metrics["rp_arm"][gate]["pass"] = True
        base_rec = json.loads((self.baseline_path / "metrics.json").read_text(encoding="utf-8"))
        metrics["test_auc_bsi"] = base_rec["test_auc_bsi"] + 0.02
        metrics["best_val_auc_bsi"] = base_rec["best_val_auc_bsi"] + 0.02
        for base_row, arm_row in zip(base_rec["per_epoch"], metrics["per_epoch"]):
            arm_row["val_auc_bsi"] = base_row["val_auc_bsi"] + 0.002          # 末段 Δval > 0（P6 过）
        (tampered_path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
        return tampered_id

    def test_tampered_fixture_persists_and_claim_and_eligible(self):
        tampered_id = self._tamper()
        result = T20.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                                  arm_run_id=tampered_id, **self.expect_kwargs)
        self.assertTrue(all(result["identity_checks"].values()), result["identity_checks"])
        self.assertTrue(all(result["verdict"]["checks"].values()), result["verdict"]["checks"])
        self.assertEqual(result["classification"], "PERSISTS")
        self.assertEqual(result["secondary"]["classification"], "POSITIVE_IMPROVEMENT")
        self.assertTrue(result["positive_persistence_claim"]["claim"],
                        result["positive_persistence_claim"])
        self.assertTrue(result["ablation_eligibility"]["eligible"])
        self.assertEqual(result["description"]["late_epoch_contradiction"], False)

    def test_analyze_identity_mismatch_flagged_invalid(self):
        result = T20.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                                  arm_run_id=self.arm["run_id"],
                                  **{**self.expect_kwargs, "expected_stage1_id": "s1-wrong"})
        self.assertEqual(result["classification"], "INVALID")
        self.assertFalse(result["identity_checks"]["stage1_id_is_pinned"])
        self.assertFalse(all(result["identity_checks"].values()))
        self.assertFalse(result["positive_persistence_claim"]["claim"])   # identity false ⇒ claim false

    def test_analyze_epochs_mismatch_flagged_invalid(self):
        result = T20.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                                  arm_run_id=self.arm["run_id"],
                                  **{**self.expect_kwargs, "expected_epochs": 20, "expected_patience": 3})
        self.assertEqual(result["classification"], "INVALID")
        self.assertFalse(result["identity_checks"]["epochs_recorded_pinned"])

    def test_analyze_tag_mismatch_flagged_invalid(self):
        result = T20.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                                  arm_run_id=self.arm["run_id"],
                                  **{**self.expect_kwargs, "expected_tag": "xlong"})
        self.assertEqual(result["classification"], "INVALID")
        self.assertFalse(result["identity_checks"]["tag_recorded_pinned"])

    def test_analyze_reference_mismatch_flagged_invalid(self):
        result = T20.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                                  arm_run_id=self.arm["run_id"],
                                  **{**self.expect_kwargs, "expected_reference_checkpoint": "other/newtask.pt"})
        self.assertEqual(result["classification"], "INVALID")
        self.assertFalse(result["identity_checks"]["reference_checkpoint_is_pinned"])

    def test_analyze_is_read_only(self):
        before = {p.name: p.read_bytes() for p in self.arm_path.iterdir() if p.is_file()}
        T20.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                         arm_run_id=self.arm["run_id"], **self.expect_kwargs)
        after = {p.name: p.read_bytes() for p in self.arm_path.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_cli_main_writes_invalid_for_non_prereg_runs(self):
        """main() 走预注册默认 expected_*；夹具 run 不满足 → INVALID（完备性分支端到端）。"""
        out_path = Path(self.arm["run_dir"]) / "cli_compare.json"
        rc = T20.main(["--root", str(self.root), "--baseline-run", self.baseline["run_id"],
                       "--arm-run", self.arm["run_id"], "--output", str(out_path)])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out_path.read_text(encoding="utf-8"))["classification"], "INVALID")


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
        extra = set(changed) - EXPECTED_WHITELIST
        self.assertEqual(extra, set(), f"白名单外改动: {extra}")


if __name__ == "__main__":
    unittest.main()
