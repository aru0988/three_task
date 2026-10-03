"""residual-prompt canonical seed-2 复现分支守卫测试（预注册 §1.4/§8：重建等式钉死 / 字节 sha 钉死 / seed2 身份与文档钉死）。

TDD：本文件先于机制移植提交编写（先红后绿）。被复现对象 = `exp/aliccp-stage2-residual-prompt-gate`
@ `79ddefa`（机制 blob `5dc7158c…`）；预注册 §1.4 的适配 A1′–A3′ 之外任何改动都会红：
  * A1′：机制文件 == 钉死 blob 经"docstring 替换 + 3 行 run-reference 常量替换"的重建等式；
  * A2′：接线二文件 == 钉死 blob 逐字节（且相对 8133d32 的 diff 与 seed1 相同）；
  * A3′：移植测试文件 == 钉死 blob 经"docstring + DOC_PATH + WHITELIST + 2 方法"替换的重建等式。
CPU 极小、纯静态（git + 文件读取）；被钉 commit 不可得时**响亮失败**（不静默跳过）。
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED1_IMPL_COMMIT = "79ddefa"
FROZEN_BASE = "8133d32"

PREREG_DOC_REL = "docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md"
GUARD_TEST_REL = "aliccp_benchmark/tests/test_residual_prompt_seed_replication.py"
MECHANISM_REL = "aliccp_benchmark/residual_prompt.py"
PORTED_TEST_REL = "aliccp_benchmark/tests/test_residual_prompt.py"

# ---- §1.1 钉死（被复现对象 @79ddefa；blob + LF-normalized sha256）----
PIN_MECHANISM_BLOB = "5dc7158ca999c9e7e6217a999c37869231411549"
PIN_MECHANISM_LF_SHA = "1c6131c7538718485ea7d7a65bd6e1d6d6a9c8dd21ba7645d66e62437020e996"
PIN_PORTED_TEST_BLOB = "6484f5b8c9393ce22651d0f9161d81a0ef383f17"
PIN_PORTED_TEST_LF_SHA = "6a4a9c6ad6911606e69dce81c3116e36461f824da3f78ac188393b65296b6956"
PIN_CLI_BLOB = "229c25a1c86719fa6fb6b057d6cff3f4120fa053"
PIN_CLI_LF_SHA = "2966ec3985e5b91e277f429c2f236fb551de7479d114def90f767f2d813ad285"
PIN_BENCH_BLOB = "bf3647686838157bf5fd7645615faad9f2d7185d"
PIN_BENCH_LF_SHA = "b063a36667173b66fc8a7ac932502cc34753c91e2bfcfdafaedfce443f1db3b4"
BASE_CLI_LF_SHA = "3940de3eb9a4d0f9d42d9d1359ccca52a55bc5e2814b4cdc4b8f8224ae5e10d0"
BASE_BENCH_LF_SHA = "95123344e1c7f107e91aa0e26ee7b316725f2b48d70e9425b2ed923e15be0dc0"

# ---- §1.4-A1′：机制文件恰 3 行 run-reference 常量替换（新值钉死；seed1 旧值见钉死 blob）----
MECHANISM_CONSTANTS = {
    "BASELINE_AUC_TEST": 0.5974422649550507,
    "BASELINE_AUC_VAL": 0.5809347091990792,
    "REFERENCE_PRED_STD": 0.005217193225189258,
}
SEED1_MECHANISM_CONSTANTS = {
    "BASELINE_AUC_TEST": 0.5988392178311113,
    "BASELINE_AUC_VAL": 0.5781533414372665,
    "REFERENCE_PRED_STD": 0.0033728455401762676,
}

# ---- 移植后工作树文件的 LF sha256 钉死（C2 提交时计算并写死；见预注册 §1.4）----
PORTED_MECHANISM_LF_SHA = "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"
PORTED_TEST_LF_SHA = "dab737a1422b88befc4c19f2bee408618061de7fd3bf0b2f83ebcdd85fd1cbc2"

# ---- §1.4-A3′：移植测试文件恰好适配（差异集合精确钉死）----
ADAPTED_TEST_METHODS = {
    ("TestPreregConstants", "test_thresholds_and_baseline_reference"),
    ("TestPreregConstants", "test_prereg_doc_tokens_and_summary_ledger"),
}
EXPECTED_WHITELIST = {
    PREREG_DOC_REL,
    "aliccp_benchmark/residual_prompt.py",
    "aliccp_benchmark/tests/test_residual_prompt.py",
    GUARD_TEST_REL,
    "run_aliccp_benchmark.py",
    "aliccp_benchmark/bench.py",
    "artifacts/aliccp_bench/SUMMARY.md",
}

# ---- §8.3 seed2 身份与文档 token 钉死 ----
DOC_TOKENS = [
    "79ddefa", "3e2f083", "221580a",
    "107221b26382da7fd44990e167d9a61da2680d92",
    "8139e067a57fd3e144846ca5d39015a1d51657d55da67b9da9e4e06c007658df",
    "847097b7cb48a56c8593dd76d63a71a21a4c9e4f",
    "5dc7158ca999c9e7e6217a999c37869231411549",
    "6484f5b8c9393ce22651d0f9161d81a0ef383f17",
    PIN_CLI_BLOB, PIN_BENCH_BLOB,
    "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
    "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07",
    "20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9",
    "20261004-0214-p2M-v500k-t1M-m1688723512-short-79ddefa-rpg",
    "0.5809347091990792", "0.5974422649550507", "0.005217193225189258",
    "0.5781533414372665", "0.5988392178311113",
    "0.005394543882337954", "0.00448174078674779",
    "0.0055", "2385", "8129", "-rpg",
    "VALID_POSITIVE", "VALID_NEGATIVE", "MECHANISM_FAIL",
    "REPLICATION_SUPPORTED", "NOT_SUPPORTED", "DIRECTION_REPLICATED_THRESHOLD_NOT_CLEARED",
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
    test.assertEqual(text.count(old), 1, f"{what}: 旧文本在钉死文本中应恰出现 1 次（实际 {text.count(old)}）")
    return text.replace(old, new)


def _line_starting_with(text: str, prefix: str) -> str:
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise AssertionError(f"期望恰 1 行以 {prefix!r} 开头，实际 {len(lines)} 行")
    return lines[0]


def _class_method(treesrc: str, cls_name: str, method: str):
    tree = ast.parse(treesrc)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method:
                    return ast.get_source_segment(treesrc, member)
    raise AssertionError(f"未找到 {cls_name}.{method}")


def _module_assign_literal(treesrc: str, name: str):
    tree = ast.parse(treesrc)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"未找到模块级赋值 {name}")


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
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
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
    raise AssertionError(f"未找到模块级赋值 {name}")


def _module_doc_segment(treesrc: str) -> str:
    """模块 docstring 的**原始源文本段**（含三引号；不经 escape 处理，供逐字节重建等式）。"""
    tree = ast.parse(treesrc)
    first = tree.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        raise AssertionError("首语句不是 docstring")
    segment = ast.get_source_segment(treesrc, first)
    if segment is None:
        raise AssertionError("docstring 源段不可得")
    return segment


class TestMechanismReconstructEquality(unittest.TestCase):
    """A1′：机制文件 == 钉死 blob + docstring 替换 + 3 行常量替换（逐字节重建等式）。"""

    def _rebuild(self, working: str) -> str:
        pinned = _git_show_text(f"{SEED1_IMPL_COMMIT}:{MECHANISM_REL}")
        modified = _replace_once(self, pinned, _module_doc_segment(pinned),
                                 _module_doc_segment(working), "模块 docstring 替换")
        for name, value in MECHANISM_CONSTANTS.items():
            old_line = _line_starting_with(modified, f"{name} = ")
            new_line = _line_starting_with(working, f"{name} = ")
            self.assertNotEqual(old_line, new_line, f"{name} 必须发生文档化的替换")
            old_value = float(old_line.split("=", 1)[1].split("#")[0].strip())
            self.assertEqual(old_value, SEED1_MECHANISM_CONSTANTS[name], f"{name} 钉死旧值")
            new_value = float(new_line.split("=", 1)[1].split("#")[0].strip())
            self.assertEqual(new_value, value, f"{name} 新值必须与预注册一致")
            modified = _replace_once(self, modified, old_line, new_line, f"{name} 常量行替换")
        return modified

    def test_mechanism_equals_pinned_blob_with_exactly_documented_edits(self):
        working = _read_lf(MECHANISM_REL)
        self.assertEqual(working, self._rebuild(working),
                         "机制文件相对钉死 blob 除 docstring + 3 行常量外必须逐字节一致")

    def test_mechanism_lf_sha_pinned(self):
        self.assertNotEqual(PORTED_MECHANISM_LF_SHA, "TBD_PORTED_MECHANISM_LF_SHA", "pin 未填写（C2 时计算）")
        self.assertEqual(_lf_sha(MECHANISM_REL), PORTED_MECHANISM_LF_SHA)

    def test_mechanism_docstring_references_seed1_pin_and_prereg(self):
        doc = ast.get_docstring(ast.parse(_read_lf(MECHANISM_REL)), clean=False) or ""
        for token in (SEED1_IMPL_COMMIT, PIN_MECHANISM_BLOB, PREREG_DOC_REL, "1688723740"):
            self.assertIn(token, doc)

    def test_pinned_mechanism_blob_available_and_matches(self):
        blob = subprocess.check_output(["git", "rev-parse", f"{SEED1_IMPL_COMMIT}:{MECHANISM_REL}"],
                                       cwd=REPO_ROOT, text=True).strip()
        self.assertEqual(blob, PIN_MECHANISM_BLOB)
        pinned = _git_show_text(f"{SEED1_IMPL_COMMIT}:{MECHANISM_REL}")
        self.assertEqual(hashlib.sha256(pinned.encode("utf-8")).hexdigest(), PIN_MECHANISM_LF_SHA)


class TestWiringByteIdentical(unittest.TestCase):
    """A2′：run_aliccp_benchmark.py 与 bench.py 逐字节 == 钉死 blob；对 8133d32 的 diff 与 seed1 相同。"""

    CASES = [("run_aliccp_benchmark.py", PIN_CLI_BLOB, PIN_CLI_LF_SHA, BASE_CLI_LF_SHA),
             ("aliccp_benchmark/bench.py", PIN_BENCH_BLOB, PIN_BENCH_LF_SHA, BASE_BENCH_LF_SHA)]

    def test_byte_identical_to_pinned_blob(self):
        for rel, blob_pin, sha_pin, _base_sha in self.CASES:
            self.assertEqual(_read_lf(rel), _git_show_text(f"{SEED1_IMPL_COMMIT}:{rel}"), rel)
            self.assertEqual(_lf_sha(rel), sha_pin, rel)
            working_blob = subprocess.check_output(["git", "hash-object", rel], cwd=REPO_ROOT, text=True).strip()
            self.assertEqual(working_blob, blob_pin, rel)

    def test_diff_vs_frozen_base_equals_seed1_diff(self):
        for rel, _blob, _sha, base_sha in self.CASES:
            self.assertEqual(hashlib.sha256(_git_show_text(f"{FROZEN_BASE}:{rel}").encode("utf-8")).hexdigest(),
                             base_sha, f"{FROZEN_BASE} 基线文件与钉死不符: {rel}")
            ours = subprocess.check_output(["git", "diff", FROZEN_BASE, "--", rel],
                                           cwd=REPO_ROOT).decode("utf-8")
            seed1 = subprocess.check_output(["git", "diff", FROZEN_BASE, SEED1_IMPL_COMMIT, "--", rel],
                                            cwd=REPO_ROOT).decode("utf-8")
            self.assertEqual(ours.replace("\r\n", "\n"), seed1.replace("\r\n", "\n"),
                             f"{rel} 相对基线的 diff 必须与 seed1 移植相同")


class TestPortedTestReconstructEquality(unittest.TestCase):
    """A3′：移植测试文件 == 钉死 blob + docstring + DOC_PATH + WHITELIST + 2 方法（逐字节重建等式）。"""

    def _rebuild(self, working: str) -> str:
        pinned = _git_show_text(f"{SEED1_IMPL_COMMIT}:{PORTED_TEST_REL}")
        modified = _replace_once(self, pinned, _module_doc_segment(pinned),
                                 _module_doc_segment(working), "模块 docstring 替换")
        for name in ("DOC_PATH", "WHITELIST"):
            # 多行集合字面量：用 AST source segment 精确框定
            old_seg = ast.get_source_segment(modified, self._assign_node(modified, name))
            new_seg = ast.get_source_segment(working, self._assign_node(working, name))
            modified = _replace_once(self, modified, old_seg, new_seg, f"{name} 赋值替换")
        for cls_name, method in sorted(ADAPTED_TEST_METHODS):
            old_seg = _class_method(modified, cls_name, method)
            new_seg = _class_method(working, cls_name, method)
            self.assertNotEqual(old_seg, new_seg, f"{cls_name}.{method} 必须发生文档化适配")
            modified = _replace_once(self, modified, old_seg, new_seg, f"{cls_name}.{method} 替换")
        return modified

    @staticmethod
    def _assign_node(treesrc: str, name: str):
        tree = ast.parse(treesrc)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                return node
        raise AssertionError(f"未找到模块级赋值 {name}")

    def test_ported_test_equals_pinned_blob_with_exactly_documented_edits(self):
        working = _read_lf(PORTED_TEST_REL)
        self.assertEqual(working, self._rebuild(working),
                         "移植测试文件相对钉死 blob 除 4 处文档化适配外必须逐字节一致")

    def test_ported_test_lf_sha_pinned(self):
        self.assertNotEqual(PORTED_TEST_LF_SHA, "TBD_PORTED_TEST_LF_SHA", "pin 未填写（C2 时计算）")
        self.assertEqual(_lf_sha(PORTED_TEST_REL), PORTED_TEST_LF_SHA)

    def test_adapted_module_values_exact(self):
        working = _read_lf(PORTED_TEST_REL)
        self.assertEqual(_module_assign_literal(working, "DOC_PATH"), PREREG_DOC_REL)
        self.assertEqual(_module_assign_set_resolved(working, "WHITELIST"), EXPECTED_WHITELIST)
        pinned = _git_show_text(f"{SEED1_IMPL_COMMIT}:{PORTED_TEST_REL}")
        # 未适配常量必须与钉死版逐字相同
        for name in ("PIN_BLOB", "PIN_SHA256_LF", "PIN_TEST_BLOB", "FROZEN_BASE"):
            self.assertEqual(_module_assign_literal(working, name),
                             _module_assign_literal(pinned, name), name)

    def test_adapted_methods_are_exactly_the_documented_set(self):
        working = _read_lf(PORTED_TEST_REL)
        pinned = _git_show_text(f"{SEED1_IMPL_COMMIT}:{PORTED_TEST_REL}")
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
        self.assertEqual(differences, ADAPTED_TEST_METHODS, "AST 差异集合必须恰为文档化的 2 个方法")

    def test_pinned_ported_test_blob_available_and_matches(self):
        blob = subprocess.check_output(["git", "rev-parse", f"{SEED1_IMPL_COMMIT}:{PORTED_TEST_REL}"],
                                       cwd=REPO_ROOT, text=True).strip()
        self.assertEqual(blob, PIN_PORTED_TEST_BLOB)
        pinned = _git_show_text(f"{SEED1_IMPL_COMMIT}:{PORTED_TEST_REL}")
        self.assertEqual(hashlib.sha256(pinned.encode("utf-8")).hexdigest(), PIN_PORTED_TEST_LF_SHA)


class TestSeed2DocAndIdentity(unittest.TestCase):
    """§8.3：预注册文档 token 钉死。"""

    def test_prereg_doc_tokens(self):
        doc = _read_lf(PREREG_DOC_REL)
        missing = [token for token in DOC_TOKENS if token not in doc]
        self.assertEqual(missing, [], f"预注册文档缺失 token: {missing}")

    def test_summary_ledger_lineage_row(self):
        summary = _read_lf("artifacts/aliccp_bench/SUMMARY.md")
        row = [line for line in summary.splitlines() if "b2e17f9" in line and "run_id" not in line]
        self.assertEqual(len(row), 1)
        self.assertIn("0.598839", row[0])
        self.assertIn("0.578153", row[0])


if __name__ == "__main__":
    unittest.main()
