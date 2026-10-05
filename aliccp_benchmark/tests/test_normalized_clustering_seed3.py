"""seed3 判定分支守卫测试（预注册 §8：字节钉死 / AST 等价守卫 / seed3 身份与文档钉死）。

TDD：本文件先于机制移植提交编写（先红后绿）。被移植对象 = `exp/aliccp-stage1-normalized-env-clustering`
@ `2058de8`（seed1 修复 tip）；§8.1 的 sha256 钉死保证"移植 == seed1 修复实现"逐字节成立，
§8.2 保证移植的测试文件与 seed1 版仅差预注册 §1.1 声明的 2 处分支适配（DOC_PATH + 白名单），
其余任何改动都会红。§8.3 钉死 seed3 身份（P1）与两份文档 token。
CPU 极小、纯静态；除 git 与文件读取外无副作用。
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
import unittest
from pathlib import Path

from aliccp_benchmark import bench, protocol

# ---- 预注册钉死值（§1.1/§3/§8；不得随实现修改）----
SEED1_COMMIT = "2058de8"
REPO_ROOT = Path(__file__).resolve().parents[2]

# 机制文件：sha256（LF 归一化后，= `git show 2058de8:<path>` 字节）与 git blob 哈希
MECHANISM_PINS = {
    "aliccp_benchmark/normalized_clustering.py": (
        "e87740a0b9cbd082b99e2538b6bd72b3103923055b76f78c877a37a8a36e7f95",
        "1a68c0b48dad7b30b849a6cb9582fc17041cfbd1",
    ),
    "aliccp_benchmark/bench.py": (
        "ee28543255dd8d6b6c629adefd5ae94522bf753f0ba9b932f925b5e180841710",
        "9c01947e1277e2e0e57c9215281ec8ad097411c6",
    ),
    "aliccp_benchmark/metrics.py": (
        "f29eb951fb0ee2ba1b768d13c740eda5b6a2d0de11e370cab4c0b9878ca0bf8e",
        "491b0847f3c9b35f622ba009b262e364763bd8ff",
    ),
    "run_aliccp_benchmark.py": (
        "534be3ca65191c2ab55898aa383e76c820691d3409e9072646856c11bbf8bb4e",
        "74230871bf39d79e7203fb5ffcbdaff812ce7e76",
    ),
    "aliccp_benchmark/audit_cluster_losses.py": (
        "335383aa3da1ffb88fd4d50bcb7013dc5e95dce18bfb403e134db0fdc17e03f3",
        "de65b2c6a817a139f6ac58fd77b7f95b94a879e9",
    ),
}

PORTED_TEST_PATH = "aliccp_benchmark/tests/test_normalized_clustering.py"
SEED1_PORTED_TEST_SHA = "652fed5a70c66c548cb886e6bc9b17fa8089234acd8a5f3e9377335ef9a76817"
ADAPTED_METHODS = {("TestStaticGuards", "test_tracked_changes_subset_of_whitelist")}
ADAPTED_MODULE_ASSIGNS = {"DOC_PATH"}
PREREG_DOC_REL = "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-seed3-design.md"
AUDIT_DOC_REL = "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-mechanism-audit.md"
NEW_WHITELIST_ENTRIES = {
    PREREG_DOC_REL,
    AUDIT_DOC_REL,
    "aliccp_benchmark/audit_prior_seeds.py",
    "aliccp_benchmark/tests/test_normalized_clustering_seed3.py",
    "aliccp_benchmark/verify_seed3_results.py",
}

FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
SEED3 = 1688738016
SEED3_RAW_CFG_HASH = "47619ce078ac75497b4da9a81f0db9de59d738f22759702788f0b4ae47640108"
SEED3_RAW_STAGE1_ID = "s1-5c060b9c-m1688738016-e3-47619ce0"
SEED3_NORM_CFG_HASH = "5f899cad0d89972af1ecd2280d9933384ddcad9db84d06069a33a0fd4843f9d3"
SEED3_NORM_STAGE1_ID = "s1-5c060b9c-m1688738016-e3-5f899cad"
SEED1_NORM_STAGE1_ID = "s1-5c060b9c-m1688723512-e3-ad3b353f"
SEED2_RAW_VAL_BSI = "0.5809347091990792"
SEED2_RAW_TEST_BSI = "0.5974422649550507"


def _lf_normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_show_bytes(spec: str) -> bytes:
    return subprocess.check_output(["git", "show", spec])


def _members(cls_node: ast.ClassDef) -> dict:
    members = {}
    for i, node in enumerate(cls_node.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members[("def", node.name)] = node
        else:
            members[("stmt", type(node).__name__, i)] = node
    return members


def _module_assign_value(module_body, name: str):
    for node in module_body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and len(value.args) == 1:  # 形如 Path("...")
                return ast.literal_eval(value.args[0])
            return ast.literal_eval(value)
    raise AssertionError(f"模块级赋值未找到: {name}")


def _method_whitelist_literal(method_node):
    for node in ast.walk(method_node):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "whitelist" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("whitelist 赋值未找到")


# ---------------------------------------------------------------- §8.1 字节钉死
class TestMechanismBytePins(unittest.TestCase):
    def test_mechanism_files_byte_identical_to_seed1_pins(self):
        for rel, (sha_pin, _blob) in MECHANISM_PINS.items():
            path = REPO_ROOT / rel
            self.assertTrue(path.is_file(), msg=f"机制文件缺失: {rel}")
            self.assertEqual(_lf_normalized_sha256(path), sha_pin, msg=f"机制文件与 seed1 修复不等价: {rel}")

    def test_mechanism_blob_hashes_match_pins_and_seed1_commit(self):
        for rel, (_sha, blob_pin) in MECHANISM_PINS.items():
            working = subprocess.check_output(["git", "hash-object", rel], text=True).strip()
            self.assertEqual(working, blob_pin, msg=f"工作树 blob 哈希与钉死值不符: {rel}")
            at_seed1 = subprocess.check_output(["git", "rev-parse", f"{SEED1_COMMIT}:{rel}"], text=True).strip()
            self.assertEqual(at_seed1, blob_pin, msg=f"{SEED1_COMMIT} 上的 blob 与钉死值不符: {rel}")


# ---------------------------------------------------------------- §8.2 AST 等价守卫
class TestPortedTestAstGuard(unittest.TestCase):
    def _trees(self):
        old = ast.parse(_git_show_bytes(f"{SEED1_COMMIT}:{PORTED_TEST_PATH}").decode("utf-8"))
        new = ast.parse((REPO_ROOT / PORTED_TEST_PATH).read_text(encoding="utf-8"))
        return old, new

    def test_ported_test_differs_from_seed1_exactly_at_declared_adaptations(self):
        old, new = self._trees()
        self.assertEqual(len(old.body), len(new.body), msg="模块级语句数发生变化")

        # 模块级：仅 DOC_PATH 赋值允许不同
        module_diff = set()
        for o, n in zip(old.body, new.body):
            if isinstance(o, ast.ClassDef) or isinstance(n, ast.ClassDef):
                self.assertIsInstance(n, ast.ClassDef)
                continue
            same = ast.dump(o, include_attributes=False) == ast.dump(n, include_attributes=False)
            if not same:
                names = {getattr(t, "id", None) for t in getattr(o, "targets", [])}
                module_diff |= names
        self.assertEqual(module_diff, ADAPTED_MODULE_ASSIGNS, msg=f"模块级意外差异: {module_diff}")

        # 类与方法：逐成员 dump；仅声明的 2 个方法允许不同
        old_classes = {n.name: n for n in old.body if isinstance(n, ast.ClassDef)}
        new_classes = {n.name: n for n in new.body if isinstance(n, ast.ClassDef)}
        self.assertEqual(set(old_classes), set(new_classes), msg="测试类集合变化")
        method_diff = set()
        for cls_name, o_cls in old_classes.items():
            o_members, n_members = _members(o_cls), _members(new_classes[cls_name])
            self.assertEqual(set(o_members), set(n_members), msg=f"{cls_name} 成员集合变化")
            for key, o_m in o_members.items():
                if ast.dump(o_m, include_attributes=False) != ast.dump(n_members[key], include_attributes=False):
                    method_diff.add((cls_name, key[1]) if key[0] == "def" else (cls_name, key))
        self.assertEqual(method_diff, ADAPTED_METHODS, msg=f"意外方法差异: {method_diff}")

    def test_declared_adaptations_have_declared_content(self):
        old, new = self._trees()
        # DOC_PATH → 本分支预注册文档
        self.assertEqual(
            _module_assign_value(new.body, "DOC_PATH"), PREREG_DOC_REL,
            msg="DOC_PATH 必须指向本分支预注册文档",
        )
        self.assertNotEqual(_module_assign_value(old.body, "DOC_PATH"), _module_assign_value(new.body, "DOC_PATH"))
        # 白名单 == seed1 白名单 ∪ 本分支 5 个新文件（只增不改）
        old_cls = {n.name: n for n in old.body if isinstance(n, ast.ClassDef)}["TestStaticGuards"]
        new_cls = {n.name: n for n in new.body if isinstance(n, ast.ClassDef)}["TestStaticGuards"]
        old_wl = _method_whitelist_literal(_members(old_cls)[("def", "test_tracked_changes_subset_of_whitelist")])
        new_wl = _method_whitelist_literal(_members(new_cls)[("def", "test_tracked_changes_subset_of_whitelist")])
        self.assertEqual(new_wl, old_wl | NEW_WHITELIST_ENTRIES, msg="白名单差异必须恰为本分支 5 个新文件")

    def test_ported_file_normalized_sha_differs_from_seed1(self):
        """非钉死断言：移植后测试文件含恰 2 处适配 ⇒ LF 归一化 sha256 必须 ≠ seed1 原版。"""
        current = _lf_normalized_sha256(REPO_ROOT / PORTED_TEST_PATH)
        self.assertNotEqual(current, SEED1_PORTED_TEST_SHA)


# ---------------------------------------------------------------- §8.3 seed3 身份与文档钉死
class TestSeed3Pins(unittest.TestCase):
    def _default_cfg(self, model_seed: int) -> dict:
        return bench._stage1_cfg(
            prefix_tag=protocol.PREFIX_TAG,
            budgets={"train": protocol.TRAIN_BUDGET, "val": protocol.VAL_BUDGET, "test": protocol.TEST_BUDGET},
            model_seed=model_seed, env_seed=protocol.ENV_SEED,
            epochs=protocol.STAGE1_EPOCHS, patience=protocol.STAGE1_PATIENCE,
            batch_size=protocol.BATCH_SIZE, lr=protocol.LR, uni_coe=protocol.UNI_COE, env_coe=protocol.ENV_COE,
            reg_embedding=protocol.REG_EMBEDDING, reg_dnn=protocol.REG_DNN,
            embedding_size=protocol.EMBEDDING_SIZE, input_size=protocol.INPUT_SIZE,
            expert_hidden=protocol.EXPERT_HIDDEN, tower_hidden=protocol.TOWER_HIDDEN,
            dropout=protocol.DROPOUT, vocab=protocol.build_vocab(),
        )

    def test_seed3_predicted_stage1_identity(self):
        raw_cfg = self._default_cfg(SEED3)
        self.assertNotIn("clustering", raw_cfg)
        self.assertEqual(protocol.config_hash(raw_cfg), SEED3_RAW_CFG_HASH)
        self.assertEqual(
            protocol.make_stage1_id(FINGERPRINT_SHA, SEED3, protocol.STAGE1_EPOCHS, SEED3_RAW_CFG_HASH),
            SEED3_RAW_STAGE1_ID,
        )
        arm_cfg = {**raw_cfg, "clustering": "rank_normalized"}
        self.assertEqual(set(arm_cfg), set(raw_cfg) | {"clustering"})  # 仅多一个键
        self.assertEqual(protocol.config_hash(arm_cfg), SEED3_NORM_CFG_HASH)
        self.assertEqual(
            protocol.make_stage1_id(FINGERPRINT_SHA, SEED3, protocol.STAGE1_EPOCHS, SEED3_NORM_CFG_HASH),
            SEED3_NORM_STAGE1_ID,
        )

    def test_prereg_doc_pins(self):
        text = (REPO_ROOT / PREREG_DOC_REL).read_text(encoding="utf-8")
        for token in (
            SEED3_RAW_STAGE1_ID, SEED3_NORM_STAGE1_ID, SEED3_RAW_CFG_HASH, SEED3_NORM_CFG_HASH,
            "1688738016", "1688723512", "1688723740",
            "0.5534062000766764", "0.5420522697385088", "0.5974422649550507", "0.5809347091990792",
            "0.0004588195", "0.0014831645",
            "0.5481837665250074", "0.5988392178311113", "0.5781533414372665",
            "0.0082428399", "0.0100592719",
            "566", "1532", "999966", "959244", "1040756",
            "+0.001", "−0.02", "0.005", "0.01",
            "POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION",
            "MECHANISM_REPAIRED_SEED3", "mechanism repair effective but utility unstable",
            "HEADROOM_PLAUSIBLE", "HEADROOM_NOT_EVIDENT",
            SEED1_NORM_STAGE1_ID, "4b983fc9f485d7cba853b8d5f0846b292e85344fc5b6f3725845dbba5ea0d52b",
            "3a30e2c0b1e8a2b4e9fecaa4d76893b775f6ee9e597922dce7303ac3b77b08c4",
            "ad3b353f6d436ac95c26703e504a318e3441933a4e00e3f3d6c5e22a05bfe96e",
            "REPAIRED", "NOT_REPAIRED", "NO_MATERIAL_DEGRADATION",
            "e87740a0b9cbd082b99e2538b6bd72b3103923055b76f78c877a37a8a36e7f95",
            "652fed5a70c66c548cb886e6bc9b17fa8089234acd8a5f3e9377335ef9a76817",
        ):
            self.assertIn(token, text, msg=f"预注册文档缺少 token: {token}")

    def test_audit_doc_pins(self):
        text = (REPO_ROOT / AUDIT_DOC_REL).read_text(encoding="utf-8")
        for token in (
            "2058de8", "7ab445c", SEED1_NORM_STAGE1_ID, "s1-5c060b9c-m1688723740-e3-820697f6",
            "1000466", "1000418", "998878", "999966", "959244", "1040756", "1106286", "893714",
            "566", "1532", "0.99977", "0.999252", "0.18375", "0.668745",
            SEED2_RAW_VAL_BSI, SEED2_RAW_TEST_BSI, "0.0082428399", "0.0100592719",
            "0.0004588195", "0.0014831645", "all_pins_pass",
        ):
            self.assertIn(token, text, msg=f"审计文档缺少 token: {token}")


if __name__ == "__main__":
    unittest.main()
