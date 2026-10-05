"""seed4/5 判定分支守卫测试（预注册 §7：字节钉死 / 不可变证据 / 白名单级联 / P1 身份与文档钉死 / 验证器机械函数）。

TDD：本文件先于 `verify_seeds45_results.py` 实现编写（先红后绿）。被钉死对象 = `exp/aliccp-stage1-normalized-clustering-seed3`
@ `f8ff4cf`（本分支起点）与其机制字节钉死 @ `2058de8`（seed1 修复 tip）。§7.1 的 sha256 钉死保证"机制 == seed1 修复实现"
逐字节成立；§7.2 保证不可变证据与白名单级联的差异**逐字节恰为**声明内容；§3.1 钉死 seed4/5 的四个 `config_hash` 与
四个 `stage1_id`（P1）；§5 钉死分类/汇总阈值与标签。CPU 极小、纯静态（freeze-p4 用合成小张量）；除 git 与文件读取外无副作用。
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from aliccp_benchmark import bench, protocol
from aliccp_benchmark import verify_seeds45_results as v45

# ---- 预注册钉死值（§3.1/§5/§7；不得随实现修改）----
SEED1_COMMIT = "2058de8"
BASE_COMMIT = "f8ff4cf"
REPO_ROOT = Path(__file__).resolve().parents[2]

# 机制文件：sha256（LF 归一化后，= `git show 2058de8:<path>` 字节）与 git blob 哈希（同 seed3 §1.1，逐字不变）
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
SEED3_GUARD_PATH = "aliccp_benchmark/tests/test_normalized_clustering_seed3.py"
SEED3_VERIFIER_PATH = "aliccp_benchmark/verify_seed3_results.py"
SEED3_DOC_REL = "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-seed3-design.md"
SEED3_AUDIT_DOC_REL = "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-mechanism-audit.md"
SEEDS45_DOC_REL = "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-seeds45-design.md"

# §7.2 不可变证据（LF 归一化 sha256 @ f8ff4cf）
SEED3_VERIFIER_SHA = "3a26aac8e6a4c39eadd67b0e13a11fefa804ba753e1ae4ced3ab660584e4c9d0"
SEED3_DOC_SHA = "13d0f03f49120c91a4c920dedbf4efd5d8f2ec30979ffeccabfa4f73b6641adc"
SEED3_AUDIT_DOC_SHA = "426dfcd889c928a38b615d0d68bbf18d39a87f36c05fb67a9cd0ca5530663896"

# §7.3 白名单级联：f8ff4cf 版本 → 本分支版本 的逐字节替换块（LF）
PORTED_WL_OLD = '''        whitelist = {
            ".gitignore",
            "aliccp_benchmark/audit_cluster_losses.py",
            "aliccp_benchmark/bench.py",
            "aliccp_benchmark/metrics.py",
            "aliccp_benchmark/normalized_clustering.py",
            "aliccp_benchmark/tests/test_normalized_clustering.py",
            "run_aliccp_benchmark.py",
            "docs/superpowers/specs/2026-10-03-aliccp-stage1-normalized-clustering-design.md",
            "artifacts/aliccp_bench/SUMMARY.md",
            "aliccp_benchmark/audit_prior_seeds.py",
            "aliccp_benchmark/tests/test_normalized_clustering_seed3.py",
            "aliccp_benchmark/verify_seed3_results.py",
            "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-mechanism-audit.md",
            "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-seed3-design.md",
        }'''
PORTED_WL_NEW = '''        whitelist = {
            ".gitignore",
            "aliccp_benchmark/audit_cluster_losses.py",
            "aliccp_benchmark/bench.py",
            "aliccp_benchmark/metrics.py",
            "aliccp_benchmark/normalized_clustering.py",
            "aliccp_benchmark/tests/test_normalized_clustering.py",
            "run_aliccp_benchmark.py",
            "docs/superpowers/specs/2026-10-03-aliccp-stage1-normalized-clustering-design.md",
            "artifacts/aliccp_bench/SUMMARY.md",
            "aliccp_benchmark/audit_prior_seeds.py",
            "aliccp_benchmark/tests/test_normalized_clustering_seed3.py",
            "aliccp_benchmark/verify_seed3_results.py",
            "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-mechanism-audit.md",
            "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-seed3-design.md",
            "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-seeds45-design.md",
            "aliccp_benchmark/tests/test_normalized_clustering_seeds45.py",
            "aliccp_benchmark/verify_seeds45_results.py",
        }'''
SEED3_GUARD_WL_OLD = '''NEW_WHITELIST_ENTRIES = {
    PREREG_DOC_REL,
    AUDIT_DOC_REL,
    "aliccp_benchmark/audit_prior_seeds.py",
    "aliccp_benchmark/tests/test_normalized_clustering_seed3.py",
    "aliccp_benchmark/verify_seed3_results.py",
}'''
SEED3_GUARD_WL_NEW = '''NEW_WHITELIST_ENTRIES = {
    PREREG_DOC_REL,
    AUDIT_DOC_REL,
    "aliccp_benchmark/audit_prior_seeds.py",
    "aliccp_benchmark/tests/test_normalized_clustering_seed3.py",
    "aliccp_benchmark/verify_seed3_results.py",
    "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-seeds45-design.md",
    "aliccp_benchmark/tests/test_normalized_clustering_seeds45.py",
    "aliccp_benchmark/verify_seeds45_results.py",
}'''
SEED3_GUARD_MSG_OLD = 'msg="白名单差异必须恰为本分支 5 个新文件"'
SEED3_GUARD_MSG_NEW = 'msg="白名单差异必须恰为本分支及其 canonical 续行（seeds45）新增文件"'

WHITELIST_14 = {
    ".gitignore",
    "aliccp_benchmark/audit_cluster_losses.py",
    "aliccp_benchmark/bench.py",
    "aliccp_benchmark/metrics.py",
    "aliccp_benchmark/normalized_clustering.py",
    "aliccp_benchmark/tests/test_normalized_clustering.py",
    "run_aliccp_benchmark.py",
    "docs/superpowers/specs/2026-10-03-aliccp-stage1-normalized-clustering-design.md",
    "artifacts/aliccp_bench/SUMMARY.md",
    "aliccp_benchmark/audit_prior_seeds.py",
    "aliccp_benchmark/tests/test_normalized_clustering_seed3.py",
    "aliccp_benchmark/verify_seed3_results.py",
    "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-mechanism-audit.md",
    "docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-seed3-design.md",
}
SEEDS45_NEW_FILES = {
    SEEDS45_DOC_REL,
    "aliccp_benchmark/tests/test_normalized_clustering_seeds45.py",
    "aliccp_benchmark/verify_seeds45_results.py",
}
WHITELIST_17 = WHITELIST_14 | SEEDS45_NEW_FILES
ADAPTED_METHODS = {("TestStaticGuards", "test_tracked_changes_subset_of_whitelist")}
ADAPTED_MODULE_ASSIGNS = {"DOC_PATH"}

# §3.1 P1（先于 run 的确定性预测；方法自证 12/12 见预注册 §3.1）
FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
SEED4, SEED5 = 1688749593, 1688762746
SEED4_RAW_CFG_HASH = "bb2b68de2dac257c20d7b50544f83e672a12eb2e04754ec476a66f90788a8cb3"
SEED4_RAW_STAGE1_ID = "s1-5c060b9c-m1688749593-e3-bb2b68de"
SEED4_NORM_CFG_HASH = "cf810ec493f52555dc7e69bcb971537c52b9f6b14dca6234759447ed1e9ac3fe"
SEED4_NORM_STAGE1_ID = "s1-5c060b9c-m1688749593-e3-cf810ec4"
SEED5_RAW_CFG_HASH = "6343490f5e16d99e31213d5bb4a91459a173b782616eda2184d9c4c3d94c75d2"
SEED5_RAW_STAGE1_ID = "s1-5c060b9c-m1688762746-e3-6343490f"
SEED5_NORM_CFG_HASH = "a8238feaa483aada38f0280971f6c5b6fd5fbfdb01ab67eda68dac99a501f0ae"
SEED5_NORM_STAGE1_ID = "s1-5c060b9c-m1688762746-e3-a8238fea"

# §5 冻结常数（预注册字面值）
T975_DF4 = 2.7764451051977987
SEED3_DELTA_TEST = 0.02347892658289208
SEED3_DELTA_VAL = 0.017557536803150087
# 合成向量 [0.01, 0.002, -0.001, 0.004, 0.0005] 的手算统计（scipy t.ppf(0.975,4) 复算）
SYNTH_V = [0.01, 0.002, -0.001, 0.004, 0.0005]
SYNTH_MEAN = 0.0031
SYNTH_SD = 0.004277849927241487
SYNTH_CI_LOW = -0.002211652244374357
SYNTH_CI_HIGH = 0.008411652244374356


def _lf_normalized_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _disk_lf(rel: str) -> bytes:
    return (REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")


def _git_show_bytes(spec: str) -> bytes:
    return subprocess.check_output(["git", "show", spec])


def _members(cls_node):
    import ast

    members = {}
    for i, node in enumerate(cls_node.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members[("def", node.name)] = node
        else:
            members[("stmt", type(node).__name__, i)] = node
    return members


def _whitelist_literal_from_ast(text: str) -> set:
    import ast

    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "whitelist" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError("whitelist 赋值未找到")


# ---------------------------------------------------------------- §7.1 字节钉死
class TestMechanismBytePins(unittest.TestCase):
    def test_mechanism_files_byte_identical_to_seed1_pins(self):
        for rel, (sha_pin, _blob) in MECHANISM_PINS.items():
            self.assertTrue((REPO_ROOT / rel).is_file(), msg=f"机制文件缺失: {rel}")
            self.assertEqual(_lf_normalized_sha256_bytes(_disk_lf(rel)), sha_pin, msg=f"机制文件与 seed1 修复不等价: {rel}")

    def test_mechanism_blob_hashes_match_pins_and_seed1_commit(self):
        for rel, (_sha, blob_pin) in MECHANISM_PINS.items():
            working = subprocess.check_output(["git", "hash-object", rel], text=True).strip()
            self.assertEqual(working, blob_pin, msg=f"工作树 blob 哈希与钉死值不符: {rel}")
            at_seed1 = subprocess.check_output(["git", "rev-parse", f"{SEED1_COMMIT}:{rel}"], text=True).strip()
            self.assertEqual(at_seed1, blob_pin, msg=f"{SEED1_COMMIT} 上的 blob 与钉死值不符: {rel}")


# ---------------------------------------------------------------- §7.2 不可变证据与 append-only
class TestImmutableEvidence(unittest.TestCase):
    def test_seed3_verifier_and_docs_unchanged_vs_f8ff4cf(self):
        self.assertEqual(_lf_normalized_sha256_bytes(_disk_lf(SEED3_VERIFIER_PATH)), SEED3_VERIFIER_SHA)
        self.assertEqual(_lf_normalized_sha256_bytes(_disk_lf(SEED3_DOC_REL)), SEED3_DOC_SHA)
        self.assertEqual(_lf_normalized_sha256_bytes(_disk_lf(SEED3_AUDIT_DOC_REL)), SEED3_AUDIT_DOC_SHA)

    def test_summary_append_only_prefix_from_f8ff4cf(self):
        old = _git_show_bytes(f"{BASE_COMMIT}:artifacts/aliccp_bench/SUMMARY.md").replace(b"\r\n", b"\n")
        self.assertEqual(len(old.splitlines()), 6, msg="f8ff4cf SUMMARY 行数应为 6")
        self.assertTrue(_disk_lf("artifacts/aliccp_bench/SUMMARY.md").startswith(old), msg="SUMMARY 既有行被改写（非 append-only）")

    def test_seed3_guard_differs_exactly_at_declared_cascade(self):
        old = _git_show_bytes(f"{BASE_COMMIT}:{SEED3_GUARD_PATH}").replace(b"\r\n", b"\n")
        expected = (
            old.replace(SEED3_GUARD_WL_OLD.encode("utf-8"), SEED3_GUARD_WL_NEW.encode("utf-8"))
            .replace(SEED3_GUARD_MSG_OLD.encode("utf-8"), SEED3_GUARD_MSG_NEW.encode("utf-8"))
        )
        self.assertNotEqual(old, expected, msg="级联替换未生效（替换块未命中原文）")
        self.assertEqual(_disk_lf(SEED3_GUARD_PATH), expected, msg="seed3 守卫文件的差异不止声明的级联替换")


# ---------------------------------------------------------------- §7.3 白名单级联（ported test）
class TestWhitelistCascade(unittest.TestCase):
    def test_ported_test_differs_exactly_at_whitelist_extension(self):
        old = _git_show_bytes(f"{BASE_COMMIT}:{PORTED_TEST_PATH}").replace(b"\r\n", b"\n")
        expected = old.replace(PORTED_WL_OLD.encode("utf-8"), PORTED_WL_NEW.encode("utf-8"))
        self.assertNotEqual(old, expected, msg="级联替换未生效（替换块未命中原文）")
        self.assertEqual(_disk_lf(PORTED_TEST_PATH), expected, msg="ported test 的差异不止声明的白名单扩展")

    def test_whitelist_values_old_and_new(self):
        old_text = _git_show_bytes(f"{BASE_COMMIT}:{PORTED_TEST_PATH}").decode("utf-8")
        new_text = _disk_lf(PORTED_TEST_PATH).decode("utf-8")
        self.assertEqual(_whitelist_literal_from_ast(old_text), WHITELIST_14)
        self.assertEqual(_whitelist_literal_from_ast(new_text), WHITELIST_17)
        self.assertEqual(WHITELIST_17 - WHITELIST_14, SEEDS45_NEW_FILES)

    def test_ported_test_ast_guard_vs_seed1_still_exactly_two_sites(self):
        import ast

        old = ast.parse(_git_show_bytes(f"{SEED1_COMMIT}:{PORTED_TEST_PATH}").decode("utf-8"))
        new = ast.parse(_disk_lf(PORTED_TEST_PATH).decode("utf-8"))
        self.assertEqual(len(old.body), len(new.body), msg="模块级语句数发生变化")
        module_diff = set()
        for o, n in zip(old.body, new.body):
            if isinstance(o, ast.ClassDef) or isinstance(n, ast.ClassDef):
                self.assertIsInstance(n, ast.ClassDef)
                continue
            if ast.dump(o, include_attributes=False) != ast.dump(n, include_attributes=False):
                module_diff |= {getattr(t, "id", None) for t in getattr(o, "targets", [])}
        self.assertEqual(module_diff, ADAPTED_MODULE_ASSIGNS, msg=f"模块级意外差异: {module_diff}")
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


# ---------------------------------------------------------------- §3.1 P1 身份
class TestSeeds45Identity(unittest.TestCase):
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

    def test_seed4_seed5_predicted_stage1_identity(self):
        for seed, raw_hash, raw_sid, norm_hash, norm_sid in (
            (SEED4, SEED4_RAW_CFG_HASH, SEED4_RAW_STAGE1_ID, SEED4_NORM_CFG_HASH, SEED4_NORM_STAGE1_ID),
            (SEED5, SEED5_RAW_CFG_HASH, SEED5_RAW_STAGE1_ID, SEED5_NORM_CFG_HASH, SEED5_NORM_STAGE1_ID),
        ):
            raw_cfg = self._default_cfg(seed)
            self.assertNotIn("clustering", raw_cfg)
            self.assertEqual(protocol.config_hash(raw_cfg), raw_hash)
            self.assertEqual(protocol.make_stage1_id(FINGERPRINT_SHA, seed, protocol.STAGE1_EPOCHS, raw_hash), raw_sid)
            arm_cfg = {**raw_cfg, "clustering": "rank_normalized"}
            self.assertEqual(set(arm_cfg), set(raw_cfg) | {"clustering"})
            self.assertEqual(protocol.config_hash(arm_cfg), norm_hash)
            self.assertEqual(protocol.make_stage1_id(FINGERPRINT_SHA, seed, protocol.STAGE1_EPOCHS, norm_hash), norm_sid)

    def test_verifier_p1_constants_match_pins(self):
        self.assertEqual(v45.P1[SEED4]["raw_cfg_hash"], SEED4_RAW_CFG_HASH)
        self.assertEqual(v45.P1[SEED4]["raw_stage1_id"], SEED4_RAW_STAGE1_ID)
        self.assertEqual(v45.P1[SEED4]["norm_cfg_hash"], SEED4_NORM_CFG_HASH)
        self.assertEqual(v45.P1[SEED4]["norm_stage1_id"], SEED4_NORM_STAGE1_ID)
        self.assertEqual(v45.P1[SEED5]["raw_cfg_hash"], SEED5_RAW_CFG_HASH)
        self.assertEqual(v45.P1[SEED5]["raw_stage1_id"], SEED5_RAW_STAGE1_ID)
        self.assertEqual(v45.P1[SEED5]["norm_cfg_hash"], SEED5_NORM_CFG_HASH)
        self.assertEqual(v45.P1[SEED5]["norm_stage1_id"], SEED5_NORM_STAGE1_ID)
        self.assertEqual(v45.SEEDS, {"seed4": SEED4, "seed5": SEED5})
        self.assertEqual(v45.T975_DF4, T975_DF4)
        self.assertEqual(v45.SEED3_DELTA_TEST, SEED3_DELTA_TEST)
        self.assertEqual(v45.SEED3_DELTA_VAL, SEED3_DELTA_VAL)


# ---------------------------------------------------------------- 预注册文档 token
class TestPreregDocTokens(unittest.TestCase):
    def test_doc_pins_identities_gates_and_labels(self):
        text = (REPO_ROOT / SEEDS45_DOC_REL).read_text(encoding="utf-8")
        for token in (
            SEED4_RAW_STAGE1_ID, SEED4_NORM_STAGE1_ID, SEED4_RAW_CFG_HASH, SEED4_NORM_CFG_HASH,
            SEED5_RAW_STAGE1_ID, SEED5_NORM_STAGE1_ID, SEED5_RAW_CFG_HASH, SEED5_NORM_CFG_HASH,
            "1688749593", "1688762746", "1688723512", "1688723740", "1688738016", "20261003",
            "0.00824283986406793", "0.010059271876175169", "0.00045881952817949934", "0.0014831644646635667",
            "0.02347892658289208", "0.017557536803150087", "0.009211716513914836",
            "+0.001", "−0.02", "−0.005", "−0.01", "2.7764451051977987",
            "POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION",
            "UTILITY_DURABLE", "MECHANISM_REPAIR_ONLY_UNSTABLE_UTILITY",
            "HEADROOM_PLAUSIBLE", "HEADROOM_NOT_EVIDENT",
            "MECHANISM_REPAIRED_SEED4", "MECHANISM_REPAIRED_SEED5",
            "SEED4_CONDITION_PASSED", "SEED5_CONDITION_PASSED",
            "MECHANISM_NOT_REPAIRED_ON_SEED4", "MECHANISM_NOT_REPAIRED_ON_SEED5",
            "mechanism repair effective but utility unstable",
            "REPAIRED", "NO_MATERIAL_DEGRADATION", "NOT_SUPPORTED",
            "MECHANISM_REPLICATED", "UTILITY_DIRECTION_REPLICATED",
            "MECHANISM_REPAIRED_SEED3", "SEED3_CONDITION_PASSED",
            "2058de8", "7ab445c", "f8ff4cf",
            "s1-550e4d92-m1688723512-e1-d069eadf",
            "freeze-p4", "aggregate",
            # seed4 P4（C3a 登记；R1 捕获后、R2 运行前）
            "1000669", "1021493", "978507",
            "523e129ca132bdab2ccfaeb22593e309dd30025ca792d3d6e945593dd8eeb2ae", "51.07465",
            # seed4 Stage-1 结果（C4a 记录）
            "c37024f", "b839e51df9cb46a6b46fa5ea5e757e006c8288e00a2716fa0d4a89967a4f2a15",
            "0.5475919346018614", "0.574534619903342", "0.5465741235184509", "0.5750194104732491",
            "0.5000354152434238", "0.4971375", "1020931", "−0.0010178110834104803",
        ):
            self.assertIn(token, text, msg=f"预注册文档缺少 token: {token}")


# ---------------------------------------------------------------- 验证器机械函数（TDD 目标）
class TestVerifierMechanics(unittest.TestCase):
    def test_classify_boundaries(self):
        self.assertEqual(v45.classify(0.02347892658289208), "POSITIVE_IMPROVEMENT")
        self.assertEqual(v45.classify(0.001), "POSITIVE_IMPROVEMENT")
        self.assertEqual(v45.classify(0.0009999), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(v45.classify(-0.0199999), "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(v45.classify(-0.02), "CLEAR_DEGRADATION")
        self.assertEqual(v45.classify(-0.5), "CLEAR_DEGRADATION")

    def test_aggregate_stats_hand_computed(self):
        stats = v45.aggregate_stats(SYNTH_V)
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["mean"], SYNTH_MEAN, places=15)
        self.assertAlmostEqual(stats["sample_std"], SYNTH_SD, places=15)
        self.assertAlmostEqual(stats["ci95_low"], SYNTH_CI_LOW, places=15)
        self.assertAlmostEqual(stats["ci95_high"], SYNTH_CI_HIGH, places=15)
        self.assertEqual(stats["positive_count"], 3)
        self.assertEqual(stats["sign_positive_count"], 4)
        self.assertEqual(stats["worst_index"], 2)
        self.assertAlmostEqual(stats["worst_value"], -0.001, places=15)

    def test_headroom_h3_generalization(self):
        priors3 = [0.00824283986406793, 0.00045881952817949934, 0.02347892658289208]
        flags = v45.headroom_flags(
            mechanism_ok=True, delta_test=0.01, delta_val=0.005, prior_deltas=priors3,
            best_epoch_raw=5, best_epoch_norm=5, epochs=5,
        )
        self.assertEqual(flags["H3_cross_seed"], "favors")  # k=4：count(4) ≥ 2 ∧ mean > 0 ∧ current > 0
        self.assertEqual(flags["H4_budget_trajectory"], "favors_longer_budget")
        self.assertEqual(flags["verdict"], "HEADROOM_PLAUSIBLE")
        flags_neg = v45.headroom_flags(
            mechanism_ok=True, delta_test=-0.01, delta_val=0.005, prior_deltas=priors3,
            best_epoch_raw=5, best_epoch_norm=5, epochs=5,
        )
        self.assertEqual(flags_neg["H3_cross_seed"], "not")
        priors4 = priors3 + [0.01]
        flags5 = v45.headroom_flags(
            mechanism_ok=True, delta_test=0.05, delta_val=-0.02, prior_deltas=priors4,
            best_epoch_raw=3, best_epoch_norm=5, epochs=5,
        )
        self.assertEqual(flags5["H3_cross_seed"], "favors")  # k=5：count(5) ≥ 3 ∧ mean > 0
        self.assertEqual(flags5["H2_validation_direction"], "contradicts")
        self.assertEqual(flags5["H4_budget_trajectory"], "not")
        self.assertEqual(flags5["verdict"], "HEADROOM_NOT_EVIDENT")

    def test_freeze_p4_round_trip_on_synthetic_capture(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            losses = torch.tensor(
                [[0.10, 0.90], [0.20, 0.80], [0.30, 0.70], [0.40, 0.60],
                 [0.50, 0.50], [0.60, 0.40], [0.70, 0.30], [0.80, 0.20],
                 [0.90, 0.10], [0.15, 0.15], [0.25, 0.35], [0.35, 0.55]],
                dtype=torch.float32,
            )
            capture = td / "cluster_losses.pt"
            torch.save(losses, capture)
            train = td / "train.csv"
            rows = ["click,purchase"] + [f"{i % 2},{(i // 3) % 2}" for i in range(losses.shape[0])]
            train.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
            out = td / "p4.json"
            p4 = v45.freeze_p4(capture, train, 20261003, out, model_seed=SEED4)
            self.assertTrue(out.is_file())
            self.assertTrue(p4["bitwise_agree"])
            self.assertEqual(p4["model_seed"], SEED4)
            self.assertEqual(p4["n"], losses.shape[0])
            self.assertEqual(p4["event"]["epoch"], 2)
            from aliccp_benchmark.normalized_clustering import per_task_rank01

            assign = torch.argmin(per_task_rank01(losses), dim=1)
            counts = torch.bincount(assign, minlength=2)
            self.assertEqual(p4["event"]["env_0"], int(counts[0]))
            self.assertEqual(p4["event"]["env_1"], int(counts[1]))
            initial = protocol.make_env_ids(losses.shape[0], 20261003)
            self.assertEqual(p4["event"]["diff_num"], int((initial != assign).sum()))
            self.assertEqual(p4["env_ids_sha256"], protocol.sha256_tensor(assign))
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), p4)

    def test_find_run_is_model_seed_scoped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "runs" / "20260101-0000-p2M-v500k-t1M-m1688749593-short-abc1234"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(
                json.dumps({"stage1_id": SEED4_RAW_STAGE1_ID, "tag": "short", "model_seed": SEED4}),
                encoding="utf-8",
            )
            self.assertEqual(v45._find_run(root, SEED4_RAW_STAGE1_ID, "short", SEED4), run_dir)
            with self.assertRaises(FileNotFoundError):
                v45._find_run(root, SEED4_RAW_STAGE1_ID, "short", SEED5)

    def test_prior_delta_literals_match_audited_values(self):
        priors = v45.prior_delta_literals()
        self.assertEqual(set(priors), {1688723512, 1688723740, 1688738016})
        self.assertEqual(priors[1688723512]["delta_test"], 0.00824283986406793)
        self.assertEqual(priors[1688723512]["delta_val"], 0.010059271876175169)
        self.assertEqual(priors[1688723740]["delta_test"], 0.00045881952817949934)
        self.assertEqual(priors[1688723740]["delta_val"], -0.0014831644646635667)
        self.assertEqual(priors[1688738016]["delta_test"], SEED3_DELTA_TEST)
        self.assertEqual(priors[1688738016]["delta_val"], SEED3_DELTA_VAL)

    def test_historical_labels_frozen(self):
        h = v45.HISTORICAL
        self.assertEqual(h[1688723512]["tip"], "2058de8")
        self.assertIn("REPAIRED", h[1688723512]["labels"])
        self.assertIn("NO_MATERIAL_DEGRADATION", h[1688723512]["labels"])
        self.assertEqual(h[1688723740]["labels"], ["NOT_SUPPORTED"])
        self.assertEqual(h[1688738016]["tip"], "f8ff4cf")
        self.assertIn("SEED3_CONDITION_PASSED", h[1688738016]["labels"])


if __name__ == "__main__":
    unittest.main()
