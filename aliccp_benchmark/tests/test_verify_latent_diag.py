"""独立验证器测试（TDD：先于实现编写）：诚实运行必过、篡改必败、常量与独立性约束。

独立性测试（AST）：verify_latent_diag.py 不得 import aliccp_benchmark 的任何模块，
AUC 只允许直接调用 sklearn.metrics.roc_auc_score（不得复用训练/评分函数）。
"""
import ast
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from aliccp_benchmark import latent_diag, verify_latent_diag
from aliccp_benchmark.tests.test_latent_diag import (
    load_json,
    run_designed_pipeline,
    run_tiny_pipeline,
)

VERIFY_SOURCE = Path(verify_latent_diag.__file__)


def _run_dir(result):
    return Path(result["run_dir"])


def _rewrite_npz(path, mutate):
    """读-改-写 npz（with 关闭句柄，Windows 下可安全覆盖）。"""
    with np.load(path) as npz:
        data = {key: npz[key] for key in npz.files}
    mutate(data)
    np.savez(path, **data)


def _force_sum(arr, target):
    """把 0/1 数组的元素和精确调整为 target（保持长度不变；测试篡改用）。"""
    arr = np.asarray(arr).copy()
    delta = int(target) - int(arr.sum())
    if delta > 0:
        idx = np.flatnonzero(arr == 0)[:delta]
        assert len(idx) == delta
        arr[idx] = 1
    elif delta < 0:
        idx = np.flatnonzero(arr == 1)[: abs(delta)]
        assert len(idx) == abs(delta)
        arr[idx] = 0
    assert int(arr.sum()) == int(target)
    return arr


class TestConstantsParity(unittest.TestCase):
    """验证器持有冻结常量的独立副本；两份副本必须逐项相等（防漂移）。"""

    def test_frozen_constants_match(self):
        v, l = verify_latent_diag, latent_diag
        self.assertEqual(v.FORMAL_STAGE1_ID, l.FORMAL_STAGE1_ID)
        self.assertEqual(v.FORMAL_FINGERPRINT_SHA256, l.FORMAL_FINGERPRINT_SHA256)
        self.assertEqual(v.SHUFFLE_SEED, l.SHUFFLE_SEED)
        self.assertEqual(v.CLIP_EPS, l.CLIP_EPS)
        self.assertEqual(v.SCALE_FLOOR, l.SCALE_FLOOR)
        self.assertEqual(v.MIN_BSI_NEGATIVES, l.MIN_BSI_NEGATIVES)
        self.assertEqual(v.VAR_FLOOR, l.VAR_FLOOR)
        self.assertEqual(
            (v.DELTA_TEST_MIN, v.DELTA_C_MIN, v.DELTA_DEV_MIN),
            (l.DELTA_TEST_MIN, l.DELTA_C_MIN, l.DELTA_DEV_MIN),
        )
        self.assertEqual((v.CLASS_POS_MIN, v.CLASS_NEG_MAX), (l.CLASS_POS_MIN, l.CLASS_NEG_MAX))
        self.assertEqual(
            (v.PROBE_C, v.PROBE_MAX_ITER, v.PROBE_SOLVER, v.PROBE_PENALTY),
            (l.PROBE_C, l.PROBE_MAX_ITER, l.PROBE_SOLVER, l.PROBE_PENALTY),
        )
        self.assertEqual(v.ARMS, l.ARMS)
        self.assertEqual(v.ARM_FEATURES, l.ARM_FEATURES)
        self.assertEqual(v.PERM_ORDER, l.PERM_ORDER)
        self.assertEqual(v.ALL_SPLITS, l.ALL_SPLITS)
        self.assertEqual(list(v.RAW_FILES), list(l.RAW_FILES))
        self.assertEqual(v.SUMMARY_FILE, l.SUMMARY_FILE)
        self.assertEqual(v.LATENT_DIAG_SUMMARY_COLUMNS, l.LATENT_DIAG_SUMMARY_COLUMNS)


class TestIndependence(unittest.TestCase):
    FORBIDDEN = {
        "aliccp_benchmark", "bench", "metrics", "latent_diag", "protocol",
        "config", "run_aliccp_benchmark", "run_aliccp_latent_diag",
    }

    def test_no_diagnostic_package_imports(self):
        tree = ast.parse(VERIFY_SOURCE.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
        bad = imported & self.FORBIDDEN
        self.assertEqual(bad, set(), f"验证器不得 import: {sorted(bad)}")
        self.assertIn("sklearn", imported)  # AUC 直接来自 sklearn

    def test_auc_uses_roc_auc_score_directly(self):
        source = VERIFY_SOURCE.read_text(encoding="utf-8")
        self.assertIn("roc_auc_score", source)
        bare = re.findall(r"\bauc_score\(", source)  # 不允许任何非 sklearn 的 auc_score 调用
        self.assertEqual(bare, [])

    def test_no_identifier_reference_to_diagnostic_modules(self):
        """除 import 外，验证器源码也不得以标识符/属性方式引用诊断侧模块（即不得
        "简单地调用 runner 助手"）。这里只覆盖诊断侧模块名；config 等是验证器内部
        合法局部名（配置字典），不在此列——其禁止性由 import 级测试覆盖。"""
        tree = ast.parse(VERIFY_SOURCE.read_text(encoding="utf-8"))
        guarded = {
            "aliccp_benchmark", "bench", "latent_diag", "protocol",
            "run_aliccp_benchmark", "run_aliccp_latent_diag",
        }
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in guarded:
                offenders.append((node.lineno, node.id))
            elif isinstance(node, ast.Attribute) and node.attr in guarded:
                offenders.append((node.lineno, node.attr))
        self.assertEqual(offenders, [], f"验证器不得引用诊断模块标识符: {offenders}")


class TestVerifierHonestRun(unittest.TestCase):
    def test_honest_smoke_run_passes_and_matches(self):
        with tempfile.TemporaryDirectory() as td:
            root, _, meta, result = run_tiny_pipeline(td)
            run_dir = _run_dir(result)
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            self.assertEqual(verdict["overall"], "PASS", verdict["checks"])
            check_ids = {c["id"] for c in verdict["checks"]}
            for required in (
                "config_sha256", "artifacts_sha256", "fingerprint_self", "fingerprint_files",
                "stage1_backbone_sha", "chain_backbone", "ranges_hashes", "split_counts_scan",
                "raw_counts_vs_scan", "perms_regenerated", "probe_prob_match", "auc_match",
                "delta_match", "gate_match", "classification_match",
            ):
                self.assertIn(required, check_ids)
            reported = load_json(run_dir / "reported.json")
            rec = verdict["recomputed"]
            for split in latent_diag.ALL_SPLITS:
                for arm in latent_diag.ARMS:
                    self.assertAlmostEqual(
                        rec["aucs"][split][arm], reported["aucs"][split][arm], places=9
                    )
            self.assertEqual(rec["verdict"], reported["verdict"])
            written = load_json(run_dir / "verification.json")
            self.assertEqual(written["overall"], "PASS")

    def test_summary_appended_once_after_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_tiny_pipeline(td)
            ledger = Path(root) / latent_diag.SUMMARY_FILE
            self.assertFalse(ledger.exists())  # 运行本身绝不写 SUMMARY
            first = verify_latent_diag.verify_run(_run_dir(result), root, append_summary=True)
            self.assertEqual(first["overall"], "PASS")
            self.assertTrue(first["summary"]["appended"])
            lines = ledger.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn(result["run_id"], lines[2])
            self.assertIn("| " + first["recomputed"]["verdict"] + " |", lines[2])
            # 重复验证幂等：不重复追加
            second = verify_latent_diag.verify_run(_run_dir(result), root, append_summary=True)
            self.assertEqual(second["overall"], "PASS")
            self.assertFalse(second["summary"]["appended"])
            self.assertTrue(second["summary"]["already_present"])
            self.assertEqual(len(ledger.read_text(encoding="utf-8").strip().splitlines()), 3)


class TestVerifierTamper(unittest.TestCase):
    def test_reported_delta_tamper_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_tiny_pipeline(td)
            run_dir = _run_dir(result)
            reported_path = run_dir / "reported.json"
            reported = load_json(reported_path)
            reported["deltas"]["test"]["delta"] += 0.05
            reported_path.write_text(json.dumps(reported, ensure_ascii=False, indent=2), encoding="utf-8")
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            self.assertEqual(verdict["overall"], "FAIL")
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertIn("delta_match", failed)

    def test_raw_tamper_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_tiny_pipeline(td)
            run_dir = _run_dir(result)

            def mutate(data):
                data["B__bsi_prob"] = data["B__bsi_prob"].copy()
                data["B__bsi_prob"][0] = 0.123456

            _rewrite_npz(run_dir / "raw.npz", mutate)
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            self.assertEqual(verdict["overall"], "FAIL")
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertIn("artifacts_sha256", failed)

    def test_perm_tamper_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_tiny_pipeline(td)
            run_dir = _run_dir(result)

            def mutate(data):
                perm = data["B__perm"].copy()
                perm[0], perm[1] = perm[1], perm[0]
                data["B__perm"] = perm

            _rewrite_npz(run_dir / "perms.npz", mutate)
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            self.assertEqual(verdict["overall"], "FAIL")
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertTrue({"artifacts_sha256", "perms_regenerated"} & failed)

    def test_probe_prob_tamper_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_tiny_pipeline(td)
            run_dir = _run_dir(result)

            def mutate(data):
                data["test__full_prob"] = data["test__full_prob"].copy()
                data["test__full_prob"][0] = 1.0 - data["test__full_prob"][0]

            _rewrite_npz(run_dir / "probe_probs.npz", mutate)
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            self.assertEqual(verdict["overall"], "FAIL")
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertTrue({"artifacts_sha256", "probe_prob_match", "auc_match"} & failed)


class TestVerifierArmSemantics(unittest.TestCase):
    """回归（臂语义缺陷，2026-10-08 编排审计）：独立验证器必须以预注册外部映射独立重建臂输入——
    base=[bsi]；full=[bsi,ctr,cvr]（行对齐原列）；shuffle=[bsi,ctr[perm],cvr[perm]]。
    历史缺陷：验证器复制了 runner 的同一缺陷（置换同时应用于 full 与 shuffle）→ 对作废运行
    30/30 PASS。本测试用字面期望值（外部规范）锁定验证器的重建语义，不调用 runner 代码。
    TestIndependence 另行锁定"不得 import / 引用诊断侧模块"。"""

    def setUp(self):
        self.perm = np.array([3, 0, 4, 1, 2])  # 已知非恒等置换（长度 5）
        self.logits = {
            "B": {
                "bsi": np.array([0.11, 0.22, 0.33, 0.44, 0.55]),
                "ctr": np.array([0.91, 0.12, 0.83, 0.24, 0.75]),
                "cvr": np.array([0.15, 0.45, 0.65, 0.35, 0.85]),
            }
        }
        self.perms = {"B": self.perm}

    def test_external_mapping_table(self):
        self.assertEqual(
            verify_latent_diag.PERMUTED_FEATURES_BY_ARM,
            {"base": frozenset(), "full": frozenset(), "shuffle": frozenset({"ctr", "cvr"})},
        )

    def test_arm_cols_matches_frozen_mapping(self):
        base = verify_latent_diag._arm_cols(self.logits, self.perms, "base", "B")
        self.assertEqual(set(base), {"bsi"})
        np.testing.assert_array_equal(base["bsi"], self.logits["B"]["bsi"])

        full = verify_latent_diag._arm_cols(self.logits, self.perms, "full", "B")
        np.testing.assert_array_equal(full["bsi"], self.logits["B"]["bsi"])
        np.testing.assert_array_equal(full["ctr"], self.logits["B"]["ctr"])  # 对齐：不得置换
        np.testing.assert_array_equal(full["cvr"], self.logits["B"]["cvr"])

        shuffle = verify_latent_diag._arm_cols(self.logits, self.perms, "shuffle", "B")
        np.testing.assert_array_equal(shuffle["bsi"], self.logits["B"]["bsi"])  # BSI 不置换
        np.testing.assert_array_equal(shuffle["ctr"], self.logits["B"]["ctr"][self.perm])
        np.testing.assert_array_equal(shuffle["cvr"], self.logits["B"]["cvr"][self.perm])

        self.assertFalse(np.array_equal(full["ctr"], shuffle["ctr"]))  # 缺陷下两者逐位相同

    def test_unknown_arm_rejected(self):
        with self.assertRaises(KeyError):
            verify_latent_diag._arm_cols(self.logits, self.perms, "both", "B")


def _buggy_arm_columns(logits, perms, arm, split):
    """历史缺陷实现（runner 与验证器曾复制同一缺陷）：置换同时应用于 full 与 shuffle 臂。"""
    cols = {"bsi": logits[split]["bsi"]}
    if arm == "base":
        return cols
    perm = perms[split]
    cols["ctr"] = logits[split]["ctr"][perm]
    cols["cvr"] = logits[split]["cvr"][perm]
    return cols


class TestVerifierRejectsBuggyArmOutputs(unittest.TestCase):
    """回归（核心）：独立验证器必须拒绝"两臂同置换"缺陷产生的自洽产物——哈希链完整、
    配置自洽，但 full 臂实为置换输入 → 独立重训探针 / 重算概率 / AUC 必然不符。"""

    def test_self_consistent_buggy_run_is_rejected_on_semantics(self):
        with tempfile.TemporaryDirectory() as td:
            # 修复后以 mock 注入历史缺陷实现复现一次缺陷运行；未修复时 runner 自身即缺陷实现
            if getattr(latent_diag, "arm_columns", None) is None:
                extra = []
            else:
                extra = [mock.patch.object(latent_diag, "arm_columns", _buggy_arm_columns)]
            root, _, _, result = run_designed_pipeline(td, extra_patches=extra)
            self.assertEqual(result["status"], "OK")
            run_dir = Path(result["run_dir"])
            if extra:  # 缺陷运行确实被复现：full 臂记录值来自置换输入
                doc = load_json(run_dir / "probes.json")["arms"]
                self.assertTrue(
                    np.allclose(
                        np.asarray(doc["full"]["coef"]), np.asarray(doc["shuffle"]["coef"]),
                        rtol=0, atol=1e-9,
                    ),
                    "mock 注入后 full 臂产物应与 shuffle 臂逐位一致（缺陷症状）",
                )
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertEqual(verdict["overall"], "FAIL", failed)
            # 产物自洽 → 拒绝必须来自臂语义的独立重算，而非完整性哈希
            self.assertNotIn("artifacts_sha256", failed)
            self.assertNotIn("config_sha256", failed)
            self.assertTrue(
                {"probe_coef_refit", "probe_prob_match", "auc_match", "delta_match"} <= failed,
                failed,
            )

    def test_mislabeled_arm_entries_rejected(self):
        """把 probes.json 的 full/shuffle 探针互换（错标臂归属）：验证器独立重训后必须拒绝。
        前置断言同时锁定缺陷症状：修复后两臂探针不得相同（缺陷下逐位相同 → 互换为空操作、
        验证器无从拒绝）。"""
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_designed_pipeline(td)
            run_dir = Path(result["run_dir"])
            probes_path = run_dir / "probes.json"
            doc = load_json(probes_path)
            full, shuffle = doc["arms"]["full"], doc["arms"]["shuffle"]
            self.assertFalse(
                np.allclose(
                    np.asarray(full["coef"]), np.asarray(shuffle["coef"]), rtol=0, atol=1e-9
                ),
                "前置条件：修复后 full/shuffle 探针必须不同（缺陷下相同，运行无效）",
            )
            doc["arms"]["full"], doc["arms"]["shuffle"] = shuffle, full
            probes_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertEqual(verdict["overall"], "FAIL", failed)
            self.assertIn("probe_coef_refit", failed)

    def test_mislabeled_raw_old_task_columns_rejected(self):
        """把 raw.npz 中 B 切分的 ctr_prob/cvr_prob 互换（错标原始旧任务列）：标签/计数不动 →
        拒绝必须来自列语义（标准化统计与独立重训探针），而非标签一致性检查。"""
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_designed_pipeline(td)
            run_dir = Path(result["run_dir"])

            def mutate(data):
                ctr = data["B__ctr_prob"].copy()
                data["B__ctr_prob"] = data["B__cvr_prob"].copy()
                data["B__cvr_prob"] = ctr

            _rewrite_npz(run_dir / "raw.npz", mutate)
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertEqual(verdict["overall"], "FAIL", failed)
            self.assertTrue({"std_stats_match", "probe_coef_refit"} <= failed, failed)
            self.assertNotIn("raw_counts_vs_scan", failed)  # 标签与计数未动：拒绝来自列语义


class TestVerifierPermNonIdentity(unittest.TestCase):
    """回归：shuffle 对照必须真正破坏行级对齐——恒等置换不破坏任何对齐（full 与 shuffle 输入
    完全相同），冻结语义下必须判 FAIL，且 P1_perm 门禁随之翻转。"""

    def test_identity_permutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_tiny_pipeline(td)
            run_dir = Path(result["run_dir"])

            def identity(lengths):
                return {s: np.arange(int(lengths[s])) for s in verify_latent_diag.PERM_ORDER}

            with mock.patch.object(verify_latent_diag, "_regenerate_perms", identity):
                verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertEqual(verdict["overall"], "FAIL", failed)
            self.assertIn("perms_nonidentity", failed)
            self.assertEqual(verdict["recomputed"]["gates"]["P1_perm"], "FAIL")


class TestVerifierDevValAlias(unittest.TestCase):
    """回归（KeyError 'dev'）：公平指纹键只有 train/val/test，诊断本地名 dev 即官方 val。
    验证器对 dev 的指纹计数比对必须精确读取 val 条目，且该比对在篡改下必须失败（不得静默放过）。"""

    def test_alias_is_exact_dev_to_val_only(self):
        self.assertEqual(verify_latent_diag.FP_KEY_ALIAS, {"dev": "val"})
        self.assertEqual(verify_latent_diag._fp_key("dev"), "val")
        for split in ("train", "val", "test", "B", "C"):
            self.assertEqual(verify_latent_diag._fp_key(split), split)

    def test_dev_counts_compared_against_val_entry_not_test(self):
        # 判别性篡改：把 raw.npz 的 dev 计数改成恰好等于指纹 test 条目（≠ val 条目）。
        # 正确别名（dev → val）下 raw_counts_vs_fingerprint 必须 FAIL；
        # 若误读 test 条目则该项被错误放过（其余失败项与别名无关，不作判据）。
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_tiny_pipeline(td)
            run_dir = _run_dir(result)
            prefix_tag = load_json(run_dir / "config.json")["config"]["fingerprint"]["prefix_tag"]
            fp = load_json(Path(root) / "splits" / prefix_tag / "prefix_fingerprint.json")
            val_c, test_c = fp["label_counts"]["val"], fp["label_counts"]["test"]
            self.assertNotEqual(
                (test_c["click1"], test_c["purchase1"], test_c["bsi_pos"]),
                (val_c["click1"], val_c["purchase1"], val_c["bsi_pos"]),
            )

            def mutate(data):
                data["dev__y"] = _force_sum(data["dev__y"], test_c["bsi_pos"])
                data["dev__click"] = _force_sum(data["dev__click"], test_c["click1"])
                data["dev__purchase"] = _force_sum(data["dev__purchase"], test_c["purchase1"])

            _rewrite_npz(run_dir / "raw.npz", mutate)
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            self.assertEqual(verdict["overall"], "FAIL")
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertIn("raw_counts_vs_fingerprint", failed)

    def test_fingerprint_val_entry_tamper_fails_dev_closure(self):
        # 篡改指纹 val 条目的标签计数并重算自哈希（篡改"内部自洽"）：
        # fingerprint_self 必须仍 PASS（不是靠自哈希抓住），交叉校验必须抓住——
        # fingerprint_files（对源文件）与 raw_counts_vs_fingerprint（dev 对 val）均 FAIL。
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_tiny_pipeline(td)
            run_dir = _run_dir(result)
            prefix_tag = load_json(run_dir / "config.json")["config"]["fingerprint"]["prefix_tag"]
            fp_path = Path(root) / "splits" / prefix_tag / "prefix_fingerprint.json"
            fp = load_json(fp_path)
            fp["label_counts"]["val"]["bsi_pos"] += 1
            body = {k: v for k, v in fp.items() if k != "fingerprint_sha256"}
            fp["fingerprint_sha256"] = hashlib.sha256(
                json.dumps(
                    body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
                ).encode()
            ).hexdigest()
            fp_path.write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")

            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            self.assertEqual(verdict["overall"], "FAIL")
            failed = {c["id"] for c in verdict["checks"] if not c["ok"]}
            self.assertNotIn("fingerprint_self", failed)
            self.assertIn("fingerprint_files", failed)
            self.assertIn("raw_counts_vs_fingerprint", failed)


class TestVerifierAbortRun(unittest.TestCase):
    def test_enforced_tiny_run_aborts_and_verifier_confirms(self):
        if verify_latent_diag._git_dirty_now() is not False:
            self.skipTest("工作树非干净（或 git 不可用）：正式语义下 git_clean_required 检查此时预期 FAIL（提交后重跑本测试）")
        with tempfile.TemporaryDirectory() as td:
            root, _, _, result = run_tiny_pipeline(td, enforce=True, require_clean_git=False)
            self.assertEqual(result["status"], "ABORT")
            codes = {r["code"] for r in result["abort_reasons"]}
            self.assertIn("IDENTITY_MISMATCH", codes)  # 微型 stage1 不可能匹配正式常量
            self.assertIn("MIN_NEGATIVES", codes)
            run_dir = _run_dir(result)
            self.assertTrue((run_dir / "abort.json").exists())
            self.assertFalse((run_dir / "probes.json").exists())
            verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=False)
            self.assertEqual(verdict["overall"], "PASS", verdict["checks"])
            self.assertEqual(verdict["recomputed"]["status"], "ABORT")
            self.assertEqual(verdict["recomputed"]["verdict"], "ABORT_NO_GO")
            for check in verdict["checks"]:
                if check["id"].startswith("abort_reason"):
                    self.assertTrue(check["ok"], check)


if __name__ == "__main__":
    unittest.main()
