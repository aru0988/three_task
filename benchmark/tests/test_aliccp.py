"""AliCCP 数据集适配器测试：常量、前缀身份（指纹/标签计数/重复）、加载器与模型构造。

移植自 infra/aliccp-fair-benchmark 分支的 aliccp_benchmark/tests/test_protocol.py（该副本在本分支被删除）。
共享协议（env_ids 独立性、stage1 产物、冻结三件套、哈希/JSON 规范化）已由
benchmark/tests/test_protocol.py 以黄金常量覆盖，此处不重复。
"""
import hashlib
import importlib.util
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import torch

from benchmark import protocol
from benchmark.datasets import aliccp
from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec

_HAVE_LEGACY = importlib.util.find_spec("aliccp_benchmark") is not None

HEADER = "click,purchase,X,121,122,301"


def write_aliccp_file(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


ROWS = [
    [0, 0, 9, 1, 0, 1],
    [1, 0, 9, 2, 1, 2],
    [0, 0, 8, 0, 2, 3],
    [1, 1, 8, 3, 1, 1],
]


class TestConstantsMatchSpec(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(aliccp.PREFIX_TAG, "p2M-v500k-t1M")
        self.assertEqual((aliccp.TRAIN_BUDGET, aliccp.VAL_BUDGET, aliccp.TEST_BUDGET),
                         (2_000_000, 500_000, 1_000_000))
        self.assertEqual((aliccp.SMOKE_TRAIN_BUDGET, aliccp.SMOKE_VAL_BUDGET, aliccp.SMOKE_TEST_BUDGET),
                         (20_000, 5_000, 10_000))
        self.assertEqual((aliccp.MODEL_SEED, aliccp.ENV_SEED), (1688723512, 20261003))
        self.assertEqual((aliccp.STAGE1_EPOCHS, aliccp.STAGE1_PATIENCE), (3, 2))
        self.assertEqual((aliccp.STAGE2_EPOCHS, aliccp.STAGE2_PATIENCE), (5, 2))
        self.assertEqual((aliccp.BATCH_SIZE, aliccp.LR), (2000, 1e-4))
        self.assertEqual((aliccp.UNI_COE, aliccp.ENV_COE), (0.9, 0.1))
        self.assertEqual((aliccp.REG_EMBEDDING, aliccp.REG_DNN), (1e-4, 7e-6))
        self.assertEqual((aliccp.INPUT_SIZE, aliccp.EMBEDDING_SIZE), (80, 5))
        self.assertEqual((aliccp.EXPERT_HIDDEN, aliccp.TOWER_HIDDEN), ((128, 64), (32, 32)))
        self.assertEqual(aliccp.DROPOUT, (0.1, 0.3))
        self.assertEqual((aliccp.NUM_TASKS, aliccp.NUM_ENVS), (2, 2))
        self.assertEqual(aliccp.NEWTASK_REP_DIM, 64)
        self.assertEqual((aliccp.AUC_FLOOR_CTR, aliccp.AUC_FLOOR_CVR, aliccp.AUC_FLOOR_BSI),
                         (0.55, 0.50, 0.53))
        self.assertEqual(aliccp.VAL_TEST_GAP_BSI, 0.05)
        self.assertEqual((aliccp.GATE_MIN, aliccp.GATE_MAX), (0.05, 0.95))
        self.assertEqual(aliccp.ENV_SHARE_MIN, 0.05)
        self.assertEqual(aliccp.IMPROVE_DELTA_AUC_TEST_BSI, 0.005)

    def test_data_files_point_at_aliccp(self):
        self.assertEqual(
            aliccp.DATA_FILES,
            {
                "train": "dataset/AliCCP/ctr_cvr.train",
                "val": "dataset/AliCCP/ctr_cvr.dev",
                "test": "dataset/AliCCP/ctr_cvr.test",
            },
        )


class TestVocabAndModel(unittest.TestCase):
    def test_build_vocab_does_not_mutate_global(self):
        vocab = aliccp.build_vocab()
        self.assertEqual(len(vocab), 16)
        self.assertNotIn("101", vocab)
        self.assertNotIn("301", vocab)
        # 全局字典未被污染（spec 2.2）
        self.assertIn("101", AliCCP_Vocabulary_Size)
        self.assertIn("301", AliCCP_Vocabulary_Size)
        self.assertEqual(len(AliCCP_Vocabulary_Size), 18)

    def test_build_mptrec_feature_names_follow_vocab(self):
        model = aliccp.build_mptrec(torch.device("cpu"))
        self.assertIsInstance(model, MPTRec)
        expected = [name for name in AliCCP_Vocabulary_Size if name not in ("101", "301")]
        self.assertEqual(model.embedding_network.feature_names, expected)


class TestPrefixIdentity(unittest.TestCase):
    def test_prefix_sha256_deterministic_and_budget_sensitive(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            write_aliccp_file(path, ROWS)
            sha_all = aliccp.prefix_sha256(path, 4)
            self.assertEqual(sha_all, aliccp.prefix_sha256(path, 4))
            sha_2 = aliccp.prefix_sha256(path, 2)
            self.assertNotEqual(sha_all, sha_2)
            # 手工等价：表头 + 前 2 条数据行的原始字节
            raw = path.read_bytes()
            n_rows = 2
            offset = len(HEADER.encode()) + 1
            for _ in range(n_rows):
                offset += raw.index(b"\n", offset) - offset + 1
            self.assertEqual(sha_2, hashlib.sha256(raw[:offset]).hexdigest())

    def test_prefix_sha256_rejects_budget_over_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            write_aliccp_file(path, ROWS)
            with self.assertRaises(ValueError):
                aliccp.prefix_sha256(path, 5)

    def test_scan_label_counts_matches_data(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            write_aliccp_file(path, ROWS)
            counts = aliccp.scan_label_counts(path, 4)
            self.assertEqual(counts["n"], 4)
            self.assertEqual(counts["click1"], 2)
            self.assertEqual(counts["purchase1"], 1)
            # raw 2 -> 0；{1,3} -> 1
            self.assertEqual(counts["bsi_pos"], 3)
            self.assertEqual(counts["bsi_raw"], {1: 2, 2: 1, 3: 1})

    def test_dataset_label_counts_matches_raw_scan_on_shared_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            write_aliccp_file(path, ROWS)
            dataset = AliCCPDataset(str(path), 4)
            loader_counts = aliccp.dataset_label_counts(dataset)
            raw_counts = aliccp.scan_label_counts(path, 4)
            for key in ("click1", "purchase1", "bsi_pos"):
                self.assertEqual(loader_counts[key], raw_counts[key], key)

    def test_duplicate_stats(self):
        with tempfile.TemporaryDirectory() as td:
            train = Path(td) / "train.csv"
            val = Path(td) / "val.csv"
            # train 内重复一行；train∩val 共享一行
            write_aliccp_file(train, [ROWS[0], ROWS[1], ROWS[0]])
            write_aliccp_file(val, [ROWS[1], ROWS[2]])
            stats = aliccp.duplicate_stats({"train": (train, 3), "val": (val, 2)})
            self.assertEqual(stats["train_within"], 1)
            self.assertEqual(stats["val_within"], 0)
            self.assertEqual(stats["train_val"], 1)

    def test_build_fingerprint_then_verify(self):
        with tempfile.TemporaryDirectory() as td:
            train = Path(td) / "train.csv"
            val = Path(td) / "val.csv"
            test = Path(td) / "test.csv"
            write_aliccp_file(train, ROWS)
            write_aliccp_file(val, ROWS[:2])
            write_aliccp_file(test, ROWS[1:])
            files = {"train": (train, 4), "val": (val, 2), "test": (test, 3)}
            fp = aliccp.build_fingerprint("tiny", files)
            self.assertEqual(fp["prefix_tag"], "tiny")
            self.assertIn("fingerprint_sha256", fp)
            self.assertEqual(fp, aliccp.build_fingerprint("tiny", files))  # 确定性
            aliccp.verify_fingerprint(fp)  # 通过
            # 篡改文件 → 校验失败
            write_aliccp_file(test, ROWS[:3])
            with self.assertRaises(AssertionError):
                aliccp.verify_fingerprint(fp)

    def test_fingerprint_roundtrip_to_disk(self):
        with tempfile.TemporaryDirectory() as td:
            train = Path(td) / "train.csv"
            write_aliccp_file(train, ROWS)
            files = {"train": (train, 4)}
            root = Path(td) / "artifacts"
            fp, created = aliccp.ensure_fingerprint(root, "tiny", files)
            self.assertTrue(created)
            fp2, created2 = aliccp.ensure_fingerprint(root, "tiny", files)
            self.assertFalse(created2)
            self.assertEqual(fp["fingerprint_sha256"], fp2["fingerprint_sha256"])


class TestVerifyLabelCounts(unittest.TestCase):
    def test_verify_len_and_counts(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            write_aliccp_file(path, ROWS)
            dataset = AliCCPDataset(str(path), 4)
            expected = aliccp.scan_label_counts(path, 4)
            aliccp.verify_label_counts(dataset, 4, expected)  # 通过
            with self.assertRaises(AssertionError):
                aliccp.verify_label_counts(dataset, 5, expected)  # 长度不符
            bad = dict(expected)
            bad["click1"] = expected["click1"] + 1
            with self.assertRaises(AssertionError):
                aliccp.verify_label_counts(dataset, 4, bad)


class TestBuildLoaders(unittest.TestCase):
    def test_build_loaders_lengths_and_iteration(self):
        with tempfile.TemporaryDirectory() as td:
            train = Path(td) / "train.csv"
            val = Path(td) / "val.csv"
            test = Path(td) / "test.csv"
            write_aliccp_file(train, ROWS)
            write_aliccp_file(val, ROWS[:2])
            write_aliccp_file(test, ROWS[1:])
            data_files = {"train": str(train), "val": str(val), "test": str(test)}
            budgets = {"train": 4, "val": 2, "test": 3}
            datasets, loaders = aliccp.build_loaders(data_files, budgets, batch_size=2)
            for split in ("train", "val", "test"):
                self.assertEqual(len(datasets[split]), budgets[split])
                seen = sum(len(batch[0]) for batch in loaders[split])
                self.assertEqual(seen, budgets[split])


class TestIdsGolden(unittest.TestCase):
    def test_recorded_run_id(self):
        run_id = protocol.make_run_id(datetime(2026, 10, 3, 15, 30), prefix=aliccp.PREFIX_TAG,
                                      model_seed=aliccp.MODEL_SEED, tag="short", commit="abc1234")
        self.assertEqual(run_id, "20261003-1530-p2M-v500k-t1M-m1688723512-short-abc1234")

    def test_stage1_id_for_delegates_to_shared_stage1_id(self):
        meta = {"fingerprint_sha256": "ab" * 32, "model_seed": aliccp.MODEL_SEED,
                "epochs": aliccp.STAGE1_EPOCHS, "config_hash": "cd" * 32}
        self.assertEqual(aliccp.stage1_id_for(meta),
                         protocol.stage1_id(fingerprint_sha=meta["fingerprint_sha256"],
                                            model_seed=meta["model_seed"], epochs=meta["epochs"],
                                            cfg_sha=meta["config_hash"]))


# ---- 门禁测试（移植自 infra branch 的 test_metrics.py）----
def a_facts(**overrides):
    facts = {
        "backbone_sha_before": "aa",
        "backbone_sha_after": "aa",
        "backbone_grads_none": True,
        "prefix_sha_ok": True,
        "fingerprint_sha_match": True,
        "len_ok": True,
        "counts_ok": True,
        "env_ids_sha_match": True,
        "backbone_sha_matches_stage1": True,
        "stage1_id_recorded": True,
    }
    facts.update(overrides)
    return facts


def b_facts(**overrides):
    facts = {
        "auc_val_ctr": 0.62,
        "auc_val_cvr": 0.55,
        "auc_val_bsi_best": 0.66,
        "auc_test_bsi": 0.65,
        "gate_mean": [0.5, 0.5],
        "cluster_events": [
            {"epoch": 2, "diff_num": 10, "env_0": 900_000, "env_1": 1_100_000},
        ],
        "train_size": 2_000_000,
    }
    facts.update(overrides)
    return facts


class TestAGates(unittest.TestCase):
    def test_all_pass(self):
        a_gates = aliccp.evaluate_a_gates(a_facts())
        self.assertEqual(set(a_gates), {"A1", "A2", "A3", "A4", "A5", "A6"})
        for key, result in a_gates.items():
            self.assertIn(result["verdict"], {"PASS", "SKIP"}, key)
        self.assertTrue(aliccp.hard_pass(a_gates, enforce_b=False))

    def test_each_failure_flips_its_gate(self):
        cases = {
            "A1": {"backbone_sha_after": "bb"},
            "A1b": {"backbone_grads_none": False},
            "A2": {"prefix_sha_ok": False},
            "A2b": {"fingerprint_sha_match": False},
            "A4": {"len_ok": False},
            "A4b": {"counts_ok": False},
            "A5": {"env_ids_sha_match": False},
            "A6": {"backbone_sha_matches_stage1": False},
            "A6b": {"stage1_id_recorded": False},
        }
        for name, override in cases.items():
            a_gates = aliccp.evaluate_a_gates(a_facts(**override))
            gate_id = name[0:2]
            self.assertEqual(a_gates[gate_id]["verdict"], "FAIL", name)
            self.assertFalse(aliccp.hard_pass(a_gates, enforce_b=False), name)

    def test_a3_is_skip(self):
        a_gates = aliccp.evaluate_a_gates(a_facts())
        self.assertEqual(a_gates["A3"]["verdict"], "SKIP")


class TestBGates(unittest.TestCase):
    def test_all_pass(self):
        b_gates = aliccp.evaluate_b_gates(b_facts())
        self.assertEqual(set(b_gates), {"B1", "B2", "B3", "B4"})
        for result in b_gates.values():
            self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(aliccp.hard_pass({"A1": {"verdict": "PASS"}}, enforce_b=False) is True)

    def test_b1_floors(self):
        b_gates = aliccp.evaluate_b_gates(b_facts(auc_val_ctr=0.54))
        self.assertEqual(b_gates["B1"]["verdict"], "FAIL")
        b_gates = aliccp.evaluate_b_gates(b_facts(auc_test_bsi=0.52))
        self.assertEqual(b_gates["B1"]["verdict"], "FAIL")
        # CVR 下限刻意设为 0.50（近乎无约束）
        b_gates = aliccp.evaluate_b_gates(b_facts(auc_val_cvr=0.499))
        self.assertEqual(b_gates["B1"]["verdict"], "FAIL")
        b_gates = aliccp.evaluate_b_gates(b_facts(auc_val_cvr=0.505))
        self.assertEqual(b_gates["B1"]["verdict"], "PASS")

    def test_b2_gap(self):
        self.assertEqual(aliccp.evaluate_b_gates(b_facts(auc_val_bsi_best=0.70, auc_test_bsi=0.65))["B2"]["verdict"], "PASS")
        self.assertEqual(aliccp.evaluate_b_gates(b_facts(auc_val_bsi_best=0.71, auc_test_bsi=0.65))["B2"]["verdict"], "FAIL")

    def test_b3_gate_collapse(self):
        self.assertEqual(aliccp.evaluate_b_gates(b_facts(gate_mean=[0.5, 0.5]))["B3"]["verdict"], "PASS")
        self.assertEqual(aliccp.evaluate_b_gates(b_facts(gate_mean=[0.99, 0.5]))["B3"]["verdict"], "FAIL")
        self.assertEqual(aliccp.evaluate_b_gates(b_facts(gate_mean=[0.01, 0.5]))["B3"]["verdict"], "FAIL")

    def test_b4_cluster_share_and_na(self):
        events = [{"epoch": 2, "diff_num": 1, "env_0": 100_000, "env_1": 1_900_000}]
        self.assertEqual(aliccp.evaluate_b_gates(b_facts(cluster_events=events))["B4"]["verdict"], "PASS")
        events = [{"epoch": 2, "diff_num": 1, "env_0": 10, "env_1": 1_999_990}]
        self.assertEqual(aliccp.evaluate_b_gates(b_facts(cluster_events=events))["B4"]["verdict"], "FAIL")
        self.assertEqual(aliccp.evaluate_b_gates(b_facts(cluster_events=[]))["B4"]["verdict"], "N/A")

    def test_hard_pass_enforce_b(self):
        failing = aliccp.evaluate_b_gates(b_facts(auc_val_ctr=0.50))
        self.assertFalse(aliccp.hard_pass({"A1": {"verdict": "PASS"}}, enforce_b=True, b_gates=failing))
        self.assertTrue(aliccp.hard_pass({"A1": {"verdict": "PASS"}}, enforce_b=False, b_gates=failing))


@unittest.skipUnless(_HAVE_LEGACY, "旧实现包已删除（迁移完成）")
class TestLegacyGateEquivalence(unittest.TestCase):
    def test_a_gates_render_matches_legacy(self):
        from aliccp_benchmark import metrics as old
        for facts in (a_facts(), a_facts(backbone_sha_after="bb"), a_facts(len_ok=False)):
            self.assertEqual(aliccp.evaluate_a_gates(facts), old.evaluate_a_gates(facts))

    def test_b_gates_render_matches_legacy(self):
        from aliccp_benchmark import metrics as old
        for facts in (b_facts(), b_facts(auc_val_ctr=0.50), b_facts(cluster_events=[]),
                      b_facts(auc_val_bsi_best=0.72, auc_test_bsi=0.65)):
            self.assertEqual(aliccp.evaluate_b_gates(facts), old.evaluate_b_gates(facts))

    def test_hard_pass_matches_legacy(self):
        from aliccp_benchmark import metrics as old
        for facts in (b_facts(), b_facts(auc_val_ctr=0.50), b_facts(cluster_events=[])):
            a_gates = aliccp.evaluate_a_gates(a_facts())
            b_gates = aliccp.evaluate_b_gates(facts)
            old_a = old.evaluate_a_gates(a_facts())
            old_b = old.evaluate_b_gates(facts)
            for enforce_b in (False, True):
                self.assertEqual(aliccp.hard_pass(a_gates, enforce_b=enforce_b, b_gates=b_gates),
                                 old.hard_pass(old_a, enforce_b=enforce_b, b_gates=old_b), enforce_b)


if __name__ == "__main__":
    unittest.main()
