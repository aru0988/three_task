"""协议层测试（TDD：先于实现编写）。CPU-only，全部使用临时合成数据。"""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from aliccp_benchmark import protocol
from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset

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
        self.assertEqual(protocol.PREFIX_TAG, "p2M-v500k-t1M")
        self.assertEqual(
            (protocol.TRAIN_BUDGET, protocol.VAL_BUDGET, protocol.TEST_BUDGET),
            (2_000_000, 500_000, 1_000_000),
        )
        self.assertEqual(
            (protocol.SMOKE_TRAIN_BUDGET, protocol.SMOKE_VAL_BUDGET, protocol.SMOKE_TEST_BUDGET),
            (20_000, 5_000, 10_000),
        )
        self.assertEqual((protocol.MODEL_SEED, protocol.ENV_SEED), (1688723512, 20261003))
        self.assertEqual((protocol.STAGE1_EPOCHS, protocol.STAGE1_PATIENCE), (3, 2))
        self.assertEqual((protocol.STAGE2_EPOCHS, protocol.STAGE2_PATIENCE), (5, 2))
        self.assertEqual((protocol.BATCH_SIZE, protocol.LR), (2000, 1e-4))
        self.assertEqual((protocol.UNI_COE, protocol.ENV_COE), (0.9, 0.1))
        self.assertEqual((protocol.REG_EMBEDDING, protocol.REG_DNN), (1e-4, 7e-6))
        self.assertEqual((protocol.INPUT_SIZE, protocol.EMBEDDING_SIZE), (80, 5))
        self.assertEqual((protocol.EXPERT_HIDDEN, protocol.TOWER_HIDDEN), ((128, 64), (32, 32)))
        self.assertEqual(protocol.DROPOUT, (0.1, 0.3))
        self.assertEqual((protocol.NUM_TASKS, protocol.NUM_ENVS), (2, 2))
        self.assertEqual(protocol.NEWTASK_REP_DIM, 64)
        self.assertEqual(
            (protocol.AUC_FLOOR_CTR, protocol.AUC_FLOOR_CVR, protocol.AUC_FLOOR_BSI),
            (0.55, 0.50, 0.53),
        )
        self.assertEqual(protocol.VAL_TEST_GAP_BSI, 0.05)
        self.assertEqual((protocol.GATE_MIN, protocol.GATE_MAX), (0.05, 0.95))
        self.assertEqual(protocol.ENV_SHARE_MIN, 0.05)
        self.assertEqual(protocol.IMPROVE_DELTA_AUC_TEST_BSI, 0.005)

    def test_data_files_point_at_aliccp(self):
        self.assertEqual(
            protocol.DATA_FILES,
            {
                "train": "dataset/AliCCP/ctr_cvr.train",
                "val": "dataset/AliCCP/ctr_cvr.dev",
                "test": "dataset/AliCCP/ctr_cvr.test",
            },
        )


class TestVocabAndSeeds(unittest.TestCase):
    def test_build_vocab_does_not_mutate_global(self):
        vocab = protocol.build_vocab()
        self.assertEqual(len(vocab), 16)
        self.assertNotIn("101", vocab)
        self.assertNotIn("301", vocab)
        # 全局字典未被污染（spec 2.2）
        self.assertIn("101", AliCCP_Vocabulary_Size)
        self.assertIn("301", AliCCP_Vocabulary_Size)
        self.assertEqual(len(AliCCP_Vocabulary_Size), 18)

    def test_env_ids_independent_of_global_rng(self):
        torch.manual_seed(0)
        x1 = torch.rand(3)
        torch.manual_seed(0)
        e1 = protocol.make_env_ids(1000, protocol.ENV_SEED)
        x2 = torch.rand(3)
        e2 = protocol.make_env_ids(1000, protocol.ENV_SEED)
        self.assertTrue(torch.equal(x1, x2))  # 未消耗全局 RNG
        self.assertTrue(torch.equal(e1, e2))  # 确定性
        self.assertEqual(e1.shape, (1000,))
        self.assertTrue(bool(((e1 == 0) | (e1 == 1)).all()))


class TestPrefixIdentity(unittest.TestCase):
    def test_prefix_sha256_deterministic_and_budget_sensitive(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            write_aliccp_file(path, ROWS)
            sha_all = protocol.prefix_sha256(path, 4)
            self.assertEqual(sha_all, protocol.prefix_sha256(path, 4))
            sha_2 = protocol.prefix_sha256(path, 2)
            self.assertNotEqual(sha_all, sha_2)
            # 手工等价：表头 + 前 2 条数据行的原始字节
            raw = path.read_bytes()
            n_rows = 2
            offset = len(HEADER.encode()) + 1
            for _ in range(n_rows):
                offset += raw.index(b"\n", offset) - offset + 1
            import hashlib

            self.assertEqual(sha_2, hashlib.sha256(raw[:offset]).hexdigest())

    def test_prefix_sha256_rejects_budget_over_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            write_aliccp_file(path, ROWS)
            with self.assertRaises(ValueError):
                protocol.prefix_sha256(path, 5)

    def test_scan_label_counts_matches_data(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            write_aliccp_file(path, ROWS)
            counts = protocol.scan_label_counts(path, 4)
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
            loader_counts = protocol.dataset_label_counts(dataset)
            raw_counts = protocol.scan_label_counts(path, 4)
            for key in ("click1", "purchase1", "bsi_pos"):
                self.assertEqual(loader_counts[key], raw_counts[key], key)

    def test_duplicate_stats(self):
        with tempfile.TemporaryDirectory() as td:
            train = Path(td) / "train.csv"
            val = Path(td) / "val.csv"
            # train 内重复一行；train∩val 共享一行
            write_aliccp_file(train, [ROWS[0], ROWS[1], ROWS[0]])
            write_aliccp_file(val, [ROWS[1], ROWS[2]])
            stats = protocol.duplicate_stats({"train": (train, 3), "val": (val, 2)})
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
            fp = protocol.build_fingerprint("tiny", files)
            self.assertEqual(fp["prefix_tag"], "tiny")
            self.assertIn("fingerprint_sha256", fp)
            self.assertEqual(fp, protocol.build_fingerprint("tiny", files))  # 确定性
            protocol.verify_fingerprint(fp)  # 通过
            # 篡改文件 → 校验失败
            write_aliccp_file(test, ROWS[:3])
            with self.assertRaises(AssertionError):
                protocol.verify_fingerprint(fp)

    def test_fingerprint_roundtrip_to_disk(self):
        with tempfile.TemporaryDirectory() as td:
            train = Path(td) / "train.csv"
            write_aliccp_file(train, ROWS)
            files = {"train": (train, 4)}
            root = Path(td) / "artifacts"
            fp, created = protocol.ensure_fingerprint(root, "tiny", files)
            self.assertTrue(created)
            fp2, created2 = protocol.ensure_fingerprint(root, "tiny", files)
            self.assertFalse(created2)
            self.assertEqual(fp["fingerprint_sha256"], fp2["fingerprint_sha256"])


class TestStage1Artifact(unittest.TestCase):
    def test_stage1_id_content_addressed(self):
        base = protocol.make_stage1_id("abcd1234", protocol.MODEL_SEED, 3, "cafe0000")
        self.assertEqual(base, f"s1-abcd1234-m{protocol.MODEL_SEED}-e3-cafe0000")
        self.assertNotEqual(base, protocol.make_stage1_id("abcd1234", protocol.MODEL_SEED, 4, "cafe0000"))

    def test_save_and_load_roundtrip_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = {"w": torch.zeros(2)}
            env_ids = torch.tensor([0, 1, 1, 0])
            meta = {"stage1_id": "s1-x", "backbone_sha256": "deadbeef"}
            path = protocol.save_stage1(root, "s1-x", backbone_state=state, env_ids=env_ids, meta=meta)
            loaded = protocol.load_stage1(root, "s1-x")
            self.assertTrue(torch.equal(loaded["backbone_state"]["w"], state["w"]))
            self.assertTrue(torch.equal(loaded["env_ids"], env_ids))
            self.assertEqual(loaded["meta"], meta)
            with self.assertRaises(FileExistsError):
                protocol.save_stage1(root, "s1-x", backbone_state=state, env_ids=env_ids, meta=meta)
            self.assertTrue(path.is_dir())

    def test_load_missing_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                protocol.load_stage1(Path(td), "s1-none")


class TestFreezeTrio(unittest.TestCase):
    def _module(self):
        m = nn.Sequential(nn.Linear(3, 2), nn.Dropout(0.5))
        m.train()
        for p in m.parameters():
            p.grad = torch.ones_like(p)
        return m

    def test_freeze_sets_eval_and_no_grad(self):
        m = self._module()
        protocol.freeze_backbone(m)
        self.assertFalse(m.training)
        self.assertTrue(all(not p.requires_grad for p in m.parameters()))

    def test_backbone_sha256_stable_and_sensitive(self):
        m = self._module()
        sha1 = protocol.backbone_sha256(m)
        self.assertEqual(sha1, protocol.backbone_sha256(m))
        with torch.no_grad():
            m[0].weight[0, 0] += 1.0
        self.assertNotEqual(sha1, protocol.backbone_sha256(m))

    def test_assert_no_grads(self):
        m = self._module()
        with self.assertRaises(AssertionError):
            protocol.assert_no_grads(m)
        m.zero_grad(set_to_none=True)
        protocol.assert_no_grads(m)


class TestGitState(unittest.TestCase):
    def test_git_state_records_commit_and_dirty_flag(self):
        state = protocol.git_state()
        self.assertIn("commit", state)
        self.assertIsInstance(state["dirty"], bool)


class TestRunIdAndSummary(unittest.TestCase):
    def test_make_run_id_format(self):
        run_id = protocol.make_run_id(
            datetime(2026, 10, 3, 15, 30), prefix_tag="p2M-v500k-t1M",
            model_seed=protocol.MODEL_SEED, tag="short", commit="abc1234",
        )
        self.assertEqual(run_id, "20261003-1530-p2M-v500k-t1M-m1688723512-short-abc1234")

    def test_append_summary_row_appends_only(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "SUMMARY.md"
            row1 = {c: "1" for c in protocol.SUMMARY_COLUMNS}
            row2 = {c: "2" for c in protocol.SUMMARY_COLUMNS}
            protocol.append_summary_row(path, row1)
            protocol.append_summary_row(path, row2)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 4)  # header + separator + 2 rows
            self.assertIn("| run_id |", lines[0])
            self.assertIn("| 1 |", lines[2])
            self.assertIn("| 2 |", lines[3])

    def test_code_commit_returns_hex(self):
        commit = protocol.code_commit()
        self.assertTrue(commit == "nogit" or all(c in "0123456789abcdef" for c in commit))


class TestVerifyLabelCounts(unittest.TestCase):
    def test_verify_len_and_counts(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            write_aliccp_file(path, ROWS)
            dataset = AliCCPDataset(str(path), 4)
            expected = protocol.scan_label_counts(path, 4)
            protocol.verify_label_counts(dataset, 4, expected)  # 通过
            with self.assertRaises(AssertionError):
                protocol.verify_label_counts(dataset, 5, expected)  # 长度不符
            bad = dict(expected)
            bad["click1"] = expected["click1"] + 1
            with self.assertRaises(AssertionError):
                protocol.verify_label_counts(dataset, 4, bad)


if __name__ == "__main__":
    unittest.main()
