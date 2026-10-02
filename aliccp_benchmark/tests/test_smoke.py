"""端到端小流程测试：stage1 产物 → stage2 加载 → 真冻结 → 门禁 → SUMMARY（TDD：先于实现编写）。"""
import tempfile
import unittest
from pathlib import Path

import torch

from aliccp_benchmark import bench, protocol

HEADER = "click,purchase,X,121,122,301"
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


def write_aliccp_file(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


def make_tiny_root(td: str):
    root = Path(td) / "artifacts"
    data_dir = Path(td) / "data"
    data_dir.mkdir()
    write_aliccp_file(data_dir / "train.csv", TRAIN_ROWS)
    write_aliccp_file(data_dir / "val.csv", VAL_ROWS)
    write_aliccp_file(data_dir / "test.csv", TEST_ROWS)
    data_files = {
        "train": str(data_dir / "train.csv"),
        "val": str(data_dir / "val.csv"),
        "test": str(data_dir / "test.csv"),
    }
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return root, data_files, budgets


class TestTinyEndToEnd(unittest.TestCase):
    def test_stage1_artifact_then_stage2_freeze_and_gates(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = make_tiny_root(td)
            meta = bench.run_stage1(
                root=root, data_files=data_files, budgets=budgets, prefix_tag="tiny",
                model_seed=123, env_seed=456, epochs=2, patience=2,
                device=torch.device("cpu"), vocab=TINY_VOCAB,
                expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
                log=lambda *a, **k: None,
            )
            self.assertEqual(meta["stage1_id"], bench.stage1_id_for(meta))
            art_dir = protocol.stage1_dir(root, meta["stage1_id"])
            self.assertTrue((art_dir / "backbone.pt").exists())
            self.assertTrue((art_dir / "env_ids.pt").exists())
            self.assertTrue((art_dir / "meta.json").exists())
            # epoch 2 触发一次 cluster（epoch%2==0），记录不得缺
            self.assertEqual(len(meta["cluster_events"]), 1)
            self.assertGreaterEqual(meta["env_acc"], 0.0)
            self.assertIn("test_auc_ctr", meta)

            result = bench.run_stage2(
                root=root, stage1_id=meta["stage1_id"], data_files=data_files, budgets=budgets,
                prefix_tag="tiny", model_seed=123, epochs=1, patience=1, tag="smoke",
                device=torch.device("cpu"), vocab=TINY_VOCAB,
                expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
                enforce_b=False, log=lambda *a, **k: None,
            )
            gates = result["gates"]
            for gate_id in ("A1", "A2", "A4", "A5", "A6"):
                self.assertEqual(gates[gate_id]["verdict"], "PASS", gate_id)
            self.assertEqual(gates["A3"]["verdict"], "SKIP")
            self.assertIn("B1", gates)  # smoke 记录 B 类但不判定
            self.assertIsInstance(result["metrics"]["test_auc_bsi"], float)
            self.assertEqual(result["metrics"]["stage1_id"], meta["stage1_id"])
            self.assertEqual(result["metrics"]["backbone_sha256_loaded"], meta["backbone_sha256"])
            # 运行产物
            run_dir = protocol.run_dir(root, result["run_id"])
            for name in ("config.json", "metrics.json", "gate_report.json", "newtask.pt"):
                self.assertTrue((run_dir / name).exists(), name)
            # 代码溯源：commit + dirty 标记必须落盘
            self.assertIn("git", result["metrics"])
            self.assertIsInstance(result["metrics"]["git"]["dirty"], bool)
            # SUMMARY 只追加一行
            summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(summary), 3)
            self.assertIn(result["run_id"], summary[2])

    def test_stage2_uses_artifact_env_ids_and_backbone_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = make_tiny_root(td)
            meta = bench.run_stage1(
                root=root, data_files=data_files, budgets=budgets, prefix_tag="tiny",
                model_seed=123, env_seed=456, epochs=1, patience=1,
                device=torch.device("cpu"), vocab=TINY_VOCAB,
                expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
                log=lambda *a, **k: None,
            )
            loaded = protocol.load_stage1(root, meta["stage1_id"])
            result = bench.run_stage2(
                root=root, stage1_id=meta["stage1_id"], data_files=data_files, budgets=budgets,
                prefix_tag="tiny", model_seed=123, epochs=1, patience=1, tag="smoke",
                device=torch.device("cpu"), vocab=TINY_VOCAB,
                expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
                enforce_b=False, log=lambda *a, **k: None,
            )
            self.assertEqual(result["metrics"]["env_ids_sha256"], protocol.sha256_tensor(loaded["env_ids"]))
            self.assertEqual(result["gates"]["A1"]["verdict"], "PASS")
            # 真冻结：训练后 backbone 参数梯度必须为 None
            self.assertTrue(result["metrics"]["backbone_grads_none"])


class TestRecordingManager(unittest.TestCase):
    def test_best_epoch_first_max(self):
        manager = bench.RecordingMPTRecTrainManager.__new__(bench.RecordingMPTRecTrainManager)
        manager.val_epoch_aucs = [[0.6, 0.6], [0.7, 0.7], [0.7, 0.7], [0.5, 0.5]]
        self.assertEqual(manager.best_epoch(), 2)


if __name__ == "__main__":
    unittest.main()
