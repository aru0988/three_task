"""pipeline.run_stage1 端到端（tiny CPU）+ 旧 run_census_benchmark.run_stage1 黄金等价。

tiny 夹具与旧 census_benchmark/tests/test_smoke.py 同口径：
train_ds 是「全量训练集」，test_ds 按 split seed 切成 val / test，与真实路径同构。
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from benchmark.datasets import census
from benchmark.pipeline import run_stage1
from benchmark import protocol as bp

_HAVE_LEGACY = importlib.util.find_spec("census_benchmark") is not None

GOLDEN_SEED = 20261010


class TinyCensus(Dataset):
    """极小 CensusIncome 形状：(income, marital, new_task, features)。"""
    def __init__(self, n, seed):
        gen = torch.Generator().manual_seed(seed)
        self.features = {"a": torch.randint(0, 3, (n,), generator=gen),
                         "b": torch.randint(0, 2, (n,), generator=gen)}
        self.labels = [torch.randint(0, 2, (n,), generator=gen).float() for _ in range(3)]

    def __len__(self):
        return int(self.features["a"].shape[0])

    def __getitem__(self, i):
        return (self.labels[0][i], self.labels[1][i], self.labels[2][i],
                {"a": self.features["a"][i], "b": self.features["b"][i]})


def tiny_model():
    """与旧 test_smoke.tiny_model 同口径：dnn_input 维度 = 2 特征 × embedding_size 4 = 8。"""
    from multitaskrec.model import MPTRec
    return MPTRec(num_tasks=2, feature_vocabulary={"a": 3, "b": 2}, embedding_size=4,
                  input_size=8, expert_dnn_hidden_units=(8, 4),
                  tower_dnn_hidden_units=(4, 2), device=torch.device("cpu"))


def tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = census.make_split(len(test_ds), census.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, census.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


def run_new(root, epochs=2):
    """新 pipeline tiny 全流程（同种子、同夹具顺序）。"""
    torch.manual_seed(GOLDEN_SEED)
    loaders, stats, indices = tiny_inputs()
    profile = census.stage1_profile(loaders=loaders, stats=stats, indices=indices)
    return run_stage1(profile, root, epochs=epochs, device=torch.device("cpu"), model=tiny_model())


# ---- 黄金常量：2026-10-09 由旧 run_census_benchmark.run_stage1（GOLDEN_SEED + tiny_inputs）捕获 ----
FROZEN = {
    "stage1_id": "s1-20de3506-m1685480945-e2-216ed6a1",
    "backbone_sha256": "0f47f574ffca32118d10853d116d592c4664ea2cd5c6d9da924a1d83612119db",
    "env_ids_sha256": "937ffce929ab2954d34700c56e59639711460106b4787042bca8b16136c99f23",
}


class TestStage1Tiny(unittest.TestCase):
    def test_stage1_smoke_cpu_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            out = run_new(root)
            stage1_path = Path(out["dir"])
            for name in ("backbone.pt", "env_ids.pt", "meta.json", "stdout.log"):
                self.assertTrue((stage1_path / name).exists(), name)
            self.assertTrue(out["stage1_id"].startswith("s1-"))
            meta = json.loads((stage1_path / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["epochs"], meta["model_seed"], meta["env_seed"]),
                             (2, census.MODEL_SEED, census.ENV_SEED))
            self.assertEqual(len(meta["epoch_records"]), 2)                # M1 每 epoch 一条
            self.assertIn("env_acc", meta["epoch_records"][0])
            self.assertEqual(len(meta["cluster_records"]), 1)              # epochs=2 → 第 2 轮后聚一次（M2）
            self.assertEqual(len(meta["fuse_loss_0_list"]), 2)
            self.assertEqual(meta["env_ids_sha256"],
                             bp.sha256_tensor(bp.load_stage1(root, out["stage1_id"])["env_ids"]))
            self.assertIn("best_epoch", meta)
            self.assertTrue((root / "splits" / str(census.SPLIT_SEED) / "split_fingerprint.json").exists())
            with self.assertRaises(FileExistsError):                      # 内容寻址 + 只读
                run_new(root)

    def test_frozen_golden(self):
        with tempfile.TemporaryDirectory() as td:
            out = run_new(Path(td) / "artifacts")
            self.assertEqual(out["stage1_id"], FROZEN["stage1_id"])
            self.assertEqual(out["meta"]["backbone_sha256"], FROZEN["backbone_sha256"])
            self.assertEqual(out["meta"]["env_ids_sha256"], FROZEN["env_ids_sha256"])


@unittest.skipUnless(_HAVE_LEGACY, "旧实现包已删除（迁移完成）")
class TestLegacyGoldenEquivalence(unittest.TestCase):
    def test_new_pipeline_matches_legacy_run_stage1(self):
        import run_census_benchmark as legacy
        with tempfile.TemporaryDirectory() as td_old, tempfile.TemporaryDirectory() as td_new:
            torch.manual_seed(GOLDEN_SEED)
            loaders, stats, indices = tiny_inputs()
            old = legacy.run_stage1(Path(td_old) / "old", epochs=2, device=torch.device("cpu"),
                                    model=tiny_model(), loaders=loaders, stats=stats,
                                    indices=indices, tag="short")
            new = run_new(Path(td_new) / "new")
            self.assertEqual(new["stage1_id"], old["stage1_id"])
            self.assertEqual((new["stage1_id"], new["meta"]["backbone_sha256"], new["meta"]["env_ids_sha256"]),
                             (FROZEN["stage1_id"], FROZEN["backbone_sha256"], FROZEN["env_ids_sha256"]))
            for key in old["meta"]:
                if key == "created":
                    continue
                self.assertEqual(new["meta"][key], old["meta"][key], key)
            # 旧路径落盘的 stdout.log 内容对比（tee 行为不变）
            self.assertEqual((Path(new["dir"]) / "stdout.log").read_text(encoding="utf-8"),
                             (Path(old["dir"]) / "stdout.log").read_text(encoding="utf-8"))
            # 旧 meta.json 与新 meta.json 逐行一致（除 created 字段）
            new_raw = (Path(new["dir"]) / "meta.json").read_text(encoding="utf-8")
            old_raw = (Path(old["dir"]) / "meta.json").read_text(encoding="utf-8")
            self.assertEqual(_strip_created(new_raw), _strip_created(old_raw))


def _strip_created(text: str) -> list:
    import re
    return [line for line in text.splitlines() if not re.search(r'"created"', line)]


if __name__ == "__main__":
    unittest.main()
