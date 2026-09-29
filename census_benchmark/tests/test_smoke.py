import json, subprocess, tempfile, unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import protocol as P
import run_census_benchmark

REPO = Path(__file__).resolve().parents[2]          # 测试依赖 cwd = 仓库根


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _ignored(rel_path):
    """git check-ignore 返回 0 表示被忽略。"""
    return subprocess.run(["git", "check-ignore", "-q", rel_path], cwd=REPO).returncode == 0


class TestStaticGuards(unittest.TestCase):
    def test_gitignore_blocks_artifacts_and_keeps_summary(self):
        for rel in ("artifacts/census_stage2/stage1", "artifacts/census_stage2/runs",
                    "artifacts/census_stage2/splits",
                    "artifacts/census_stage2/stage1/s1-deadbeef-m1-e2-cafe0000/backbone.pt",
                    "artifacts/census_stage2/runs/20260929-1530-s20260929-m1685480945-short-fd75198/metrics.json"):
            self.assertTrue(_ignored(rel), f"应被忽略: {rel}")
        self.assertFalse(_ignored("artifacts/census_stage2/SUMMARY.md"))     # SUMMARY 必须能入库

    def test_static_guards_master_untouched_and_no_flops(self):
        diff = _git("diff", "--name-only", "master", "--", "multitaskrec", "config.py",
                    "CensusIncome_MPTRec.py", "CensusIncome_NewTask.py").stdout.strip()
        self.assertEqual(diff, "", f"协议分支不得改动模型/master 文件: {diff}")
        protocol_src = (REPO / "census_benchmark" / "protocol.py").read_text(encoding="utf-8")
        runner_src = (REPO / "run_census_benchmark.py").read_text(encoding="utf-8")
        for src in (protocol_src, runner_src):
            self.assertNotIn("fvcore", src)                    # 不做 FLOPs（spec 2.2.4）
        self.assertIn("random_state=split_seed", protocol_src)  # 划分只由 split seed 决定（spec 5.3.5）
        self.assertNotIn("random_state=model_seed", protocol_src)


class TinyCensus(Dataset):
    """极小 CensusIncome 形状：(income, marital, new_task, features)。"""
    def __init__(self, n, seed):
        gen = torch.Generator().manual_seed(seed)
        self.features = {"a": torch.randint(0, 3, (n,), generator=gen),
                         "b": torch.randint(0, 2, (n,), generator=gen)}
        self.labels = [torch.randint(0, 2, (n,), generator=gen).float() for _ in range(3)]
    def __len__(self): return int(self.features["a"].shape[0])
    def __getitem__(self, i):
        return (self.labels[0][i], self.labels[1][i], self.labels[2][i],
                {"a": self.features["a"][i], "b": self.features["b"][i]})


def tiny_model():
    """与 test_protocol.tiny_mptrec 同口径：dnn_input 维度 = 2 个特征 × embedding_size 4 = 8。"""
    return run_census_benchmark.MPTRec(num_tasks=2, feature_vocabulary={"a": 3, "b": 2}, embedding_size=4,
                                       input_size=8, expert_dnn_hidden_units=(8, 4),
                                       tower_dnn_hidden_units=(4, 2), device=torch.device("cpu"))


def tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    """train_ds 是「全量训练集」，test_ds 按 P.SPLIT_SEED 切成 val / test，与真实路径同构。"""
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


class TestStage1Smoke(unittest.TestCase):
    def test_stage1_smoke_cpu_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            loaders, stats, indices = tiny_inputs()
            out = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_model(),
                                                  loaders=loaders, stats=stats, indices=indices, tag="short")
            stage1_path = Path(out["dir"])
            for name in ("backbone.pt", "env_ids.pt", "meta.json"):
                self.assertTrue((stage1_path / name).exists())
            self.assertTrue(out["stage1_id"].startswith("s1-"))
            meta = json.loads((stage1_path / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["epochs"], meta["model_seed"], meta["env_seed"]), (2, P.MODEL_SEED, P.ENV_SEED))
            self.assertEqual(len(meta["epoch_records"]), 2)                 # M1 每 epoch 一条
            self.assertIn("env_acc", meta["epoch_records"][0])
            self.assertEqual(len(meta["cluster_records"]), 1)               # epochs=2 → 第 2 轮后聚一次（M2）
            self.assertEqual(len(meta["fuse_loss_0_list"]), 2)
            self.assertEqual(meta["env_ids_sha256"], P.sha256_tensor(P.load_stage1(root, out["stage1_id"])["env_ids"]))
            self.assertTrue((root / "splits" / str(P.SPLIT_SEED) / "split_fingerprint.json").exists())
            with self.assertRaises(FileExistsError):                        # 内容寻址 + 只读
                run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_model(),
                                                loaders=loaders, stats=stats, indices=indices, tag="short")


class TestStage2Smoke(unittest.TestCase):
    def test_stage2_smoke_cpu_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            loaders, stats, indices = tiny_inputs()
            stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_model(),
                                                     loaders=loaders, stats=stats, indices=indices)
            out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=3, device=device,
                                                  model=tiny_model(), loaders=loaders, stats=stats,
                                                  indices=indices, input_size=8, rep_dim=4)
            run_path = Path(out["run_dir"])
            for name in ("config.json", "metrics.json", "split_fingerprint.json", "env_ids.pt",
                         "newtask.pt", "gate_report.json", "stdout.log"):
                self.assertTrue((run_path / name).exists())
            report = json.loads((run_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["A1"]["pass"])                      # 同一 checkpoint + 真冻结
            self.assertTrue(report["A1"]["detail"]["backbone_sha_equal"])
            self.assertTrue(report["A2"]["pass"])
            self.assertTrue(report["A5"]["pass"])                      # env_ids 只读取、不重抽
            self.assertIn("overall_pass", report)                      # 小夹具上 B 类可能为 False，属预期
            self.assertEqual(report["A3"]["status"], "on_demand")      # spec 8：A3 按需触发
            metrics_json = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metrics_json["stage2"]["epoch_records"]), 3)
            self.assertGreaterEqual(metrics_json["stage2"]["test_auc"], 0.0)
            self.assertEqual(len(metrics_json["mechanism"]["gate_mean"]), 2)
            self.assertEqual(metrics_json["backbone_sha256_before"], metrics_json["backbone_sha256_after"])
            self.assertEqual(metrics_json["stage1_id"], stage1["stage1_id"])
            summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(summary), 3)                          # 表头 + 分隔 + 1 行
            self.assertIn(out["run_id"], summary[2])

    def test_stage2_missing_stage1_dir_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                run_census_benchmark.run_stage2(Path(td) / "artifacts",
                                                stage1_dir=Path(td) / "no_such_stage1",
                                                device=torch.device("cpu"))
