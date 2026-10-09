"""pipeline.run_stage1/2 的 aliccp tiny e2e + 旧 aliccp_benchmark.bench 黄金等价。

tiny 夹具与 infra/aliccp-fair-benchmark 分支 tests/test_smoke.py 同口径。
旧实现包删除后 TestLegacyGoldenEquivalence 自动跳过，FROZEN 常量继续守护位级一致。
注意：前缀指纹含数据文件路径 → stage1_id 依 tempdir 而变，故 FROZEN 只收录路径无关量
（config_hash / backbone_sha256 / env_ids_sha256 / 各 AUC / per_epoch / cluster_events）；
新旧对比测试共用同一数据目录，因此可以逐项比对 stage1_id 与指纹文件本身。
"""
import importlib.util
import json
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import torch

from benchmark.datasets import aliccp
from benchmark.pipeline import run_stage1, run_stage2

_HAVE_LEGACY = importlib.util.find_spec("aliccp_benchmark") is not None

FIXED_NOW = datetime(2026, 10, 9, 12, 30)

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


def tiny_inputs(td) -> tuple:
    """同一 td 下写 tiny 三切分：新旧对比必须共用同一数据路径（前缀指纹含文件路径）。"""
    data_dir = Path(td) / "data"
    data_dir.mkdir()
    write_aliccp_file(data_dir / "train.csv", TRAIN_ROWS)
    write_aliccp_file(data_dir / "val.csv", VAL_ROWS)
    write_aliccp_file(data_dir / "test.csv", TEST_ROWS)
    data_files = {"train": str(data_dir / "train.csv"), "val": str(data_dir / "val.csv"),
                  "test": str(data_dir / "test.csv")}
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return data_files, budgets


def tiny_profile(*, data_files, budgets, **overrides):
    kwargs = dict(prefix_tag="tiny", model_seed=123, env_seed=456, vocab=TINY_VOCAB,
                  expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
                  stage1_epochs=2, stage1_patience=2, stage2_epochs=1, stage2_patience=1,
                  enforce_b=False, log=lambda *a, **k: None)
    kwargs.update(overrides)
    return aliccp.build_profile(data_files=data_files, budgets=budgets, **kwargs)


def run_new_stage1(root, data_files, budgets):
    return run_stage1(tiny_profile(data_files=data_files, budgets=budgets), root,
                      device=torch.device("cpu"))


def run_new_stage2(root, data_files, budgets, stage1_dir, now=FIXED_NOW):
    return run_stage2(tiny_profile(data_files=data_files, budgets=budgets), root,
                      stage1_dir=stage1_dir, tag="smoke", device=torch.device("cpu"),
                      now=now, log=lambda *a, **k: None)


# ---- 黄金常量：2026-10-09 由旧 aliccp_benchmark.bench（tiny 夹具）捕获；仅路径无关量 ----
FROZEN1 = {
    "config_hash": "40095029f95dbcce4aec376be5f4d1661ef91682359db552a22a35ebb8d7fd7d",
    "backbone_sha256": "f6aed806b9c53ddff6420a1ead002c4bc9389d9f4aa362df59bb26cda430db39",
    "env_ids_sha256": "db9a96d26f925f70e907553f4838863d514b070576ed40bdaf798e4998672ab3",
    "best_epoch": 1,
    "best_val_auc_ctr": 0.4444444444444445,
    "best_val_auc_cvr": 0.4,
    "test_auc_ctr": 0.3125,
    "test_auc_cvr": 0.5714285714285714,
    "env_acc": 0.4166666666666667,
    "cluster_events": [{"epoch": 2, "diff_num": 5, "env_0": 2, "env_1": 10}],
    "per_epoch": [
        {"epoch": 1, "auc_val_ctr": 0.4444444444444445, "auc_val_cvr": 0.4,
         "uni_loss_0": 0.6947970986366272, "uni_loss_1": 0.6140116453170776,
         "fuse_loss_0": 0.6959519386291504, "fuse_loss_1": 0.6029359698295593,
         "env_loss": 0.7161352038383484},
        {"epoch": 2, "auc_val_ctr": 0.4444444444444445, "auc_val_cvr": 0.4,
         "uni_loss_0": 0.6931105256080627, "uni_loss_1": 0.6072444915771484,
         "fuse_loss_0": 0.6948403716087341, "fuse_loss_1": 0.5995516180992126,
         "env_loss": 0.8037147521972656},
    ],
}

FROZEN2 = {
    "best_epoch": 1,
    "best_val_auc_bsi": 0.5,
    "test_auc_bsi": 0.6666666666666667,
    "gate_mean": [0.6167663335800171, 0.3832337061564128],
    "per_epoch": [{"epoch": 1, "train_loss": 0.6365289092063904, "val_auc_bsi": 0.5}],
}


class TestStage1Tiny(unittest.TestCase):
    def test_stage1_smoke_cpu_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            data_files, budgets = tiny_inputs(td)
            out = run_new_stage1(root, data_files, budgets)
            stage1_path = Path(out["dir"])
            for name in ("backbone.pt", "env_ids.pt", "meta.json"):
                self.assertTrue((stage1_path / name).exists(), name)
            self.assertFalse((stage1_path / "stdout.log").exists())     # aliccp 现状：无 stdout.log
            meta = out["meta"]
            self.assertTrue(out["stage1_id"].startswith("s1-"))
            self.assertEqual(out["stage1_id"], aliccp.stage1_id_for(meta))
            self.assertEqual(meta["stage1_id"],
                             f"s1-{meta['fingerprint_sha256'][:8]}-m123-e2-{meta['config_hash'][:8]}")
            self.assertEqual(len(meta["per_epoch"]), 2)
            self.assertEqual(len(meta["cluster_events"]), 1)            # epochs=2 → 第 2 轮前聚一次
            self.assertEqual(set(meta["cluster_events"][0]),
                             {"epoch", "diff_num", "env_0", "env_1"})
            self.assertGreaterEqual(meta["env_acc"], 0.0)
            self.assertTrue((root / "splits" / "tiny" / "prefix_fingerprint.json").exists())
            with self.assertRaises(FileExistsError):                    # 内容寻址 + 只读
                run_new_stage1(root, data_files, budgets)

    def test_frozen_golden(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            data_files, budgets = tiny_inputs(td)
            meta = run_new_stage1(root, data_files, budgets)["meta"]
            for key, expected in FROZEN1.items():
                self.assertEqual(meta[key], expected, key)


class TestStage2Tiny(unittest.TestCase):
    def test_stage2_smoke_cpu_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            data_files, budgets = tiny_inputs(td)
            stage1 = run_new_stage1(root, data_files, budgets)
            out = run_new_stage2(root, data_files, budgets, stage1["dir"])
            run_path = Path(out["run_dir"])
            for name in ("config.json", "metrics.json", "gate_report.json", "newtask.pt"):
                self.assertTrue((run_path / name).exists(), name)
            gates = out["gates"]
            for gate_id in ("A1", "A2", "A4", "A5", "A6"):
                self.assertEqual(gates[gate_id]["verdict"], "PASS", gate_id)
            self.assertEqual(gates["A3"]["verdict"], "SKIP")
            self.assertIn("B1", gates)                                  # B 类只记录不判定
            self.assertTrue(out["hard_pass"])                           # enforce_b=False，A 全过
            metrics = out["metrics"]
            self.assertEqual(metrics["stage1_id"], stage1["stage1_id"])
            self.assertEqual(metrics["backbone_sha256_loaded"], stage1["meta"]["backbone_sha256"])
            self.assertEqual(metrics["env_ids_sha256"], stage1["meta"]["env_ids_sha256"])
            self.assertIsInstance(metrics["test_auc_bsi"], float)
            self.assertIsInstance(metrics["git"]["dirty"], bool)
            self.assertTrue(out["run_id"].startswith("20261009-1230-tiny-m123-smoke-"))
            summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(summary), 3)
            self.assertIn(out["run_id"], summary[2])

    def test_frozen_golden(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            data_files, budgets = tiny_inputs(td)
            stage1 = run_new_stage1(root, data_files, budgets)
            out = run_new_stage2(root, data_files, budgets, stage1["dir"])
            self.assertEqual(out["run_id"], "20261009-1230-tiny-m123-smoke-" + _commit())
            metrics = out["metrics"]
            for key, expected in FROZEN2.items():
                self.assertEqual(metrics[key], expected, key)


def _commit() -> str:
    from benchmark.protocol import code_commit
    return code_commit()


def _strip_volatile(text: str) -> list:
    return [line for line in text.splitlines()
            if not re.search(r'"(wall_seconds|peak_vram_mb)"', line)]


@unittest.skipUnless(_HAVE_LEGACY, "旧实现包已删除（迁移完成）")
class TestLegacyGoldenEquivalence(unittest.TestCase):
    def test_new_pipeline_matches_legacy_stage1(self):
        import aliccp_benchmark.bench as legacy
        with tempfile.TemporaryDirectory() as td:
            data_files, budgets = tiny_inputs(td)                    # 新旧共用数据路径
            root_old, root_new = Path(td) / "old", Path(td) / "new"
            old_meta = legacy.run_stage1(
                root=root_old, data_files=data_files, budgets=budgets, prefix_tag="tiny",
                model_seed=123, env_seed=456, epochs=2, patience=2, device=torch.device("cpu"),
                vocab=TINY_VOCAB, expert_hidden=(4,), tower_hidden=(4,), embedding_size=2,
                input_size=4, log=lambda *a, **k: None)
            new = run_new_stage1(root_new, data_files, budgets)
            self.assertEqual(new["stage1_id"], old_meta["stage1_id"])
            self.assertEqual(new["meta"]["config_hash"], FROZEN1["config_hash"])
            self.assertEqual(new["meta"]["backbone_sha256"], FROZEN1["backbone_sha256"])
            self.assertEqual(new["meta"]["env_ids_sha256"], FROZEN1["env_ids_sha256"])
            for key in old_meta:
                if key in ("wall_seconds", "peak_vram_mb"):
                    continue
                self.assertEqual(new["meta"][key], old_meta[key], key)
            # 指纹文件与 meta.json 文本逐行一致（除易变的 wall/peak 行）
            old_fp = (root_old / "splits" / "tiny" / "prefix_fingerprint.json").read_text(encoding="utf-8")
            new_fp = (root_new / "splits" / "tiny" / "prefix_fingerprint.json").read_text(encoding="utf-8")
            self.assertEqual(new_fp, old_fp)
            new_raw = (Path(new["dir"]) / "meta.json").read_text(encoding="utf-8")
            old_raw = (root_old / "stage1" / old_meta["stage1_id"] / "meta.json").read_text(encoding="utf-8")
            self.assertEqual(_strip_volatile(new_raw), _strip_volatile(old_raw))

    def test_new_pipeline_matches_legacy_stage2(self):
        import aliccp_benchmark.bench as legacy
        from benchmark import protocol as bp
        with tempfile.TemporaryDirectory() as td:
            data_files, budgets = tiny_inputs(td)
            root_old, root_new = Path(td) / "old", Path(td) / "new"
            st_old = run_new_stage1(root_old, data_files, budgets)   # 新 stage1 先证与旧一致
            st_new = run_new_stage1(root_new, data_files, budgets)
            self.assertEqual(st_old["stage1_id"], st_new["stage1_id"])
            fixed_run_id = bp.make_run_id(FIXED_NOW, prefix="tiny", model_seed=123, tag="smoke",
                                          commit=bp.code_commit())
            old = legacy.run_stage2(
                root=root_old, stage1_id=st_old["stage1_id"], data_files=data_files,
                budgets=budgets, prefix_tag="tiny", model_seed=123, epochs=1, patience=1,
                tag="smoke", device=torch.device("cpu"), vocab=TINY_VOCAB, expert_hidden=(4,),
                tower_hidden=(4,), embedding_size=2, input_size=4, enforce_b=False,
                run_id=fixed_run_id, log=lambda *a, **k: None)
            new = run_new_stage2(root_new, data_files, budgets, st_new["dir"])
            self.assertEqual(new["run_id"], old["run_id"])
            self.assertEqual(new["run_id"], fixed_run_id)
            self.assertEqual(new["gates"], old["gates"])
            self.assertEqual(new["hard_pass"], old["hard_pass"])
            self.assertEqual(new["metrics"]["test_auc_bsi"], FROZEN2["test_auc_bsi"])
            old_metrics = json.loads((Path(old["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
            new_metrics = json.loads((Path(new["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
            for key in old_metrics:
                if key in ("wall_seconds", "peak_vram_mb"):
                    continue
                self.assertEqual(new_metrics[key], old_metrics[key], key)
            self.assertEqual(json.loads((Path(new["run_dir"]) / "config.json").read_text(encoding="utf-8")),
                             json.loads((Path(old["run_dir"]) / "config.json").read_text(encoding="utf-8")))
            self.assertEqual(json.loads((Path(new["run_dir"]) / "gate_report.json").read_text(encoding="utf-8")),
                             json.loads((Path(old["run_dir"]) / "gate_report.json").read_text(encoding="utf-8")))
            self.assertEqual((root_new / "SUMMARY.md").read_text(encoding="utf-8"),
                             (root_old / "SUMMARY.md").read_text(encoding="utf-8"))
            # newtask.pt 逐参数一致
            old_newtask = torch.load(Path(old["run_dir"]) / "newtask.pt", map_location="cpu")
            new_newtask = torch.load(Path(new["run_dir"]) / "newtask.pt", map_location="cpu")
            self.assertEqual(list(old_newtask), list(new_newtask))
            for name in old_newtask:
                self.assertTrue(torch.equal(old_newtask[name], new_newtask[name]), name)


if __name__ == "__main__":
    unittest.main()
