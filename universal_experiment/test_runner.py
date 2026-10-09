"""runner.train_head 黄金等价 + profiles 常量固定 + stub profile 端到端（run_screen 编排）。

train_head 与旧实现（universal_experiment/run.py，两数据集共用同一函数体）逐值一致：
- arm B + cpu_loss=False（census 口径）与 arm U + cpu_loss=True（aliccp 口径，覆盖 shuffle 路径）。
黄金常量于 2026-10-09 由旧实现 tiny 夹具捕获，旧模块删除后仍持续断言。
"""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from multitaskrec.model import MPTRec, NewTask
from universal_experiment import profiles, runner
from universal_experiment.model import ResidualHead

_HAVE_LEGACY_RUN = importlib.util.find_spec("universal_experiment.run") is not None

REPO = Path(__file__).resolve().parents[1]
FIXTURE_SEED = 424242
HEAD_SEED = 20261009


def tiny_fixture(n_train=32, n_val=16, n_test=16):
    """train_head 的极小缓存夹具（列形状与真实四臂一致：x 8 / g,s,u,r 4）。"""
    gen = torch.Generator().manual_seed(FIXTURE_SEED)

    def split(n):
        return {"x": torch.randn(n, 8, generator=gen), "g": torch.randn(n, 4, generator=gen),
                "s0": torch.randn(n, 4, generator=gen), "s1": torch.randn(n, 4, generator=gen),
                "u": torch.randn(n, 4, generator=gen), "r": torch.randn(n, 4, generator=gen),
                "y": torch.randint(0, 2, (n,), generator=gen).float()}

    caches = {"train": split(n_train), "val": split(n_val), "test": split(n_test)}
    envs = [torch.randn(4, generator=gen), torch.randn(4, generator=gen)]
    return caches, envs


def make_head(arm):
    """同一种子下的头部构造；legacy 与新 runner 各调用一次 → 权重逐位一致。"""
    device = torch.device("cpu")
    torch.manual_seed(HEAD_SEED)
    original = NewTask(8, 4, [4, 2], 0.0, device)
    if arm == "B":
        return copy.deepcopy(original)
    torch.manual_seed(HEAD_SEED + 1)
    return ResidualHead(original, 4)


# ---- 黄金常量：2026-10-09 由旧 universal_experiment.run.train_head（tiny 夹具）捕获 ----
GOLDEN_B = {  # census 口径：lr=1e-3, batch=256, patience=2, cpu_loss=False
    "best_epoch": 1,
    "epochs": [
        {"epoch": 1, "val_auc": 0.39682539682539686, "loss": 0.7905197739601135, "steps": 1},
        {"epoch": 2, "val_auc": 0.36507936507936506, "loss": 0.7900040745735168, "steps": 1},
        {"epoch": 3, "val_auc": 0.36507936507936506, "loss": 0.7894918918609619, "steps": 1},
    ],
    "parameters": 85,
    "val_auc": 0.39682539682539686,
    "test_auc": 0.2909090909090909,
}
GOLDEN_U = {  # aliccp 口径：lr=1e-4, batch=2000, patience=2, cpu_loss=True（含 shuffle 路径）
    "best_epoch": 1,
    "epochs": [
        {"epoch": 1, "val_auc": 0.39682539682539686, "loss": 0.7913469076156616, "steps": 1},
        {"epoch": 2, "val_auc": 0.39682539682539686, "loss": 0.7912939786911011, "steps": 1},
        {"epoch": 3, "val_auc": 0.39682539682539686, "loss": 0.7912410497665405, "steps": 1},
    ],
    "parameters": 102,
    "val_auc": 0.39682539682539686,
    "test_auc": 0.2909090909090909,
    "val_shuffled_u_auc": 0.39682539682539686,
    "test_shuffled_u_auc": 0.3090909090909091,
}


class TestTrainHeadGolden(unittest.TestCase):
    """冻结黄金：旧模块删除后仍持续断言（train_head ≠ 旧实现即失败）。"""

    def _run(self, arm, out, **kwargs):
        caches, envs = tiny_fixture()
        return runner.train_head(make_head(arm), caches, envs, arm, torch.device("cpu"), 3, out, **kwargs)

    def test_arm_b_census_flavor(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = self._run("B", out, lr=1e-3, batch_size=256, patience=2)
            for key, value in GOLDEN_B.items():
                self.assertEqual(result[key], value, key)
            self.assertIn("wall_seconds", result)
            for name in ("B_predictions.npz", "B_head.pt", "B_metrics.json"):
                self.assertTrue((out / name).exists(), name)

    def test_arm_u_aliccp_flavor(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = self._run("U", out, lr=1e-4, batch_size=2000, patience=2, cpu_loss=True)
            for key, value in GOLDEN_U.items():
                self.assertEqual(result[key], value, key)
            for name in ("U_predictions.npz", "U_head.pt", "U_metrics.json"):
                self.assertTrue((out / name).exists(), name)


@unittest.skipUnless(_HAVE_LEGACY_RUN, "旧 universal_experiment.run 已删除（迁移完成）")
class TestLegacyTrainHeadEquivalence(unittest.TestCase):
    def _assert_equivalent(self, old, new, old_out, new_out, arm):
        self.assertEqual({k: v for k, v in new.items() if k != "wall_seconds"},
                         {k: v for k, v in old.items() if k != "wall_seconds"})
        with np.load(old_out / f"{arm}_predictions.npz") as old_npz, \
                np.load(new_out / f"{arm}_predictions.npz") as new_npz:
            self.assertEqual(set(old_npz.files), set(new_npz.files))
            for key in old_npz.files:
                self.assertTrue(np.array_equal(old_npz[key], new_npz[key]), key)

        def load_metrics(path):                    # wall_seconds 为计时量，剔除后逐字段一致
            metrics = json.loads(path.read_text(encoding="utf-8"))
            metrics.pop("wall_seconds", None)
            return metrics

        self.assertEqual(load_metrics(new_out / f"{arm}_metrics.json"),
                         load_metrics(old_out / f"{arm}_metrics.json"))
        old_state = torch.load(old_out / f"{arm}_head.pt", map_location="cpu")
        new_state = torch.load(new_out / f"{arm}_head.pt", map_location="cpu")
        self.assertEqual(set(old_state), set(new_state))
        for key in old_state:
            self.assertTrue(torch.equal(old_state[key], new_state[key]), key)

    def test_arm_b_matches_legacy(self):
        import universal_experiment.run as legacy
        caches, envs = tiny_fixture()
        with tempfile.TemporaryDirectory() as td_old, tempfile.TemporaryDirectory() as td_new:
            old_out, new_out = Path(td_old), Path(td_new)
            old = legacy.train_head(make_head("B"), caches, envs, "B", torch.device("cpu"), 3, old_out)
            new = runner.train_head(make_head("B"), caches, envs, "B", torch.device("cpu"), 3, new_out,
                                    lr=legacy.P.LR, batch_size=legacy.P.BATCH_SIZE,
                                    patience=legacy.P.PATIENCE)
            self._assert_equivalent(old, new, old_out, new_out, "B")
            for key, value in GOLDEN_B.items():
                self.assertEqual(new[key], value, key)

    def test_arm_u_matches_legacy_aliccp_flavor(self):
        from types import SimpleNamespace
        import universal_experiment.run as legacy
        caches, envs = tiny_fixture()
        saved = legacy.P
        legacy.P = SimpleNamespace(LR=1e-4, BATCH_SIZE=2000, PATIENCE=2)
        try:
            with tempfile.TemporaryDirectory() as td_old, tempfile.TemporaryDirectory() as td_new:
                old_out, new_out = Path(td_old), Path(td_new)
                old = legacy.train_head(make_head("U"), caches, envs, "U", torch.device("cpu"), 3,
                                        old_out, cpu_loss=True)
                new = runner.train_head(make_head("U"), caches, envs, "U", torch.device("cpu"), 3,
                                        new_out, lr=1e-4, batch_size=2000, patience=2, cpu_loss=True)
                self._assert_equivalent(old, new, old_out, new_out, "U")
        finally:
            legacy.P = saved


class TestProfileConstants(unittest.TestCase):
    """profiles 的预期哈希/种子必须与已提交 results/universal/* 一致（只读）。"""

    def test_census_constants_match_committed_results(self):
        cdir = REPO / "results" / "universal" / "census-seed1685480945"
        cfg = json.loads((cdir / "config.json").read_text(encoding="utf-8"))
        stage1 = json.loads((cdir / "stage1.json").read_text(encoding="utf-8"))
        self.assertEqual(profiles.CENSUS_SCREEN.expected_base_hash, stage1["base_hash"])
        self.assertEqual(profiles.CENSUS_SCREEN.model_seed, cfg["model_seed"])
        self.assertEqual(profiles.CENSUS_SCREEN.env_seed, cfg["env_seed"])
        self.assertTrue(stage1["matches_historical"])

    def test_aliccp_constants_match_committed_results(self):
        adir = REPO / "results" / "universal" / "aliccp-seed1688723512"
        cfg = json.loads((adir / "config.json").read_text(encoding="utf-8"))
        stage1 = json.loads((adir / "stage1.json").read_text(encoding="utf-8"))
        self.assertEqual(profiles.ALICCP_SCREEN.expected_base_hash, stage1["base_hash"])
        self.assertEqual(profiles.ALICCP_SCREEN.model_seed, cfg["model_seed"])
        self.assertEqual(profiles.ALICCP_SCREEN.env_seed, cfg["env_seed"])
        self.assertTrue(stage1["matches_historical"])


class _TinyDataset(Dataset):
    """极小两特征三标签数据集（与 census 数据集同构）。"""

    def __init__(self, n, seed=7):
        gen = torch.Generator().manual_seed(seed)
        self.features = {"a": torch.randint(0, 3, (n,), generator=gen),
                         "b": torch.randint(0, 2, (n,), generator=gen)}
        self.labels = [torch.randint(0, 2, (n,), generator=gen).float() for _ in range(3)]

    def __len__(self):
        return int(self.features["a"].shape[0])

    def __getitem__(self, i):
        return (self.labels[0][i], self.labels[1][i], self.labels[2][i],
                {"a": self.features["a"][i], "b": self.features["b"][i]})


def _tiny_profile():
    """stub 适配器：只测 run_screen 编排（不覆盖真实数据集适配器）。"""

    def prepare(source, smoke):
        train, val_test = _TinyDataset(64, 1), _TinyDataset(32, 2)
        val_idx, test_idx = np.arange(16), np.arange(16, 32)
        loaders = {"train": DataLoader(train, batch_size=16),
                   "val": DataLoader(Subset(val_test, val_idx.tolist()), batch_size=16),
                   "test": DataLoader(Subset(val_test, test_idx.tolist()), batch_size=16)}
        return {"loaders": loaders, "datasets": {"train": train}, "sets": {"train": train},
                "fp": {"fingerprint_sha256": "0" * 64}, "indices": (val_idx, test_idx)}

    def build_s1(*, base_hash, universal_hash, manager, model):
        return {"base_hash": base_hash, "universal_hash": universal_hash,
                "matches_historical": False, "epochs": len(manager.val_epoch_aucs)}

    def build_original_head(device):
        torch.manual_seed(1234)
        return NewTask(8, 4, [4, 2], 0.0, device)

    return profiles.ScreenProfile(
        name="tiny", expected_base_hash="", model_seed=20260929, env_seed=20260929,
        resolve_device=lambda: torch.device("cpu"),
        prepare=prepare, save_pre_artifacts=lambda out, ctx: None,
        build_config=lambda ctx, meta: {"stage1_epochs": 1, "stage2_epochs": 1},
        build_base=lambda device: MPTRec(num_tasks=2, feature_vocabulary={"a": 3, "b": 2},
                                         embedding_size=4, input_size=8,
                                         expert_dnn_hidden_units=(8, 4),
                                         tower_dnn_hidden_units=(4, 2), device=device),
        u_widths=lambda ctx, base: [4, 4], u_rep_dim=4,
        env_id_count=lambda ctx: len(ctx["sets"]["train"]),
        task_names=["T0", "T1"], lr=1e-2, batch_size=16, stage1_patience=2,
        uni_coe=0.9, env_coe=0.1, record_env_acc=False, cluster_epoch_offset=0,
        build_s1=build_s1, cache_split=runner.memory_cache_split,
        old_task_names=("t0", "t1"),
        mechanism=lambda cache, u, device, out, widths: {
            "u_std_mean": 1.0, "u_live_fraction": 0.9, "ug_squared_correlation": 0.0,
            "reconstruction_train_probe": {"reconstruction": 0.1, "zero": 0.5}},
        finalize_stage1=lambda out, s1, mech, old_auc: None,
        build_original_head=build_original_head, cpu_loss=True,
        stage2_lr=1e-2, stage2_batch_size=16, stage2_patience=2, rep_dim=4)


def run_tiny_screen(td):
    return runner.run_screen(_tiny_profile(), source=REPO, out=Path(td) / "out", smoke=True)


class TestRunScreenTiny(unittest.TestCase):
    def test_stub_profile_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            report = run_tiny_screen(td)
            expected_keys = {"arms", "u_minus", "classification", "preregistered_go", "mechanism",
                             "stage1_old_tasks", "stage1_matches_historical", "frozen_base_hash_after",
                             "frozen_u_hash_after", "frozen_gradients_none", "wall_seconds",
                             "prediction_hashes"}
            self.assertEqual(set(report), expected_keys)
            for name in ("config.json", "stage1.json", "stage1.pt", "random_u.pt", "old_tasks.npz",
                         "mechanism.json", "report.json"):
                self.assertTrue((out_dir / name).exists(), name)
            for arm in ("B", "U", "R", "G"):
                for suffix in ("_predictions.npz", "_metrics.json", "_head.pt"):
                    self.assertTrue((out_dir / f"{arm}{suffix}").exists(), arm + suffix)
            arms = report["arms"]
            self.assertEqual(arms["U"]["parameters"], arms["R"]["parameters"])
            self.assertEqual(arms["U"]["parameters"], arms["G"]["parameters"])
            self.assertIn("test_shuffled_u_auc", arms["U"])
            stage1 = json.loads((out_dir / "stage1.json").read_text(encoding="utf-8"))
            self.assertEqual(report["frozen_base_hash_after"], stage1["base_hash"])
            self.assertTrue(report["frozen_gradients_none"])
            self.assertEqual(set(report["stage1_old_tasks"]), {"val", "test"})
            self.assertEqual(set(report["stage1_old_tasks"]["val"]), {"t0", "t1"})
            self.assertIn(report["classification"], ("positive", "clear_decline", "no_clear_improvement"))
            self.assertEqual(set(report["prediction_hashes"]),
                             {p.name for p in out_dir.glob("*.npz")})

    def test_stub_profile_deterministic(self):
        """两次全流程（同种子）→ 哈希与四臂结果位级一致（钉住 seed→base→u 的 RNG 顺序）。"""
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            r1, r2 = run_tiny_screen(td1), run_tiny_screen(td2)
            for key in ("frozen_base_hash_after", "frozen_u_hash_after", "classification",
                        "preregistered_go", "u_minus", "stage1_matches_historical"):
                self.assertEqual(r1[key], r2[key], key)
            for arm in ("B", "U", "R", "G"):
                self.assertEqual({k: v for k, v in r1["arms"][arm].items() if k != "wall_seconds"},
                                 {k: v for k, v in r2["arms"][arm].items() if k != "wall_seconds"}, arm)


if __name__ == "__main__":
    unittest.main()
