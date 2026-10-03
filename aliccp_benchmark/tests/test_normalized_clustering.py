"""不变量测试：归一化聚类处理臂（TDD：先于实现编写；见
docs/superpowers/specs/2026-10-03-aliccp-stage1-normalized-clustering-design.md §7）。

CPU 极小夹具，不构成任何性能证据，只验证语义与接线。
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from aliccp_benchmark import bench, protocol

# ---- 预注册钉死值（§3；本文件在实现之前写死，不得随实现修改）----
BASE_COMMIT = "8133d32"
BASELINE_CFG_HASH = "3a30e2c0b1e8a2b4e9fecaa4d76893b775f6ee9e597922dce7303ac3b77b08c4"
TREATMENT_CFG_HASH = "ad3b353f6d436ac95c26703e504a318e3441933a4e00e3f3d6c5e22a05bfe96e"
FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
PREDICTED_STAGE1_ID = "s1-5c060b9c-m1688723512-e3-ad3b353f"
PREDICTED_ENV_IDS_SHA = "4b983fc9f485d7cba853b8d5f0846b292e85344fc5b6f3725845dbba5ea0d52b"
PREDICTED_EVENT = {"diff_num": 999966, "env_0": 959244, "env_1": 1040756}
DOC_PATH = Path("docs/superpowers/specs/2026-10-03-aliccp-stage1-normalized-clustering-seed-replication-design.md")

HEADER = "click,purchase,X,121,122,301"
TINY_VOCAB = {"121": 5, "122": 4}
TRAIN_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2], [0, 0, 9, 2, 2, 1], [1, 1, 9, 0, 1, 3],
    [0, 0, 8, 3, 0, 2], [1, 0, 8, 1, 1, 1], [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2],
]
VAL_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2],
]
TEST_ROWS = [
    [0, 0, 9, 2, 2, 1], [1, 0, 9, 0, 1, 3], [0, 0, 8, 3, 0, 2], [1, 1, 8, 1, 1, 1],
    [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2],
]


def _write_rows(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


def make_tiny_root(td: str):
    root = Path(td) / "artifacts"
    data_dir = Path(td) / "data"
    data_dir.mkdir()
    _write_rows(data_dir / "train.csv", TRAIN_ROWS)
    _write_rows(data_dir / "val.csv", VAL_ROWS)
    _write_rows(data_dir / "test.csv", TEST_ROWS)
    data_files = {s: str(data_dir / f"{s}.csv") for s in ("train", "val", "test")}
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return root, data_files, budgets


def _run_tiny_stage1(root, data_files, budgets, clustering_arm):
    return bench.run_stage1(
        root=root, data_files=data_files, budgets=budgets, prefix_tag="tiny",
        model_seed=123, env_seed=456, epochs=2, patience=2,
        device=torch.device("cpu"), vocab=TINY_VOCAB,
        expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
        log=lambda *a, **k: None, clustering_arm=clustering_arm,
    )


# ---------------------------------------------------------------- I1/I2：rank01 语义
class TestPerTaskRank01(unittest.TestCase):
    def setUp(self):
        from aliccp_benchmark import normalized_clustering
        self.mod = normalized_clustering

    def test_hand_computed_average_ranks(self):
        t = torch.tensor([[1.0], [1.0], [2.0], [2.0], [2.0], [3.0]])
        q = self.mod.per_task_rank01(t).flatten().tolist()
        self.assertEqual(q, [0.1, 0.1, 0.6, 0.6, 0.6, 1.0])

    def test_bounds_dtype_and_two_columns(self):
        l = torch.rand(101, 2)
        q = self.mod.per_task_rank01(l)
        self.assertEqual(q.dtype, torch.float64)
        self.assertEqual(tuple(q.shape), (101, 2))
        self.assertTrue(bool((q >= 0).all() and (q <= 1).all()))

    def test_monotone_invariance(self):
        l = torch.rand(500, 2).double() * 7 - 3
        q1 = self.mod.per_task_rank01(l)
        q2 = self.mod.per_task_rank01(torch.exp(l))          # 严格单调变换
        q3 = self.mod.per_task_rank01(1.0 / (1.0 + torch.exp(-l)))  # sigmoid
        self.assertTrue(torch.equal(q1, q2))
        self.assertTrue(torch.equal(q1, q3))

    def test_affine_invariance(self):
        l = torch.rand(500, 2)
        self.assertTrue(torch.equal(self.mod.per_task_rank01(l), self.mod.per_task_rank01(5.0 * l + 2.0)))

    def test_zero_variance_and_single_row(self):
        z = torch.zeros(8, 2)
        q = self.mod.per_task_rank01(z)
        self.assertTrue(torch.equal(q, torch.full((8, 2), 0.5, dtype=torch.float64)))
        q1 = self.mod.per_task_rank01(torch.tensor([[5.0, 7.0]]))
        self.assertTrue(torch.equal(q1, torch.zeros(1, 2, dtype=torch.float64)))
        # 单列全并列、另一列有区分 → 常数列恒 0.5
        mixed = torch.stack([torch.ones(6), torch.arange(6.0)], dim=1)
        qm = self.mod.per_task_rank01(mixed)
        self.assertTrue(torch.equal(qm[:, 0], torch.full((6,), 0.5, dtype=torch.float64)))

    def test_matches_audit_reference_bitwise(self):
        from aliccp_benchmark.audit_cluster_losses import ref_rank01
        torch.manual_seed(0)
        cases = [
            torch.rand(300, 2),
            torch.rand(300, 2) * 1e-3,                       # 小尺度
            torch.cat([torch.rand(150, 2), torch.rand(150, 2).round(decimals=2)], dim=0),  # 含并列
            torch.stack([torch.full((50,), 2.0), torch.rand(50)], dim=1),  # 常数列
        ]
        for case in cases:
            self.assertTrue(torch.equal(self.mod.per_task_rank01(case), ref_rank01(case)))

    def test_chosen_rule_reproduces_preregistered_predictions(self):
        """§3 预测的机制部分：在钉死的合成损失上，实现与审计参考给出同一 env_ids sha 语义。"""
        from aliccp_benchmark.audit_cluster_losses import ref_rank01
        torch.manual_seed(7)
        l = torch.rand(1000, 2)
        a1 = torch.argmin(self.mod.per_task_rank01(l), dim=1)
        a2 = torch.argmin(ref_rank01(l), dim=1)
        self.assertTrue(torch.equal(a1, a2))
        self.assertEqual(protocol.sha256_tensor(a1), protocol.sha256_tensor(a2))


# ---------------------------------------------------------------- I3：cluster 函数
class _StubModel(torch.nn.Module):
    """预测两端极值（0/1），用于检查 loss ops 链的有限性闭合。"""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def cluster_predict(self, x):
        return [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])]


class _InfBCE:
    """模拟 BCE 返回非有限损失（真实 ops 链中不可达；防御性守卫的定向测试）。"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pred, target):
        return torch.full_like(pred, float("inf"))


class TestRankNormalizedCluster2(unittest.TestCase):
    def _tiny_model_loader(self):
        from multitaskrec.model import MPTRec
        from torch.utils.data import DataLoader
        from multitaskrec.dataset import AliCCPDataset
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "t.csv"
            _write_rows(data, TRAIN_ROWS)
            ds = AliCCPDataset(str(data), len(TRAIN_ROWS))
            return ds, DataLoader(ds, batch_size=5, shuffle=False, num_workers=0)

    def test_matches_reference_pipeline_and_is_deterministic(self):
        from aliccp_benchmark import normalized_clustering as nc
        from aliccp_benchmark.audit_cluster_losses import loss_vectors, ref_rank01
        from multitaskrec.model import MPTRec
        ds, loader = self._tiny_model_loader()
        torch.manual_seed(0)
        model = MPTRec(
            num_tasks=2, feature_vocabulary=dict(TINY_VOCAB), embedding_size=2, input_size=4,
            expert_dnn_hidden_units=[4], tower_dnn_hidden_units=[4], dropout=[0.0, 0.0],
            reg_embedding=1e-4, reg_dnn=7e-6, device=torch.device("cpu"),
        )
        device = torch.device("cpu")
        assign, diag = nc.rank_normalized_cluster_2(model, loader, device)
        losses, _ = loss_vectors(model, loader, device)
        expected = torch.argmin(ref_rank01(losses), dim=1)
        self.assertEqual(assign.dtype, torch.int64)
        self.assertEqual(len(assign), len(ds))
        self.assertTrue(torch.equal(assign, expected))
        self.assertEqual(int(diag["env_0"]) + int(diag["env_1"]), len(ds))
        self.assertTrue(all(
            isinstance(v, float) and torch.isfinite(torch.tensor(v))
            for v in diag["raw_loss_quantiles"]["ctr_task0"].values()
        ))
        assign2, _ = nc.rank_normalized_cluster_2(model, loader, device)
        self.assertTrue(torch.equal(assign, assign2))  # 同状态两次调用逐位一致

    def test_extreme_valid_predictions_stay_finite(self):
        """pred ∈ {0,1} 的极端输入：BCE 有 clamp（−100），损失仍有限 → 不触发守卫。"""
        from aliccp_benchmark import normalized_clustering as nc
        model = _StubModel()
        loader = [(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), None,
                   {"121": torch.zeros(2, dtype=torch.long)})]
        assign, diag = nc.rank_normalized_cluster_2(model, loader, torch.device("cpu"))
        self.assertEqual(len(assign), 2)
        self.assertTrue(all(
            torch.isfinite(torch.tensor(v))
            for task in ("ctr_task0", "cvr_task1")
            for v in diag["raw_loss_quantiles"][task].values()
        ))

    def test_non_finite_losses_raise(self):
        """防御性守卫：BCE 若返回非有限损失，必须在 argmin 前抛 AssertionError（真实 ops 链不可达）。"""
        from aliccp_benchmark import normalized_clustering as nc
        model = _StubModel()
        loader = [(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), None,
                   {"121": torch.zeros(2, dtype=torch.long)})]
        with mock.patch.object(nc.nn, "BCELoss", _InfBCE):
            with self.assertRaises(AssertionError):
                nc.rank_normalized_cluster_2(model, loader, torch.device("cpu"))


# ---------------------------------------------------------------- I4/I5/I6：端到端与管理器
class TestArmWiring(unittest.TestCase):
    def test_default_run_unchanged_schema_and_cfg(self):
        from aliccp_benchmark import normalized_clustering  # noqa: F401  (确保误引用会被发现)
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = make_tiny_root(td)
            meta = _run_tiny_stage1(root, data_files, budgets, clustering_arm=None)
            self.assertNotIn("clustering", meta["config"])
            self.assertNotIn("clustering", meta)
            self.assertNotIn("cluster_diagnostics", meta)
            self.assertNotIn("env_bal_acc", meta)
            self.assertEqual(len(meta["cluster_events"]), 1)
            self.assertEqual(set(meta["cluster_events"][0]), {"epoch", "diff_num", "env_0", "env_1"})

    def test_arm_run_records_diagnostics_and_config_marker(self):
        from aliccp_benchmark.normalized_clustering import RANK_NORMALIZED_ARM
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = make_tiny_root(td)
            meta = _run_tiny_stage1(root, data_files, budgets, clustering_arm=RANK_NORMALIZED_ARM)
            self.assertEqual(meta["clustering"], "rank_normalized")
            self.assertEqual(meta["config"]["clustering"], "rank_normalized")
            self.assertEqual(len(meta["cluster_events"]), 1)
            event = meta["cluster_events"][0]
            self.assertEqual(set(event), {"epoch", "diff_num", "env_0", "env_1"})
            self.assertEqual(event["env_0"] + event["env_1"], budgets["train"])
            self.assertEqual(len(meta["cluster_diagnostics"]), 1)
            self.assertEqual(meta["cluster_diagnostics"][0]["epoch"], event["epoch"])
            self.assertTrue(0.0 <= meta["env_bal_acc"] <= 1.0)
            self.assertTrue(torch.isfinite(torch.tensor(meta["env_bal_acc"])))
            # 同配置下 arm 的 stage1_id 与默认路径不同（cfg 含 clustering 键）
            meta_default = _run_tiny_stage1(root, data_files, budgets, clustering_arm=None)
            self.assertNotEqual(meta["stage1_id"], meta_default["stage1_id"])

    def test_arm_manager_only_overrides_cluster2(self):
        from aliccp_benchmark.normalized_clustering import RankNormalizedClusteringMPTRecTrainManager
        self.assertTrue(issubclass(RankNormalizedClusteringMPTRecTrainManager, bench.RecordingMPTRecTrainManager))
        self.assertIn("cluster_2", RankNormalizedClusteringMPTRecTrainManager.__dict__)
        overridden = set(RankNormalizedClusteringMPTRecTrainManager.__dict__) - {"cluster_2", "__doc__", "__module__", "__qualname__", "__init__"}
        self.assertEqual(overridden, set())

    def test_preregistered_config_hashes_and_stage1_id(self):
        cfg = bench._stage1_cfg(
            prefix_tag=protocol.PREFIX_TAG,
            budgets={"train": protocol.TRAIN_BUDGET, "val": protocol.VAL_BUDGET, "test": protocol.TEST_BUDGET},
            model_seed=protocol.MODEL_SEED, env_seed=protocol.ENV_SEED,
            epochs=protocol.STAGE1_EPOCHS, patience=protocol.STAGE1_PATIENCE,
            batch_size=protocol.BATCH_SIZE, lr=protocol.LR, uni_coe=protocol.UNI_COE, env_coe=protocol.ENV_COE,
            reg_embedding=protocol.REG_EMBEDDING, reg_dnn=protocol.REG_DNN,
            embedding_size=protocol.EMBEDDING_SIZE, input_size=protocol.INPUT_SIZE,
            expert_hidden=protocol.EXPERT_HIDDEN, tower_hidden=protocol.TOWER_HIDDEN,
            dropout=protocol.DROPOUT, vocab=protocol.build_vocab(),
        )
        self.assertEqual(protocol.config_hash(cfg), BASELINE_CFG_HASH)          # 默认关：哈希不变
        arm_cfg = {**cfg, "clustering": "rank_normalized"}
        self.assertEqual(set(arm_cfg), set(cfg) | {"clustering"})                # 仅多一个键
        self.assertEqual(protocol.config_hash(arm_cfg), TREATMENT_CFG_HASH)
        self.assertEqual(
            protocol.make_stage1_id(FINGERPRINT_SHA, protocol.MODEL_SEED, protocol.STAGE1_EPOCHS, TREATMENT_CFG_HASH),
            PREDICTED_STAGE1_ID,
        )


# ---------------------------------------------------------------- I7：CLI
class TestCli(unittest.TestCase):
    def test_defaults_and_tag_norm(self):
        from run_aliccp_benchmark import build_parser
        parser = build_parser()
        args = parser.parse_args(["stage1"])
        self.assertEqual(args.clustering, "raw")
        args = parser.parse_args(["stage1", "--clustering", "rank_normalized", "--tag", "norm"])
        self.assertEqual(args.clustering, "rank_normalized")
        self.assertEqual(args.tag, "norm")
        args2 = parser.parse_args(["stage2", "--stage1-id", "s1-x", "--tag", "norm"])
        self.assertEqual(args2.tag, "norm")
        with self.assertRaises(SystemExit):
            parser.parse_args(["stage1", "--clustering", "bogus"])

    def test_main_threads_arm_and_none(self):
        from run_aliccp_benchmark import main
        fake_meta = {"stage1_id": "s1-x", "best_epoch": 1, "test_auc_ctr": 0.5, "test_auc_cvr": 0.5,
                     "env_acc": 0.5, "cluster_events": [], "wall_seconds": 1.0}
        for cli_args, expect in (
            (["stage1", "--clustering", "rank_normalized"], "rank_normalized"),
            (["stage1"], None),
        ):
            captured = {}
            with tempfile.TemporaryDirectory() as td, mock.patch.object(
                bench, "run_stage1", lambda **kw: (captured.update(kw), fake_meta)[1]
            ):
                rc = main(cli_args + ["--root", td])
            self.assertEqual(rc, 0)
            arm = captured.get("clustering_arm")
            if expect is None:
                self.assertIsNone(arm)
            else:
                self.assertIsNotNone(arm)
                self.assertEqual(arm.name, expect)


# ---------------------------------------------------------------- I8：静态守卫
class TestStaticGuards(unittest.TestCase):
    def _git_diff_names(self, *paths):
        out = subprocess.check_output(
            ["git", "diff", "--name-only", BASE_COMMIT, "--", *paths], text=True
        )
        return {line.strip() for line in out.splitlines() if line.strip()}

    def test_forbidden_paths_zero_diff(self):
        self.assertEqual(
            self._git_diff_names("multitaskrec", "config.py", "AliCCP_MPTRec.py", "AliCCP_NewTask.py", "baseline"),
            set(),
        )
        self.assertEqual(self._git_diff_names("aliccp_benchmark/protocol.py"), set())

    def test_tracked_changes_subset_of_whitelist(self):
        whitelist = {
            ".gitignore",
            "aliccp_benchmark/audit_cluster_losses.py",
            "aliccp_benchmark/bench.py",
            "aliccp_benchmark/metrics.py",
            "aliccp_benchmark/normalized_clustering.py",
            "aliccp_benchmark/tests/test_normalized_clustering.py",
            "run_aliccp_benchmark.py",
            "docs/superpowers/specs/2026-10-03-aliccp-stage1-normalized-clustering-design.md",
            "artifacts/aliccp_bench/SUMMARY.md",
            "docs/superpowers/specs/2026-10-03-aliccp-stage1-normalized-clustering-seed-replication-design.md",
            "aliccp_benchmark/tests/test_normalized_clustering_replication.py",
        }
        changed = self._git_diff_names(".")
        self.assertTrue(changed <= whitelist, msg=f"白名单外改动: {sorted(changed - whitelist)}")


# ---------------------------------------------------------------- I9：预注册文档 token
class TestPreregDocTokens(unittest.TestCase):
    def test_doc_pins_predictions_and_gates(self):
        text = DOC_PATH.read_text(encoding="utf-8")
        for token in (
            PREDICTED_STAGE1_ID, PREDICTED_ENV_IDS_SHA, "999966", "959244", "1040756",
            BASELINE_CFG_HASH, TREATMENT_CFG_HASH, "0.005", "0.01",
            "0.5481837665250074", "0.5988392178311113", "0.5781533414372665",
            "REPAIRED", "NOT_REPAIRED", "NO_MATERIAL_DEGRADATION",
        ):
            self.assertIn(token, text, msg=f"预注册文档缺少 token: {token}")


if __name__ == "__main__":
    unittest.main()
