"""RecordingMPTRecTrainManager 等价性测试。

黄金常量于 2026-10-09 由旧实现在同一 tiny CPU 夹具上捕获（捕获脚本同时确认新旧逐值相等后冻结）：
census 侧为 run_census_benchmark.Stage1HookTrainManager（record_env_acc=True, offset=0），
aliccp 侧为 aliccp_benchmark.bench.RecordingMPTRecTrainManager（record_env_acc=False, offset=1）。
旧实现直接对照类在旧包删除后自动跳过（find_spec 判断）。
"""
import contextlib
import importlib.util
import io
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from benchmark import protocol as bp
from benchmark.recording import RecordingMPTRecTrainManager
from multitaskrec.model import MPTRec

SEED = 20261009
VOCAB = {"a": 3, "b": 2}
ENV_SEED = 123
GOLDEN = {
    "val_aucs": [[0.2421875, 0.65625], [0.6953125, 0.65625], [0.7734375, 0.65625], [0.765625, 0.65625]],
    "env_accs": [0.34375, 0.625, 0.625, 0.625],
    "census_epoch_records": [
        {"epoch": 1, "auc_val_income": 0.2421875, "auc_val_marital": 0.65625, "env_acc": 0.34375},
        {"epoch": 2, "auc_val_income": 0.6953125, "auc_val_marital": 0.65625, "env_acc": 0.625},
        {"epoch": 3, "auc_val_income": 0.7734375, "auc_val_marital": 0.65625, "env_acc": 0.625},
        {"epoch": 4, "auc_val_income": 0.765625, "auc_val_marital": 0.65625, "env_acc": 0.625},
    ],
    "census_cluster_records": [
        {"epoch": 1, "diff_num": 13, "env_counts": [11, 21]},
        {"epoch": 3, "diff_num": 0, "env_counts": [11, 21]},
    ],
    "aliccp_cluster_events": [
        {"epoch": 2, "diff_num": 13, "env_0": 11, "env_1": 21},
        {"epoch": 4, "diff_num": 0, "env_0": 11, "env_1": 21},
    ],
    "best_epoch": 3,
    "env_ids_sha": "a4c91cbeabb1c25a719cbfc15db092ea31d32b47a73328caed50820885879f4a",
}
_HAVE_LEGACY = (importlib.util.find_spec("census_benchmark") is not None
                and importlib.util.find_spec("aliccp_benchmark") is not None)


class TinyDataset(Dataset):
    def __init__(self, n, data_seed):
        g = torch.Generator().manual_seed(data_seed)
        self.f = torch.randint(0, 3, (n, 2), generator=g)
        self.f[:, 1] = self.f[:, 1] % 2
        self.y = torch.randint(0, 2, (n, 2), generator=g)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        return (self.y[i, 0], self.y[i, 1], self.y[i, 0], {"a": self.f[i, 0], "b": self.f[i, 1]})


def make_model():
    return MPTRec(num_tasks=2, feature_vocabulary=dict(VOCAB), embedding_size=4, input_size=8,
                  expert_dnn_hidden_units=(8, 4), tower_dnn_hidden_units=(4, 2),
                  device=torch.device("cpu"))


def make_loaders():
    return (DataLoader(TinyDataset(32, 7), batch_size=8), DataLoader(TinyDataset(16, 8), batch_size=8))


def train_merged(cls=RecordingMPTRecTrainManager, **kwargs):
    """与黄金捕获脚本完全同序：seed → 模型 → 数据 → env_ids → 训练。"""
    torch.manual_seed(SEED)
    train_loader, val_loader = make_loaders()
    mgr = cls(model=make_model(), train_loader=train_loader, val_loader=val_loader,
              env_ids=bp.make_env_ids(32, ENV_SEED), task_name=["T0", "T1"], lr=1e-3, batch_size=8,
              uni_coe=0.9, env_coe=0.1, epochs=4, patience=10, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mgr.train_two_task()
    return mgr, train_loader, val_loader


def census_epoch_records(mgr):
    """census 侧记录 schema（旧 Stage1HookTrainManager.epoch_records 的等价重建）。"""
    return [{"epoch": i + 1, "auc_val_income": a[0], "auc_val_marital": a[1], "env_acc": mgr.env_accs[i]}
            for i, a in enumerate(mgr.val_epoch_aucs)]


def aliccp_cluster_events(mgr):
    """aliccp 侧记录 schema（旧 RecordingMPTRecTrainManager.cluster_events 的等价重建）。"""
    return [{"epoch": r["epoch"], "diff_num": r["diff_num"],
             "env_0": r["env_counts"][0], "env_1": r["env_counts"][1]} for r in mgr.cluster_records]


class TestGoldenCensus(unittest.TestCase):
    def test_records_env_acc_and_env_ids(self):
        mgr, _, _ = train_merged(record_env_acc=True, cluster_epoch_offset=0)
        self.assertEqual(mgr.val_epoch_aucs, GOLDEN["val_aucs"])
        self.assertEqual(mgr.env_accs, GOLDEN["env_accs"])
        self.assertEqual(census_epoch_records(mgr), GOLDEN["census_epoch_records"])
        self.assertEqual(mgr.cluster_records, GOLDEN["census_cluster_records"])
        self.assertEqual(bp.sha256_tensor(mgr.env_ids.clone()), GOLDEN["env_ids_sha"])
        self.assertEqual(mgr.best_epoch(), GOLDEN["best_epoch"])  # census 选点（sum 最大）同值

    def test_test_loader_eval_not_recorded(self):
        mgr, train_loader, _ = train_merged(record_env_acc=True, cluster_epoch_offset=0)
        before = list(mgr.val_epoch_aucs)
        mgr.evaluation_two_task(train_loader)  # 非 val_loader → 不追加
        self.assertEqual(mgr.val_epoch_aucs, before)


class TestGoldenAliccp(unittest.TestCase):
    def test_aucs_events_best_epoch(self):
        mgr, _, _ = train_merged(record_env_acc=False, cluster_epoch_offset=1)
        self.assertEqual(mgr.val_epoch_aucs, GOLDEN["val_aucs"])
        self.assertEqual(mgr.env_accs, [])                        # 不开 per-epoch env_acc
        self.assertEqual(aliccp_cluster_events(mgr), GOLDEN["aliccp_cluster_events"])
        self.assertEqual(mgr.best_epoch(), GOLDEN["best_epoch"])
        self.assertEqual(bp.sha256_tensor(mgr.env_ids.clone()), GOLDEN["env_ids_sha"])


class TestBestEpochRule(unittest.TestCase):
    def test_first_maximum_wins_on_ties(self):
        train_loader, val_loader = make_loaders()
        mgr = RecordingMPTRecTrainManager(model=make_model(), train_loader=train_loader, val_loader=val_loader,
                                          env_ids=bp.make_env_ids(32, ENV_SEED), task_name=["T0", "T1"],
                                          lr=1e-3, batch_size=8, uni_coe=0.9, env_coe=0.1, epochs=4, patience=10)
        mgr.val_epoch_aucs = [[0.5, 0.5], [0.7, 0.6], [0.6, 0.7]]   # sum 1.0 / 1.3 / 1.3 → 首次最大
        self.assertEqual(mgr.best_epoch(), 2)


@unittest.skipUnless(_HAVE_LEGACY, "旧实现包已删除（迁移完成）")
class TestLegacyManagerEquivalence(unittest.TestCase):
    def test_census_manager_matches_stage1_hook(self):
        from run_census_benchmark import Stage1HookTrainManager
        old, _, _ = train_merged(cls=Stage1HookTrainManager)
        new, _, _ = train_merged(record_env_acc=True, cluster_epoch_offset=0)
        self.assertEqual(old.epoch_records, census_epoch_records(new))
        self.assertEqual(old.cluster_records, new.cluster_records)
        self.assertEqual(bp.sha256_tensor(old.env_ids.clone()), bp.sha256_tensor(new.env_ids.clone()))

    def test_aliccp_manager_matches_recording(self):
        from aliccp_benchmark.bench import RecordingMPTRecTrainManager as OldMgr
        old, _, _ = train_merged(cls=OldMgr)
        new, _, _ = train_merged(record_env_acc=False, cluster_epoch_offset=1)
        self.assertEqual(old.val_epoch_aucs, new.val_epoch_aucs)
        self.assertEqual(old.cluster_events, aliccp_cluster_events(new))
        self.assertEqual(old.best_epoch(), new.best_epoch())
        self.assertEqual(bp.sha256_tensor(old.env_ids.clone()), bp.sha256_tensor(new.env_ids.clone()))


if __name__ == "__main__":
    unittest.main()
