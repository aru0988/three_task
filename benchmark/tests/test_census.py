import importlib.util
import tempfile, unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from benchmark.datasets import census
from multitaskrec.model import MPTRec

_HAVE_LEGACY = importlib.util.find_spec("census_benchmark") is not None

# ---- 黄金常量：2026-10-09 由旧实现（census_benchmark，n_train 取自已记录 stage1 meta）验证一致 ----
FULL = dict(n_test=99762, n_train=199523, n_val=49881, n_test_side=49881,
            fingerprint="096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c",
            val_sha="6f2c5868b5bf261a7a04fe82ad1e039598b38b71b49cd52f5665014a27f62722",
            test_sha="68e027924fb1127ef97720e02e32b5a3363e71aacd9538b11673f53fc3a8e5da")
SMALL = dict(n_test=1000, n_train=2000, fingerprint="98ad603bc22ed1c4ea6de4d89734fb9673b5aa52d9d16b1be4eb5a9b02492b02",
             val_head=[0, 1, 3, 4, 7], test_head=[2, 5, 6, 8, 10])


class ListDataset(Dataset):
    """最小 list 型 Dataset，复现 CensusIncomeDataset 的索引行为（无 .shape）。"""
    def __init__(self, n): self.data = [(i, {"x": i}) for i in range(n)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


class TestSplitAndSeeds(unittest.TestCase):
    def test_index_split_matches_dataset_split(self):
        ds = ListDataset(37)
        direct_val, direct_test = train_test_split(ds, test_size=0.5, random_state=census.SPLIT_SEED)
        perm_val, perm_test = train_test_split(np.arange(len(ds)), test_size=0.5, random_state=census.SPLIT_SEED)
        self.assertEqual([ds[int(i)] for i in perm_val], direct_val)      # 同一排列
        self.assertEqual([ds[int(i)] for i in perm_test], direct_test)
        val_idx, test_idx = census.make_split(len(ds), census.SPLIT_SEED)
        self.assertEqual(sorted(perm_val.tolist()), val_idx.tolist())     # 规范形式 = 排序后的索引
        self.assertEqual(sorted(perm_test.tolist()), test_idx.tolist())

    def test_split_disjoint_complete_and_fingerprint_deterministic(self):
        val_idx, test_idx = census.make_split(1000, census.SPLIT_SEED)
        stats = census.split_stats(val_idx, test_idx, n_train=2000)
        self.assertTrue(stats["disjoint"]); self.assertTrue(stats["union_complete"])
        self.assertEqual((stats["n_train"], stats["n_val"], stats["n_test"]), (2000, 500, 500))
        fp1 = census.split_fingerprint(split_seed=census.SPLIT_SEED, stats=stats, val_idx=val_idx, test_idx=test_idx)
        fp2 = census.split_fingerprint(split_seed=census.SPLIT_SEED, stats=stats, val_idx=val_idx, test_idx=test_idx)
        self.assertEqual(fp1, fp2)                                        # 同 split seed → 指纹恒定（A2）
        v2, t2 = census.make_split(1000, census.SPLIT_SEED + 1)
        fp3 = census.split_fingerprint(split_seed=census.SPLIT_SEED + 1,
                                       stats=census.split_stats(v2, t2, 2000), val_idx=v2, test_idx=t2)
        self.assertNotEqual(fp1["fingerprint_sha256"], fp3["fingerprint_sha256"])

    def test_protocol_constants_match_spec(self):
        self.assertEqual((census.SPLIT_SEED, census.MODEL_SEED, census.ENV_SEED), (20260929, 1685480945, 20260929))
        self.assertEqual((census.STAGE1_EPOCHS, census.STAGE2_EPOCHS, census.PATIENCE), (2, 5, 2))
        self.assertEqual((census.BATCH_SIZE, census.LR), (256, 1e-3))
        self.assertEqual((census.UNI_COE, census.ENV_COE), (0.9, 0.1))
        self.assertEqual((census.REG_EMBEDDING, census.REG_DNN), (0.006, 3e-5))
        self.assertEqual((census.INPUT_SIZE, census.EMBEDDING_SIZE), (123, 4))
        self.assertEqual((census.EXPERT_HIDDEN, census.TOWER_HIDDEN), ((256, 128), (64, 32)))
        self.assertEqual((census.AUC_FLOOR, census.VAL_TEST_GAP), (0.60, 0.03))


class TestSplitGoldens(unittest.TestCase):
    def test_full_scale_fingerprint_matches_recorded(self):
        """纯 numpy 复现已记录 stage1 的划分指纹（EXPECTED_SPLIT），不依赖数据文件。"""
        val_idx, test_idx = census.make_split(FULL["n_test"], census.SPLIT_SEED)
        stats = census.split_stats(val_idx, test_idx, n_train=FULL["n_train"])
        fp = census.split_fingerprint(split_seed=census.SPLIT_SEED, stats=stats,
                                      val_idx=val_idx, test_idx=test_idx)
        self.assertEqual((stats["n_val"], stats["n_test"]), (FULL["n_val"], FULL["n_test_side"]))
        self.assertEqual(fp["fingerprint_sha256"], FULL["fingerprint"])
        self.assertEqual(fp["val_sha256"], FULL["val_sha"])
        self.assertEqual(fp["test_sha256"], FULL["test_sha"])

    def test_small_scale_golden(self):
        val_idx, test_idx = census.make_split(SMALL["n_test"], census.SPLIT_SEED)
        fp = census.split_fingerprint(split_seed=census.SPLIT_SEED,
                                      stats=census.split_stats(val_idx, test_idx, n_train=SMALL["n_train"]),
                                      val_idx=val_idx, test_idx=test_idx)
        self.assertEqual(fp["fingerprint_sha256"], SMALL["fingerprint"])
        self.assertEqual(fp["val_indices_head"], SMALL["val_head"])
        self.assertEqual(fp["test_indices_head"], SMALL["test_head"])


class TestSplitArtifacts(unittest.TestCase):
    def test_write_load_verify_roundtrip(self):
        val_idx, test_idx = census.make_split(1000, census.SPLIT_SEED)
        stats = census.split_stats(val_idx, test_idx, 2000)
        fp = census.split_fingerprint(split_seed=census.SPLIT_SEED, stats=stats, val_idx=val_idx, test_idx=test_idx)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertIsNone(census.load_split_fingerprint(root, census.SPLIT_SEED))
            path = census.write_split_fingerprint(root, census.SPLIT_SEED, fp)
            self.assertTrue(path.exists())
            loaded = census.load_split_fingerprint(root, census.SPLIT_SEED)
            self.assertEqual(loaded, fp)
            self.assertTrue(census.verify_split_fingerprint(fp, loaded))
            v2, t2 = census.make_split(1000, census.SPLIT_SEED + 1)
            fp2 = census.split_fingerprint(split_seed=census.SPLIT_SEED + 1,
                                           stats=census.split_stats(v2, t2, 2000), val_idx=v2, test_idx=t2)
            self.assertFalse(census.verify_split_fingerprint(fp2, loaded))   # 基准不匹配（A2 拒绝）
            self.assertFalse(census.verify_split_fingerprint(fp, None))

    def test_save_split_indices_roundtrip(self):
        val_idx, test_idx = census.make_split(1000, census.SPLIT_SEED)
        with tempfile.TemporaryDirectory() as td:
            path = census.save_split_indices(Path(td), census.SPLIT_SEED, val_idx, test_idx)
            with np.load(path) as data:                                    # Windows 下需及时关闭句柄
                self.assertTrue(np.array_equal(data["val"], val_idx))
                self.assertTrue(np.array_equal(data["test"], test_idx))


class TestBuildMptrec(unittest.TestCase):
    def test_builds_without_education_feature(self):
        from config import CensusIncome_Vocabulary_Size
        model = census.build_mptrec(torch.device("cpu"))
        self.assertIsInstance(model, MPTRec)
        # spec 2.2.1：vocab = CensusIncome_Vocabulary_Size 去掉 education，其余同序保留
        expected = [name for name in CensusIncome_Vocabulary_Size if name != "education"]
        self.assertEqual(model.embedding_network.feature_names, expected)


def _ok_kwargs(**over):
    """全部门禁通过的基线参数；单个测试按需覆盖（移植自 census_benchmark/tests/test_metrics.py）。"""
    base = dict(backbone_sha_equal=True, grads_all_none=True, split_ok=True, split_stats_ok=True,
                env_ids_ok=True, auc_val_income=0.90, auc_val_marital=0.90, auc_test_education=0.90,
                auc_val_education_best=0.90, gate_mean=[0.5, 0.5], env_shares=[0.40, 0.60])
    base.update(over)
    return base


class TestJudge(unittest.TestCase):
    def test_a_class_hard_gates(self):
        report = census.judge(**_ok_kwargs())
        self.assertTrue(report["overall_pass"]); self.assertEqual(report["failures"], [])
        for key in ("A1", "A2", "A4", "A5"):
            self.assertTrue(report[key]["pass"])
        self.assertFalse(census.judge(**_ok_kwargs(backbone_sha_equal=False))["A1"]["pass"])
        self.assertFalse(census.judge(**_ok_kwargs(grads_all_none=False))["A1"]["pass"])
        self.assertFalse(census.judge(**_ok_kwargs(split_ok=False))["overall_pass"])
        self.assertFalse(census.judge(**_ok_kwargs(env_ids_ok=False))["A5"]["pass"])

    def test_b1_b2_thresholds(self):
        self.assertTrue(census.judge(**_ok_kwargs(auc_val_income=0.60))["B1"]["pass"])     # 边界含等号
        self.assertFalse(census.judge(**_ok_kwargs(auc_val_income=0.599))["B1"]["pass"])
        self.assertFalse(census.judge(**_ok_kwargs(auc_test_education=0.59,
                                                   auc_val_education_best=0.59))["B1"]["pass"])
        self.assertTrue(census.judge(**_ok_kwargs(auc_val_education_best=0.90,
                                                  auc_test_education=0.88))["B2"]["pass"])   # 差 = 0.02
        self.assertFalse(census.judge(**_ok_kwargs(auc_val_education_best=0.90,
                                                   auc_test_education=0.86))["B2"]["pass"])  # 差 = 0.04
        # 阈值 0.03 在二进制浮点下不可精确表示（0.9 - 0.87 = 0.030000000000000027），故用 0.02 / 0.04 覆盖两侧

    def test_b3_b4_degenerate(self):
        self.assertTrue(census.judge(**_ok_kwargs(gate_mean=[0.05, 0.95]))["B3"]["pass"])
        self.assertFalse(census.judge(**_ok_kwargs(gate_mean=[0.02, 0.98]))["B3"]["pass"])    # 坍缩到单一分支
        self.assertTrue(census.judge(**_ok_kwargs(env_shares=[0.05, 0.95]))["B4"]["pass"])
        self.assertFalse(census.judge(**_ok_kwargs(env_shares=[0.049, 0.951]))["B4"]["pass"])  # 聚类退化


@unittest.skipUnless(_HAVE_LEGACY, "旧实现包已删除（迁移完成）")
class TestLegacyJudgeEquivalence(unittest.TestCase):
    def test_judge_matches_legacy(self):
        from census_benchmark import metrics as old
        cases = ({}, {"backbone_sha_equal": False}, {"grads_all_none": False}, {"split_ok": False},
                 {"split_stats_ok": False}, {"env_ids_ok": False}, {"auc_val_income": 0.599},
                 {"auc_val_education_best": 0.90, "auc_test_education": 0.86},
                 {"gate_mean": [0.02, 0.98]}, {"env_shares": [0.049, 0.951]})
        for over in cases:
            kwargs = _ok_kwargs(**over)
            self.assertEqual(census.judge(**kwargs), old.judge(**kwargs), over)


if __name__ == "__main__":
    unittest.main()
