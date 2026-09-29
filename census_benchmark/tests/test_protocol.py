import tempfile, unittest
from datetime import datetime
from pathlib import Path

import numpy as np, torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from census_benchmark import protocol
from multitaskrec.model import MPTRec

# EmbeddingNetwork 输出宽度 = 特征个数 × embedding_size（vocab 大小只决定 embedding 表行数）；
# 该夹具无 dense 特征 → input_size = 2 × 4 = 8（计划原文写的 (3+2)*4=20 会令首个 Linear 形状不匹配）。
TINY_VOCAB, TINY_INPUT_SIZE = {"a": 3, "b": 2}, 8


class ListDataset(Dataset):
    """最小 list 型 Dataset，复现 CensusIncomeDataset 的索引行为（无 .shape）。"""
    def __init__(self, n): self.data = [(i, {"x": i}) for i in range(n)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


class TestSplitAndSeeds(unittest.TestCase):
    def test_index_split_matches_dataset_split(self):
        ds = ListDataset(37)
        direct_val, direct_test = train_test_split(ds, test_size=0.5, random_state=protocol.SPLIT_SEED)
        perm_val, perm_test = train_test_split(np.arange(len(ds)), test_size=0.5, random_state=protocol.SPLIT_SEED)
        self.assertEqual([ds[int(i)] for i in perm_val], direct_val)      # 同一排列
        self.assertEqual([ds[int(i)] for i in perm_test], direct_test)
        val_idx, test_idx = protocol.make_split(len(ds), protocol.SPLIT_SEED)
        self.assertEqual(sorted(perm_val.tolist()), val_idx.tolist())     # 规范形式 = 排序后的索引
        self.assertEqual(sorted(perm_test.tolist()), test_idx.tolist())

    def test_split_disjoint_complete_and_fingerprint_deterministic(self):
        val_idx, test_idx = protocol.make_split(1000, protocol.SPLIT_SEED)
        stats = protocol.split_stats(val_idx, test_idx, n_train=2000)
        self.assertTrue(stats["disjoint"]); self.assertTrue(stats["union_complete"])
        self.assertEqual((stats["n_train"], stats["n_val"], stats["n_test"]), (2000, 500, 500))
        fp1 = protocol.split_fingerprint(split_seed=protocol.SPLIT_SEED, stats=stats, val_idx=val_idx, test_idx=test_idx)
        fp2 = protocol.split_fingerprint(split_seed=protocol.SPLIT_SEED, stats=stats, val_idx=val_idx, test_idx=test_idx)
        self.assertEqual(fp1, fp2)                                        # 同 split seed → 指纹恒定（A2）
        v2, t2 = protocol.make_split(1000, protocol.SPLIT_SEED + 1)
        fp3 = protocol.split_fingerprint(split_seed=protocol.SPLIT_SEED + 1,
                                         stats=protocol.split_stats(v2, t2, 2000), val_idx=v2, test_idx=t2)
        self.assertNotEqual(fp1["fingerprint_sha256"], fp3["fingerprint_sha256"])

    def test_protocol_constants_match_spec(self):
        self.assertEqual((protocol.SPLIT_SEED, protocol.MODEL_SEED, protocol.ENV_SEED), (20260929, 1685480945, 20260929))
        self.assertEqual((protocol.STAGE1_EPOCHS, protocol.STAGE2_EPOCHS, protocol.PATIENCE), (2, 5, 2))
        self.assertEqual((protocol.BATCH_SIZE, protocol.LR), (256, 1e-3))
        self.assertEqual((protocol.UNI_COE, protocol.ENV_COE), (0.9, 0.1))
        self.assertEqual((protocol.REG_EMBEDDING, protocol.REG_DNN), (0.006, 3e-5))
        self.assertEqual((protocol.INPUT_SIZE, protocol.EMBEDDING_SIZE), (123, 4))
        self.assertEqual((protocol.EXPERT_HIDDEN, protocol.TOWER_HIDDEN), ((256, 128), (64, 32)))

    def test_env_ids_independent_of_global_rng(self):
        torch.manual_seed(0); x1 = torch.rand(3)
        torch.manual_seed(0); e1 = protocol.make_env_ids(1000, protocol.ENV_SEED); x2 = torch.rand(3)
        e2 = protocol.make_env_ids(1000, protocol.ENV_SEED)
        self.assertTrue(torch.equal(x1, x2))                              # 全局 RNG 未被消耗（spec 5.3.2）
        self.assertTrue(torch.equal(e1, e2))                              # 同 env seed 可复现
        self.assertEqual(set(e1.tolist()), {0, 1})
        self.assertNotEqual(protocol.sha256_tensor(e1),
                            protocol.sha256_tensor(protocol.make_env_ids(1000, protocol.ENV_SEED + 1)))


def tiny_mptrec():
    return MPTRec(num_tasks=2, feature_vocabulary=dict(TINY_VOCAB), embedding_size=4,
                  input_size=TINY_INPUT_SIZE, expert_dnn_hidden_units=(8, 4),
                  tower_dnn_hidden_units=(4, 2), device=torch.device("cpu"))


class TestStage1ArtifactsAndFreeze(unittest.TestCase):
    def test_freeze_backbone_and_backbone_sha256(self):
        model = tiny_mptrec()
        before = protocol.backbone_sha256(model)
        protocol.freeze_backbone(model)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(before, protocol.backbone_sha256(model))          # 冻结不改变参数值
        with torch.no_grad():
            model({"a": torch.zeros(2, dtype=torch.long), "b": torch.zeros(2, dtype=torch.long)})
        protocol.assert_no_grads(model)                                   # 无梯度 → 通过
        param = model.shared_expert_network.mlp[0].weight                  # MLP.mlp 是 nn.Sequential
        param.grad = torch.zeros_like(param)
        with self.assertRaises(AssertionError):
            protocol.assert_no_grads(model)
        with torch.no_grad():
            param.add_(0.1)
        self.assertNotEqual(before, protocol.backbone_sha256(model))       # 参数一变哈希即变

    def test_stage1_id_content_addressed(self):
        base = dict(split_sha="ab" * 32, model_seed=1685480945, epochs=2, cfg_sha="cd" * 32)
        sid = protocol.stage1_id(**base)
        self.assertTrue(sid.startswith("s1-"))
        self.assertEqual(sid, protocol.stage1_id(**base))
        for changed in ({"model_seed": 1}, {"epochs": 3}, {"cfg_sha": "ef" * 32}):
            self.assertNotEqual(sid, protocol.stage1_id(**{**base, **changed}))

    def test_stage1_save_load_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root, sid = Path(td), "s1-deadbeef-m1685480945-e2-cafe0000"
            model, env_ids = tiny_mptrec(), protocol.make_env_ids(16, protocol.ENV_SEED)
            out_dir = protocol.save_stage1(root, sid, backbone_state=model.state_dict(),
                                           env_ids=env_ids, meta={"stage1_id": sid})
            for name in ("backbone.pt", "env_ids.pt", "meta.json"):
                self.assertTrue((out_dir / name).exists())
            loaded = protocol.load_stage1(root, sid)
            self.assertEqual(loaded["meta"]["stage1_id"], sid)
            self.assertTrue(torch.equal(loaded["env_ids"], env_ids))
            self.assertEqual(protocol.sha256_tensor(loaded["env_ids"]), protocol.sha256_tensor(env_ids))
            with self.assertRaises(FileExistsError):                      # 只读产物，永不覆盖（spec 4.2）
                protocol.save_stage1(root, sid, backbone_state=model.state_dict(), env_ids=env_ids, meta={})
            with self.assertRaises(FileNotFoundError):
                protocol.load_stage1(root, "s1-missing")

    def test_run_id_format_and_summary_append(self):
        rid = protocol.make_run_id(datetime(2026, 9, 29, 15, 30), split_seed=20260929,
                                   model_seed=1685480945, tag="short", commit="fd75198")
        self.assertEqual(rid, "20260929-1530-s20260929-m1685480945-short-fd75198")
        self.assertRegex(rid, r"^\d{8}-\d{4}-s\d+-m\d+-(short|full)-[0-9a-f]{7}$")
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "SUMMARY.md"
            row = {"run_id": rid, "commit": "fd75198", "auc_test_education": "0.900000", "stage1_id": "s1-x",
                   **{k: "PASS" for k in ("A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4")}}
            protocol.append_summary_row(summary, row)
            protocol.append_summary_row(summary, {**row, "run_id": rid.replace("1530", "1545")})
            lines = summary.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 4)                                # 表头 + 分隔 + 2 行
            self.assertIn(rid, lines[2]); self.assertIn("1545", lines[3])  # 只追加，不重写历史（spec 7.3）
