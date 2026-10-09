import tempfile, unittest
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from benchmark import protocol
from multitaskrec.model import MPTRec

# ---- 黄金常量：2026-10-09 由旧实现（census_benchmark/aliccp_benchmark，两处逐字节相同）捕获并冻结 ----
GOLDEN = {
    "sha_tensor_123": "1b192f8ab8f5b460374d6711e768c26f32d29115ed9f1b7c1821a81313837a04",
    "sha_tensor_range": "2821a902fcfe5cd0b6b8a771ee2a5d44ad88cab557ab4ac9b45d64a45c56e00e",
    "config_hash": "90c55acda7f6bfedef368ec2278bb86404fb88c0eccf3a627a3ab831666bd1ea",
    "env_ids_sha": "01b67ee34795116fd81d70064118753e061390f8184da124aa05844e712bf289",
    "backbone_sha": "c5ea0292e2cce85f6aa568fb0fa315eaaeea144322e2ec2d10295668b8bcf37b",
    "stage1_id": "s1-abababab-m1685480945-e2-cdcdcdcd",
    "run_id": "20260929-1530-s20260929-m1685480945-short-fd75198",
}
ENV_SEED = 20260929

CENSUS_COLUMNS = ["run_id", "commit", "auc_test_education",
                  "A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4", "stage1_id"]


class TestGoldenEquivalence(unittest.TestCase):
    def test_hash_and_json_golden(self):
        self.assertEqual(protocol.sha256_tensor(torch.tensor([1, 2, 3])), GOLDEN["sha_tensor_123"])
        self.assertEqual(protocol.sha256_tensor(torch.arange(6).reshape(2, 3)), GOLDEN["sha_tensor_range"])
        self.assertEqual(protocol.canonical_json({"b": 2, "a": 1, "s": "x"}), '{"a":1,"b":2,"s":"x"}')
        self.assertEqual(protocol.config_hash({"b": 2, "a": 1, "s": "x"}), GOLDEN["config_hash"])

    def test_env_ids_golden(self):
        self.assertEqual(protocol.sha256_tensor(protocol.make_env_ids(1000, ENV_SEED)), GOLDEN["env_ids_sha"])

    def test_backbone_sha_golden(self):
        lin = nn.Linear(2, 3)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
            lin.bias.copy_(torch.tensor([0.5, -0.5, 1.5]))
        self.assertEqual(protocol.backbone_sha256(lin), GOLDEN["backbone_sha"])

    def test_id_goldens(self):
        self.assertEqual(protocol.stage1_id(fingerprint_sha="ab" * 32, model_seed=1685480945,
                                            epochs=2, cfg_sha="cd" * 32), GOLDEN["stage1_id"])
        self.assertEqual(protocol.make_run_id(datetime(2026, 9, 29, 15, 30), prefix="s20260929",
                                              model_seed=1685480945, tag="short", commit="fd75198"),
                         GOLDEN["run_id"])


class TestEnvAndSeeds(unittest.TestCase):
    def test_env_ids_independent_of_global_rng(self):
        torch.manual_seed(0); x1 = torch.rand(3)
        torch.manual_seed(0); e1 = protocol.make_env_ids(1000, ENV_SEED); x2 = torch.rand(3)
        e2 = protocol.make_env_ids(1000, ENV_SEED)
        self.assertTrue(torch.equal(x1, x2))                      # 全局 RNG 未被消耗
        self.assertTrue(torch.equal(e1, e2))                      # 同 env seed 可复现
        self.assertEqual(set(e1.tolist()), {0, 1})
        self.assertNotEqual(protocol.sha256_tensor(e1),
                            protocol.sha256_tensor(protocol.make_env_ids(1000, ENV_SEED + 1)))

    def test_env_ids_num_envs(self):
        e = protocol.make_env_ids(200, 7, num_envs=3)
        self.assertEqual(set(e.tolist()), {0, 1, 2})

    def test_seed_model_seeds_global_rng(self):
        protocol.seed_model(123); a = (torch.rand(4), torch.get_rng_state())
        protocol.seed_model(123); b = (torch.rand(4), torch.get_rng_state())
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))


# EmbeddingNetwork 输出宽度 = 特征个数 × embedding_size（vocab 大小只决定 embedding 表行数）
TINY_VOCAB, TINY_INPUT_SIZE = {"a": 3, "b": 2}, 8


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
        protocol.assert_no_grads(model)                                    # 无梯度 → 通过
        param = model.shared_expert_network.mlp[0].weight
        param.grad = torch.zeros_like(param)
        with self.assertRaises(AssertionError):
            protocol.assert_no_grads(model)
        with torch.no_grad():
            param.add_(0.1)
        self.assertNotEqual(before, protocol.backbone_sha256(model))       # 参数一变哈希即变

    def test_stage1_id_content_addressed(self):
        base = dict(fingerprint_sha="ab" * 32, model_seed=1685480945, epochs=2, cfg_sha="cd" * 32)
        sid = protocol.stage1_id(**base)
        self.assertTrue(sid.startswith("s1-"))
        self.assertEqual(sid, protocol.stage1_id(**base))
        for changed in ({"model_seed": 1}, {"epochs": 3}, {"cfg_sha": "ef" * 32},
                        {"fingerprint_sha": "ef" * 32}):
            self.assertNotEqual(sid, protocol.stage1_id(**{**base, **changed}))

    def test_stage1_save_load_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root, sid = Path(td), "s1-deadbeef-m1685480945-e2-cafe0000"
            model, env_ids = tiny_mptrec(), protocol.make_env_ids(16, ENV_SEED)
            meta = {"stage1_id": sid, "epochs": 2}
            out_dir = protocol.save_stage1(root, sid, backbone_state=model.state_dict(),
                                           env_ids=env_ids, meta=meta)
            for name in ("backbone.pt", "env_ids.pt", "meta.json"):
                self.assertTrue((out_dir / name).exists())
            # meta.json 必须与旧实现逐字节一致（纯 JSON 值下 default=str 不生效）
            import json as _json
            self.assertEqual((out_dir / "meta.json").read_text(encoding="utf-8"),
                             _json.dumps(meta, ensure_ascii=False, indent=2))
            loaded = protocol.load_stage1(root, sid)
            self.assertEqual(loaded["meta"]["stage1_id"], sid)
            self.assertTrue(torch.equal(loaded["env_ids"], env_ids))
            self.assertEqual(protocol.sha256_tensor(loaded["env_ids"]), protocol.sha256_tensor(env_ids))
            with self.assertRaises(FileExistsError):                       # 只读产物，永不覆盖
                protocol.save_stage1(root, sid, backbone_state=model.state_dict(), env_ids=env_ids, meta={})
            with self.assertRaises(FileNotFoundError):
                protocol.load_stage1(root, "s1-missing")

    def test_load_stage1_missing_meta_raises(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "stage1" / "s1-x").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                protocol.load_stage1(Path(td), "s1-x")

    def test_save_stage1_meta_default_str(self):
        with tempfile.TemporaryDirectory() as td:
            sid = "s1-deadbeef-m1-e1-cafe0000"
            protocol.save_stage1(Path(td), sid, backbone_state={}, env_ids=torch.zeros(1, dtype=torch.long),
                                 meta={"when": datetime(2026, 10, 9)})
            meta = protocol.load_stage1(Path(td), sid)["meta"]
            self.assertEqual(meta["when"], "2026-10-09 00:00:00")


class TestRunIdAndSummary(unittest.TestCase):
    def test_run_id_format(self):
        rid = protocol.make_run_id(datetime(2026, 9, 29, 15, 30), prefix="s20260929", model_seed=1685480945,
                                   tag="short", commit="fd75198")
        self.assertEqual(rid, GOLDEN["run_id"])
        self.assertRegex(rid, r"^\d{8}-\d{4}-s\d+-m\d+-(short|full)-[0-9a-f]{7}$")

    def test_summary_append_only(self):
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "SUMMARY.md"
            rid = GOLDEN["run_id"]
            row = {"run_id": rid, "commit": "fd75198", "auc_test_education": "0.900000", "stage1_id": "s1-x",
                   **{k: "PASS" for k in ("A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4")}}
            protocol.append_summary_row(summary, row, CENSUS_COLUMNS)
            protocol.append_summary_row(summary, {**row, "run_id": rid.replace("1530", "1545")}, CENSUS_COLUMNS)
            lines = summary.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0], "| " + " | ".join(CENSUS_COLUMNS) + " |")
            self.assertEqual(lines[1], "|" + "---|" * len(CENSUS_COLUMNS))
            self.assertEqual(len(lines), 4)                                # 表头 + 分隔 + 2 行
            self.assertIn(rid, lines[2]); self.assertIn("1545", lines[3])  # 只追加，不重写历史


if __name__ == "__main__":
    unittest.main()
