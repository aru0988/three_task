"""run_benchmark 统一 CLI：解析（数据集 × 阶段）、数据集差异透传、enforce_b 规则。"""
import unittest
from types import SimpleNamespace

import run_benchmark
from benchmark.datasets import aliccp, census


class TestParser(unittest.TestCase):
    def test_census_stage1_defaults(self):
        args = run_benchmark.build_parser().parse_args(["census", "stage1"])
        self.assertEqual((args.dataset, args.command), ("census", "stage1"))
        self.assertEqual(args.root, census.ARTIFACT_ROOT)
        self.assertEqual((args.gpu, args.tag, args.epochs), (0, "short", census.STAGE1_EPOCHS))
        self.assertEqual((args.split_seed, args.model_seed, args.env_seed),
                         (census.SPLIT_SEED, census.MODEL_SEED, census.ENV_SEED))

    def test_census_stage2_requires_stage1_dir(self):
        parser = run_benchmark.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["census", "stage2"])
        args = parser.parse_args(["census", "stage2",
                                  "--stage1-dir", "artifacts/census_stage2/stage1/s1-x"])
        self.assertEqual((args.tag, args.epochs), ("short", census.STAGE2_EPOCHS))

    def test_aliccp_stage1_defaults_and_tag_choices(self):
        parser = run_benchmark.build_parser()
        args = parser.parse_args(["aliccp", "stage1"])
        self.assertEqual(args.root, aliccp.ARTIFACT_ROOT)
        self.assertEqual((args.tag, args.epochs, args.patience),
                         ("short", aliccp.STAGE1_EPOCHS, aliccp.STAGE1_PATIENCE))
        self.assertEqual((args.train_budget, args.val_budget, args.test_budget),
                         (aliccp.TRAIN_BUDGET, aliccp.VAL_BUDGET, aliccp.TEST_BUDGET))
        self.assertEqual((args.model_seed, args.env_seed), (aliccp.MODEL_SEED, aliccp.ENV_SEED))
        with self.assertRaises(SystemExit):                # full 是 census 的 tag，不是 aliccp 的
            parser.parse_args(["aliccp", "stage1", "--tag", "full"])

    def test_aliccp_stage2_requires_stage1_id_and_defaults(self):
        parser = run_benchmark.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["aliccp", "stage2"])
        args = parser.parse_args(["aliccp", "stage2", "--stage1-id", "s1-x"])
        self.assertEqual((args.tag, args.epochs, args.patience),
                         ("short", aliccp.STAGE2_EPOCHS, aliccp.STAGE2_PATIENCE))
        self.assertFalse(args.no_enforce_b)
        smoke = parser.parse_args(["aliccp", "stage2", "--tag", "smoke", "--stage1-id", "s1-x",
                                   "--train-budget", "20000", "--val-budget", "5000",
                                   "--test-budget", "10000", "--epochs", "1", "--patience", "1",
                                   "--no-enforce-b"])
        self.assertEqual((smoke.epochs, smoke.patience), (1, 1))
        self.assertTrue(smoke.no_enforce_b)


class TestResolveEnforceB(unittest.TestCase):
    def test_truth_table(self):
        self.assertTrue(run_benchmark.resolve_enforce_b("short", False))    # 正式 tag 默认 enforce
        self.assertFalse(run_benchmark.resolve_enforce_b("smoke", False))   # smoke 默认只记录
        self.assertFalse(run_benchmark.resolve_enforce_b("short", True))
        self.assertFalse(run_benchmark.resolve_enforce_b("smoke", True))


class TestDatasetParity(unittest.TestCase):
    def test_aliccp_prefix_tag_for(self):
        self.assertEqual(aliccp.prefix_tag_for(), aliccp.PREFIX_TAG)        # 默认即全量预算
        self.assertEqual(aliccp.prefix_tag_for(train=aliccp.TRAIN_BUDGET, val=aliccp.VAL_BUDGET,
                                               test=aliccp.TEST_BUDGET), aliccp.PREFIX_TAG)
        self.assertEqual(aliccp.prefix_tag_for(train=aliccp.SMOKE_TRAIN_BUDGET,
                                               val=aliccp.SMOKE_VAL_BUDGET,
                                               test=aliccp.SMOKE_TEST_BUDGET), "p20000-v5000-t10000")

    def test_census_profile_seed_overrides(self):
        profile = census.build_profile(model_seed=7, env_seed=8)
        self.assertEqual((profile.model_seed, profile.env_seed), (7, 8))
        cfg = profile.build_cfg({"loaders": {"train": SimpleNamespace(batch_size=256)}}, 2)
        self.assertEqual((cfg["model_seed"], cfg["env_seed"]), (7, 8))


if __name__ == "__main__":
    unittest.main()
