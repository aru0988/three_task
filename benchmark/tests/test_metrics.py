"""共享指标测试：黄金值 + 与旧实现（census_benchmark / aliccp_benchmark 的 metrics）逐值等价。

黄金值于 2026-10-09 由旧实现在固定夹具上捕获；旧实现包删除后等价类自动跳过（find_spec 判断）。
"""
import importlib.util
import unittest

import torch

from benchmark import metrics

_HAVE_LEGACY = (importlib.util.find_spec("census_benchmark") is not None
                and importlib.util.find_spec("aliccp_benchmark") is not None)

Y_TRUE = torch.tensor([0, 0, 1, 1, 1, 0, 1, 0])
Y_HAT = torch.tensor([0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.7, 0.3])
ENV_IDS = torch.tensor([0, 1, 1, 0])
ENV_PRED = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])


class TestGoldenValues(unittest.TestCase):
    def test_auc_orientation_and_value(self):
        self.assertEqual(metrics.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(metrics.auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(metrics.auc(Y_TRUE, Y_HAT), 0.9375)              # 旧实现捕获值
        self.assertEqual(metrics.auc(Y_TRUE.numpy(), Y_HAT.numpy()), 0.9375)

    def test_env_accuracy_value_and_truncation(self):
        # argmax: [0, 1, 0, 0] vs ids [0, 1, 1, 0] → 3/4
        self.assertEqual(metrics.env_accuracy(ENV_PRED, ENV_IDS), 0.75)
        self.assertAlmostEqual(metrics.env_accuracy(ENV_PRED, ENV_IDS[:3]), 2 / 3, places=6)


@unittest.skipUnless(_HAVE_LEGACY, "旧实现包已删除（迁移完成）")
class TestLegacyEquivalence(unittest.TestCase):
    def test_auc_matches_legacy(self):
        from aliccp_benchmark import metrics as old_aliccp
        from census_benchmark import metrics as old_census
        self.assertEqual(metrics.auc(Y_TRUE, Y_HAT), old_census.auc(Y_TRUE, Y_HAT))
        self.assertEqual(metrics.auc(Y_TRUE, Y_HAT), old_aliccp.auc_score(Y_TRUE, Y_HAT))

    def test_env_accuracy_matches_legacy(self):
        from aliccp_benchmark import metrics as old_aliccp
        from census_benchmark import metrics as old_census
        self.assertEqual(metrics.env_accuracy(ENV_PRED, ENV_IDS), old_census.env_accuracy(ENV_PRED, ENV_IDS))
        self.assertEqual(metrics.env_accuracy(ENV_PRED, ENV_IDS), old_aliccp.env_accuracy(ENV_PRED, ENV_IDS))
        # 截断路径为 aliccp 旧语义的超集（census 旧实现对长度不等会广播报错）
        self.assertEqual(metrics.env_accuracy(ENV_PRED, ENV_IDS[:3]), old_aliccp.env_accuracy(ENV_PRED, ENV_IDS[:3]))


if __name__ == "__main__":
    unittest.main()
