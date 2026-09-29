import unittest

import torch
from sklearn.metrics import roc_auc_score

from census_benchmark import metrics


def _ok_kwargs(**over):
    """全部门禁通过的基线参数；单个测试按需覆盖。"""
    base = dict(backbone_sha_equal=True, grads_all_none=True, split_ok=True, split_stats_ok=True,
                env_ids_ok=True, auc_val_income=0.90, auc_val_marital=0.90, auc_test_education=0.90,
                auc_val_education_best=0.90, gate_mean=[0.5, 0.5], env_shares=[0.40, 0.60])
    base.update(over)
    return base


class TestMetrics(unittest.TestCase):
    def test_auc_matches_sklearn(self):
        gen = torch.Generator().manual_seed(0)
        y = torch.randint(0, 2, (500,), generator=gen)
        score = torch.rand(500, generator=gen) + y.float() * 0.5
        self.assertAlmostEqual(metrics.auc(y, score), roc_auc_score(y.tolist(), score.tolist()), places=12)

    def test_env_accuracy(self):
        log_prob = torch.log(torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]]))
        self.assertAlmostEqual(metrics.env_accuracy(log_prob, torch.tensor([0, 1, 1, 1])), 0.75)

    def test_gate_mean_weights(self):
        stats = metrics.GateStats(num_tasks=2)
        stats.update([torch.tensor([[0.8, 0.2], [0.6, 0.4]]), torch.tensor([[0.1, 0.9], [0.3, 0.7]])])
        out = stats.result()
        self.assertAlmostEqual(out[0], 0.70, places=6); self.assertAlmostEqual(out[1], 0.20, places=6)

    def test_rep_geometry_and_constant_rep(self):
        stats = metrics.RepStats(num_tasks=2)
        stats.update(torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
                     [torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]])])
        out = stats.result()
        self.assertAlmostEqual(out["cos_gen_spec"][0], 0.5, places=6)     # cos: 1.0 与 0.0
        self.assertAlmostEqual(out["cos_gen_spec"][1], 1.0, places=6)
        self.assertAlmostEqual(out["gen_std"], 0.75, places=6)
        flat = metrics.RepStats(num_tasks=1)
        flat.update(torch.ones(4, 3), [torch.ones(4, 3)])
        self.assertAlmostEqual(flat.result()["gen_std"], 0.0, places=9)   # 常量表征 → std 0（防塌缩诊断）

    def test_judge_a_class_hard_gates(self):
        report = metrics.judge(**_ok_kwargs())
        self.assertTrue(report["overall_pass"]); self.assertEqual(report["failures"], [])
        for key in ("A1", "A2", "A4", "A5"):
            self.assertTrue(report[key]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(backbone_sha_equal=False))["A1"]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(grads_all_none=False))["A1"]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(split_ok=False))["overall_pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(env_ids_ok=False))["A5"]["pass"])

    def test_judge_b1_b2_thresholds(self):
        self.assertTrue(metrics.judge(**_ok_kwargs(auc_val_income=0.60))["B1"]["pass"])     # 边界含等号
        self.assertFalse(metrics.judge(**_ok_kwargs(auc_val_income=0.599))["B1"]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(auc_test_education=0.59,
                                                    auc_val_education_best=0.59))["B1"]["pass"])
        self.assertTrue(metrics.judge(**_ok_kwargs(auc_val_education_best=0.90,
                                                   auc_test_education=0.88))["B2"]["pass"])   # 差 = 0.02
        self.assertFalse(metrics.judge(**_ok_kwargs(auc_val_education_best=0.90,
                                                    auc_test_education=0.86))["B2"]["pass"])  # 差 = 0.04
        # 阈值 0.03 在二进制浮点下不可精确表示（0.9 - 0.87 = 0.030000000000000027），故用 0.02 / 0.04 覆盖两侧

    def test_judge_b3_b4_degenerate(self):
        self.assertTrue(metrics.judge(**_ok_kwargs(gate_mean=[0.05, 0.95]))["B3"]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(gate_mean=[0.02, 0.98]))["B3"]["pass"])    # 坍缩到单一分支
        self.assertTrue(metrics.judge(**_ok_kwargs(env_shares=[0.05, 0.95]))["B4"]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(env_shares=[0.049, 0.951]))["B4"]["pass"])  # 聚类退化


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 才能复现累加器 device mismatch")
class TestCudaAccumulatorRegression(unittest.TestCase):
    """回归：CUDA 张量喂进 CPU 累加器 → "Expected all tensors to be on the same device"。

    实际崩溃路径：stage2 五个 epoch 后 evaluate_newtask(mechanism=True) 在 CUDA 上抽表征，
    GateStats.update 的 self.total[i] += <cuda reduction> 直接抛错；RepStats 的 cos_sum 同理。
    用例数值与 CPU 版测试逐一对应：同一输入在不同设备下必须给出同一结果。
    """

    def test_gate_stats_cuda_gate_outs(self):
        stats = metrics.GateStats(num_tasks=2)
        stats.update([torch.tensor([[0.8, 0.2], [0.6, 0.4]], device="cuda"),
                      torch.tensor([[0.1, 0.9], [0.3, 0.7]], device="cuda")])
        out = stats.result()
        self.assertAlmostEqual(out[0], 0.70, places=6)
        self.assertAlmostEqual(out[1], 0.20, places=6)

    def test_rep_stats_cuda_tensors(self):
        stats = metrics.RepStats(num_tasks=2)
        stats.update(torch.tensor([[1.0, 0.0], [0.0, 2.0]], device="cuda"),
                     [torch.tensor([[1.0, 0.0], [1.0, 0.0]], device="cuda"),
                      torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda")])
        out = stats.result()
        self.assertAlmostEqual(out["cos_gen_spec"][0], 0.5, places=6)
        self.assertAlmostEqual(out["cos_gen_spec"][1], 1.0, places=6)
        self.assertAlmostEqual(out["gen_std"], 0.75, places=6)

    def test_rep_stats_mixed_device_batches(self):
        """累加器必须与输入设备无关：CPU batch 之后再喂 CUDA batch 不得 device mismatch。"""
        stats = metrics.RepStats(num_tasks=1)
        gen = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        spec = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        stats.update(gen, [spec])
        stats.update(gen.cuda(), [spec.cuda()])
        out = stats.result()
        self.assertAlmostEqual(out["cos_gen_spec"][0], 0.5, places=6)
        self.assertAlmostEqual(out["gen_std"], 0.75, places=6)
