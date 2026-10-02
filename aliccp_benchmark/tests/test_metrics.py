"""指标与门禁判定测试（TDD：先于实现编写）。"""
import unittest

import torch

from aliccp_benchmark import metrics, protocol


def a_facts(**overrides):
    facts = {
        "backbone_sha_before": "aa",
        "backbone_sha_after": "aa",
        "backbone_grads_none": True,
        "prefix_sha_ok": True,
        "fingerprint_sha_match": True,
        "len_ok": True,
        "counts_ok": True,
        "env_ids_sha_match": True,
        "backbone_sha_matches_stage1": True,
        "stage1_id_recorded": True,
    }
    facts.update(overrides)
    return facts


def b_facts(**overrides):
    facts = {
        "auc_val_ctr": 0.62,
        "auc_val_cvr": 0.55,
        "auc_val_bsi_best": 0.66,
        "auc_test_bsi": 0.65,
        "gate_mean": [0.5, 0.5],
        "cluster_events": [
            {"epoch": 2, "diff_num": 10, "env_0": 900_000, "env_1": 1_100_000},
        ],
        "train_size": 2_000_000,
    }
    facts.update(overrides)
    return facts


class TestAucAndEnvAcc(unittest.TestCase):
    def test_auc_score_perfect_and_random_orientation(self):
        y = [0, 0, 1, 1]
        self.assertEqual(metrics.auc_score(y, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(metrics.auc_score(y, [0.9, 0.8, 0.2, 0.1]), 0.0)

    def test_env_accuracy(self):
        env_ids = torch.tensor([0, 1, 1, 0])
        env_pred = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
        # argmax: [0, 1, 0, 0] vs ids [0, 1, 1, 0] → 3/4
        self.assertAlmostEqual(metrics.env_accuracy(env_pred, env_ids), 0.75)


class TestAGates(unittest.TestCase):
    def test_all_pass(self):
        gates = metrics.evaluate_a_gates(a_facts())
        self.assertEqual(set(gates), {"A1", "A2", "A3", "A4", "A5", "A6"})
        for key, result in gates.items():
            self.assertIn(result["verdict"], {"PASS", "SKIP"}, key)
        self.assertTrue(metrics.hard_pass(gates, enforce_b=False))

    def test_each_failure_flips_its_gate(self):
        cases = {
            "A1": {"backbone_sha_after": "bb"},
            "A1b": {"backbone_grads_none": False},
            "A2": {"prefix_sha_ok": False},
            "A2b": {"fingerprint_sha_match": False},
            "A4": {"len_ok": False},
            "A4b": {"counts_ok": False},
            "A5": {"env_ids_sha_match": False},
            "A6": {"backbone_sha_matches_stage1": False},
            "A6b": {"stage1_id_recorded": False},
        }
        for name, override in cases.items():
            gates = metrics.evaluate_a_gates(a_facts(**override))
            gate_id = name[0:2]
            self.assertEqual(gates[gate_id]["verdict"], "FAIL", name)
            self.assertFalse(metrics.hard_pass(gates, enforce_b=False), name)

    def test_a3_is_skip(self):
        gates = metrics.evaluate_a_gates(a_facts())
        self.assertEqual(gates["A3"]["verdict"], "SKIP")


class TestBGates(unittest.TestCase):
    def test_all_pass(self):
        gates = metrics.evaluate_b_gates(b_facts())
        self.assertEqual(set(gates), {"B1", "B2", "B3", "B4"})
        for result in gates.values():
            self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(metrics.hard_pass({"A1": {"verdict": "PASS"}}, enforce_b=False) is True)

    def test_b1_floors(self):
        gates = metrics.evaluate_b_gates(b_facts(auc_val_ctr=0.54))
        self.assertEqual(gates["B1"]["verdict"], "FAIL")
        gates = metrics.evaluate_b_gates(b_facts(auc_test_bsi=0.52))
        self.assertEqual(gates["B1"]["verdict"], "FAIL")
        # CVR 下限刻意设为 0.50（近乎无约束）
        gates = metrics.evaluate_b_gates(b_facts(auc_val_cvr=0.499))
        self.assertEqual(gates["B1"]["verdict"], "FAIL")
        gates = metrics.evaluate_b_gates(b_facts(auc_val_cvr=0.505))
        self.assertEqual(gates["B1"]["verdict"], "PASS")

    def test_b2_gap(self):
        self.assertEqual(metrics.evaluate_b_gates(b_facts(auc_val_bsi_best=0.70, auc_test_bsi=0.65))["B2"]["verdict"], "PASS")
        self.assertEqual(metrics.evaluate_b_gates(b_facts(auc_val_bsi_best=0.71, auc_test_bsi=0.65))["B2"]["verdict"], "FAIL")

    def test_b3_gate_collapse(self):
        self.assertEqual(metrics.evaluate_b_gates(b_facts(gate_mean=[0.5, 0.5]))["B3"]["verdict"], "PASS")
        self.assertEqual(metrics.evaluate_b_gates(b_facts(gate_mean=[0.99, 0.5]))["B3"]["verdict"], "FAIL")
        self.assertEqual(metrics.evaluate_b_gates(b_facts(gate_mean=[0.01, 0.5]))["B3"]["verdict"], "FAIL")

    def test_b4_cluster_share_and_na(self):
        events = [{"epoch": 2, "diff_num": 1, "env_0": 100_000, "env_1": 1_900_000}]
        self.assertEqual(metrics.evaluate_b_gates(b_facts(cluster_events=events))["B4"]["verdict"], "PASS")
        events = [{"epoch": 2, "diff_num": 1, "env_0": 10, "env_1": 1_999_990}]
        self.assertEqual(metrics.evaluate_b_gates(b_facts(cluster_events=events))["B4"]["verdict"], "FAIL")
        self.assertEqual(metrics.evaluate_b_gates(b_facts(cluster_events=[]))["B4"]["verdict"], "N/A")

    def test_hard_pass_enforce_b(self):
        failing = metrics.evaluate_b_gates(b_facts(auc_val_ctr=0.50))
        self.assertFalse(metrics.hard_pass({"A1": {"verdict": "PASS"}}, enforce_b=True, b_gates=failing))
        self.assertTrue(metrics.hard_pass({"A1": {"verdict": "PASS"}}, enforce_b=False, b_gates=failing))

    def test_gate_constants_match_protocol(self):
        self.assertEqual(metrics.GATE_MIN, protocol.GATE_MIN)
        self.assertEqual(metrics.AUC_FLOORS, (protocol.AUC_FLOOR_CTR, protocol.AUC_FLOOR_CVR, protocol.AUC_FLOOR_BSI))


if __name__ == "__main__":
    unittest.main()
