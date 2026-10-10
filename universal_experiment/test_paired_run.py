import unittest

from universal_experiment.paired_run import summarize_stage1


class PairedRunnerTests(unittest.TestCase):
    def test_stage1_summary_uses_first_strict_max_and_convergence_rule(self):
        records = [
            {"epoch": 1, "auc_val_0": .50, "auc_val_1": .60},
            {"epoch": 2, "auc_val_0": .55, "auc_val_1": .61},
            {"epoch": 3, "auc_val_0": .54, "auc_val_1": .60},
            {"epoch": 4, "auc_val_0": .53, "auc_val_1": .59},
            {"epoch": 5, "auc_val_0": .52, "auc_val_1": .58},
            {"epoch": 6, "auc_val_0": .51, "auc_val_1": .57},
            {"epoch": 7, "auc_val_0": .50, "auc_val_1": .56},
        ]
        summary = summarize_stage1(records, budget=20, patience=5)
        self.assertEqual(summary["best_epoch"], 2)
        self.assertTrue(summary["convergence"]["converged"])

    def test_stage1_summary_marks_cap_as_unresolved(self):
        records = [
            {"epoch": 1, "auc_val_0": .50, "auc_val_1": .60},
            {"epoch": 2, "auc_val_0": .55, "auc_val_1": .61},
            {"epoch": 3, "auc_val_0": .56, "auc_val_1": .62},
        ]
        summary = summarize_stage1(records, budget=3, patience=5)
        self.assertFalse(summary["convergence"]["converged"])
        self.assertEqual(summary["convergence"]["reason"], "best_epoch_at_cap")


if __name__ == "__main__":
    unittest.main()
