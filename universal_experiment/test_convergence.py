import unittest

from universal_experiment.convergence import convergence_decision


class ConvergenceTests(unittest.TestCase):
    def test_cap_hit_is_not_converged(self):
        decision = convergence_decision(
            [.50, .51, .52], best_epoch=3, budget=3,
            patience=5, stopped_early=False,
        )
        self.assertFalse(decision.converged)
        self.assertEqual(decision.reason, "best_epoch_at_cap")

    def test_early_stopped_curve_is_converged(self):
        curve = [.50, .52, .519, .518, .517, .516, .515]
        decision = convergence_decision(
            curve, best_epoch=2, budget=40,
            patience=5, stopped_early=True,
        )
        self.assertTrue(decision.converged)
        self.assertEqual(decision.reason, "patience_exhausted")

    def test_budget_end_before_patience_is_not_convergence(self):
        decision = convergence_decision(
            [.50, .52, .519], best_epoch=2, budget=3,
            patience=5, stopped_early=False,
        )
        self.assertFalse(decision.converged)
        self.assertEqual(decision.reason, "budget_exhausted_before_patience")

    def test_best_in_final_two_epochs_is_not_converged(self):
        decision = convergence_decision(
            [.50, .51, .52, .519, .518, .517, .516, .515],
            best_epoch=3, budget=8, patience=5, stopped_early=True,
        )
        self.assertTrue(decision.converged)
        late = convergence_decision(
            [.50, .51, .52], best_epoch=2, budget=3,
            patience=1, stopped_early=True,
        )
        self.assertFalse(late.converged)
        self.assertEqual(late.reason, "best_epoch_too_late")


if __name__ == "__main__":
    unittest.main()
