import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from universal_experiment.verify_paired import verify


class PairedVerifierTests(unittest.TestCase):
    def test_recomputes_auc_and_delta_from_raw_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.array([0, 0, 1, 1])
            b = np.array([.1, .4, .35, .8])
            u = np.array([.1, .2, .7, .8])
            for arm, prediction in (("B", b), ("U", u)):
                np.savez_compressed(root / f"{arm}_predictions.npz",
                                    val_y=labels, val_p=prediction,
                                    test_y=labels, test_p=prediction)
            stage1 = {
                arm: {"seed": 7, "budget": 20, "patience": 5,
                      "convergence": {"converged": True}}
                for arm in ("B", "U")
            }
            stage2 = {
                "B": {"val_auc": .75, "test_auc": .75, "best_epoch": 2,
                      "epochs": [{"epoch": 1, "val_auc": .5}, {"epoch": 2, "val_auc": .75}],
                      "budget": 20, "patience": 5, "convergence": {"converged": True}},
                "U": {"val_auc": 1.0, "test_auc": 1.0, "best_epoch": 2,
                      "epochs": [{"epoch": 1, "val_auc": .5}, {"epoch": 2, "val_auc": 1.0}],
                      "budget": 20, "patience": 5, "convergence": {"converged": True}},
            }
            report = {"config": {"dirty": False, "smoke": False}, "stage1": stage1,
                      "stage2": stage2, "u_minus_b": {"val": .25, "test": .25},
                      "classification": "positive", "all_converged": True}
            (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
            result = verify(root, write=False)
            self.assertTrue(result["passed"], result["failures"])


if __name__ == "__main__":
    unittest.main()
