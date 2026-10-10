import copy
import unittest

import torch

from multitaskrec.model import MPTRec
from universal_experiment.model import UniversalExpert, UniversalStage1
from universal_experiment.paired import ReadDetachedStage1, make_stage1_model


class PairedUniversalTests(unittest.TestCase):
    def make_base(self):
        return MPTRec(
            2, {"c": 3}, 2, 3, [8, 4], [4],
            device=torch.device("cpu"),
        )

    def features(self, n=16):
        return {"d": torch.randn(n), "c": torch.randint(3, (n,))}

    def make_u(self):
        return UniversalExpert([1, 2], rep_dim=4, hidden=8, seed=41)

    def test_read_detached_supervised_loss_cannot_update_u(self):
        torch.manual_seed(12)
        model = ReadDetachedStage1(self.make_base(), self.make_u())
        target = torch.randint(2, (16,)).float()
        output = model(self.features())
        supervised = sum(
            torch.nn.functional.binary_cross_entropy(pred, target)
            for key in ("gen_preds", "fused_preds") for pred in output[key]
        )
        supervised.backward()
        self.assertTrue(all(p.grad is None for p in model.universal.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.projections.parameters()))

    def test_read_detached_auxiliary_loss_updates_u_only_through_ssl(self):
        torch.manual_seed(12)
        model = ReadDetachedStage1(self.make_base(), self.make_u())
        model(self.features())
        model.auxiliary["total"].backward()
        self.assertTrue(any(p.grad is not None for p in model.universal.parameters()))

    def test_no_read_wrapper_preserves_old_predictions(self):
        torch.manual_seed(12)
        base = self.make_base()
        plain = copy.deepcopy(base).eval()
        wrapped = UniversalStage1(base, self.make_u()).eval()
        features = self.features()
        with torch.no_grad():
            expected = plain.predict(features)
            actual = wrapped.predict(features)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    def test_make_stage1_model_supports_only_b_and_u(self):
        base = self.make_base()
        universal = self.make_u()
        self.assertIs(make_stage1_model(base, None, "B", "no_read"), base)
        no_read = make_stage1_model(self.make_base(), universal, "U", "no_read")
        self.assertIsInstance(no_read, UniversalStage1)
        read = make_stage1_model(self.make_base(), self.make_u(), "U", "read_detached")
        self.assertIsInstance(read, ReadDetachedStage1)
        with self.assertRaises(ValueError):
            make_stage1_model(self.make_base(), self.make_u(), "R", "no_read")
        with self.assertRaises(ValueError):
            make_stage1_model(self.make_base(), self.make_u(), "U", "unknown")


if __name__ == "__main__":
    unittest.main()
