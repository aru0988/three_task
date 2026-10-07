"""Pre-run verifier regression tests; no dataset or training required."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from aliccp_benchmark import audit_env_weighting as audit


class AuditEnvWeightingTests(unittest.TestCase):
    def test_parity_reads_utf8_precedent_on_windows(self):
        self.assertTrue(audit.parity_check()["all_bit_equal"])

    def test_clip_ess_floor_checks_both_weight_directions(self):
        env_ids = torch.tensor([0] * 48 + [1] * 52)
        result = audit.weight_space(env_ids)
        self.assertAlmostEqual(result["ess_floor_at_clip"] / 100, 0.4904032063193153)

    def test_parity_rejects_unpinned_precedent_blob(self):
        with patch.object(audit, "PRECEDENT_BLOB", "0" * 40):
            with self.assertRaisesRegex((AssertionError, RuntimeError), "blob|pin|hash"):
                audit.parity_check()

    def test_stage1_backbone_hash_mismatch_aborts_audit(self):
        self.assertTrue(hasattr(audit, "require_backbone_match"))
        with self.assertRaisesRegex(AssertionError, "backbone.*sha"):
            audit.require_backbone_match(nn.Linear(1, 1), {"backbone_sha256": "0" * 64})

    def test_stage1_data_fingerprint_mismatch_aborts_audit(self):
        self.assertTrue(hasattr(audit, "require_fingerprint_match"))
        with self.assertRaisesRegex(AssertionError, "fingerprint"):
            audit.require_fingerprint_match({"fingerprint_sha256": "a"}, {"fingerprint_sha256": "b"})


if __name__ == "__main__":
    unittest.main()
