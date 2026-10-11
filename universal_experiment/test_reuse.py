import unittest
from universal_experiment.reuse import validate_config


class ReuseTests(unittest.TestCase):
    def config(self):
        return dict(dataset='census', seed=42, stage1_epochs=20, stage2_epochs=20,
                    patience=5, smoke=False, dirty=False, source_fingerprints={'split': 'x'})

    def test_baseline_ignores_u_only_factors(self):
        a = self.config(); b = self.config()
        a.update(read_policy='no_read', u_mode='frozen')
        b.update(read_policy='read_detached', u_mode='trainable')
        validate_config(a, b)

    def test_rejects_each_mismatch_and_dirty_evidence(self):
        for key in ('dataset', 'seed', 'stage1_epochs', 'stage2_epochs',
                    'patience', 'source_fingerprints', 'smoke'):
            a = self.config(); b = self.config(); b[key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(a, b)
        a = self.config(); a['dirty'] = True
        with self.assertRaises(ValueError):
            validate_config(a, self.config())

    def test_stage1_reuse_allows_new_stage2_budget_only(self):
        a = self.config(); b = self.config(); b['stage2_epochs'] = 30
        validate_config(a, b, stage1_only=True)
        b['stage1_epochs'] = 30
        with self.assertRaises(ValueError):
            validate_config(a, b, stage1_only=True)
