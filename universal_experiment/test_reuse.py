import unittest
import json
import copy
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

    def test_aliccp_fingerprints_survive_json_key_roundtrip(self):
        fresh = self.config()
        fresh['dataset'] = 'aliccp'
        fresh['source_fingerprints'] = {
            'registered': {'fingerprint_sha256': 'same'},
            'actual': {split: {'prefix_sha256': split,
                              'label_counts': {'n': 100, 'bsi_raw': {1: 20, 2: 30, 3: 50}}}
                       for split in ('train', 'val', 'test')},
        }
        saved = json.loads(json.dumps(fresh))
        self.assertNotEqual(saved['source_fingerprints'], fresh['source_fingerprints'])
        for stage1_only in (False, True):
            with self.subTest(stage1_only=stage1_only):
                validate_config(saved, fresh, stage1_only=stage1_only)
                validate_config(fresh, saved, stage1_only=stage1_only)

    def test_normalization_does_not_hide_real_fingerprint_changes(self):
        saved = self.config()
        saved['source_fingerprints'] = {'actual': {'train': {
            'prefix_sha256': 'same', 'label_counts': {'bsi_raw': {'1': 20, '2': 30, '3': 50}}}}}
        for change in ('hash', 'count', 'missing'):
            fresh = copy.deepcopy(saved)
            train = fresh['source_fingerprints']['actual']['train']
            train['label_counts']['bsi_raw'] = {1: 20, 2: 30, 3: 50}
            if change == 'hash':
                train['prefix_sha256'] = 'different'
            elif change == 'count':
                train['label_counts']['bsi_raw'][1] += 1
            else:
                del train['label_counts']['bsi_raw'][3]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(saved, fresh)

    def test_normalization_is_non_mutating_and_rejects_colliding_keys(self):
        a = self.config(); b = self.config()
        b['source_fingerprints'] = {'bsi_raw': {1: 20}}
        a['source_fingerprints'] = {'bsi_raw': {'1': 20}}
        before = copy.deepcopy(b)
        validate_config(a, b)
        self.assertEqual(before, b)
        b['source_fingerprints']['bsi_raw']['1'] = 99
        with self.assertRaises(ValueError):
            validate_config(a, b)
