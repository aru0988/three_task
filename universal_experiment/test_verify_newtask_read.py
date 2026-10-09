"""Tests for the independent raw-artifact verifier of the new-task read screen.

A tiny synthetic formal run (built with the runner on tiny models) must verify
clean; every tampered copy must be rejected with the specific failed check.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from census_benchmark import protocol as P
from universal_experiment import newtask_read as N
from universal_experiment import verify_newtask_read as V
from universal_experiment.test_newtask_read import ARMS, CPU, tiny_build, tiny_spec, write_stage1_arm


def build_golden_root(root):
    """Tiny but structurally complete formal run: src data, stage1 arms, run output."""
    source = root / 'src'
    (source / 'dataset/Census-income').mkdir(parents=True)
    # Rows 0 and 3 carry the positive new-task label; val=[0,1,2], test=[3,4,5].
    pd.DataFrame({'education': [9, 1, 1, 9, 1, 1]}).to_csv(
        source / 'dataset/Census-income/test.gz', index=False)
    stage1 = root / 'stage1'
    stage1.mkdir()
    for i, arm in enumerate(ARMS):
        torch.manual_seed(400 + i)
        write_stage1_arm(stage1, arm, tiny_build(CPU))
    stats = {'n_train': 10, 'n_val': 3, 'n_test': 3, 'disjoint': True, 'union_complete': True}
    fingerprint = P.split_fingerprint(split_seed=P.SPLIT_SEED, stats=stats,
                                      val_idx=np.array([0, 1, 2]), test_idx=np.array([3, 4, 5]))
    spec = tiny_spec(shared_split=([0, 1, 2], [3, 4, 5]), fingerprints=fingerprint)
    N.run(spec, 'census', stage1, root / 'run', source=source, commit='test', dirty=False,
          smoke=False, epochs=2, patience=3, device=CPU)
    return root / 'run', fingerprint['fingerprint_sha256']


class VerifierGoldenPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.golden_root = Path(cls._tmp.name) / 'golden'
        cls.golden_root.mkdir()
        cls.run_dir, cls.expected_split = build_golden_root(cls.golden_root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def copy_of_golden(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        root = Path(tmp) / 'root'
        shutil.copytree(self.golden_root, root)
        run_dir = root / 'run'
        config_path = run_dir / 'config.json'
        config = json.loads(config_path.read_text())
        config['stage1_dir'] = str(root / 'stage1')
        config['source'] = str(root / 'src')
        config_path.write_text(json.dumps(config), encoding='utf-8')
        return run_dir

    def test_golden_run_verifies(self):
        result = V.verify(self.run_dir, build=tiny_build, protocol=P, expected_split=self.expected_split)
        self.assertTrue(result['passed'], result['failures'])
        self.assertTrue(all(result['checks'].values()))
        self.assertTrue((Path(self.run_dir) / 'verification.json').exists())

    def test_split_identity_pin_is_active(self):
        result = V.verify(self.run_dir, build=tiny_build, protocol=P)
        self.assertFalse(result['passed'])
        self.assertIn('split_identity', result['failures'])

    def test_tampered_label_vector_is_rejected(self):
        run_dir = self.copy_of_golden()
        with np.load(run_dir / 'U_predictions.npz') as data:
            arrays = {k: data[k] for k in data.files}
        arrays['test_y'] = arrays['test_y'].copy()
        arrays['test_y'][1] = 1.0 - arrays['test_y'][1]  # keeps both classes present
        np.savez_compressed(run_dir / 'U_predictions.npz', **arrays)
        result = V.verify(run_dir, build=tiny_build, protocol=P, expected_split=self.expected_split)
        self.assertFalse(result['passed'])
        self.assertIn('U_test_labels_equal', result['failures'])
        self.assertIn('hash_U_predictions.npz', result['failures'])

    def test_tampered_reported_auc_is_rejected(self):
        run_dir = self.copy_of_golden()
        path = run_dir / 'report.json'
        report = json.loads(path.read_text())
        report['arms']['B']['val_auc'] += 0.01
        path.write_text(json.dumps(report), encoding='utf-8')
        result = V.verify(run_dir, build=tiny_build, protocol=P, expected_split=self.expected_split)
        self.assertIn('B_val_auc', result['failures'])

    def test_tampered_selection_is_rejected(self):
        run_dir = self.copy_of_golden()
        path = run_dir / 'report.json'
        report = json.loads(path.read_text())
        report['arms']['G']['best_epoch'] = 99
        path.write_text(json.dumps(report), encoding='utf-8')
        result = V.verify(run_dir, build=tiny_build, protocol=P, expected_split=self.expected_split)
        self.assertIn('G_selection', result['failures'])

    def test_tampered_head_artifact_is_rejected(self):
        run_dir = self.copy_of_golden()
        state = torch.load(run_dir / 'B_head.pt', map_location='cpu', weights_only=True)
        state['extra_input.weight'] = torch.zeros(3, 3)
        torch.save(state, run_dir / 'B_head.pt')
        result = V.verify(run_dir, build=tiny_build, protocol=P, expected_split=self.expected_split)
        self.assertIn('B_head_contract', result['failures'])
        self.assertIn('B_head_file_sha', result['failures'])

    def test_tampered_checkpoint_is_rejected(self):
        run_dir = self.copy_of_golden()
        config = json.loads((run_dir / 'config.json').read_text())
        checkpoint = Path(config['stage1_dir']) / 'B_model.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        key = next(k for k in state if k.endswith('weight'))
        state[key] = state[key] + 1.0
        torch.save(state, checkpoint)
        result = V.verify(run_dir, build=tiny_build, protocol=P, expected_split=self.expected_split)
        self.assertFalse(result['passed'])
        self.assertIn('B_checkpoint_sha', result['failures'])
        self.assertIn('B_checkpoint_base_hash', result['failures'])

    def test_tampered_row_count_is_rejected(self):
        run_dir = self.copy_of_golden()
        with np.load(run_dir / 'R_predictions.npz') as data:
            arrays = {k: data[k] for k in data.files}
        arrays['val_y'] = arrays['val_y'][:2] + 0.0
        arrays['val_p'] = arrays['val_p'][:2] + 0.0
        np.savez_compressed(run_dir / 'R_predictions.npz', **arrays)
        result = V.verify(run_dir, build=tiny_build, protocol=P, expected_split=self.expected_split)
        self.assertIn('R_val_rows', result['failures'])

    def test_dirty_flag_is_rejected(self):
        run_dir = self.copy_of_golden()
        path = run_dir / 'config.json'
        config = json.loads(path.read_text())
        config['dirty'] = True
        path.write_text(json.dumps(config), encoding='utf-8')
        result = V.verify(run_dir, build=tiny_build, protocol=P, expected_split=self.expected_split)
        self.assertIn('clean_worktree', result['failures'])

    def test_unequal_head_init_is_rejected(self):
        run_dir = self.copy_of_golden()
        path = run_dir / 'report.json'
        report = json.loads(path.read_text())
        report['head_init_hashes']['U'] = '0' * 64
        report['arms']['U']['head_init_hash'] = '0' * 64
        path.write_text(json.dumps(report), encoding='utf-8')
        result = V.verify(run_dir, build=tiny_build, protocol=P, expected_split=self.expected_split)
        self.assertIn('head_init_across_arms', result['failures'])
        self.assertIn('U_head_init_hash', result['failures'])

    def test_decisions_are_recomputed_from_raw_auc(self):
        run_dir = self.copy_of_golden()
        path = run_dir / 'report.json'
        report = json.loads(path.read_text())
        report['verdict']['decision'] = 'POSITIVE_TRANSFER_SIGNAL' if report['verdict']['decision'] == 'NO_GO' else 'NO_GO'
        path.write_text(json.dumps(report), encoding='utf-8')
        result = V.verify(run_dir, build=tiny_build, protocol=P, expected_split=self.expected_split)
        self.assertIn('decision', result['failures'])


class AliCCPLabelProvenanceTests(unittest.TestCase):
    def test_derived_bsi_labels_match_raw_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ctr_cvr.dev'
            rows = ['click,purchase,bsi']
            for i in range(12):
                rows.append(f'{i % 2},{i % 3},{2 if i % 4 == 0 else i % 2}')
            path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
            labels = V.read_aliccp_labels(path, 12)
            self.assertEqual(labels.tolist(), [0 if i % 4 == 0 else 1 for i in range(12)])
            with self.assertRaises(ValueError):
                V.read_aliccp_labels(path, 13)


if __name__ == '__main__':
    unittest.main()
