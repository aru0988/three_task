"""Focused tests for the Read-U new-task transfer runner (frozen Stage-1 arms).

Covers: checkpoint/base/encoder provenance refusal, canonical head-init pinning,
the no-extra-input head contract, validation selection with the declared tie
rule, patience, base freezing, and the frozen decision rule.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import protocol as P
from multitaskrec.model import MPTRec, NewTask
from universal_experiment import newtask_read as N
from universal_experiment.model import UniversalExpert
from universal_experiment.verify import rank_auc

ARMS = ('B', 'U', 'R', 'G')
CPU = torch.device('cpu')


def tiny_build(device):
    return MPTRec(2, {'c': 3}, 2, 3, [8, 4], [4], reg_embedding=0.0, reg_dnn=0.0, device=device)


def tiny_protocol():
    return SimpleNamespace(seed_model=P.seed_model, backbone_sha256=P.backbone_sha256,
                           split_fingerprint=P.split_fingerprint, SPLIT_SEED=P.SPLIT_SEED,
                           INPUT_SIZE=3, TOWER_HIDDEN=(4,), REG_DNN=0.01, NUM_TASKS=2)


class TinyData(Dataset):
    """y0, y1, new-task label, features - mirrors the dataset tuple contract."""

    def __init__(self, n, offset=0):
        self.n, self.offset = n, offset

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        i = i + self.offset
        return (i % 2, (i // 2) % 2, float(i % 3 == 0),
                {'c': torch.tensor(i % 3), 'd': torch.tensor(float(i % 5))})


def tiny_spec(*, n_train=10, n_val=3, n_test=3, batch_size=4, model_seed=321, lr=0.01, rep_dim=4,
              fingerprints=None, shared_split=None):
    if shared_split is None:
        sets = {'train': TinyData(n_train), 'val': TinyData(n_val, 0), 'test': TinyData(n_test, 3)}
    else:
        source = TinyData(n_train)
        sets = {'train': source, 'val': Subset(source, shared_split[0]), 'test': Subset(source, shared_split[1])}
    loaders = {k: DataLoader(v, batch_size=batch_size, shuffle=False) for k, v in sets.items()}
    return SimpleNamespace(P=tiny_protocol(), build=tiny_build, sets=sets, loaders=loaders,
                           task_names=['A', 'B'], lr=lr, batch_size=batch_size, model_seed=model_seed,
                           rep_dim=rep_dim, make_u=lambda base: (UniversalExpert([1, 2], rep_dim=rep_dim), [1, 2]),
                           fingerprints=fingerprints or {})


def write_stage1_arm(folder, arm, base, *, selected_epoch=2, records=None, tasks=('A', 'B')):
    """Write one Stage-1 arm's checkpoint, metrics and raw old-task predictions."""
    prefix = '' if arm == 'B' else 'base.'
    state = {prefix + k: v for k, v in base.state_dict().items()}
    if arm in ('U', 'R'):
        encoder = UniversalExpert([1, 2], rep_dim=4)
        with torch.no_grad():
            encoder.encoder[0].weight.normal_()
        key = 'universal.' if arm == 'U' else 'random_u.'
        state.update({key + k: v for k, v in encoder.state_dict().items()})
        universal_hash = P.backbone_sha256(encoder)
    else:
        universal_hash = None
    if arm in ('U', 'R', 'G'):
        state['projections.0.weight'] = torch.eye(4) * 0.5
        state['projections.1.weight'] = torch.eye(4) * 0.5
        state['logits.0'] = torch.tensor(-2.0)
        state['logits.1'] = torch.tensor(-2.0)
    torch.save(state, folder / f'{arm}_model.pt')
    raw, scores = {}, {'val': {}, 'test': {}}
    for index, task in enumerate(tasks):
        for split, shift in (('val', 0.0), ('test', 0.01)):
            y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
            p = np.linspace(0.1, 0.9, 6) + 0.02 * index + shift
            raw[f'{split}_y{index}'], raw[f'{split}_p{index}'] = y, p
            scores[split][task] = rank_auc(y, p)
    np.savez_compressed(folder / f'{arm}_predictions.npz', **raw)
    records = records or [{'epoch': 1, 'auc_val_0': 0.60, 'auc_val_1': 0.60, 'wall_seconds': 1.0},
                          {'epoch': 2, 'auc_val_0': 0.70, 'auc_val_1': 0.65, 'wall_seconds': 2.0},
                          {'epoch': 3, 'auc_val_0': 0.65, 'auc_val_1': 0.65, 'wall_seconds': 3.0}]
    metrics = {'arm': arm, 'base_hash': P.backbone_sha256(base), 'universal_hash': universal_hash,
               'selected_epoch': selected_epoch, 'epochs': records, 'wall_seconds': 3.0,
               'val_auc': scores['val'], 'test_auc': scores['test']}
    (folder / f'{arm}_metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    return metrics


class Stage1SelectionTests(unittest.TestCase):
    def test_first_maximum_tie_rule(self):
        records = [{'epoch': 1, 'auc_val_0': 0.5, 'auc_val_1': 0.5},
                   {'epoch': 2, 'auc_val_0': 0.6, 'auc_val_1': 0.5},
                   {'epoch': 3, 'auc_val_0': 0.55, 'auc_val_1': 0.55}]
        self.assertEqual(N.selected_stage1_epoch(records), 2)


class ArmCheckpointLoadingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.bases, self.metrics = {}, {}
        for i, arm in enumerate(ARMS):
            torch.manual_seed(100 + i)
            self.bases[arm] = tiny_build(CPU)
            self.metrics[arm] = write_stage1_arm(self.folder, arm, self.bases[arm])

    def tearDown(self):
        self._tmp.cleanup()

    def rewrite(self, arm, **changes):
        metrics = json.loads((self.folder / f'{arm}_metrics.json').read_text())
        metrics.update(changes)
        (self.folder / f'{arm}_metrics.json').write_text(json.dumps(metrics), encoding='utf-8')

    def test_loads_each_arms_own_base(self):
        for arm in ARMS:
            base, info = N.load_arm_base(tiny_build, P, self.folder, arm, CPU,
                                         make_u=lambda b: (UniversalExpert([1, 2], rep_dim=4), [1, 2]))
            self.assertEqual(P.backbone_sha256(base), P.backbone_sha256(self.bases[arm]))
            self.assertEqual(info['base_hash_loaded'], info['base_hash_recorded'])
            self.assertEqual(info['selected_epoch'], 2)
            self.assertEqual(info['checkpoint_sha256'], N.C.file_hash(self.folder / f'{arm}_model.pt'))
            self.assertFalse(base.training)
            self.assertTrue(all(not p.requires_grad for p in base.parameters()))
            if arm in ('U', 'R'):
                self.assertEqual(info['encoder_hash_loaded'], info['encoder_hash_recorded'])

    def test_refuses_changed_base_hash(self):
        self.rewrite('G', base_hash='0' * 64)
        with self.assertRaises(RuntimeError):
            N.load_arm_base(tiny_build, P, self.folder, 'G', CPU)

    def test_refuses_changed_encoder_hash(self):
        self.rewrite('U', universal_hash='0' * 64)
        with self.assertRaises(RuntimeError):
            N.load_arm_base(tiny_build, P, self.folder, 'U', CPU,
                            make_u=lambda b: (UniversalExpert([1, 2], rep_dim=4), [1, 2]))

    def test_refuses_selection_inconsistent_with_epoch_history(self):
        self.rewrite('B', selected_epoch=3)
        with self.assertRaises(RuntimeError):
            N.load_arm_base(tiny_build, P, self.folder, 'B', CPU)

    def test_refuses_missing_checkpoint(self):
        (self.folder / 'R_model.pt').unlink()
        with self.assertRaises(FileNotFoundError):
            N.load_arm_base(tiny_build, P, self.folder, 'R', CPU)


class HeadInitTests(unittest.TestCase):
    kwargs = dict(model_seed=7, input_size=3, rep_dim=4, tower_hidden=[4], reg_dnn=0.01, device=CPU)

    def test_canonical_sequence_is_deterministic(self):
        first, hash_one = N.canonical_head_init(P, tiny_build, **self.kwargs)
        second, hash_two = N.canonical_head_init(P, tiny_build, **self.kwargs)
        self.assertEqual(hash_one, hash_two)
        for (_, a), (_, b) in zip(first.state_dict().items(), second.state_dict().items()):
            self.assertTrue(torch.equal(a, b))

    def test_arm_heads_share_init_hash_and_plain_newtask_contract(self):
        original, init_hash = N.canonical_head_init(P, tiny_build, **self.kwargs)
        heads, hashes = N.build_arm_heads(P, original, init_hash)
        self.assertEqual(set(hashes), set(ARMS))
        self.assertEqual(set(hashes.values()), {init_hash})
        canonical = NewTask(3, 4, [4], 0.01, CPU)
        for head in heads.values():
            self.assertEqual(set(head.state_dict()), set(canonical.state_dict()))
            for key, value in head.state_dict().items():
                self.assertEqual(tuple(value.shape), tuple(canonical.state_dict()[key].shape))
            self.assertEqual(sum(p.numel() for p in head.parameters()),
                             sum(p.numel() for p in canonical.parameters()))
            self.assertFalse(any(k == 'logit' or k.startswith(('projection.', 'universal.', 'random_u.'))
                                 for k in head.state_dict()))

    def test_refuses_unequal_head_init(self):
        original, _ = N.canonical_head_init(P, tiny_build, **self.kwargs)
        with self.assertRaises(RuntimeError):
            N.build_arm_heads(P, original, '0' * 64)
        tampered = copy.deepcopy(original)
        with torch.no_grad():
            tampered.tower_network.mlp[0].weight.add_(1.0)
        _, init_hash = N.canonical_head_init(P, tiny_build, **self.kwargs)
        with self.assertRaises(RuntimeError):
            N.build_arm_heads(P, tampered, init_hash)


class TrainArmTests(unittest.TestCase):
    y = np.array([0, 0, 1, 1])

    def _train(self, spec, arm, tmp, *, epochs, patience, fake=None):
        base = tiny_build(CPU)
        base.eval()
        for param in base.parameters():
            param.requires_grad_(False)
        head, init_hash = N.canonical_head_init(spec.P, tiny_build, model_seed=spec.model_seed,
                                                input_size=3, rep_dim=4, tower_hidden=[4],
                                                reg_dnn=0.01, device=CPU)
        real_hash = P.backbone_sha256(base)
        provenance = dict(checkpoint_file=f'{arm}_model.pt', checkpoint_sha256='x' * 64,
                          base_hash_recorded=real_hash, base_hash_loaded=real_hash, selected_epoch=2,
                          stage1_wall_seconds=1.0, stage1_old_tasks_val={'A': 0.9},
                          stage1_old_tasks_test={'A': 0.8}, head_init_hash=init_hash,
                          encoder_hash_recorded=None, encoder_hash_loaded=None)
        with patch.object(N, 'predict_newtask', fake) if fake else patch.object(N, 'predict_newtask', N.predict_newtask):
            metrics = N.train_arm(base, head, spec, arm, Path(tmp), epochs=epochs, patience=patience,
                                  device=CPU, provenance=provenance, task_name='Education')
        return metrics, base, provenance

    def test_selection_uses_first_max_val_auc_and_stops_at_patience(self):
        spec = tiny_spec()
        vectors = {'high': np.array([.1, .2, .9, .3]), 'mid': np.array([.5, .2, .9, .3]),
                   'low': np.array([.5, .6, .4, .3])}
        # Four scripted epochs (high, high, mid, low -> 3 stale epochs) plus the
        # post-training evaluation of the reloaded best head.
        script = ['high', 'high', 'mid', 'low', 'high']
        calls = {'val': 0}

        def fake(head, base, loader, device):
            if loader is spec.loaders['val']:
                index = min(calls['val'], len(script) - 1)
                calls['val'] += 1
                return self.y, vectors[script[index]]
            return self.y, vectors['high']

        with tempfile.TemporaryDirectory() as tmp:
            metrics, base, provenance = self._train(spec, 'B', tmp, epochs=5, patience=3, fake=fake)
            self.assertEqual([e['val_auc'] for e in metrics['epochs']],
                             [roc_auc_score(self.y, vectors[k]) for k in script[:4]])
            self.assertEqual(metrics['best_epoch'], 1)
            self.assertEqual(metrics['epoch_count'], 4)
            self.assertEqual(metrics['val_auc'], metrics['epochs'][0]['val_auc'])
            self.assertEqual(metrics['test_auc'], roc_auc_score(self.y, vectors['high']))
            with np.load(Path(tmp) / 'B_predictions.npz') as saved:
                self.assertTrue(np.array_equal(saved['test_y'], self.y))
                self.assertTrue(np.array_equal(saved['test_p'], vectors['high']))
            self.assertTrue((Path(tmp) / 'B_head.pt').exists())
            self.assertEqual(P.backbone_sha256(base), provenance['base_hash_loaded'])
            self.assertEqual(metrics['frozen_base_hash_after'], provenance['base_hash_loaded'])
            self.assertTrue(metrics['frozen_gradients_none'])
            self.assertTrue(all(not p.requires_grad for p in base.parameters()))
            self.assertEqual(metrics['parameters'], sum(p.numel() for p in NewTask(3, 4, [4], 0.01, CPU).parameters()))
            self.assertEqual([e['steps'] for e in metrics['epochs']], [3, 3, 3, 3])

    def test_no_early_stop_and_last_epoch_wins_when_improving(self):
        spec = tiny_spec()
        vectors = {'low': np.array([.5, .6, .4, .3]), 'mid': np.array([.5, .2, .9, .3]),
                   'high': np.array([.1, .2, .9, .3])}
        script = ['low', 'mid', 'high', 'high', 'high', 'high']
        calls = {'val': 0}

        def fake(head, base, loader, device):
            if loader is spec.loaders['val']:
                index = min(calls['val'], len(script) - 1)
                calls['val'] += 1
                return self.y, vectors[script[index]]
            return self.y, vectors['high']

        with tempfile.TemporaryDirectory() as tmp:
            metrics, _, _ = self._train(spec, 'B', tmp, epochs=5, patience=3, fake=fake)
            self.assertEqual(metrics['epoch_count'], 5)
            self.assertEqual(metrics['best_epoch'], 3)

    def test_real_training_writes_freezes_and_selects(self):
        spec = tiny_spec(n_train=12, n_val=6, n_test=6)
        with tempfile.TemporaryDirectory() as tmp:
            metrics, base, provenance = self._train(spec, 'U', tmp, epochs=3, patience=3)
            self.assertLessEqual(metrics['epoch_count'], 3)
            self.assertIn(metrics['best_epoch'], [e['epoch'] for e in metrics['epochs']])
            with np.load(Path(tmp) / 'U_predictions.npz') as saved:
                self.assertEqual(len(saved['val_y']), 6)
                self.assertEqual(len(saved['test_y']), 6)
                self.assertTrue(((saved['val_p'] >= 0) & (saved['val_p'] <= 1)).all())
            best = max(metrics['epochs'], key=lambda e: e['val_auc'])
            self.assertEqual(metrics['val_auc'], best['val_auc'])
            self.assertEqual(metrics['head_init_hash'], provenance['head_init_hash'])
            for param in base.parameters():
                self.assertIsNone(param.grad)


class DecisionRuleTests(unittest.TestCase):
    def test_positive_requires_both_test_and_val(self):
        verdict = N.delta_verdict({'B': {'test': .002, 'val': .001}, 'R': {'test': .0015, 'val': .0005},
                                   'G': {'test': .003, 'val': .002}})
        self.assertEqual(verdict['decision'], 'POSITIVE_TRANSFER_SIGNAL')
        self.assertFalse(verdict['unstable'])
        self.assertEqual(verdict['per_arm']['B']['classification'], 'positive')

    def test_no_go_when_content_control_flat(self):
        verdict = N.delta_verdict({'B': {'test': .002, 'val': .001}, 'R': {'test': .0005, 'val': .001},
                                   'G': {'test': .002, 'val': .001}})
        self.assertEqual(verdict['decision'], 'NO_GO')
        self.assertFalse(verdict['unstable'])
        self.assertEqual(verdict['per_arm']['R']['classification'], 'no_clear_improvement')

    def test_test_only_positive_is_flagged_unstable(self):
        verdict = N.delta_verdict({'B': {'test': .002, 'val': -.001}, 'R': {'test': .002, 'val': -.0001},
                                   'G': {'test': .002, 'val': .001}})
        self.assertEqual(verdict['decision'], 'NO_GO')
        self.assertTrue(verdict['unstable'])

    def test_classification_boundaries(self):
        self.assertEqual(N.classify_delta(.001), 'positive')
        self.assertEqual(N.classify_delta(.0009), 'no_clear_improvement')
        self.assertEqual(N.classify_delta(-.0199), 'no_clear_improvement')
        self.assertEqual(N.classify_delta(-.02), 'clear_decline')


class RunEndToEndTests(unittest.TestCase):
    def test_run_writes_arms_report_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage1 = root / 'stage1'
            stage1.mkdir()
            for i, arm in enumerate(ARMS):
                torch.manual_seed(200 + i)
                write_stage1_arm(stage1, arm, tiny_build(CPU))
            spec = tiny_spec(shared_split=([0, 1, 2], [3, 4, 5]))
            out = root / 'out'
            report = N.run(spec, 'census', stage1, out, source=root / 'src', commit='test',
                           dirty=False, smoke=False, epochs=2, patience=3, device=CPU)
            for name in ('config.json', 'report.json', 'indices.npz'):
                self.assertTrue((out / name).exists(), name)
            for arm in ARMS:
                self.assertTrue((out / f'{arm}_metrics.json').exists())
                self.assertTrue((out / f'{arm}_predictions.npz').exists())
                self.assertTrue((out / f'{arm}_head.pt').exists())
            config = json.loads((out / 'config.json').read_text())
            self.assertEqual(set(config['stage1_files']), set(ARMS))
            for arm in ARMS:
                self.assertEqual(config['stage1_files'][arm]['checkpoint_sha256'],
                                 N.C.file_hash(stage1 / f'{arm}_model.pt'))
                self.assertEqual(report['arms'][arm]['head_init_hash'], config['head_init_hash'])
            self.assertEqual(set(report['u_minus']), {'B', 'R', 'G'})
            self.assertIn(report['verdict']['decision'], ('POSITIVE_TRANSFER_SIGNAL', 'NO_GO'))
            self.assertEqual(set(report['prediction_hashes']), {'indices.npz'} | {f'{a}_predictions.npz' for a in ARMS})
            with self.assertRaises(RuntimeError):
                N.run(spec, 'census', stage1, out, commit='test', dirty=False, smoke=False,
                      epochs=2, patience=3, device=CPU)
            with self.assertRaises(RuntimeError):
                N.run(spec, 'census', stage1, root / 'out-dirty', commit='test', dirty=True,
                      smoke=False, epochs=2, patience=3, device=CPU)

    def test_smoke_run_allows_dirty_tree_and_short_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage1 = root / 'stage1'
            stage1.mkdir()
            for arm in ARMS:
                write_stage1_arm(stage1, arm, tiny_build(CPU))
            spec = tiny_spec()
            out = root / 'smoke'
            report = N.run(spec, 'census', stage1, out, commit='test', dirty=True, smoke=True,
                           epochs=2, patience=3, device=CPU)
            self.assertTrue(report['smoke'])
            config = json.loads((out / 'config.json').read_text())
            self.assertTrue(config['smoke'] and config['dirty'])
            with self.assertRaises(RuntimeError):
                N.run(spec, 'census', stage1, root / 'smoke-too-long', commit='test', dirty=True,
                      smoke=True, epochs=3, patience=3, device=CPU)


READ_U_DIRS = {
    'census': Path('results/universal/census-read-u-seed1685480945'),
    'aliccp': Path('results/universal/aliccp-read-u-seed1688723512'),
}
# Supervisor-verified 2026-10-10: (base_hash, selected_epoch) per arm.
READ_U_PINS = {
    'census': {'B': ('a24f55841770c464901ebb15cfce824f7559e500092f59353cbf5573483aed83', 18),
               'U': ('497976270a99362521fb62eb0df33f96e7ca38f3934a818642cc368755f411b3', 15),
               'R': ('4b3032ef462f8067a5071f4e68ac1ff28b5699b774b8ae4eac8904ff647fcc17', 18),
               'G': ('5c21d4d0421878fee288900a67b0be0674221acc0c9dc54ecb46a6c450fdbe02', 15)},
    'aliccp': {'B': ('29ebe43ba10f628fd89d98ffa2dda66e9e9ea47517565ba2504b142e23d48213', 20),
               'U': ('2da87217d70b74e6d0d3a2fd73e33edb30da175dabf04a1d96c6d6db2753b6e6', 20),
               'R': ('f79261b1c3515ea12fc4bba5b65ac616108416ecfcaa689ae91882fd156a567f', 19),
               'G': ('2ac0c379f7a7be468ea967085646e7abc9515fd3846f37094912cff7fb21dbf4', 19)},
}


class LocalReadUArtifactTests(unittest.TestCase):
    """Preflight the frozen read-u checkpoints on disk against the pinned hashes.

    Skipped when the (gitignored) local artifacts are absent.  Census encoder
    widths are reconstructed from the raw CSV header + vocabulary; the [5]*16
    AliCCP widths follow `stage1_read.prepare`.
    """

    def make_u(self, dataset, base, source):
        if dataset == 'census':
            import pandas as pd
            from config import CensusIncome_Vocabulary_Size
            columns = list(pd.read_csv(source / 'dataset/Census-income/train.gz', nrows=1).columns)
            vocab = {k: v for k, v in CensusIncome_Vocabulary_Size.items() if k != 'education'}
            labels = ('education', 'label_income', 'label_marital')
            dense = [k for k in columns if k not in vocab and k not in labels]
            widths = [1] * len(dense) + [4] * len(vocab)
            self.assertEqual(sum(widths), 123)
            return UniversalExpert(widths).to(CPU), widths
        self.assertEqual([5] * 16, [5] * 16)
        return UniversalExpert([5] * 16, rep_dim=64).to(CPU), [5] * 16

    def test_frozen_checkpoints_match_pinned_hashes_and_epochs(self):
        for dataset, folder in READ_U_DIRS.items():
            if not folder.is_dir():
                self.skipTest(f'{dataset} read-u artifacts missing')
            if dataset == 'census':
                from run_census_benchmark import build_mptrec as build
                from census_benchmark import protocol as protocol
            else:
                from universal_experiment.aliccp import build
                from aliccp_benchmark import protocol as protocol
            source = Path(json.loads((folder / 'config.json').read_text())['source'])
            for arm in ARMS:
                base, info = N.load_arm_base(build, protocol, folder, arm, CPU,
                                             make_u=lambda b, d=dataset, s=source: self.make_u(d, b, s))
                expected_hash, expected_epoch = READ_U_PINS[dataset][arm]
                self.assertEqual(info['base_hash_loaded'], expected_hash, f'{dataset} {arm}')
                self.assertEqual(info['selected_epoch'], expected_epoch, f'{dataset} {arm}')
                if arm in ('U', 'R'):
                    self.assertEqual(info['encoder_hash_loaded'], info['encoder_hash_recorded'])


if __name__ == '__main__':
    unittest.main()
