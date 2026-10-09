"""Independent raw-artifact verifier for the Read-U new-task transfer screen.

Recomputes every AUC from the saved raw labels/predictions with rank statistics
(never the training metric function), re-derives split labels from the raw data
files, re-hashes the input checkpoints and their base states, re-simulates the
declared validation selection and patience, and re-evaluates the frozen decision
rule.  `build`/`protocol`/`expected_split` are injectable for synthetic tests.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from multitaskrec.model import NewTask
from universal_experiment.verify import rank_auc

ARMS = ('B', 'U', 'R', 'G')
EXPECTED_SPLIT = '096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c'  # census canonical split


def resolve(dataset):
    if dataset == 'census':
        from census_benchmark import protocol as protocol
        from run_census_benchmark import build_mptrec as build
    else:
        from aliccp_benchmark import protocol as protocol
        from universal_experiment.aliccp import build
    return protocol, build


def read_census_labels(path):
    """Independent label derivation: education == 9 is the positive class."""
    import pandas as pd
    frame = pd.read_csv(path, usecols=['education'])
    return (frame['education'] == 9).astype(int).to_numpy()


def read_aliccp_labels(path, n):
    """Independent raw prefix scan: bsi positive unless the raw last field is 2."""
    labels = np.empty(n, dtype=np.int64)
    with open(path, encoding='utf-8') as handle:
        handle.readline()
        for i in range(n):
            line = handle.readline()
            if not line:
                raise ValueError(f'{path} has fewer than {n} data rows')
            labels[i] = 0 if int(line.strip().split(',')[-1]) == 2 else 1
    return labels


def classify_delta(delta):
    if delta >= 0.001:
        return 'positive'
    return 'clear_decline' if delta <= -0.02 else 'no_clear_improvement'


def verify(folder, *, build=None, protocol=None, expected_split=EXPECTED_SPLIT):
    folder = Path(folder)
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    dataset = config['dataset']
    if build is None or protocol is None:
        default_protocol, default_build = resolve(dataset)
        protocol = protocol or default_protocol
        build = build or default_build
    checks = {}

    checks['mode'] = config['mode'] == 'newtask-read-u'
    checks['clean_worktree'] = (not config['dirty']) and (not config['smoke'])
    checks['budget'] = config['epochs'] <= 20 and config['patience'] == 3
    for name, digest in report['prediction_hashes'].items():
        checks[f'hash_{name}'] = hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest

    # Canonical head-init reproduction: same seed -> build -> NewTask sequence on CPU.
    protocol.seed_model(config['model_seed'])
    dummy = build(torch.device('cpu'))
    canonical = NewTask(config['input_size'], config['rep_dim'], list(config['tower_hidden']),
                        config['reg_dnn'], torch.device('cpu'))
    checks['head_init_reproduces'] = (protocol.backbone_sha256(canonical)
                                      == config['head_init_hash'] == report['head_init_hash'])
    del dummy
    checks['head_init_across_arms'] = (len(set(report['head_init_hashes'].values())) == 1
                                       and set(report['head_init_hashes'].values()) == {config['head_init_hash']})

    labels, scores = {}, {}
    for arm in ARMS:
        metrics = report['arms'][arm]
        with np.load(folder / f'{arm}_predictions.npz') as raw:
            for split in ('val', 'test'):
                y, p = raw[f'{split}_y'], raw[f'{split}_p']
                score = rank_auc(y, p)
                scores[(arm, split)] = score
                checks[f'{arm}_{split}_auc'] = abs(score - metrics[f'{split}_auc']) < 1e-12
                checks[f'{arm}_{split}_shape'] = y.shape == p.shape and y.ndim == 1
                checks[f'{arm}_{split}_rows'] = (len(y) == config['rows'][split]
                                                 == metrics[f'{split}_n'])
                if split not in labels:
                    labels[split] = y
                else:
                    checks[f'{arm}_{split}_labels_equal'] = bool(np.array_equal(y, labels[split]))
        history = metrics['epochs']
        best_value = max(e['val_auc'] for e in history)
        first_best = min(e['epoch'] for e in history if e['val_auc'] == best_value)
        checks[f'{arm}_selection'] = (metrics['best_epoch'] == first_best
                                      and abs(best_value - metrics['val_auc']) < 1e-12)
        best, stale, stop = -1.0, 0, None
        for record in history:
            if record['val_auc'] > best:
                best, stale = record['val_auc'], 0
            else:
                stale += 1
                if stale == config['patience']:
                    stop = record['epoch']
                    break
        checks[f'{arm}_early_stop'] = len(history) == (config['epochs'] if stop is None else stop)
        expected_steps = math.ceil(config['rows']['train'] / config['batch_size'])
        checks[f'{arm}_steps'] = bool(history) and all(e['steps'] == expected_steps for e in history)
        # No-U-input contract: the saved head is exactly the canonical NewTask, nothing more.
        head_state = torch.load(folder / f'{arm}_head.pt', map_location='cpu', weights_only=True)
        reference = canonical.state_dict()
        checks[f'{arm}_head_contract'] = (set(head_state) == set(reference)
                                          and all(tuple(head_state[k].shape) == tuple(v.shape)
                                                  for k, v in reference.items()))
        checks[f'{arm}_head_parameters'] = metrics['parameters'] == sum(p.numel() for p in canonical.parameters())
        checks[f'{arm}_head_file_sha'] = (hashlib.sha256((folder / f'{arm}_head.pt').read_bytes()).hexdigest()
                                          == metrics['head_state_sha256'])
        checks[f'{arm}_head_init_hash'] = metrics['head_init_hash'] == config['head_init_hash']
        # Input checkpoint provenance: recorded file hash, base lineage, Stage-1 selection.
        stage1 = Path(config['stage1_dir'])
        recorded = config['stage1_files'][arm]
        checkpoint = stage1 / recorded['checkpoint_file']
        checks[f'{arm}_checkpoint_sha'] = (hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                                           == recorded['checkpoint_sha256'])
        stage1_metrics = json.loads((stage1 / f'{arm}_metrics.json').read_text(encoding='utf-8'))
        checks[f'{arm}_base_lineage'] = (metrics['base_hash'] == metrics['base_hash_recorded']
                                         == recorded['base_hash_recorded'] == recorded['base_hash_loaded']
                                         == stage1_metrics['base_hash'])
        sums = [r['auc_val_0'] + r['auc_val_1'] for r in stage1_metrics['epochs']]
        stage1_first_max = min(r['epoch'] for r in stage1_metrics['epochs']
                               if r['auc_val_0'] + r['auc_val_1'] == max(sums))
        checks[f'{arm}_stage1_selection'] = (stage1_metrics['selected_epoch'] == stage1_first_max
                                             == recorded['selected_epoch'] == metrics['stage1_selected_epoch'])
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        prefix = '' if arm == 'B' else 'base.'
        base_state = ({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                      if prefix else dict(state))
        model = build(torch.device('cpu'))
        model.load_state_dict(base_state)
        checks[f'{arm}_checkpoint_base_hash'] = (protocol.backbone_sha256(model)
                                                 == stage1_metrics['base_hash'])
        checks[f'{arm}_frozen'] = (metrics['frozen_base_hash_before'] == metrics['frozen_base_hash_after']
                                   == metrics['base_hash'] and metrics['frozen_gradients_none'])
        # Old-task context must be a faithful copy of the Stage-1 raw predictions.
        with np.load(stage1 / f'{arm}_predictions.npz') as old:
            for index, task in enumerate(config['stage1_task_names']):
                for split in ('val', 'test'):
                    expected = metrics[f'stage1_old_tasks_{split}'][task]
                    checks[f'{arm}_{split}_{task}_context'] = abs(
                        rank_auc(old[f'{split}_y{index}'], old[f'{split}_p{index}']) - expected) < 1e-12

    deltas = {arm: {split: scores[('U', split)] - scores[(arm, split)] for split in ('val', 'test')}
              for arm in ('B', 'R', 'G')}
    for arm in ('B', 'R', 'G'):
        for split in ('val', 'test'):
            checks[f'delta_{arm}_{split}'] = abs(deltas[arm][split] - report['u_minus'][arm][split]) < 1e-12
    test_ok = all(deltas[arm]['test'] >= 0.001 for arm in ('B', 'R'))
    val_ok = all(deltas[arm]['val'] > 0 for arm in ('B', 'R'))
    checks['decision'] = report['verdict']['decision'] == ('POSITIVE_TRANSFER_SIGNAL' if (test_ok and val_ok)
                                                           else 'NO_GO')
    checks['unstable'] = bool(report['verdict']['unstable']) == bool(test_ok and not val_ok)
    for arm in ('B', 'R', 'G'):
        checks[f'classification_{arm}'] = (report['verdict']['per_arm'][arm]['classification']
                                           == classify_delta(deltas[arm]['test']))

    source = Path(config['source'])
    if dataset == 'census':
        fingerprint = config['fingerprints']  # prepare() stores the census split fingerprint flat
        with np.load(folder / 'indices.npz') as indices:
            val_idx, test_idx = indices['val'], indices['test']
        stats = {'n_train': config['rows']['train'], 'n_val': int(len(val_idx)),
                 'n_test': int(len(test_idx)),
                 'disjoint': bool(set(val_idx.tolist()).isdisjoint(test_idx.tolist())),
                 'union_complete': bool(sorted(val_idx.tolist() + test_idx.tolist())
                                        == list(range(len(val_idx) + len(test_idx))))}
        recomputed = protocol.split_fingerprint(split_seed=config['split_seed'], stats=stats,
                                                val_idx=val_idx, test_idx=test_idx)
        checks['split_fingerprint'] = recomputed == fingerprint
        if expected_split is not None:
            checks['split_identity'] = fingerprint['fingerprint_sha256'] == expected_split
        derived = read_census_labels(source / 'dataset/Census-income/test.gz')
        checks['val_labels_provenance'] = bool(np.array_equal(labels['val'], derived[val_idx]))
        checks['test_labels_provenance'] = bool(np.array_equal(labels['test'], derived[test_idx]))
    else:
        fingerprint = config['fingerprints']['prefix_fingerprint']
        checks['prefix_fingerprint_selfhash'] = protocol.fingerprint_digest(fingerprint) == fingerprint['fingerprint_sha256']
        for split in ('val', 'test'):
            derived = read_aliccp_labels(source / protocol.DATA_FILES[split], config['rows'][split])
            checks[f'{split}_labels_provenance'] = bool(np.array_equal(labels[split], derived))
            checks[f'{split}_label_counts'] = int(derived.sum()) == fingerprint['label_counts'][split]['bsi_pos']
            checks[f'{split}_prefix_recorded'] = (config['fingerprints']['actual'][split]['prefix_sha256']
                                                  == fingerprint['files'][split]['prefix_sha256'])

    result = dict(passed=all(checks.values()), checks={k: bool(v) for k, v in checks.items()},
                  independent_auc={f'{arm}_{split}': scores[(arm, split)] for arm in ARMS
                                   for split in ('val', 'test')},
                  u_minus=deltas, failures=[k for k, v in checks.items() if not v])
    (folder / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path, help='formal output directory of the new-task read screen')
    folder = parser.parse_args().run
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    expected = EXPECTED_SPLIT if config['dataset'] == 'census' else None
    result = verify(folder, expected_split=expected)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
