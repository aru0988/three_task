"""Explicit verified artifact reuse; never silently fall back to training B."""
import json
import shutil
import ast
import hashlib
import subprocess
from pathlib import Path

import torch

BASELINE_FILES = ('multitaskrec/model.py', 'multitaskrec/train.py', 'multitaskrec/dataset.py',
                  'census_benchmark/protocol.py', 'aliccp_benchmark/protocol.py',
                  'run_census_benchmark.py', 'universal_experiment/aliccp.py')
LEGACY_BASELINES = {'b619f193cc50b9acc2e606bd63dc6c3933e418d6',
                    '427282dd2a968a78a34f0b329c11ecf51ba4608d',
                    'bb00456e35022ff5ca4f5d43c0210cdac9939f53'}


def semantic_hash(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def baseline_code_identity(commit=None):
    root = Path(__file__).resolve().parents[1]
    identity = {name: semantic_hash(
        subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root).decode('utf-8')
        if commit else (root / name).read_text(encoding='utf-8')) for name in BASELINE_FILES}
    runner = (subprocess.check_output(['git', 'show', f'{commit}:universal_experiment/paired_run.py'], cwd=root).decode('utf-8')
              if commit else (root / 'universal_experiment/paired_run.py').read_text(encoding='utf-8'))
    tree = ast.parse(runner)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('prepare', 'train_stage1', 'iter_batches'):
            identity[f'runner:{node.name}'] = semantic_hash(ast.unparse(node))
        if isinstance(node, ast.FunctionDef) and node.name == 'run_frozen_pair':
            # Match the original deterministic head initialization without hashing U-only orchestration.
            selected, active = [], False
            for statement in node.body:
                text = ast.unparse(statement)
                if text == 'spec.P.seed_model(spec.model_seed)':
                    active = True
                if active:
                    selected.append(statement)
                if active and isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'heads' for t in statement.targets):
                    break
            if not selected or not any('make_stage2_heads' in ast.unparse(s) for s in selected):
                raise ValueError('head initialization contract cannot be identified')
            identity['runner:head_initialization'] = semantic_hash(ast.unparse(ast.Module(body=selected, type_ignores=[])))
    return identity


def normalize_fingerprints(value):
    """Normalize only BSI category keys affected by JSON, without changing counts."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'bsi_raw' and isinstance(item, dict):
                normalized = {str(category): count for category, count in item.items()}
                if len(normalized) != len(item):
                    raise ValueError('ambiguous bsi_raw category keys')
                result[key] = normalized
            else:
                result[key] = normalize_fingerprints(item)
        return result
    if isinstance(value, list):
        return [normalize_fingerprints(item) for item in value]
    return value


def validate_config(source, target, *, stage1_only=False):
    keys = ['dataset', 'seed', 'stage1_epochs', 'patience', 'smoke', 'source_fingerprints']
    if not stage1_only:
        keys.append('stage2_epochs')
    if source['dirty'] and not source['smoke']:
        raise ValueError('cannot reuse dirty formal evidence')
    for key in keys:
        left, right = source[key], target[key]
        if key == 'source_fingerprints':
            left, right = normalize_fingerprints(left), normalize_fingerprints(right)
        if left != right:
            raise ValueError(f'reuse mismatch: {key}')


def checked_report(folder, target, *, stage1_only=False):
    from universal_experiment.verify_paired import verify
    from universal_experiment.paired_run import file_hash
    folder = Path(folder).resolve()
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    validate_config(report['config'], target, stage1_only=stage1_only)
    source_config = report['config']
    if 'baseline_code_identity' in target:
        if 'baseline_code_identity' in source_config:
            source_identity = source_config['baseline_code_identity']
        else:
            if source_config['commit'] not in LEGACY_BASELINES:
                raise ValueError('legacy baseline has not been admitted for reuse')
            source_identity = baseline_code_identity(source_config['commit'])
        if source_identity != target['baseline_code_identity']:
            raise ValueError('baseline model/protocol code changed; train a new B')
        for key in ('torch', 'cuda', 'device'):
            if source_config[key] != target[key]:
                raise ValueError(f'reuse runtime mismatch: {key}')
    verify(folder, write=False)
    for arm in ('B', 'U'):
        for phase, checkpoint, predictions in (
            ('stage1', f'{arm}_stage1.pt', f'{arm}_old_tasks.npz'),
            ('stage2', f'{arm}_head.pt', f'{arm}_predictions.npz'),
        ):
            metrics = report[phase][arm]
            disk_metrics = json.loads((folder / f'{arm}_{phase}.json').read_text(encoding='utf-8'))
            if disk_metrics != metrics:
                raise ValueError(f'reuse metric/report mismatch: {arm}_{phase}')
            for name, key in ((checkpoint, 'checkpoint_sha256'),
                              (predictions, 'predictions_sha256')):
                if file_hash(folder / name) != metrics[key]:
                    raise ValueError(f'reuse artifact hash mismatch: {name}')
    return folder, report


def copy_baseline(folder, output, target):
    folder, report = checked_report(folder, target)
    for suffix in ('stage1.pt', 'stage1.json', 'old_tasks.npz',
                   'stage2.json', 'head.pt', 'predictions.npz'):
        shutil.copy2(folder / f'B_{suffix}', Path(output) / f'B_{suffix}')
    return report['stage1']['B'], report['stage2']['B']


def load_stage1(spec, folder, arm, read_policy, device):
    from universal_experiment.paired import make_stage1_model
    spec.P.seed_model(spec.model_seed)
    base = spec.build(device).to(device)
    universal = spec.make_u(base)[0].to(device) if arm == 'U' else None
    model = make_stage1_model(base, universal, arm, read_policy).to(device)
    model.load_state_dict(torch.load(Path(folder) / f'{arm}_stage1.pt',
                                    map_location=device, weights_only=True), strict=True)
    model.eval()
    return model, base, universal


def record_old_after(spec, model, output, device, before):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from universal_experiment.paired_run import predict_old_tasks, dump, file_hash
    raw, scores, deltas = {}, {}, {}
    for split in ('val', 'test'):
        labels, predictions = predict_old_tasks(model, spec.loaders[split], device)
        scores[split], deltas[split] = {}, {}
        for i, task in enumerate(spec.task_names):
            raw[f'{split}_y{i}'], raw[f'{split}_p{i}'] = labels[i], predictions[i]
            score = float(roc_auc_score(labels[i].astype(int), predictions[i]))
            scores[split][task] = score
            deltas[split][task] = score - before[split][task]
    path = Path(output) / 'U_old_tasks_after_stage2.npz'
    np.savez_compressed(path, **raw)
    result = dict(auc=scores, delta_from_stage1=deltas, predictions_sha256=file_hash(path))
    dump(Path(output) / 'U_old_tasks_after_stage2.json', result)
    return result
