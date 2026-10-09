"""Read-U Stage-1 -> new-task transfer screen (preregistered 2026-10-10).

Four arms B/U/R/G each load their own frozen Stage-1 read-u base
(`results/universal/{dataset}-read-u-seed*/{arm}_model.pt` at the epoch named in
`{arm}_metrics.json`) and train the same original NewTask head, initialized from
one canonical `seed_model -> build -> NewTask` sequence and deep-copied per arm.
No arm feeds U/R/G to the new-task head: only `base.get_infos` outputs enter it.
The runner refuses a dirty formal worktree, a nonempty output directory, missing
or changed checkpoints, an inconsistent Stage-1 selection, and unequal head
initializations.  Selection is validation AUC under the declared tie rule; test
is evaluated once on the selected head.
"""
import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from multitaskrec.model import NewTask
from universal_experiment import run as C

ARMS = ('B', 'U', 'R', 'G')
EPOCHS, PATIENCE, SMOKE_EPOCHS = 20, 3, 2
TIE_RULE = ('best epoch = earliest epoch attaining the strictly maximum validation AUC; '
            'an epoch must strictly improve to replace the current best')
LOSS = 'BCELoss(pred.cpu(), target.float()) + head.get_l2_reg() (original NewTask path)'
HEAD_INIT = 'seed_model -> build -> NewTask, deep-copied per arm'
TASK_NAMES = {'census': 'Education', 'aliccp': 'BSI'}
BASE_PREFIX = {'B': '', 'U': 'base.', 'R': 'base.', 'G': 'base.'}
ENCODER_PREFIX = {'U': 'universal.', 'R': 'random_u.'}


def selected_stage1_epoch(records):
    """First epoch attaining the strictly maximum old-task validation sum.

    Matches both `ReadTrainManager` records and `MPTRecTrainManager`'s strict-`>`
    best-weight update: ties resolve to the earliest epoch.
    """
    best = max(r['auc_val_0'] + r['auc_val_1'] for r in records)
    return min(r['epoch'] for r in records if r['auc_val_0'] + r['auc_val_1'] == best)


def load_arm_base(build, protocol, stage1_dir, arm, device, make_u=None):
    """Load one arm's own frozen Stage-1 base; refuse missing/changed artifacts."""
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm!r}')
    stage1_dir = Path(stage1_dir)
    metrics_path = stage1_dir / f'{arm}_metrics.json'
    checkpoint_path = stage1_dir / f'{arm}_model.pt'
    for path in (metrics_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f'missing Stage-1 artifact: {path}')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    selected = selected_stage1_epoch(metrics['epochs'])
    if selected != metrics['selected_epoch']:
        raise RuntimeError(f'{arm}: metrics selected_epoch={metrics["selected_epoch"]} inconsistent '
                           f'with its epoch history (first maximum = {selected})')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    prefix = BASE_PREFIX[arm]
    base_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)} if prefix else dict(state)
    if not base_state:
        raise RuntimeError(f'{arm}: checkpoint {checkpoint_path} holds no base weights (prefix {prefix!r})')
    base = build(device).to(device)
    base.load_state_dict(base_state)
    base_hash = protocol.backbone_sha256(base)
    if base_hash != metrics['base_hash']:
        raise RuntimeError(f'{arm}: loaded base hash {base_hash} != recorded {metrics["base_hash"]} '
                           f'({checkpoint_path})')
    info = dict(checkpoint_file=checkpoint_path.name,
                checkpoint_sha256=C.file_hash(checkpoint_path),
                base_hash_recorded=metrics['base_hash'], base_hash_loaded=base_hash,
                selected_epoch=int(metrics['selected_epoch']),
                stage1_wall_seconds=float(metrics.get('wall_seconds', float('nan'))),
                stage1_old_tasks_val=metrics.get('val_auc'), stage1_old_tasks_test=metrics.get('test_auc'),
                encoder_hash_recorded=metrics.get('universal_hash'), encoder_hash_loaded=None)
    if arm in ('U', 'R'):
        if make_u is None:
            raise ValueError(f'{arm} needs make_u to verify its auxiliary encoder')
        encoder, _widths = make_u(base)
        encoder.to(device)
        key = ENCODER_PREFIX[arm]
        encoder_state = {k[len(key):]: v for k, v in state.items() if k.startswith(key)}
        if not encoder_state:
            raise RuntimeError(f'{arm}: checkpoint {checkpoint_path} holds no {key!r} encoder weights')
        encoder.load_state_dict(encoder_state)
        encoder_hash = protocol.backbone_sha256(encoder)
        if encoder_hash != metrics.get('universal_hash'):
            raise RuntimeError(f'{arm}: loaded encoder hash {encoder_hash} != recorded '
                               f'{metrics.get("universal_hash")} ({checkpoint_path})')
        info['encoder_hash_loaded'] = encoder_hash
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False); param.grad = None
    return base, info


def canonical_head_init(protocol, build, *, model_seed, input_size, rep_dim, tower_hidden, reg_dnn, device):
    """The original stage-2 init sequence; returns (head, parameter hash)."""
    protocol.seed_model(model_seed)
    dummy = build(device)
    head = NewTask(input_size, rep_dim, list(tower_hidden), reg_dnn, device).to(device)
    del dummy
    return head, protocol.backbone_sha256(head)


def build_arm_heads(protocol, original, init_hash, arms=ARMS):
    """One deep copy of the pinned head per arm; refuse any unequal initialization."""
    heads = {arm: copy.deepcopy(original) for arm in arms}
    hashes = {arm: protocol.backbone_sha256(head) for arm, head in heads.items()}
    if any(value != init_hash for value in hashes.values()):
        raise RuntimeError(f'head initialization differs across arms: {hashes} != {init_hash}')
    return heads, hashes


@torch.no_grad()
def predict_newtask(head, base, loader, device):
    """Evaluate the head on the original G/S+environment path of the frozen base."""
    head.eval(); base.eval()
    labels, preds = [], []
    for _, _, y, features in loader:
        features = {k: v.to(device) for k, v in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = base.get_infos(features)
        pred = head(dnn_input, gen_rep, spec_reps, env_embs)
        labels.append(y.reshape(-1))
        preds.append(pred.detach().cpu().reshape(-1))
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def train_arm(base, head, spec, arm, out, *, epochs, patience, device, provenance, task_name):
    """Train one arm's head on its own frozen base; select on validation AUC."""
    out = Path(out)
    protocol = spec.P
    protocol.seed_model(spec.model_seed)  # per-arm reseed: arm training is reproducible standalone
    base_hash_before = protocol.backbone_sha256(base)
    optimizer = torch.optim.Adam(head.parameters(), lr=spec.lr)
    loss_func = torch.nn.BCELoss()
    best, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train()
        total, steps = 0.0, 0
        for _, _, y, features in spec.loaders['train']:
            features = {k: v.to(device) for k, v in features.items()}
            with torch.no_grad():
                dnn_input, gen_rep, spec_reps, env_embs = base.get_infos(features)
            pred = head(dnn_input, gen_rep, spec_reps, env_embs)
            loss = loss_func(pred.cpu(), y.float()) + head.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()); steps += 1
        y_val, p_val = predict_newtask(head, base, spec.loaders['val'], device)
        val_auc = float(roc_auc_score(y_val.astype(int), p_val))
        history.append(dict(epoch=epoch, loss=total / max(steps, 1), steps=steps, val_auc=val_auc,
                            wall_seconds=time.perf_counter() - start))
        print(f'[{arm}] epoch={epoch} loss={history[-1]["loss"]:.6f} val_auc={val_auc:.9f}', flush=True)
        if val_auc > best:
            best, best_epoch, stale = val_auc, epoch, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
            if stale == patience:
                print(f'[{arm}] early stop after epoch {epoch} ({patience} stale epochs)', flush=True)
                break
    head.load_state_dict(best_state)
    raw, scores = {}, {}
    for split in ('val', 'test'):
        y_split, p_split = predict_newtask(head, base, spec.loaders[split], device)
        raw[f'{split}_y'], raw[f'{split}_p'] = y_split, p_split
        scores[split] = float(roc_auc_score(y_split.astype(int), p_split))
    np.savez_compressed(out / f'{arm}_predictions.npz', **raw)
    torch.save(head.state_dict(), out / f'{arm}_head.pt')
    head_state_sha256 = C.file_hash(out / f'{arm}_head.pt')
    base_hash_after = protocol.backbone_sha256(base)
    if base_hash_after != base_hash_before:
        raise RuntimeError(f'{arm}: frozen base changed during head training')
    metrics = dict(arm=arm, task=task_name, best_epoch=best_epoch, val_auc=scores['val'], test_auc=scores['test'],
                   epochs=history, epoch_count=len(history),
                   parameters=sum(p.numel() for p in head.parameters()),
                   head_init_hash=provenance['head_init_hash'],
                   head_state_sha256=head_state_sha256,
                   val_n=int(len(raw['val_y'])), test_n=int(len(raw['test_y'])),
                   base_hash=provenance['base_hash_loaded'], base_hash_recorded=provenance['base_hash_recorded'],
                   frozen_base_hash_before=base_hash_before, frozen_base_hash_after=base_hash_after,
                   frozen_gradients_none=all(p.grad is None for p in base.parameters()),
                   lr=spec.lr, batch_size=spec.batch_size, epoch_budget=epochs, patience=patience,
                   tie_rule=TIE_RULE, loss=LOSS,
                   checkpoint_file=provenance['checkpoint_file'], checkpoint_sha256=provenance['checkpoint_sha256'],
                   stage1_selected_epoch=provenance['selected_epoch'],
                   stage1_wall_seconds=provenance['stage1_wall_seconds'],
                   stage1_old_tasks_val=provenance['stage1_old_tasks_val'],
                   stage1_old_tasks_test=provenance['stage1_old_tasks_test'],
                   encoder_hash_recorded=provenance['encoder_hash_recorded'],
                   encoder_hash_loaded=provenance['encoder_hash_loaded'],
                   stage2_wall_seconds=time.perf_counter() - start)
    C.dump(out / f'{arm}_metrics.json', metrics)
    print(f'[{arm}] best_epoch={best_epoch} val_auc={scores["val"]:.9f} test_auc={scores["test"]:.9f} '
          f'wall={metrics["stage2_wall_seconds"]:.1f}s', flush=True)
    return metrics


def classify_delta(delta):
    """Project classification rule: positive / no clear improvement / clear decline."""
    if delta >= 0.001:
        return 'positive'
    return 'clear_decline' if delta <= -0.02 else 'no_clear_improvement'


def delta_verdict(deltas):
    """Frozen decision rule: positive only if U-B and U-R test >= +0.001 with positive val deltas."""
    per_arm = {arm: dict(val=deltas[arm]['val'], test=deltas[arm]['test'],
                         classification=classify_delta(deltas[arm]['test'])) for arm in ('B', 'R', 'G')}
    test_ok = all(deltas[arm]['test'] >= 0.001 for arm in ('B', 'R'))
    val_ok = all(deltas[arm]['val'] > 0 for arm in ('B', 'R'))
    return dict(per_arm=per_arm, decision='POSITIVE_TRANSFER_SIGNAL' if (test_ok and val_ok) else 'NO_GO',
                unstable=bool(test_ok and not val_ok),
                note='U-B is the complete Stage-1 design effect; U-R the learned-content control; '
                     'U-G secondary specificity; single seed, not a generalization claim')


def run(spec, dataset, stage1_dir, out, *, source=None, commit='unknown', dirty=True, smoke=False,
        epochs=EPOCHS, patience=PATIENCE, device=None):
    """Execute the four-arm screen and write config.json / per-arm artifacts / report.json."""
    if dirty and not smoke:
        raise RuntimeError('formal run requires a clean worktree; commit first')
    if smoke and epochs > SMOKE_EPOCHS:
        raise RuntimeError(f'smoke runs use at most {SMOKE_EPOCHS} epochs, got {epochs}')
    out, stage1_dir = Path(out), Path(stage1_dir)
    if out.exists():
        if any(out.iterdir()):
            raise RuntimeError(f'output directory is not empty: {out}')
    else:
        out.mkdir(parents=True)
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    task_name = TASK_NAMES.get(dataset, dataset)
    start = time.perf_counter()
    loaded = {arm: load_arm_base(spec.build, spec.P, stage1_dir, arm, device, make_u=spec.make_u)
              for arm in ARMS}
    base_hashes = [info['base_hash_loaded'] for _, info in loaded.values()]
    base_hashes_distinct = len(set(base_hashes)) == len(ARMS)
    if not base_hashes_distinct:
        raise RuntimeError('the four arms did not load four distinct Stage-1 bases')
    original, init_hash = canonical_head_init(spec.P, spec.build, model_seed=spec.model_seed,
                                              input_size=spec.P.INPUT_SIZE, rep_dim=spec.rep_dim,
                                              tower_hidden=list(spec.P.TOWER_HIDDEN),
                                              reg_dnn=spec.P.REG_DNN, device=device)
    heads, init_hashes = build_arm_heads(spec.P, original, init_hash)
    del original
    for arm in ARMS:
        loaded[arm][1]['head_init_hash'] = init_hashes[arm]
    rows = {name: len(data) for name, data in spec.sets.items()}
    config = dict(dataset=dataset, mode='newtask-read-u', task_name=task_name,
                  commit=commit, dirty=dirty, smoke=smoke, epochs=epochs, patience=patience, tie_rule=TIE_RULE,
                  loss=LOSS, head_init_protocol=HEAD_INIT, head_init_hash=init_hash, head_init_hashes=init_hashes,
                  head_parameters=sum(p.numel() for p in heads['B'].parameters()),
                  model_seed=spec.model_seed, lr=spec.lr, batch_size=spec.batch_size,
                  input_size=spec.P.INPUT_SIZE, rep_dim=spec.rep_dim,
                  tower_hidden=list(spec.P.TOWER_HIDDEN), reg_dnn=spec.P.REG_DNN,
                  frozen_base=True, extra_head_input=False, per_arm_reseed=True,
                  rows=rows, split_seed=getattr(spec.P, 'SPLIT_SEED', None),
                  stage1_dir=str(stage1_dir), stage1_task_names=list(spec.task_names),
                  stage1_files={arm: {key: info[key] for key in
                                      ('checkpoint_file', 'checkpoint_sha256', 'base_hash_recorded',
                                       'base_hash_loaded', 'selected_epoch', 'encoder_hash_recorded',
                                       'encoder_hash_loaded')}
                                for arm, (_, info) in loaded.items()},
                  base_hashes_distinct=base_hashes_distinct,
                  fingerprints=spec.fingerprints, source=str(source) if source else None,
                  created=datetime.now().isoformat(), device=str(device), torch=torch.__version__,
                  cuda=torch.version.cuda, python=sys.version,
                  preflight_seconds=time.perf_counter() - start)
    C.dump(out / 'config.json', config)
    if not smoke and hasattr(spec.sets['val'], 'indices') and hasattr(spec.sets['test'], 'indices'):
        np.savez_compressed(out / 'indices.npz',
                            val=np.asarray(spec.sets['val'].indices, dtype=np.int64),
                            test=np.asarray(spec.sets['test'].indices, dtype=np.int64))
    results = {}
    for arm in ARMS:
        print(f'[{arm}] base {loaded[arm][1]["base_hash_loaded"][:12]} at Stage-1 epoch '
              f'{loaded[arm][1]["selected_epoch"]}; head init {init_hashes[arm][:12]}', flush=True)
        base, info = loaded[arm]
        results[arm] = train_arm(base, heads.pop(arm), spec, arm, out, epochs=epochs, patience=patience,
                                 device=device, provenance=info, task_name=task_name)
        del base
    deltas = {arm: {split: results['U'][f'{split}_auc'] - results[arm][f'{split}_auc']
                    for split in ('val', 'test')} for arm in ('B', 'R', 'G')}
    verdict = delta_verdict(deltas)
    report = dict(dataset=dataset, mode='newtask-read-u', task_name=task_name, commit=commit,
                  dirty=dirty, smoke=smoke, epochs=epochs, patience=patience, tie_rule=TIE_RULE,
                  head_init_hash=init_hash, head_init_hashes=init_hashes,
                  head_parameters=config['head_parameters'],
                  arms={arm: results[arm] for arm in ARMS},
                  u_minus={arm: {split: deltas[arm][split] for split in ('val', 'test')}
                           for arm in ('B', 'R', 'G')},
                  verdict=verdict,
                  old_task_context_note='Stage-1 old-task AUCs copied from each arm\'s Stage-1 metrics as '
                                        'contextual trade-offs (the verifier re-derives them from the Stage-1 '
                                        'raw predictions); not Stage-2 measurements',
                  stage1_wall_seconds={arm: loaded[arm][1]['stage1_wall_seconds'] for arm in ARMS},
                  stage2_wall_seconds={arm: results[arm]['stage2_wall_seconds'] for arm in ARMS},
                  frozen_base_gradients_none=all(results[arm]['frozen_gradients_none'] for arm in ARMS),
                  frozen_base_hashes_unchanged=all(results[arm]['frozen_base_hash_after']
                                                   == results[arm]['frozen_base_hash_before'] for arm in ARMS),
                  base_hashes_distinct=base_hashes_distinct,
                  prediction_hashes={p.name: C.file_hash(p) for p in sorted(out.glob('*.npz'))},
                  total_wall_seconds=time.perf_counter() - start)
    C.dump(out / 'report.json', report)
    print(json.dumps({arm: (results[arm]['best_epoch'], results[arm]['val_auc'], results[arm]['test_auc'])
                      for arm in ARMS}, indent=2), flush=True)
    print(json.dumps(deltas, indent=2), flush=True)
    print(json.dumps(verdict, indent=2), flush=True)
    return report


def main(dataset):
    parser = argparse.ArgumentParser(prog=f'python -m universal_experiment {dataset} --newtask-read')
    parser.add_argument('--source', type=Path, required=True, help='checkout holding dataset/ and artifacts/')
    parser.add_argument('--stage1', type=Path, required=True,
                        help='results/universal/{dataset}-read-u-seed* directory holding the four arms')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    argv = sys.argv[1:]
    if argv and argv[0] in ('census', 'aliccp'):
        argv = argv[1:]
    args = parser.parse_args(argv)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    if dirty and not args.smoke:
        raise RuntimeError('formal run requires a clean worktree; commit first')
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f'output directory is not empty: {args.out}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    from universal_experiment.stage1_read import prepare
    spec = prepare(dataset, args.source, args.smoke, device)
    epochs = SMOKE_EPOCHS if args.smoke else EPOCHS
    run(spec, dataset, args.stage1, args.out, source=args.source, commit=commit, dirty=dirty,
        smoke=args.smoke, epochs=epochs, patience=PATIENCE, device=device)


if __name__ == '__main__':
    main(sys.argv[1])
