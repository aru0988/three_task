"""Single-seed full-data short screen; immutable outputs, no test selection."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from census_benchmark import protocol as P
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import NewTask
from run_census_benchmark import build_mptrec, Stage1HookTrainManager
from universal_experiment.model import UniversalExpert, UniversalStage1, ResidualHead


EXPECTED_BASE = 'a12a5f5369a7002fb12f0ead7576e4dad3375a060ba2f90d45bfb1d566153f85'
EXPECTED_SPLIT = '096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c'


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def fit_normalizer(base, u, loader, device):
    # This iterator must not perturb the original Stage-1 random stream.
    with torch.random.fork_rng(devices=[]):
        total = torch.zeros(u.input_dim, dtype=torch.float64, device=device)
        sq = total.clone(); n = 0
        for _, _, _, features in loader:
            x = base.embedding_network({k: v.to(device) for k, v in features.items()}).double()
            total += x.sum(0); sq += x.square().sum(0); n += len(x)
        mean = total / n
        u.set_normalizer(mean.float(), (sq / n - mean.square()).clamp_min(0).sqrt().float())


@torch.no_grad()
def cache_split(base, u, random_u, loader, device, old_tasks):
    columns = {k: [] for k in ('x', 'g', 's0', 's1', 'u', 'r', 'y')}
    old = {k: [] for k in ('y0', 'y1', 'p0', 'p1')}
    base.eval(); u.eval(); random_u.eval()
    envs = None
    for y0, y1, y, features in loader:
        features = {k: v.to(device) for k, v in features.items()}
        x, g, specs, envs = base.get_infos(features)
        values = (x, g, specs[0], specs[1], u(x), random_u(x), y.float())
        for key, value in zip(columns, values):
            columns[key].append(value.detach().cpu())
        if old_tasks:
            p = base.predict(features)
            for key, value in zip(old, (y0, y1, p[0], p[1])):
                old[key].append(value.detach().cpu().reshape(-1))
    cache = {k: torch.cat(v) for k, v in columns.items()}
    old = {k: torch.cat(v).numpy() for k, v in old.items()} if old_tasks else {}
    return cache, [e.detach() for e in envs], old


def batches(cache, device, batch_size):
    for start in range(0, len(cache['y']), batch_size):
        yield {k: v[start:start + batch_size].to(device) for k, v in cache.items()}


def predict(head, batch, envs, arm):
    args = (batch['x'], batch['g'], [batch['s0'], batch['s1']], envs)
    if arm == 'B':
        return head(*args).reshape(-1)
    return head(*args, extra=batch[{'U': 'u', 'R': 'r', 'G': 'g'}[arm]])


@torch.no_grad()
def evaluate(head, cache, envs, arm, device, batch_size):
    head.eval()
    p = torch.cat([predict(head, b, envs, arm).cpu() for b in batches(cache, device, batch_size)]).numpy()
    return float(roc_auc_score(cache['y'].numpy(), p)), p


def train_head(head, caches, envs, arm, device, epochs, out, cpu_loss=False,
               *, lr=None, batch_size=None, patience=None):
    lr = P.LR if lr is None else lr
    batch_size = P.BATCH_SIZE if batch_size is None else batch_size
    patience = P.PATIENCE if patience is None else patience
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    best = -1.; state = None; stale = 0; history = []; best_epoch = 0
    sync(); start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        sync(); epoch_start = time.perf_counter()
        head.train(); total = 0.; steps = 0
        for b in batches(caches['train'], device, batch_size):
            opt.zero_grad()
            prediction = predict(head, b, envs, arm)
            loss = torch.nn.functional.binary_cross_entropy(prediction.cpu() if cpu_loss else prediction, b['y'].cpu() if cpu_loss else b['y']) + head.get_l2_reg()
            loss.backward(); opt.step()
            total += float(loss.detach()); steps += 1
        sync(); train_end = time.perf_counter()
        va, _ = evaluate(head, caches['val'], envs, arm, device, batch_size)
        sync(); val_end = time.perf_counter()
        history.append(dict(epoch=epoch, val_auc=va, loss=total / steps, steps=steps,
                            train_seconds=train_end-epoch_start, val_seconds=val_end-train_end,
                            cumulative_seconds=val_end-start))
        print(f'{arm}: epoch={epoch} val={va:.9f}', flush=True)
        if va > best:
            best = va; best_epoch = epoch; state = copy.deepcopy(head.state_dict()); stale = 0
        else:
            stale += 1
            if stale == patience:
                break
    head.load_state_dict(state)
    result = dict(best_epoch=best_epoch, epochs=history, parameters=sum(p.numel() for p in head.parameters()),
                  wall_seconds=time.perf_counter() - start)
    raw = {}
    for split in ('val', 'test'):
        score, preds = evaluate(head, caches[split], envs, arm, device, batch_size)
        result[split + '_auc'] = score
        raw[split + '_y'] = caches[split]['y'].numpy()
        raw[split + '_p'] = preds
    if arm == 'U':
        for split in ('val', 'test'):
            shuffled = dict(caches[split])
            order = torch.randperm(len(shuffled['y']), generator=torch.Generator().manual_seed(20261009))
            shuffled['u'] = shuffled['u'][order]
            score, preds = evaluate(head, shuffled, envs, arm, device, batch_size)
            result[split + '_shuffled_u_auc'] = score
            raw[split + '_shuffled_u_p'] = preds
    np.savez_compressed(out / f'{arm}_predictions.npz', **raw)
    torch.save(state, out / f'{arm}_head.pt')
    dump(out / f'{arm}_metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='original checkout holding dataset/artifacts')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    if dirty and not args.smoke:
        raise RuntimeError('Formal experiment requires a clean worktree')
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train = CensusIncomeDataset(str(args.source / 'dataset/Census-income/train.gz'), 'education')
    test = CensusIncomeDataset(str(args.source / 'dataset/Census-income/test.gz'), 'education')
    vi, ti = P.make_split(len(test), P.SPLIT_SEED)
    fp = P.split_fingerprint(split_seed=P.SPLIT_SEED, stats=P.split_stats(vi, ti, len(train)), val_idx=vi, test_idx=ti)
    assert fp['fingerprint_sha256'] == EXPECTED_SPLIT
    saved = np.load(args.source / 'artifacts/census_stage2/splits/20260929/split_indices.npz')
    assert np.array_equal(vi, saved['val']) and np.array_equal(ti, saved['test'])
    sets = dict(train=train, val=Subset(test, vi.tolist()), test=Subset(test, ti.tolist()))
    if args.smoke:
        sets = {k: Subset(v, list(range(1024))) for k, v in sets.items()}
    loaders = {k: DataLoader(v, batch_size=P.BATCH_SIZE, shuffle=False) for k, v in sets.items()}
    config = dict(commit=commit, dirty=dirty, model_seed=P.MODEL_SEED, env_seed=P.ENV_SEED,
                  split_seed=P.SPLIT_SEED, smoke=args.smoke, fingerprint=fp,
                  stage1_epochs=1 if args.smoke else 2, stage2_epochs=1 if args.smoke else 5,
                  source=str(args.source), created=datetime.now().isoformat(), device=str(device),
                  python=os.sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  data_sha256={n: file_hash(args.source / 'dataset/Census-income' / n) for n in ('train.gz', 'test.gz')})
    dump(args.out / 'config.json', config)
    np.savez_compressed(args.out / 'indices.npz', val=vi, test=ti)
    P.seed_model(P.MODEL_SEED)
    base = build_mptrec(device).to(device)
    feat = train[0][-1]
    assert 'education' not in feat
    widths = [1 for k in feat if k not in base.embedding_network.feature_names] + [P.EMBEDDING_SIZE] * len(base.embedding_network.feature_names)
    assert sum(widths) == P.INPUT_SIZE
    u = UniversalExpert(widths).to(device)
    fit_normalizer(base, u, loaders['train'], device)
    random_u = copy.deepcopy(u)
    model = UniversalStage1(base, u).to(device)
    env_ids = P.make_env_ids(len(sets['train']), P.ENV_SEED)
    manager = Stage1HookTrainManager(model, loaders['train'], loaders['val'], env_ids,
                                    ['Income', 'Marital'], P.LR, P.BATCH_SIZE,
                                    P.UNI_COE, P.ENV_COE, epochs=config['stage1_epochs'], patience=P.PATIENCE)
    print('Stage 1: supervised G/S plus detached masked-feature U', flush=True)
    manager.train_two_task()
    model.load_state_dict(manager.best_weight)
    base_hash = P.backbone_sha256(base)
    s1 = dict(base_hash=base_hash, matches_historical=base_hash == EXPECTED_BASE,
              selected_epoch=max(manager.epoch_records, key=lambda r: r['auc_val_income'] + r['auc_val_marital'])['epoch'],
              epochs=manager.epoch_records, clusters=manager.cluster_records,
              auxiliary_steps=model.loss_history, universal_hash=P.backbone_sha256(u))
    dump(args.out / 'stage1.json', s1)
    torch.save(model.state_dict(), args.out / 'stage1.pt')
    torch.save(random_u.state_dict(), args.out / 'random_u.pt')
    if not args.smoke and base_hash != EXPECTED_BASE:
        raise RuntimeError('G/S hash differs from historical baseline; stop and audit before Stage 2')
    for module in (base, u, random_u):
        module.eval()
        for p in module.parameters():
            p.grad = None; p.requires_grad_(False)
    caches = {}; old_raw = {}
    for split, loader in loaders.items():
        print(f'Caching frozen {split} representations', flush=True)
        caches[split], envs, old = cache_split(base, u, random_u, loader, device, split != 'train')
        for k, v in old.items():
            old_raw[f'{split}_{k}'] = v
    np.savez_compressed(args.out / 'old_tasks.npz', **old_raw)
    s1['old_task_auc'] = {s: {t: float(roc_auc_score(old_raw[f'{s}_y{i}'], old_raw[f'{s}_p{i}']))
                              for i, t in enumerate(('income', 'marital'))} for s in ('val', 'test')}
    z = caches['train']['u'].double(); g = caches['train']['g'].double()
    std = z.std(0, unbiased=False)
    zc = (z-z.mean(0)) / std.clamp_min(.01)
    gc = (g-g.mean(0)) / g.std(0, unbiased=False).clamp_min(.01)
    corr = zc.T @ gc / len(z)
    probe_mask = u.sample_mask(min(4096, len(z)))
    with torch.no_grad():
        diag = u.losses(caches['train']['x'][:4096].to(device), caches['train']['g'][:4096].to(device), probe_mask)
    mech = dict(u_std_mean=float(std.mean()), u_live_fraction=float((std > .01).double().mean()),
                ug_squared_correlation=float(corr.square().mean()),
                reconstruction_train_probe={k:float(v) for k,v in diag.items()})
    np.savez_compressed(args.out / 'mechanism.npz', u_std=std.numpy(), ug_correlation=corr.numpy(),
                        n=np.array(len(z)), u_sum=z.sum(0).numpy(), g_sum=g.sum(0).numpy(),
                        u_sq=z.square().sum(0).numpy(), g_sq=g.square().sum(0).numpy(), ug_sum=(z.T@g).numpy(),
                        probe_x=caches['train']['x'][:4096].numpy(), probe_g=g[:4096].float().numpy(),
                        probe_u=z[:4096].float().numpy(), probe_mask=probe_mask.numpy(), widths=np.array(widths))
    dump(args.out / 'stage1.json', s1); dump(args.out / 'mechanism.json', mech)
    del z, g, zc, gc
    # Match original stage2 initialization: seed -> construct MPTRec -> construct NewTask.
    P.seed_model(P.MODEL_SEED)
    dummy = build_mptrec(device)
    original = NewTask(P.INPUT_SIZE, P.EXPERT_HIDDEN[-1], list(P.TOWER_HIDDEN), P.REG_DNN, device).to(device)
    del dummy
    results = {}
    for arm in ('B', 'U', 'R', 'G'):
        head = copy.deepcopy(original) if arm == 'B' else ResidualHead(original, P.EXPERT_HIDDEN[-1]).to(device)
        results[arm] = train_head(head, caches, envs, arm, device, config['stage2_epochs'], args.out)
        del head
    assert results['U']['parameters'] == results['R']['parameters'] == results['G']['parameters']
    assert P.backbone_sha256(base) == base_hash
    assert all(p.grad is None for m in (base, u, random_u) for p in m.parameters())
    deltas = {a: {s: results['U'][s+'_auc'] - results[a][s+'_auc'] for s in ('val','test')} for a in ('B','R','G')}
    delta = deltas['B']['test']
    classification = 'positive' if delta >= .001 else 'clear_decline' if delta <= -.02 else 'no_clear_improvement'
    go = all(d['test'] >= .001 and d['val'] > 0 for d in deltas.values()) and mech['u_live_fraction'] >= .5 and float(diag['reconstruction']) < float(diag['zero'])
    report = dict(arms=results, u_minus=deltas, classification=classification, preregistered_go=go,
                  mechanism=mech, stage1_old_tasks=s1['old_task_auc'],
                  frozen_base_hash_after=P.backbone_sha256(base), frozen_u_hash_after=P.backbone_sha256(u),
                  frozen_gradients_none=all(p.grad is None for m in (base,u,random_u) for p in m.parameters()),
                  stage1_matches_historical=s1['matches_historical'], wall_seconds=time.perf_counter()-start,
                  prediction_hashes={p.name:file_hash(p) for p in args.out.glob('*.npz')})
    dump(args.out / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
