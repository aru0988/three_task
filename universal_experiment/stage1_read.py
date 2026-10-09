"""Stage-1 old-task read-only-U four-arm screen.

The four arms mirror the branch's Stage-2 B/U/R/G design, but move the extra-input
residual onto the Stage-1 old-task heads.  Inside `MPTRec.forward`, after each task's
fused representation is formed and before its tower:

    fused_rep = fused_rep + sigmoid(alpha_i) * proj_i(extra.detach())

`extra` differs by arm: B has no residual at all; U reads the learned UniversalExpert
output; R reads a frozen random twin (identical init, never trained); G reads the
general expert output (duplicate access).  The old-task loss can never reach the extra
source (detach), and the U encoder trains only through its own self-supervised losses
on detached embeddings.  20 epochs, no early stop, per-epoch validation AUCs recorded,
test evaluated once on the validation-selected checkpoint.
"""
import argparse
import copy
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from multitaskrec.model import ReverseLayerF
from multitaskrec.train import MPTRecTrainManager
from universal_experiment.model import UniversalExpert

EXPECTED_SPLIT = '096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c'


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


class ReadModel(nn.Module):
    """MPTRec forward replicated verbatim with one residual line added per task."""

    def __init__(self, base, extra_source, rep_dim, mode):
        super().__init__()
        assert mode in ('U', 'R', 'G')
        self.base = base
        self.mode = mode
        self.rep_dim = rep_dim
        self.universal = extra_source if mode == 'U' else None
        self.random_u = extra_source if mode == 'R' else None
        self.projections = nn.ModuleList()
        self.logits = nn.ParameterList()
        with torch.random.fork_rng(devices=[]):
            for _ in range(base.num_tasks):
                projection = nn.Linear(rep_dim, rep_dim, bias=False)
                with torch.no_grad():
                    projection.weight.copy_(torch.eye(rep_dim))
                self.projections.append(projection)
                self.logits.append(nn.Parameter(torch.tensor(math.log(.1 / .9))))
        self.loss_history = []
        self.auxiliary = None

    def _extra(self, dnn_input, gen_rep):
        if self.mode == 'G':
            return gen_rep
        source = self.universal if self.mode == 'U' else self.random_u
        with torch.no_grad():
            return source(dnn_input)

    def forward(self, x, alpha=1):
        base = self.base
        dnn_input = base.embedding_network(x)
        gen_rep = base.shared_expert_network(dnn_input)

        gen_preds = []
        for i in range(base.num_tasks):
            output = base.tower_networks[i](gen_rep)
            gen_preds.append(output.squeeze())

        gate_outs = []
        for gate in base.gate_networks:
            gate_outs.append(gate(dnn_input))

        extra = self._extra(dnn_input, gen_rep)

        fused_preds = []
        for i in range(base.num_tasks):
            spec_rep = base.specific_expert_networks[i](dnn_input)
            env_embedding = base.env_embedding_network(torch.tensor(i).to(base.device))
            env_aware_rep = spec_rep * env_embedding
            all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
            fused_rep = torch.matmul(all_reps, gate_outs[i].unsqueeze(dim=2)).squeeze()
            fused_rep = fused_rep + self.logits[i].sigmoid() * self.projections[i](extra.detach())
            output = base.tower_networks[i](fused_rep)
            fused_preds.append(output.squeeze())

        rev_gen_rep = ReverseLayerF.apply(gen_rep, alpha)
        env_pred = base.env_classifier(rev_gen_rep)

        if self.mode == 'U' and self.training and torch.is_grad_enabled():
            # Mirror the branch's detached-U SSL wiring: inputs recomputed under a
            # forked RNG so the extra forward cannot perturb base dropout streams.
            devices = [next(base.parameters()).device.index] if next(base.parameters()).is_cuda else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                embedded = base.embedding_network(x)
                general = base.shared_expert_network(embedded)
            self.auxiliary = self.universal.losses(embedded, general)
            self.loss_history.append({k: float(v.detach()) for k, v in self.auxiliary.items()})
        return {'gen_preds': gen_preds, 'fused_preds': fused_preds, 'env_pred': env_pred}

    def predict(self, x):
        return self.forward(x)['fused_preds']

    def cluster_predict(self, x):
        return self.predict(x)

    def get_l2_reg(self):
        reg = self.base.get_l2_reg()
        if self.mode == 'U':
            reg = reg + self.auxiliary['total']
        return reg


class ReadTrainManager(MPTRecTrainManager):
    """MPTRecTrainManager with per-epoch val-AUC/wall records; no early stop when patience>=epochs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_records = []
        self.cluster_records = []
        self._t0 = time.perf_counter()

    def evaluation_two_task(self, data_loader):
        aucs = super().evaluation_two_task(data_loader)
        if data_loader is self.val_loader:
            self.epoch_records.append({'epoch': len(self.epoch_records) + 1,
                                       'auc_val_0': float(aucs[0]), 'auc_val_1': float(aucs[1]),
                                       'wall_seconds': time.perf_counter() - self._t0})
        return aucs

    def cluster_2(self):
        previous = self.env_ids.clone()
        updated = super().cluster_2()
        counts = torch.bincount(updated, minlength=2).tolist()
        self.cluster_records.append({'epoch': len(self.epoch_records) + 1,
                                     'diff_num': int((previous != updated).sum()),
                                     'env_counts': [int(c) for c in counts]})
        return updated


def prepare(dataset, source, smoke, device):
    if dataset == 'census':
        from census_benchmark import protocol as P
        from multitaskrec.dataset import CensusIncomeDataset
        from run_census_benchmark import build_mptrec
        train = CensusIncomeDataset(str(source / 'dataset/Census-income/train.gz'), 'education')
        test = CensusIncomeDataset(str(source / 'dataset/Census-income/test.gz'), 'education')
        vi, ti = P.make_split(len(test), P.SPLIT_SEED)
        fp = P.split_fingerprint(split_seed=P.SPLIT_SEED, stats=P.split_stats(vi, ti, len(train)),
                                 val_idx=vi, test_idx=ti)
        assert fp['fingerprint_sha256'] == EXPECTED_SPLIT
        saved = np.load(source / 'artifacts/census_stage2/splits/20260929/split_indices.npz')
        assert np.array_equal(vi, saved['val']) and np.array_equal(ti, saved['test'])
        sets = dict(train=train, val=Subset(test, vi.tolist()), test=Subset(test, ti.tolist()))
        if smoke:
            sets = {k: Subset(v, list(range(1024))) for k, v in sets.items()}
        loaders = {k: DataLoader(v, batch_size=P.BATCH_SIZE, shuffle=False) for k, v in sets.items()}

        def make_u(base):
            feat = train[0][-1]
            assert 'education' not in feat
            widths = [1 for k in feat if k not in base.embedding_network.feature_names] + \
                     [P.EMBEDDING_SIZE] * len(base.embedding_network.feature_names)
            assert sum(widths) == P.INPUT_SIZE
            return UniversalExpert(widths).to(device), widths

        return SimpleNamespace(P=P, build=build_mptrec, sets=sets, loaders=loaders,
                               task_names=['Income', 'Marital'], lr=P.LR, batch_size=P.BATCH_SIZE,
                               uni_coe=P.UNI_COE, env_coe=P.ENV_COE, model_seed=P.MODEL_SEED,
                               env_seed=P.ENV_SEED, rep_dim=P.EXPERT_HIDDEN[-1], make_u=make_u,
                               fingerprints=fp)
    from aliccp_benchmark import protocol as A
    from universal_experiment.aliccp import build
    from multitaskrec.dataset import AliCCPDataset
    budgets = dict(zip(('train', 'val', 'test'), (20000, 5000, 10000) if smoke else (2000000, 500000, 1000000)))
    fp = json.loads((source / 'artifacts/aliccp_bench/splits/p2M-v500k-t1M/prefix_fingerprint.json').read_text())
    assert A.fingerprint_digest(fp) == fp['fingerprint_sha256']
    actual = {}
    sets, loaders = {}, {}
    for split, n in budgets.items():
        data_file = source / Path(A.DATA_FILES[split])
        h = A.prefix_sha256(data_file, n)
        counts = A.scan_label_counts(data_file, n)
        if not smoke:
            assert h == fp['files'][split]['prefix_sha256']
            assert A._normalize_counts(counts) == A._normalize_counts(fp['label_counts'][split])
        actual[split] = {'prefix_sha256': h, 'label_counts': counts}
        sets[split] = AliCCPDataset(str(data_file), n)
        A.verify_label_counts(sets[split], n, counts)
        assert set(sets[split][0][-1]) == set(A.build_vocab()) and '301' not in sets[split][0][-1]
        loaders[split] = DataLoader(sets[split], batch_size=2000, shuffle=False)

    def make_u(base):
        return UniversalExpert([5] * 16, rep_dim=64).to(device), [5] * 16

    return SimpleNamespace(P=A, build=build, sets=sets, loaders=loaders,
                           task_names=['CTR', 'CVR'], lr=A.LR, batch_size=2000,
                           uni_coe=A.UNI_COE, env_coe=A.ENV_COE, model_seed=A.MODEL_SEED,
                           env_seed=A.ENV_SEED, rep_dim=64, make_u=make_u,
                           fingerprints={'prefix_fingerprint': fp, 'actual': actual})


@torch.no_grad()
def predict_two_task(model, loader, device):
    model.eval()
    ys = [[], []]; ps = [[], []]
    for y_0, y_1, _, features in loader:
        features = {k: v.to(device) for k, v in features.items()}
        pred = model.predict(features)
        for i, (y, p) in enumerate(zip((y_0, y_1), pred)):
            ys[i].append(y.reshape(-1))
            ps[i].append(p.detach().cpu().reshape(-1))
    return [torch.cat(y).numpy() for y in ys], [torch.cat(p).numpy() for p in ps]


def run_arm(spec, arm, out, epochs, device):
    from universal_experiment.run import fit_normalizer
    P = spec.P
    P.seed_model(spec.model_seed)
    base = spec.build(device).to(device)
    extra_source = None
    if arm == 'B':
        model = base
    else:
        u, _widths = spec.make_u(base)
        fit_normalizer(base, u, spec.loaders['train'], device)
        if arm == 'U':
            extra_source = u
        elif arm == 'R':
            extra_source = copy.deepcopy(u)
            for p in extra_source.parameters():
                p.requires_grad_(False)
        model = ReadModel(base, extra_source, spec.rep_dim, mode=arm).to(device)
    env_ids = P.make_env_ids(len(spec.sets['train']), spec.env_seed)
    manager = ReadTrainManager(model, spec.loaders['train'], spec.loaders['val'], env_ids,
                               spec.task_names, spec.lr, spec.batch_size, spec.uni_coe, spec.env_coe,
                               epochs=epochs, patience=epochs)
    print(f'[{arm}] training {epochs} epochs (no early stop)', flush=True)
    manager.train_two_task()
    model.load_state_dict(manager.best_weight)
    records = manager.epoch_records
    selected = max(records, key=lambda r: r['auc_val_0'] + r['auc_val_1'])
    raw = {}
    scores = {}
    for split in ('val', 'test'):
        ys, ps = predict_two_task(model, spec.loaders[split], device)
        for i, (y, p) in enumerate(zip(ys, ps)):
            raw[f'{split}_y{i}'] = y
            raw[f'{split}_p{i}'] = p
            scores[f'{split}_{i}'] = float(roc_auc_score(y.astype(int), p))
    np.savez_compressed(out / f'{arm}_predictions.npz', **raw)
    aux_means = None
    if arm == 'U':
        steps = len(spec.loaders['train'])
        hist = model.loss_history
        keys = list(hist[0]) if hist else []
        aux_means = {k: [float(np.mean([h[k] for h in hist[e * steps:(e + 1) * steps]]))
                         for e in range(epochs)] for k in keys}
    core_params = sum(p.numel() for p in base.parameters())
    encoder_params = sum(p.numel() for p in extra_source.parameters()) if extra_source is not None else 0
    total_params = sum(p.numel() for p in model.parameters())
    metrics = dict(
        arm=arm, selected_epoch=selected['epoch'], epochs=records,
        val_auc={spec.task_names[i]: scores[f'val_{i}'] for i in (0, 1)},
        test_auc={spec.task_names[i]: scores[f'test_{i}'] for i in (0, 1)},
        train_loss_components=dict(uni_0=manager.uni_loss_0_list, uni_1=manager.uni_loss_1_list,
                                   fused_0=manager.fused_loss_0_list, fused_1=manager.fused_loss_1_list,
                                   env=manager.env_loss_list),
        cluster_records=manager.cluster_records,
        auxiliary_epoch_means=aux_means,
        wall_seconds=records[-1]['wall_seconds'],
        params=dict(total=total_params, core=core_params, encoder=encoder_params,
                    residual=total_params - core_params - encoder_params),
        base_hash=P.backbone_sha256(base),
        universal_hash=P.backbone_sha256(extra_source) if extra_source is not None else None,
    )
    if arm in ('U', 'R', 'G'):
        metrics['residual_gates'] = dict(
            alpha=[float(torch.sigmoid(l)) for l in model.logits],
            identity_deviation=[float((proj.weight - torch.eye(spec.rep_dim, device=proj.weight.device)).norm())
                                for proj in model.projections])
    dump(out / f'{arm}_metrics.json', metrics)
    torch.save(model.state_dict(), out / f'{arm}_model.pt')
    print(f'[{arm}] selected_epoch={selected["epoch"]} val={metrics["val_auc"]} '
          f'test={metrics["test_auc"]} wall={records[-1]["wall_seconds"]:.1f}s', flush=True)
    return metrics


def main(dataset):
    parser = argparse.ArgumentParser(prog=f'python -m universal_experiment {dataset} --stage1-read')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=('B', 'U', 'R', 'G', 'all'), default='all')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    argv = sys.argv[1:]
    if argv and argv[0] in ('census', 'aliccp'):
        argv = argv[1:]
    args = parser.parse_args(argv)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    if dirty and not args.smoke:
        raise RuntimeError('Formal experiment requires a clean worktree; commit first')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.out.mkdir(parents=True, exist_ok=False)
    spec = prepare(dataset, args.source, args.smoke, device)
    config = dict(dataset=dataset, mode='stage1-read-u', commit=commit, dirty=dirty, smoke=args.smoke,
                  epochs=args.epochs, patience=args.epochs, no_early_stop=True,
                  task_names=spec.task_names, lr=spec.lr, batch_size=spec.batch_size,
                  uni_coe=spec.uni_coe, env_coe=spec.env_coe, model_seed=spec.model_seed,
                  env_seed=spec.env_seed, rep_dim=spec.rep_dim, source=str(args.source),
                  rows={k: len(v) for k, v in spec.sets.items()},
                  fingerprints=spec.fingerprints, created=datetime.now().isoformat(),
                  device=str(device), torch=torch.__version__, cuda=torch.version.cuda,
                  python=sys.version)
    dump(args.out / 'config.json', config)
    arms = 'BURG' if args.arm == 'all' else args.arm
    summary = {}
    for arm in arms:
        summary[arm] = run_arm(spec, arm, args.out, args.epochs, device)
    if args.arm == 'all':
        deltas = {a: {t: {s: summary['U'][f'{s}_auc'][t] - summary[a][f'{s}_auc'][t]
                          for s in ('val', 'test')} for t in spec.task_names}
                  for a in ('B', 'R', 'G')}
        report = dict(dataset=dataset, epochs=args.epochs, task_names=spec.task_names,
                      arms={a: {'selected_epoch': summary[a]['selected_epoch'],
                                'val_auc': summary[a]['val_auc'], 'test_auc': summary[a]['test_auc'],
                                'wall_seconds': summary[a]['wall_seconds'],
                                'residual_gates': summary[a].get('residual_gates')} for a in arms},
                      read_minus=deltas,
                      note='old-task metrics; U/R/G read extra with detached gradient; '
                           'U encoder trained only by its self-supervised losses')
        dump(args.out / 'report.json', report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
