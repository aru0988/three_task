"""Longer Stage-2 budget, reusing the original frozen Stage-1 experiment."""
import argparse
import copy
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader, Subset

from universal_experiment import run as C
from universal_experiment.model import UniversalExpert, UniversalStage1, ResidualHead
from multitaskrec.model import NewTask


def main(dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Commit implementation before a formal run')
    old = json.loads((args.previous/'report.json').read_text())
    old_config = json.loads((args.previous/'config.json').read_text())
    stage1 = json.loads((args.previous/'stage1.json').read_text())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if dataset == 'census':
        from census_benchmark import protocol as P
        build = C.build_mptrec
    else:
        from aliccp_benchmark import protocol as P
        from universal_experiment.aliccp import build
    widths = np.load(args.previous/'mechanism.npz')['widths'].tolist()
    base = build(device).to(device)
    u = UniversalExpert(widths, rep_dim=P.EXPERT_HIDDEN[-1]).to(device)
    wrapper = UniversalStage1(base, u).to(device)
    wrapper.load_state_dict(torch.load(args.previous/'stage1.pt', map_location=device, weights_only=True))
    random_u = copy.deepcopy(u)
    random_u.load_state_dict(torch.load(args.previous/'random_u.pt', map_location=device, weights_only=True))
    assert P.backbone_sha256(base) == stage1['base_hash']
    assert P.backbone_sha256(u) == stage1['universal_hash']
    for module in (base, u, random_u):
        module.eval().requires_grad_(False)
    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(dataset=dataset, previous=str(args.previous), source=str(args.source),
                  model_seed=old_config['model_seed'], epochs=20, patience=3,
                  batch_size=P.BATCH_SIZE, lr=P.LR, stage1_retrained=False,
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  dirty=False, created=datetime.now().isoformat(), device=str(device),
                  stage1_base=stage1['base_hash'], stage1_u=stage1['universal_hash'],
                  torch=torch.__version__, arm_order=['B','U','R','G'],
                  timing_note='cached Stage-2; single sequential run, not an end-to-end speed claim')
    C.dump(args.out/'config.json', config)
    start = time.perf_counter()
    caches = {}
    if dataset == 'aliccp':
        for split in ('train','val','test'):
            caches[split] = {k: torch.from_numpy(np.load(args.previous/'cache'/f'{split}_{k}.npy', mmap_mode='c'))
                             for k in ('x','g','s0','s1','u','r','y')}
        # Check cache reuse against saved predictions and a small encoder probe.
        for split in ('val','test'):
            raw = np.load(args.previous/'B_predictions.npz')
            assert np.array_equal(caches[split]['y'].numpy(), raw[split+'_y'])
        with torch.no_grad():
            probe = caches['train']
            assert torch.allclose(u(probe['x'][:128].to(device)).cpu(), probe['u'][:128], atol=2e-5, rtol=2e-5)
    else:
        from multitaskrec.dataset import CensusIncomeDataset
        train = CensusIncomeDataset(str(args.source/'dataset/Census-income/train.gz'), 'education')
        test = CensusIncomeDataset(str(args.source/'dataset/Census-income/test.gz'), 'education')
        indices = np.load(args.previous/'indices.npz')
        sets = dict(train=train, val=Subset(test, indices['val'].tolist()), test=Subset(test, indices['test'].tolist()))
        for split, data in sets.items():
            print('Caching '+split, flush=True)
            caches[split], _, _ = C.cache_split(base, u, random_u, DataLoader(data,batch_size=P.BATCH_SIZE,shuffle=False),device,False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    config['cache_preparation_seconds'] = time.perf_counter()-start
    config['rows'] = {s:len(c['y']) for s,c in caches.items()}
    C.dump(args.out/'config.json', config)
    with torch.no_grad():
        envs = [base.env_embedding_network(torch.tensor(i,device=device)).detach() for i in range(2)]
    P.seed_model(old_config['model_seed'])
    dummy = build(device)
    original = NewTask(P.INPUT_SIZE,P.EXPERT_HIDDEN[-1],list(P.TOWER_HIDDEN),P.REG_DNN,device).to(device)
    del dummy
    results = {}
    for arm in ('B','U','R','G'):
        head = copy.deepcopy(original) if arm == 'B' else ResidualHead(original,P.EXPERT_HIDDEN[-1]).to(device)
        results[arm] = C.train_head(head,caches,envs,arm,device,20,args.out,cpu_loss=dataset=='aliccp',
                                    lr=P.LR,batch_size=P.BATCH_SIZE,patience=3)
        metric = results[arm]
        reached = [e for e in metric['epochs'] if e['val_auc'] >= old['arms']['B']['val_auc']]
        metric['historical_baseline_val_target'] = old['arms']['B']['val_auc']
        metric['first_target_epoch'] = reached[0]['epoch'] if reached else None
        metric['first_target_seconds'] = reached[0]['cumulative_seconds'] if reached else None
        metric['train_seconds'] = sum(e['train_seconds'] for e in metric['epochs'])
        metric['val_seconds'] = sum(e['val_seconds'] for e in metric['epochs'])
        metric['frozen_base_parameters'] = sum(p.numel() for p in base.parameters())
        metric['additional_encoder_parameters'] = sum(p.numel() for p in u.encoder.parameters()) if arm in ('U','R') else 0
        metric['retained_base_plus_head_plus_encoder_parameters'] = metric['parameters']+metric['frozen_base_parameters']+metric['additional_encoder_parameters']
        C.dump(args.out/f'{arm}_metrics.json', metric)
        del head
    assert P.backbone_sha256(base) == stage1['base_hash']
    assert P.backbone_sha256(u) == stage1['universal_hash']
    deltas = {a:{s:results['U'][s+'_auc']-results[a][s+'_auc'] for s in ('val','test')} for a in ('B','R','G')}
    delta = deltas['B']['test']
    report = dict(arms=results,u_minus=deltas,stage1_reused=True,stage1_frozen=True,
                  classification='positive' if delta>=.001 else 'clear_decline' if delta<=-.02 else 'no_clear_improvement',
                  incremental_signal=all(d['test']>=.001 and d['val']>0 for d in deltas.values()),
                  stability='single seed; not evaluated')
    C.dump(args.out/'report.json',report)
    print(json.dumps(deltas),flush=True)
