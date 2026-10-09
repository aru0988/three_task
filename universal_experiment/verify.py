"""Independent file parser: rank-statistic AUC, sufficient-statistic mechanisms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def checkpoint_audit(path, mech, report, stage1, checks):
    # Rebuild the probe with NumPy, not UniversalExpert.forward/losses.
    import torch
    state = torch.load(path/'stage1.pt', map_location='cpu', weights_only=True)
    u = {k.removeprefix('universal.'): v.numpy() for k,v in state.items() if k.startswith('universal.')}
    x = np.clip((mech['probe_x']-u['mean'])/u['scale'], -10, 10)
    mask = mech['probe_mask']; expanded = np.repeat(mask, mech['widths'], axis=1)
    def encode(a):
        h = np.maximum(a @ u['encoder.0.weight'].T + u['encoder.0.bias'], 0)
        return h @ u['encoder.2.weight'].T + u['encoder.2.bias']
    clean = encode(np.concatenate([x,np.zeros_like(mask,dtype=np.float32)],axis=1))
    hidden = encode(np.concatenate([np.where(expanded,0,x),mask.astype(np.float32)],axis=1))
    pred = hidden @ u['decoder.weight'].T + u['decoder.bias']
    ends = np.cumsum(mech['widths'])[:-1]
    errors = np.stack([a.mean(1) for a in np.split((pred-x)**2,ends,axis=1)],axis=1)
    zeros = np.stack([a.mean(1) for a in np.split(x*x,ends,axis=1)],axis=1)
    expected = report['mechanism']['reconstruction_train_probe']
    checks['probe_clean_encoding'] = np.allclose(clean,mech['probe_u'],atol=2e-5,rtol=2e-5)
    checks['probe_reconstruction'] = bool(np.isclose((errors*mask).sum()/mask.sum(),expected['reconstruction'],atol=2e-6,rtol=2e-5))
    checks['probe_zero'] = bool(np.isclose((zeros*mask).sum()/mask.sum(),expected['zero'],atol=2e-6,rtol=2e-5))
    checks['probe_mask_count'] = bool(np.all(mask.sum(1)==round(.3*len(mech['widths']))))
    for prefix,key in [('base.','base_hash'),('universal.','universal_hash')]:
        digest = hashlib.sha256()
        for name,t in sorted(state.items()):
            if not name.startswith(prefix) or (prefix=='universal.' and name.removeprefix(prefix) in ('mean','scale')) or name=='base.env_indices':
                continue
            h = hashlib.sha256(str(t.dtype).encode()+str(tuple(t.shape)).encode()+t.contiguous().numpy().tobytes()).hexdigest()
            digest.update(name.removeprefix(prefix).encode()); digest.update(h.encode())
        checks['checkpoint_'+key] = digest.hexdigest()==stage1[key]
    checks['checkpoint_env_buffer'] = np.array_equal(state['base.env_indices'].numpy(),[0,1])


def rank_auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    assert y.shape == p.shape and np.isfinite(p).all()
    assert np.isin(y, [0, 1]).all() and ((p >= 0) & (p <= 1)).all()
    pos = y == 1; n1 = int(pos.sum()); n0 = len(y)-n1
    assert n1 > 0 and n0 > 0
    return float((rankdata(p)[pos].sum() - n1*(n1+1)/2) / (n1*n0))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    path = parser.parse_args().run
    report = json.loads((path/'report.json').read_text())
    config = json.loads((path/'config.json').read_text())
    stage1 = json.loads((path/'stage1.json').read_text())
    checks = {}; scores = {}; labels = {}
    for name, expected in report['prediction_hashes'].items():
        checks['hash_'+name] = hashlib.sha256((path/name).read_bytes()).hexdigest() == expected
    for arm in ('B','U','R','G'):
        data = np.load(path/f'{arm}_predictions.npz')
        scores[arm] = {}
        for split in ('val','test'):
            y, p = data[split+'_y'],data[split+'_p']
            score = rank_auc(y,p); scores[arm][split]=score
            checks[f'{arm}_{split}_auc'] = abs(score-report['arms'][arm][split+'_auc']) < 1e-12
            checks[f'{arm}_{split}_n'] = len(y) == (1024 if config['smoke'] else 49881)
            if split in labels:
                checks[f'{arm}_{split}_labels'] = np.array_equal(y, labels[split])
            labels[split] = y
            if arm == 'U':
                checks[f'shuffled_{split}'] = abs(rank_auc(y,data[split+'_shuffled_u_p']) - report['arms'][arm][split+'_shuffled_u_auc']) < 1e-12
        hist = report['arms'][arm]['epochs']
        selected = max(hist, key=lambda h:h['val_auc'])
        checks[arm+'_selection'] = selected['epoch'] == report['arms'][arm]['best_epoch'] and abs(selected['val_auc']-scores[arm]['val'])<1e-12
    old = np.load(path/'old_tasks.npz')
    for s in ('val','test'):
        for i,t in enumerate(('income','marital')):
            checks[f'old_{s}_{t}'] = abs(rank_auc(old[f'{s}_y{i}'],old[f'{s}_p{i}']) - report['stage1_old_tasks'][s][t]) < 1e-12
    mech = np.load(path/'mechanism.npz'); n = int(mech['n'])
    us = np.sqrt(np.maximum(mech['u_sq']/n-(mech['u_sum']/n)**2,0))
    gs = np.sqrt(np.maximum(mech['g_sq']/n-(mech['g_sum']/n)**2,0))
    corr = (mech['ug_sum']/n-np.outer(mech['u_sum']/n,mech['g_sum']/n))/np.outer(np.maximum(us,.01),np.maximum(gs,.01))
    checks['u_std'] = np.allclose(us,mech['u_std'],atol=1e-8,rtol=1e-8)
    checks['ug_corr'] = np.allclose(corr,mech['ug_correlation'],atol=1e-8,rtol=1e-8)
    checks['live_fraction'] = abs(float(np.mean(us > .01))-report['mechanism']['u_live_fraction']) < 1e-12
    checkpoint_audit(path,mech,report,stage1,checks)
    checks['canonical_seeds'] = (config['model_seed'],config['env_seed'],config['split_seed']) == (1685480945,20260929,20260929)
    checks['budget'] = config['stage1_epochs']==2 and config['stage2_epochs']==5 and all(len(report['arms'][a]['epochs'])<=5 and all(e['steps']==780 for e in report['arms'][a]['epochs']) for a in scores)
    checks['old_historical_auc'] = all(abs(report['stage1_old_tasks'][s][t]-v)<1e-12 for s,t,v in [('val','income',.9373971773636002),('val','marital',.9909744586701101),('test','income',.9381688583685694),('test','marital',.9908426142246791)])
    checks['baseline_historical_auc'] = abs(scores['B']['val']-.8527881905614896)<1e-12 and abs(scores['B']['test']-.8500685307175756)<1e-12
    checks['positive_runtime'] = report['wall_seconds']>0 and all(report['arms'][a]['wall_seconds']>0 for a in scores)
    expected_go = all(scores['U']['test']-scores[a]['test']>=.001 and scores['U']['val']>scores[a]['val'] for a in ('B','R','G')) and float(np.mean(us>.01))>=.5 and report['mechanism']['reconstruction_train_probe']['reconstruction']<report['mechanism']['reconstruction_train_probe']['zero'] and checks['old_historical_auc']
    checks['decision'] = bool(expected_go)==report['preregistered_go']
    checks['capacity'] = len({report['arms'][a]['parameters'] for a in ('U','R','G')}) == 1
    checks['freeze_base'] = report['frozen_base_hash_after'] == stage1['base_hash']
    checks['freeze_u'] = report['frozen_u_hash_after'] == stage1['universal_hash']
    checks['no_frozen_grads'] = report['frozen_gradients_none']
    checks['clean_formal_commit'] = not config['dirty'] or config['smoke']
    for a in ('B','R','G'):
        for s in ('val','test'):
            checks[f'delta_{a}_{s}'] = abs(scores['U'][s]-scores[a][s]-report['u_minus'][a][s])<1e-12
    idx = np.load(path/'indices.npz')
    checks['split_disjoint_complete'] = len(np.intersect1d(idx['val'],idx['test'])) == 0 and np.array_equal(np.sort(np.concatenate([idx['val'],idx['test']])),np.arange(99762))
    for s in ('val','test'):
        x = idx[s]
        h = hashlib.sha256(str('torch.int64').encode()+str(tuple(x.shape)).encode()+x.tobytes()).hexdigest()
        checks[s+'_index_hash'] = h == config['fingerprint'][s+'_sha256']
    result = dict(passed=all(checks.values()), checks=checks, independent_auc=scores,
                  failures=[k for k,v in checks.items() if not v])
    (path/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
