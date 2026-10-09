"""Independent file parser: rank-statistic AUC, sufficient-statistic mechanisms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


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
