"""Recompute saved budget-screen AUC with rank statistics, independently."""
import argparse
import json
from pathlib import Path
import numpy as np
from universal_experiment.verify import rank_auc


def verify(folder):
    report = json.loads((folder/'report.json').read_text())
    config = json.loads((folder/'config.json').read_text())
    prior = Path(config['previous'])
    old = json.loads((prior/'report.json').read_text())
    checks = dict(clean_start=not config['dirty'],budget=config['epochs']==20 and config['patience']==3,
                  stage1_reused=not config['stage1_retrained'])
    scores = {}
    reference = np.load(folder/'B_predictions.npz')
    for arm in ('B','U','R','G'):
        raw = np.load(folder/f'{arm}_predictions.npz')
        m = report['arms'][arm]
        scores[arm] = {}
        for split in ('val','test'):
            y,p = raw[split+'_y'],raw[split+'_p']
            score = rank_auc(y,p)
            scores[arm][split] = score
            checks[f'{arm}_{split}_auc'] = abs(score-m[split+'_auc'])<1e-12
            checks[f'{arm}_{split}_labels'] = np.array_equal(y,reference[split+'_y']) and len(y)==config['rows'][split]
        history = m['epochs']
        selected = max(history,key=lambda e:e['val_auc'])
        checks[arm+'_selection'] = selected['epoch']==m['best_epoch'] and abs(selected['val_auc']-m['val_auc'])<1e-12
        checks[arm+'_budget'] = len(history)<=20 and all(e['steps']==int(np.ceil(config['rows']['train']/config['batch_size'])) for e in history)
        checks[arm+'_historical_first5'] = all(abs(a['val_auc']-b['val_auc'])<1e-10 for a,b in zip(history[:5],old['arms'][arm]['epochs'][:5]))
        best=-1.; stale=0; expected_stop=None
        for e in history:
            if e['val_auc']>best: best=e['val_auc']; stale=0
            else: stale+=1
            if stale==3:
                expected_stop=e['epoch']; break
        checks[arm+'_early_stop'] = (len(history)==20 if expected_stop is None else len(history)==expected_stop)
        reached=[e for e in history if e['val_auc']>=old['arms']['B']['val_auc']]
        checks[arm+'_target'] = m['first_target_epoch']==(reached[0]['epoch'] if reached else None)
        checks[arm+'_timing'] = all(e['train_seconds']>0 and e['val_seconds']>0 for e in history)
    for arm in ('B','R','G'):
        for split in ('val','test'):
            checks[f'delta_{arm}_{split}']=abs(scores['U'][split]-scores[arm][split]-report['u_minus'][arm][split])<1e-12
    d=scores['U']['test']-scores['B']['test']
    checks['classification']=report['classification']==('positive' if d>=.001 else 'clear_decline' if d<=-.02 else 'no_clear_improvement')
    result=dict(passed=all(checks.values()),checks=checks,independent_auc=scores,failures=[k for k,v in checks.items() if not v])
    (folder/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']: raise RuntimeError('Independent verification failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('folder',type=Path)
    verify(parser.parse_args().folder)
