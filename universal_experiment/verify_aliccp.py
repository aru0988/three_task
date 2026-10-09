"""AliCCP audit independent of runner metric/loss implementations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from universal_experiment.verify import rank_auc, checkpoint_audit


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);path=p.parse_args().run
    report=json.loads((path/'report.json').read_text());config=json.loads((path/'config.json').read_text())
    s1=json.loads((path/'stage1.json').read_text());checks={};scores={};ys={}
    for name,h in report['prediction_hashes'].items():
        checks['sha_'+name]=hashlib.sha256((path/name).read_bytes()).hexdigest()==h
    for a in ('B','U','R','G'):
        raw=np.load(path/f'{a}_predictions.npz');scores[a]={}
        for s in ('val','test'):
            y,pred=raw[s+'_y'],raw[s+'_p'];auc=rank_auc(y,pred);scores[a][s]=auc
            checks[a+s+'_auc']=abs(auc-report['arms'][a][s+'_auc'])<1e-12
            checks[a+s+'_count']=len(y)==config['budgets'][s]
            checks[a+s+'_positives']=int(y.sum())==config['actual_prefixes'][s]['label_counts']['bsi_pos']
            if s in ys:checks[a+s+'_alignment']=np.array_equal(y,ys[s])
            ys[s]=y
            if a=='U':checks['shuffle_'+s]=abs(rank_auc(y,raw[s+'_shuffled_u_p'])-report['arms'][a][s+'_shuffled_u_auc'])<1e-12
        hist=report['arms'][a]['epochs'];best=max(hist,key=lambda x:x['val_auc'])
        checks[a+'_selection']=best['epoch']==report['arms'][a]['best_epoch'] and abs(best['val_auc']-scores[a]['val'])<1e-12
        checks[a+'_budget']=len(hist)<=config['stage2_epochs'] and all(e['steps']==config['budgets']['train']//2000 for e in hist)
    old=np.load(path/'old_tasks.npz');expected={'val':[.5493130789281618,.5132119172500262],'test':[.5481837665250074,.5280080347106456]}
    for s in ('val','test'):
        for i,t in enumerate(('ctr','cvr')):
            auc=rank_auc(old[f'{s}_y{i}'],old[f'{s}_p{i}'])
            checks[f'{s}_{t}_raw']=abs(auc-report['stage1_old_tasks'][s][t])<1e-12
            checks[f'{s}_{t}_historical']=abs(auc-expected[s][i])<1e-12
    mech=np.load(path/'mechanism.npz');n=int(mech['n'])
    us=np.sqrt(np.maximum(mech['u_sq']/n-(mech['u_sum']/n)**2,0));gs=np.sqrt(np.maximum(mech['g_sq']/n-(mech['g_sum']/n)**2,0))
    corr=(mech['ug_sum']/n-np.outer(mech['u_sum']/n,mech['g_sum']/n))/np.outer(np.maximum(us,.01),np.maximum(gs,.01))
    checks['std']=np.allclose(us,mech['u_std'],atol=1e-8);checks['corr']=np.allclose(corr,mech['ug_correlation'],atol=1e-8)
    checks['live_fraction']=abs(float(np.mean(us>.01))-report['mechanism']['u_live_fraction'])<1e-12
    checkpoint_audit(path,mech,report,s1,checks)
    checks['base_matches']=s1['base_hash']==report['frozen_base_hash_after']=='5553640bc1f43c7af0065f4f1d3f2d719022b3764751e2e4a6cb57f663f2c6ee'
    checks['u_frozen']=s1['universal_hash']==report['frozen_u_hash_after'] and report['frozen_gradients_none']
    checks['capacity']=len({report['arms'][a]['parameters'] for a in ('U','R','G')})==1
    checks['clean']=not config['dirty'];checks['canonical']=config['model_seed']==1688723512 and config['env_seed']==20261003 and config['budgets']==dict(train=2000000,val=500000,test=1000000)
    checks['historical_B_val']=abs(scores['B']['val']-.5781533414372665)<1e-12
    checks['historical_B_test']=abs(scores['B']['test']-.5988392178311113)<1e-12
    for a in ('B','R','G'):
        for s in ('val','test'):checks[a+s+'_delta']=abs(scores['U'][s]-scores[a][s]-report['u_minus'][a][s])<1e-12
    rec=report['mechanism']['reconstruction_train_probe']
    go=all(scores['U']['test']-scores[a]['test']>=.001 and scores['U']['val']>scores[a]['val'] for a in ('B','R','G')) and np.mean(us>.01)>=.5 and rec['reconstruction']<rec['zero']
    checks['decision']=bool(go)==report['preregistered_go']
    result=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},independent_auc=scores,failures=[k for k,v in checks.items() if not v])
    (path/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
