"""Shared AliCCP adapter retained for the paired B/U experiment runner."""
import argparse
import copy
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from aliccp_benchmark import protocol as A
from aliccp_benchmark.bench import RecordingMPTRecTrainManager
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec, NewTask
from universal_experiment import run as C
from universal_experiment.model import UniversalExpert, UniversalStage1, ResidualHead

EXPECTED = '5553640bc1f43c7af0065f4f1d3f2d719022b3764751e2e4a6cb57f663f2c6ee'


def build(device):
    return MPTRec(2,A.build_vocab(),5,80,[128,64],[32,32],dropout=[.1,.3],
                  reg_embedding=A.REG_EMBEDDING,reg_dnn=A.REG_DNN,device=device).to(device)


@torch.no_grad()
def disk_cache(base,u,r,loader,device,folder,split):
    folder.mkdir(parents=True,exist_ok=True)
    n=len(loader.dataset); dims={'x':80,'g':64,'s0':64,'s1':64,'u':64,'r':64,'y':None}
    arrays={k:np.lib.format.open_memmap(folder/f'{split}_{k}.npy',mode='w+',dtype='float32',shape=(n,d) if d else (n,)) for k,d in dims.items()}
    old={k:np.empty(n,dtype=np.float32) for k in ('y0','y1','p0','p1')} if split!='train' else {}
    offset=0
    for y0,y1,y,features in loader:
        f={k:v.to(device) for k,v in features.items()}; x,g,s,envs=base.get_infos(f)
        end=offset+len(y)
        for k,v in zip(arrays,(x,g,s[0],s[1],u(x),r(x),y)):
            arrays[k][offset:end]=v.detach().cpu().numpy()
        if old:
            preds=base.predict(f)
            for k,v in zip(old,(y0,y1,preds[0],preds[1])):
                old[k][offset:end]=v.cpu().numpy().reshape(-1)
        offset=end
    for arr in arrays.values(): arr.flush()
    return {k:torch.from_numpy(v) for k,v in arrays.items()},old,[e.detach() for e in envs]


@torch.no_grad()
def mechanism(cache,u,device,out):
    n=len(cache['y']); su=torch.zeros(64,dtype=torch.float64); sg=su.clone()
    qu=su.clone(); qg=su.clone(); cross=torch.zeros(64,64,dtype=torch.float64)
    for lo in range(0,n,10000):
        z=cache['u'][lo:lo+10000].double(); g=cache['g'][lo:lo+10000].double()
        su+=z.sum(0);sg+=g.sum(0);qu+=z.square().sum(0);qg+=g.square().sum(0);cross+=z.T@g
    us=(qu/n-(su/n).square()).clamp_min(0).sqrt(); gs=(qg/n-(sg/n).square()).clamp_min(0).sqrt()
    corr=(cross/n-torch.outer(su/n,sg/n))/torch.outer(us.clamp_min(.01),gs.clamp_min(.01))
    m=u.sample_mask(min(n,4096)); x=cache['x'][:4096]; g=cache['g'][:4096]
    diag={k:float(v) for k,v in u.losses(x.to(device),g.to(device),m).items()}
    result={'u_std_mean':float(us.mean()),'u_live_fraction':float((us>.01).double().mean()),
            'ug_squared_correlation':float(corr.square().mean()),'reconstruction_train_probe':diag}
    np.savez_compressed(out/'mechanism.npz',n=np.array(n),u_sum=su.numpy(),g_sum=sg.numpy(),
        u_sq=qu.numpy(),g_sq=qg.numpy(),ug_sum=cross.numpy(),u_std=us.numpy(),ug_correlation=corr.numpy(),
        probe_x=x.numpy(),probe_g=g.numpy(),probe_u=cache['u'][:4096].numpy(),probe_mask=m.numpy(),widths=np.array([5]*16))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); start=time.perf_counter()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    if dirty and not args.smoke: raise RuntimeError('formal start must be clean')
    args.out.mkdir(parents=True,exist_ok=False)
    budgets=dict(zip(('train','val','test'),(20000,5000,10000) if args.smoke else (2000000,500000,1000000)))
    fp=json.loads((args.source/'artifacts/aliccp_bench/splits/p2M-v500k-t1M/prefix_fingerprint.json').read_text())
    assert A.fingerprint_digest(fp)==fp['fingerprint_sha256']
    actual={}; datasets={};loaders={}
    for split,n in budgets.items():
        source=args.source/Path(A.DATA_FILES[split]); h=A.prefix_sha256(source,n)
        counts=A.scan_label_counts(source,n)
        if not args.smoke:
            assert h==fp['files'][split]['prefix_sha256']
            assert A._normalize_counts(counts)==A._normalize_counts(fp['label_counts'][split])
        actual[split]={'prefix_sha256':h,'label_counts':counts}
        datasets[split]=AliCCPDataset(str(source),n)
        A.verify_label_counts(datasets[split],n,counts)
        assert set(datasets[split][0][-1])==set(A.build_vocab()) and '301' not in datasets[split][0][-1]
        loaders[split]=DataLoader(datasets[split],batch_size=2000,shuffle=False)
    config={'dataset':'AliCCP','commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dirty':dirty,'smoke':args.smoke,'model_seed':A.MODEL_SEED,'env_seed':A.ENV_SEED,
        'budgets':budgets,'fingerprint':fp,'source':str(args.source),'actual_prefixes':actual,
        'stage1_epochs':1 if args.smoke else 3,'stage2_epochs':1 if args.smoke else 5,
        'batch_size':2000,'lr':.0001,'created':datetime.now().isoformat(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'u_seed':1685480946,'mask_seed':20261009}
    C.dump(args.out/'config.json',config)
    device=torch.device('cuda:0'); A.seed_model(A.MODEL_SEED);base=build(device)
    u=UniversalExpert([5]*16,rep_dim=64).to(device)
    C.fit_normalizer(base,u,loaders['train'],device);r=copy.deepcopy(u)
    wrapper=UniversalStage1(base,u)
    mgr=RecordingMPTRecTrainManager(wrapper,loaders['train'],loaders['val'],A.make_env_ids(budgets['train'],A.ENV_SEED),
        ['CTR','CVR'],A.LR,2000,A.UNI_COE,A.ENV_COE,epochs=config['stage1_epochs'],patience=2)
    print('Stage1 G/S + detached U',flush=True);mgr.train_two_task();wrapper.load_state_dict(mgr.best_weight)
    s1={'base_hash':A.backbone_sha256(base),'universal_hash':A.backbone_sha256(u),
        'selected_epoch':mgr.best_epoch(),'val_epochs':mgr.val_epoch_aucs,'clusters':mgr.cluster_events,'auxiliary_steps':wrapper.loss_history}
    s1['matches_historical']=s1['base_hash']==EXPECTED
    C.dump(args.out/'stage1.json',s1);torch.save(wrapper.state_dict(),args.out/'stage1.pt');torch.save(r.state_dict(),args.out/'random_u.pt')
    if not args.smoke and not s1['matches_historical']: raise RuntimeError('base hash changed: stop interpretation')
    for m in (base,u,r):
        m.eval()
        for p in m.parameters():p.requires_grad_(False);p.grad=None
    del mgr
    caches={};old={}
    for split in budgets:
        print('Caching '+split,flush=True)
        caches[split],raw,envs=disk_cache(base,u,r,loaders[split],device,args.out/'cache',split)
        old.update({f'{split}_{k}':v for k,v in raw.items()})
    np.savez_compressed(args.out/'old_tasks.npz',**old)
    old_auc={s:{t:float(roc_auc_score(old[f'{s}_y{i}'],old[f'{s}_p{i}'])) for i,t in enumerate(('ctr','cvr'))} for s in ('val','test')}
    mech=mechanism(caches['train'],u,device,args.out);C.dump(args.out/'mechanism.json',mech)
    A.seed_model(A.MODEL_SEED);dummy=build(device)
    original=NewTask(80,64,[32,32],A.REG_DNN,device).to(device);del dummy
    C.P=SimpleNamespace(LR=A.LR,BATCH_SIZE=2000,PATIENCE=2)
    arms={}
    for arm in ('B','U','R','G'):
        h=copy.deepcopy(original) if arm=='B' else ResidualHead(original,64).to(device)
        arms[arm]=C.train_head(h,caches,envs,arm,device,config['stage2_epochs'],args.out,cpu_loss=True);del h
    delta={a:{s:arms['U'][s+'_auc']-arms[a][s+'_auc'] for s in ('val','test')} for a in ('B','R','G')}
    d=delta['B']['test'];rec=mech['reconstruction_train_probe']
    report={'arms':arms,'u_minus':delta,'classification':'positive' if d>=.001 else 'clear_decline' if d<=-.02 else 'no_clear_improvement',
        'preregistered_go':all(v['test']>=.001 and v['val']>0 for v in delta.values()) and mech['u_live_fraction']>=.5 and rec['reconstruction']<rec['zero'],
        'mechanism':mech,'stage1_old_tasks':old_auc,'stage1_matches_historical':s1['matches_historical'],
        'frozen_base_hash_after':A.backbone_sha256(base),'frozen_u_hash_after':A.backbone_sha256(u),
        'frozen_gradients_none':all(p.grad is None for m in (base,u,r) for p in m.parameters()),'wall_seconds':time.perf_counter()-start,
        'prediction_hashes':{p.name:C.file_hash(p) for p in args.out.glob('*.npz')}}
    C.dump(args.out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
