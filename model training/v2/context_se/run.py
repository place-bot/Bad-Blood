"""Bounded seed42 source-training and personal-development screen. No target access."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from context_se.model import ContextResidual
from paired_se.run import restore
from shared.data import load_personal_cohort
from shared.metrics import summarize
from se_personalization.run import dense_support
from personalized_forecasting.run import atomic_json,file_hash,canonical_hash

VARIANTS=('attention','context','kernel')

def caches(directory):
    result={}
    for role in ('train','validation'):
        path=directory/f'{role}_features.npz';stamp=json.loads((directory/f'{role}_features.json').read_text())
        assert file_hash(path)==stamp['sha256']
        with np.load(path) as f:result[role]={k:f[k] for k in f.files}
    return result

def sample(data,rng,people):
    support=[];query=[]
    for person in people:
        positions=rng.choice(data['rows'][person],48,replace=False)
        support.append(positions[:32]);query.append(positions[32:])
    s,q=np.array(support),np.array(query)
    return [torch.from_numpy(v.astype(np.float32)) for v in
        (data['h'][q],data['h'][s],data['residual'][s],data['p'][q],data['y'][q])]

def val_score(model,data,c,scale):
    sums=[];rng=np.random.default_rng(20261012)
    with torch.inference_mode():
        for i in range(0,len(data['people']),16):
            q,s,r,p,y=sample(data,rng,data['people'][i:i+16])
            pred=p+model(q,s,r)
            # Equal number of query windows per person => participant macro.
            sums.extend(((pred-y).abs()*torch.from_numpy(scale)).mean(1).numpy())
    return np.mean(sums,0)

def encode(base,x):
    hs=[];ps=[]
    with torch.inference_mode():
        for i in range(0,len(x),256):
            h,p=base.encode(torch.from_numpy(x[i:i+256]));hs.append(h.numpy());ps.append(p.numpy())
    return np.concatenate(hs),np.concatenate(ps)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path);p.add_argument('--individual-root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        f=a.output/'summary.json'
        ok=f.exists() and json.loads(f.read_text())['development_pass']
        print('PASS' if ok else 'NOT_MET');return
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
    saved=torch.load(ROOT/'paired_se/models/paired_population_seed42.pt',map_location='cpu',weights_only=False)
    base=restore(saved,'cpu');c,scale=np.asarray(saved['center']),np.asarray(saved['scale'])
    fc,fs=base.feature_center.numpy(),base.feature_scale.numpy()
    data=caches(a.cache)
    for role,d in data.items():
        d['h']=(d['h']-fc)/fs;d['y']=(d['y']-c)/scale;d['residual']=d['y']-d['p']
        d['people']=np.unique(d['ids']);d['rows']={s:np.flatnonzero(d['ids']==s) for s in d['people']}
    # Derive a development-only audit so no target signal or label file is read.
    audit=json.loads((ROOT/'audit_snapshot.json').read_text())
    protocol=audit['final_personalization_protocol']
    protocol['patients']=[v for v in protocol['patients'] if v['role']=='development']
    protocol['target_subjects']=[]
    atomic_json(a.output/'development_only_audit.json',audit)
    dev=load_personal_cohort(a.individual_root,a.output/'development_only_audit.json')['development']
    assert len(dev)==4
    population_ids=set(data['train']['people'])|set(data['validation']['people'])
    assert not population_ids & {v['patient_id'] for v in dev}
    features=[]
    for v in dev:
        s=dense_support(v);q=v['query_indices'];h,pred=encode(base,v['x'])
        features.append(dict(patient=v['patient_id'],h=(h-fc)/fs,p=pred,y=(v['y']-c)/scale,s=s,q=q))
    manifest=dict(date='20261012',seed=42,variants=list(VARIANTS),protocol_sha256=file_hash(Path(__file__).with_name('PROTOCOL.md')),
        population_checkpoint_sha256=file_hash(ROOT/'paired_se/models/paired_population_seed42.pt'),
        source_sha256={f:file_hash(Path(__file__).with_name(f)) for f in ('run.py','model.py')},
        train_people=len(data['train']['people']),validation_people=len(data['validation']['people']),
        development_people=[v['patient'] for v in features],target_predictions=False)
    mp=a.output/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:atomic_json(mp,manifest)
    controls={}
    for method in ('calibration_mean','bias'):
        rows=[]
        for v in features:
            s,q=v['s'],v['q'];y=v['y'];pr=v['p']
            pred=np.tile(y[s].mean(0),(len(q),1)) if method=='calibration_mean' else pr[q]+(y[s]-pr[s]).mean(0)
            rows.append(summarize(y[q]*scale+c,pred*scale+c,np.repeat(v['patient'],len(q))))
        controls[method]=np.mean([r['macro']['mae'] for r in rows],0).tolist()
    outcomes=[]
    for variant in VARIANTS:
        directory=a.output/variant;directory.mkdir(exist_ok=True)
        if (directory/'development.json').exists():outcomes.append(json.loads((directory/'development.json').read_text()));continue
        torch.manual_seed(42);model=ContextResidual(variant);opt=torch.optim.Adam(model.parameters(),lr=.001)
        best=float('inf');stale=0;history=[]
        for epoch in range(1,9):
            begin=time.perf_counter();rng=np.random.default_rng(42+epoch);total=0.
            for step in range(128):
                people=rng.choice(data['train']['people'],16,replace=False)
                q,s,r,pred,y=sample(data['train'],rng,people)
                opt.zero_grad(set_to_none=True);out=pred+model(q,s,r);loss=torch.nn.functional.mse_loss(out,y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();total+=loss.item()
            val=val_score(model,data['validation'],c,scale);score=val.mean()
            row=dict(variant=variant,epoch=epoch,loss=total/128,validation_mae=val.tolist(),seconds=time.perf_counter()-begin)
            history.append(row);atomic_json(directory/'history.json',history);print(json.dumps(row),flush=True)
            if score<best:
                best=score;stale=0;torch.save(dict(variant=variant,state_dict=model.state_dict(),epoch=epoch,manifest_hash=canonical_hash(manifest)),directory/'best.pt')
            else:stale+=1
            if stale>=3:break
        checkpoint=torch.load(directory/'best.pt',weights_only=False);model.load_state_dict(checkpoint['state_dict']);model.eval();rows=[]
        for v in features:
            s,q=v['s'],v['q'];h,pr,y=v['h'],v['p'],v['y'];parts=[]
            with torch.inference_mode():
                for i in range(0,len(q),128):
                    idx=q[i:i+128]
                    residual=model(torch.from_numpy(h[idx][None]),torch.from_numpy(h[s][None]),torch.from_numpy((y[s]-pr[s])[None]))[0].numpy()
                    parts.append((pr[idx]+residual)*scale+c)
            prediction=np.concatenate(parts);truth=y[q]*scale+c
            metrics=summarize(truth,prediction,np.repeat(v['patient'],len(q)))
            np.savez_compressed(directory/(v['patient'].replace(':','_')+'.npz'),y=truth,pred=prediction,support=s,query=q)
            rows.append(dict(patient=v['patient'],metrics=metrics))
        mae=np.mean([r['metrics']['macro']['mae'] for r in rows],0)
        score=float(max(np.r_[mae/controls['bias'],mae/controls['calibration_mean']]))
        result=dict(variant=variant,seed=42,selected_epoch=checkpoint['epoch'],mae=mae.tolist(),worst_control_ratio=score,development_pass=score<1,people=rows,checkpoint_sha256=file_hash(directory/'best.pt'))
        atomic_json(directory/'development.json',result);outcomes.append(result);print('DEVELOPMENT',variant,mae.tolist(),score,flush=True)
    chosen=min(outcomes,key=lambda r:(r['worst_control_ratio'],VARIANTS.index(r['variant'])))
    summary=dict(status='development_screen_complete',controls=controls,outcomes=outcomes,selected=chosen['variant'],development_pass=chosen['development_pass'],independent_validation_complete=False)
    atomic_json(a.output/'summary.json',summary);print('SCREEN COMPLETE',chosen['variant'],chosen['development_pass'],flush=True)

if __name__=='__main__':main()
