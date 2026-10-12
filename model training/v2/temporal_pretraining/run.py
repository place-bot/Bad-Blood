"""Matched random/chronological source learning; no target-data reads."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from paired_se.run import restore
from paired_se.model import chronological_pairs
from context_se.adapt import MaskedContext
from context_se.run import encode
from shared.data import load_personal_cohort
from se_personalization.run import dense_support
from shared.metrics import summarize
from personalized_forecasting.run import atomic_json, file_hash, array_hash

def episode_indices(d,rng,mode):
    ns=min(32,len(d['support']));nq=min(32,len(d['query']))
    if mode=='chronological':
        s=rng.choice(d['support'],ns,replace=False);q=rng.choice(d['query'],nq,replace=False)
        assert np.max(d['end'][s])+10<=np.min(d['start'][q])+1e-6
    elif mode=='random':
        idx=rng.choice(len(d['h']),ns+nq,replace=False);s,q=idx[:ns],idx[ns:]
    else:raise ValueError(mode)
    assert not set(s)&set(q)
    return s,q

def tensor(a):return torch.as_tensor(a,dtype=torch.float32)

def episode_people(n,epoch):
    # Independent of reference/query RNG consumption in the two sampling modes.
    return np.random.default_rng(4200+epoch).integers(n,size=64)

def bank(model,kind,h,p,ys,hs,ps,fc,fs):
    """Only known reference labels ys are accepted, never query labels."""
    if kind=='context':
        residual=model(((h-fc)/fs)[None],((hs-fc)/fs)[None],(ys-ps)[None])[0]
        return p+residual
    total=torch.zeros_like(p)
    for i in range(0,len(hs),64):
        hr,pr,yr=hs[i:i+64],ps[i:i+64],ys[i:i+64];n=len(hr);b=len(h)
        out=model.from_features(h[:,None].expand(-1,n,-1).reshape(-1,48),
            p[:,None].expand(-1,n,-1).reshape(-1,2),hr.repeat(b,1),pr.repeat(b,1),yr.repeat(b,1))
        total=total+out.reshape(b,n,2).sum(1)
    return total/len(hs)

def predictions(model,kind,d,fc,fs):
    s,q=d['support'],d['query'];parts=[]
    with torch.inference_mode():
        for i in range(0,len(q),32):
            idx=q[i:i+32]
            parts.append(bank(model,kind,tensor(d['h'][idx]),tensor(d['p'][idx]),tensor(d['yn'][s]),tensor(d['h'][s]),tensor(d['p'][s]),fc,fs).numpy())
    return np.concatenate(parts)

def evaluate(model,kind,records,center,scale,fc,fs,directory=None):
    y=[];p=[];ids=[]
    for d in records:
        q=d['query'];pred=predictions(model,kind,d,fc,fs)*scale+center
        y.append(d['y'][q]);p.append(pred);ids.extend([d['patient']]*len(q))
        if directory:
            directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/(d['patient'].replace(':','_')+'.npz'),y=d['y'][q],pred=pred,support=d['support'],query=q)
    return summarize(np.concatenate(y),np.concatenate(p),np.array(ids))

def control_metrics(records,center,scale):
    out={}
    for method in ('population','calibration_mean','bias'):
        ys=[];ps=[];ids=[]
        for d in records:
            s,q=d['support'],d['query'];pr=d['p']*scale+center
            prediction=pr[q] if method=='population' else np.tile(d['y'][s].mean(0),(len(q),1)) if method=='calibration_mean' else pr[q]+(d['y'][s]-pr[s]).mean(0)
            ys.append(d['y'][q]);ps.append(prediction);ids.extend([d['patient']]*len(q))
        out[method]=summarize(np.concatenate(ys),np.concatenate(ps),np.array(ids))
    return out

def adapt(model,kind,d,lr,steps,fc,fs):
    # Receives an already sliced support-only record; later labels unavailable.
    fitted=copy.deepcopy(model)
    params=list(fitted.change.parameters()) if kind=='pair' else list(fitted.parameters())
    opt=torch.optim.Adam(params,lr=lr)
    h,p,y=tensor(d['h']),tensor(d['p']),tensor(d['yn'])
    q,r=chronological_pairs(d['start'],d['end'])
    mask=d['end'][None,:]+10<=d['start'][:,None]+1e-6
    trainq=np.flatnonzero(mask.any(1));history=[]
    if kind=='context':
        hs=(h-fc)/fs;count=len(trainq);st=hs[None].expand(count,-1,-1);rt=(y-p)[None].expand(count,-1,-1)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        if kind=='pair':
            loss=F.mse_loss(fitted.from_features(h[q],p[q],h[r],p[r],y[r]),y[q])
        else:
            pred=p[trainq]+fitted(hs[trainq][:,None],st,rt,torch.from_numpy(mask[trainq]))[:,0]
            loss=F.mse_loss(pred,y[trainq])
        if not torch.isfinite(loss):raise ValueError('Nonfinite personal loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);opt.step();history.append(loss.item())
    return fitted,history

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--features',type=Path,required=True);parser.add_argument('--individual-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);a.output.mkdir(parents=True,exist_ok=True)
    lock=json.loads((HERE/'sources.json').read_text());assert lock['protocol_sha256']==file_hash(HERE/'PROTOCOL.md')
    prepared=json.loads((a.features/'complete.json').read_text());assert prepared['sources_sha256']==file_hash(HERE/'sources.json')
    assert not any(r['status']=='download_failed' for r in prepared['audit']), 'Resolve downloads before training'
    saved=torch.load(ROOT/'paired_se/models/paired_population_seed42.pt',map_location='cpu',weights_only=False)
    assert prepared['checkpoint_sha256']==file_hash(ROOT/'paired_se/models/paired_population_seed42.pt')
    base=restore(saved,'cpu');center=np.asarray(saved['center'],np.float32);scale=np.asarray(saved['scale'],np.float32)
    fc,fs=base.feature_center,base.feature_scale
    for p in base.encoder.parameters():p.requires_grad_(False)
    initcontext=MaskedContext();ctxpath=ROOT/'context_se/models/screen/context/best.pt'
    initcontext.load_state_dict(torch.load(ctxpath,map_location='cpu',weights_only=False)['state_dict'])
    initializers={'pair':base,'context':initcontext};data={'train':[],'validation':[]}
    for row in sorted(prepared['audit'],key=lambda x:x['patient']):
        if row['status']!='included':continue
        path=a.features/(Path(row['file']).stem+'.npz');assert file_hash(path)==row['features_sha256']
        with np.load(path) as f:d={k:f[k] for k in f.files}
        d['patient']=row['patient'];d['yn']=(d['y']-center)/scale;data[row['role']].append(d)
    assert len(data['train'])>=8 and len(data['validation'])>=2,'Insufficient timestamped source people'
    manifest=dict(protocol_sha256=file_hash(HERE/'PROTOCOL.md'),code_sha256=file_hash(Path(__file__)),
        sources_sha256=file_hash(HERE/'sources.json'),prepared_sha256=file_hash(a.features/'complete.json'),
        initializer_sha256=dict(pair=file_hash(ROOT/'paired_se/models/paired_population_seed42.pt'),context=file_hash(ctxpath)),
        seed=42,train_people=len(data['train']),validation_people=len(data['validation']),
        train_windows=sum(len(d['y']) for d in data['train']),validation_windows=sum(len(d['y']) for d in data['validation']),
        encoder_frozen=True,target_people_read=False)
    mp=a.output/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Existing run differs'
    else:atomic_json(mp,manifest)
    controls=control_metrics(data['validation'],center,scale);atomic_json(a.output/'validation_controls.json',controls)
    print('COHORT',json.dumps(manifest),flush=True)
    results=[]
    for kind in ('pair','context'):
        for mode in ('random','chronological'):
            name=kind+'_'+mode;directory=a.output/name;directory.mkdir(exist_ok=True)
            if (directory/'result.json').exists():results.append(json.loads((directory/'result.json').read_text()));continue
            model=copy.deepcopy(initializers[kind]);params=list(model.change.parameters()) if kind=='pair' else list(model.parameters())
            opt=torch.optim.Adam(params,lr=.0003);init=evaluate(model,kind,data['validation'],center,scale,fc,fs)
            best=float(np.mean(init['macro']['mae']));stale=0;history=[dict(epoch=0,validation=init,loss=None)]
            torch.save(dict(kind=kind,mode=mode,epoch=0,state_dict=model.state_dict(),center=center,scale=scale),directory/'best.pt')
            print(name,'INITIAL',init['macro']['mae'],flush=True)
            for epoch in range(1,21):
                tic=time.perf_counter();rng=np.random.default_rng(42+epoch);losses=[]
                for person_index in episode_people(len(data['train']),epoch):
                    d=data['train'][int(person_index)];s,q=episode_indices(d,rng,mode)
                    opt.zero_grad(set_to_none=True)
                    pred=bank(model,kind,tensor(d['h'][q]),tensor(d['p'][q]),tensor(d['yn'][s]),tensor(d['h'][s]),tensor(d['p'][s]),fc,fs)
                    loss=F.mse_loss(pred,tensor(d['yn'][q]))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite source loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);opt.step();losses.append(loss.item())
                metric=evaluate(model,kind,data['validation'],center,scale,fc,fs);score=float(np.mean(metric['macro']['mae']))
                history.append(dict(epoch=epoch,validation=metric,loss=float(np.mean(losses)),seconds=time.perf_counter()-tic))
                atomic_json(directory/'history.json',history);print(name,epoch,metric['macro']['mae'],'seconds',round(time.perf_counter()-tic,2),flush=True)
                if score<best:
                    best=score;stale=0;torch.save(dict(kind=kind,mode=mode,epoch=epoch,state_dict=model.state_dict(),center=center,scale=scale),directory/'best.pt')
                else:stale+=1
                if stale>=5:break
            chosen=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(chosen['state_dict'])
            metric=evaluate(model,kind,data['validation'],center,scale,fc,fs,directory/'predictions')
            passed=bool(np.all(np.array(metric['macro']['mae'])<np.array(init['macro']['mae'])))
            result=dict(name=name,kind=kind,mode=mode,selected_epoch=chosen['epoch'],initial=init,validation=metric,
                both_validation_endpoints_improved=passed,checkpoint_sha256=file_hash(directory/'best.pt'))
            atomic_json(directory/'result.json',result);results.append(result)
    candidates=[r for r in results if r['mode']=='chronological']
    chosen=min(candidates,key=lambda r:np.mean(r['validation']['macro']['mae']))
    atomic_json(a.output/'source_selection.json',dict(selected=chosen,comparisons=results,controls=controls))
    print('SOURCE SELECTED',chosen['name'],chosen['selected_epoch'],chosen['both_validation_endpoints_improved'],flush=True)
    # Separate original personal DEVELOPMENT cohort, never historic/new test people.
    audit=json.loads((ROOT/'audit_snapshot.json').read_text());ap=audit['final_personalization_protocol']
    ap['patients']=[r for r in ap['patients'] if r['role']=='development'];ap['target_subjects']=[]
    atomic_json(a.output/'development_only_audit.json',audit)
    dev=[]
    for v in load_personal_cohort(a.individual_root,a.output/'development_only_audit.json')['development']:
        h,p=encode(base,v['x']);dev.append(dict(patient=v['patient_id'],h=h,p=p,y=v['y'],yn=(v['y']-center)/scale,
            support=dense_support(v),query=v['query_indices'],start=v['start'],end=v['end_exclusive']))
    assert not {d['patient'] for d in dev}&{d['patient'] for role in data.values() for d in role}
    devcontrols=control_metrics(dev,center,scale);kind=chosen['kind'];model=copy.deepcopy(initializers[kind])
    model.load_state_dict(torch.load(a.output/chosen['name']/'best.pt',map_location='cpu',weights_only=False)['state_dict'])
    settings=[None]+([dict(lr=lr,steps=n) for lr in (.0001,.0003) for n in (10,25,100)] if chosen['both_validation_endpoints_improved'] else [])
    outcomes=[]
    for config in settings+['old_paired']:
        name='unadapted' if config is None else 'old_paired' if config=='old_paired' else f"lr{config['lr']:g}_s{config['steps']}"
        directory=a.output/'development'/name;directory.mkdir(parents=True,exist_ok=True);ys=[];ps=[];ids=[];rows=[]
        for d in dev:
            s,q=d['support'],d['query'];k='pair' if config=='old_paired' else kind;m=base if config=='old_paired' else model
            if config is not None:
                cfg=dict(lr=.0003,steps=25) if config=='old_paired' else config
                supportonly={key:d[key][s] for key in ('h','p','yn','start','end')}
                m,history=adapt(m,k,supportonly,cfg['lr'],cfg['steps'],fc,fs)
            else:history=[]
            pred=predictions(m,k,d,fc,fs)*scale+center;ys.append(d['y'][q]);ps.append(pred);ids.extend([d['patient']]*len(q))
            filename=d['patient'].replace(':','_')
            np.savez_compressed(directory/(filename+'.npz'),y=d['y'][q],pred=pred,support=s,query=q)
            torch.save(dict(kind=k,config=config,state_dict=m.state_dict(),center=center,scale=scale,patient=d['patient']),directory/(filename+'.pt'))
            rows.append(dict(patient=d['patient'],support_hash=array_hash(s),query_hash=array_hash(q),loss_history=history))
        metric=summarize(np.concatenate(ys),np.concatenate(ps),np.array(ids));mae=np.array(metric['macro']['mae'])
        ratio=float(max(np.r_[mae/devcontrols['bias']['macro']['mae'],mae/devcontrols['calibration_mean']['macro']['mae']]))
        out=dict(name=name,config=config,metrics=metric,worst_control_ratio=ratio,people=rows)
        atomic_json(directory/'result.json',out);outcomes.append(out);print('DEVELOPMENT',name,mae.tolist(),ratio,flush=True)
    eligible=[r for r in outcomes if r['name']!='old_paired'];bestdev=min(eligible,key=lambda r:r['worst_control_ratio'])
    atomic_json(a.output/'summary.json',dict(status='complete',source_selection=chosen,source_results=results,
        validation_controls=controls,development_controls=devcontrols,development_results=outcomes,selected_development=bestdev,
        development_pass=bestdev['worst_control_ratio']<1,independent_test_complete=False,
        note='All results are seed42 development only; frozen waveform encoder means zero-calibration predictions are unchanged.'))
    print('COMPLETE',bestdev['name'],bestdev['worst_control_ratio'],flush=True)

if __name__=='__main__':main()
