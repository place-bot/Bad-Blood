"""Population pair-head training, development selection, then locked comparison."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paired_se.model import PairSE,same_person_references,chronological_pairs,predict_bank
from shared.data import load_population,load_personal_cohort
from shared.models import BPNet
from shared.metrics import summarize,quick_score
from se_personalization.run import load_se,dense_support,summarize_phase,SEEDS,DEV_IDS,TARGET_IDS
from personalized_forecasting.adaptation import predict
from personalized_forecasting.run import atomic_json,canonical_hash,file_hash,array_hash

EVENT_LOG=None


def configurations():
    return [None]+[dict(scope=scope,lr=lr,steps=steps) for scope in ('head','full')
                  for lr in (.0003,.001) for steps in (25,100)]


def name(config):
    return 'paired_unadapted' if config is None else f"paired_{config['scope']}_lr{config['lr']:g}_s{config['steps']}"


def emit(event):
    print(json.dumps(event),flush=True)
    if EVENT_LOG is not None:
        with EVENT_LOG.open('a') as handle:
            handle.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),**event))+'\n')


def encode_population(pair,part,device):
    indices=part['indices'];hs=[];ps=[]
    pair.to(device).eval()
    with torch.inference_mode():
        for first in range(0,len(indices),512):
            x=torch.as_tensor(np.asarray(part['waves'][indices[first:first+512]]),device=device)
            h,p=pair.encode(x);hs.append(h.cpu().numpy());ps.append(p.cpu().numpy())
            if first%51200==0:emit(dict(stage='encode',completed=first,total=len(indices)))
    return dict(h=np.concatenate(hs),p=np.concatenate(ps),
                y=part['y'][indices],ids=part['subjects'][indices])


def pair_validation(model,data,refs,center,scale,device):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(refs),2048):
            q=np.arange(i,min(i+2048,len(refs)));r=refs[q]
            args=[data['h'][q],data['p'][q],data['h'][r],data['p'][r],(data['y'][r]-center)/scale]
            out.append(model.from_features(*[torch.as_tensor(v,device=device) for v in args]).cpu().numpy()*scale+center)
    return quick_score(data['y'],np.concatenate(out),data['ids'])


def train_population(seed,saved,population,directory,manifest_hash,device):
    directory.mkdir(parents=True,exist_ok=True)
    done=directory/'complete.json'
    if done.exists():
        record=json.loads(done.read_text())
        if record['manifest_hash']!=manifest_hash or file_hash(directory/'best.pt')!=record['sha256']:
            raise ValueError('Changed population completion')
        return torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    base=BPNet(variant='se_last');base.load_state_dict(saved['state_dict'],strict=True)
    torch.manual_seed(seed)
    pair=PairSE(base).to(device).eval()
    caches={}
    for role in ('train','validation'):
        path=directory/f'{role}_features.npz';stamp=directory/f'{role}_features.json'
        if path.exists() and stamp.exists():
            record=json.loads(stamp.read_text())
            if record['manifest_hash']!=manifest_hash or record['sha256']!=file_hash(path):raise ValueError('Changed feature cache')
            with np.load(path,allow_pickle=False) as z:caches[role]={k:z[k] for k in z.files}
        else:
            caches[role]=encode_population(pair,population[role],device)
            np.savez(path,**caches[role]);atomic_json(stamp,dict(manifest_hash=manifest_hash,sha256=file_hash(path)))
    train,val=caches['train'],caches['validation']
    with torch.no_grad():
        pair.feature_center.copy_(torch.as_tensor(train['h'].mean(0),device=device))
        pair.feature_scale.copy_(torch.as_tensor(np.maximum(train['h'].std(0),.01),device=device))
    for param in pair.encoder.parameters():param.requires_grad_(False)
    center,scale=np.asarray(saved['center'],np.float32),np.asarray(saved['scale'],np.float32)
    optimizer=torch.optim.Adam(pair.change.parameters(),lr=.001)
    validation_refs=same_person_references(val['ids'],seed+10000)
    best=float('inf');stale=0;history=[]
    for epoch in range(1,21):
        started=time.perf_counter();refs=same_person_references(train['ids'],seed+epoch*1000)
        order=np.random.default_rng(seed+epoch).permutation(len(refs));loss_sum=0.
        for first in range(0,len(order),2048):
            q=order[first:first+2048];r=refs[q]
            args=[train['h'][q],train['p'][q],train['h'][r],train['p'][r],(train['y'][r]-center)/scale]
            optimizer.zero_grad(set_to_none=True)
            pred=pair.from_features(*[torch.as_tensor(v,device=device) for v in args])
            truth=torch.as_tensor((train['y'][q]-center)/scale,device=device)
            loss=torch.nn.functional.mse_loss(pred,truth)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite population loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(pair.change.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            loss_sum+=float(loss.detach().cpu())*len(q)
        metric=pair_validation(pair,val,validation_refs,center,scale,device)
        score=float(np.mean(metric['macro_mae']))
        row=dict(seed=seed,epoch=epoch,train_loss=loss_sum/len(order),validation=metric,seconds=time.perf_counter()-started)
        history.append(row);atomic_json(directory/'history.json',history);emit(dict(stage='population',**row))
        if score<best:
            best=score;stale=0
            torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in pair.state_dict().items()},
                center=center,scale=scale,seed=seed,epoch=epoch,manifest_hash=manifest_hash),directory/'best.pt')
        else:stale+=1
        if stale>=5:break
    result=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    atomic_json(done,dict(manifest_hash=manifest_hash,sha256=file_hash(directory/'best.pt'),
        best_epoch=result['epoch'],epochs=len(history),best_validation_macro_mae=best))
    return result


def restore(saved,device):
    pair=PairSE(BPNet(variant='se_last')).to(device).eval()
    pair.load_state_dict(saved['state_dict'],strict=True)
    return pair


def fit_personal(pair,x,y,start,end,center,scale,config,device):
    """Only support waveforms, labels and times; no held-out arguments."""
    model=copy.deepcopy(pair).to(device).eval()
    if config is None:return model,dict(steps=0,support_count=len(y))
    x,y=np.asarray(x,np.float32),np.asarray(y,np.float32)
    center,scale=np.asarray(center,np.float32),np.asarray(scale,np.float32)
    if x.shape!=(len(y),2,1250) or y.shape!=(len(x),2) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid support data')
    q,r=chronological_pairs(start,end)
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for n,p in model.named_parameters():p.requires_grad_(config['scope']=='full' or n.startswith('change.'))
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['lr'])
    xt=torch.as_tensor(x,device=device);yt=torch.as_tensor((y-center)/scale,device=device)
    qt,rt=torch.as_tensor(q,device=device),torch.as_tensor(r,device=device)
    cached=None
    if config['scope']=='head':
        with torch.no_grad():cached=model.encode(xt)
    history=[]
    for step in range(config['steps']):
        optimizer.zero_grad(set_to_none=True)
        h,p=model.encode(xt) if cached is None else cached
        pred=model.from_features(h[qt],p[qt],h[rt],p[rt],yt[rt])
        loss=torch.nn.functional.mse_loss(pred,yt[qt])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite personal loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
        optimizer.step();history.append(float(loss.detach().cpu()))
    params=dict(model.named_parameters())
    changed=[]
    for k,v in model.state_dict().items():
        if not torch.equal(before[k],v.detach().cpu()):
            if k not in params or not params[k].requires_grad:raise AssertionError('Frozen buffer/parameter changed')
            changed.append(k)
    return model,dict(config=config,support_count=len(y),pair_count=len(q),
        all_support_rows_participate=True,query_used=False,weight_decay=0,L2_SP=False,
        loss_history=history,changed=changed,batchnorm_running_stats_frozen=True)


def selection(rows):
    expected={(p,s) for p in DEV_IDS for s in SEEDS}
    maes={}
    for method in ('calibration_mean','bias')+tuple(name(c) for c in configurations()):
        units=[r for r in rows if r['method']==method]
        if {(r['patient'],r['seed']) for r in units}!=expected or len(units)!=len(expected):raise ValueError('Incomplete development')
        maes[method]=np.mean([r['metrics']['macro']['mae'] for r in units],0)
    ranking=[]
    for config in configurations():
        m=name(config);ratios=np.r_[maes[m]/maes['bias'],maes[m]/maes['calibration_mean']]
        ranking.append(dict(method=m,config=config,mae=maes[m].tolist(),worst_ratio=float(max(ratios)),mean_mae=float(maes[m].mean())))
    chosen=min(ranking,key=lambda r:(r['worst_ratio'],r['mean_mae'],[name(c) for c in configurations()].index(r['method'])))
    return dict(selected=chosen,ranking=ranking,controls={m:maes[m].tolist() for m in ('calibration_mean','bias')})


def personal_phase(cohort,models,originals,output,manifest_hash,device,phase,locked=None):
    rows=[];configs=configurations() if phase=='development' else [locked['selected']['config']]
    for seed in SEEDS:
        pair=restore(models[seed],device)
        original=BPNet(variant='se_last');original.load_state_dict(originals[seed]['state_dict'],strict=True)
        center,scale=models[seed]['center'],models[seed]['scale']
        for patient in cohort['development' if phase=='development' else 'target']:
            support=dense_support(patient);q=patient['query_indices'];x=patient['x'];y=patient['y']
            sp,_=predict(original,x[support],center,scale,device);qp,_=predict(original,x[q],center,scale,device)
            controls=dict(population=qp,calibration_mean=np.tile(y[support].astype(float).mean(0),(len(q),1)),
                          bias=qp+(y[support].astype(float)-sp).mean(0))
            for method,config in [(m,'control') for m in controls]+[(name(c),c) for c in configs]:
                directory=output/phase/patient['patient_id']/f'seed{seed}'/method;directory.mkdir(parents=True,exist_ok=True)
                path=directory/'unit.json'
                unit_config=dict(manifest_hash=manifest_hash,selection_hash=canonical_hash(locked) if locked else None,
                    patient=patient['patient_id'],seed=seed,method=method,config=config,
                    support_hash=array_hash(support),query_hash=array_hash(q))
                if path.exists():
                    row=json.loads(path.read_text())
                    if row['config']!=unit_config:raise ValueError('Changed completed unit')
                    for fn,digest in row['hashes'].items():
                        if file_hash(directory/fn)!=digest:raise ValueError('Changed artifact')
                else:
                    artifacts=['predictions.npz']
                    if config=='control':prediction=controls[method];audit=dict(query_used=False,support_count=len(support))
                    else:
                        fitted,audit=fit_personal(pair,x[support],y[support],patient['start'][support],
                            patient['end_exclusive'][support],center,scale,config,device)
                        prediction=predict_bank(fitted,x[q],x[support],y[support],center,scale,device)
                        torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in fitted.state_dict().items()},
                            center=center,scale=scale,seed=seed,config=unit_config),directory/'model.pt')
                        artifacts.append('model.pt')
                    np.savez_compressed(directory/'predictions.npz',y=y[q],pred=prediction,support_indices=support,query_indices=q)
                    row=dict(config=unit_config,patient=patient['patient_id'],seed=seed,method=method,audit=audit,
                        metrics=summarize(y[q],prediction,np.repeat(patient['patient_id'],len(q))),
                        hashes={fn:file_hash(directory/fn) for fn in artifacts})
                    atomic_json(path,row)
                row['_prediction_path']=str(directory/'predictions.npz');rows.append(row)
                emit(dict(stage=phase,seed=seed,patient=patient['patient_id'],method=method,
                    mae=row['metrics']['macro']['mae'],units=len(rows)))
    if phase=='development':
        locked=selection(rows);locked['manifest_hash']=manifest_hash
        path=output/'selection.json'
        if path.exists() and json.loads(path.read_text())!=locked:raise ValueError('Selection changed')
        atomic_json(path,locked)
    methods=summarize_phase(rows)
    result=dict(complete=True,phase=phase,methods=methods,selected=locked['selected'],
        manifest_hash=manifest_hash,selection_hash=canonical_hash(locked))
    selected=methods[locked['selected']['method']]
    result['beats_both_controls_both_pressures']=bool(all(np.all(np.array(selected['macro']['mae'])<methods[m]['macro']['mae']) for m in ('bias','calibration_mean')))
    result['paired_differences']={}
    for method in ('bias','calibration_mean'):
        ids=sorted(selected['per_patient_mae'])
        d=np.array([np.array(selected['per_patient_mae'][p])-methods[method]['per_patient_mae'][p] for p in ids])
        rng=np.random.default_rng(20261012)
        boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
        result['paired_differences'][method]=dict(delta_mae=d.mean(0).tolist(),ci95=np.quantile(boot,[.025,.975],axis=0).tolist(),
            people_improved=(d<0).sum(0).tolist(),people=len(d))
    atomic_json(output/f'{phase}_summary.json',result)
    return locked,result


def main():
    global EVENT_LOG
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--individual-root',type=Path,required=True)
    p.add_argument('--population-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--phase',choices=('all','population','development','test'),default='all')
    a=p.parse_args();torch.set_num_threads(4)
    log=ROOT.parents[1]/'autoresearch'/'classic-261012-paired-se'
    log.mkdir(parents=True,exist_ok=True);EVENT_LOG=log/'events.jsonl'
    population=load_population(a.data_root);cohort=load_personal_cohort(a.individual_root)
    originals={s:load_se(a.population_root/f'se_last_seed{s}'/'best.pt',s) for s in SEEDS}
    for s in SEEDS:
        if originals[s]['hashes']['manifest']!=population['manifest_hash']:raise ValueError('Different population source')
    manifest=dict(protocol='paired-se-20261012',population=population['manifest'],configs=configurations(),seeds=list(SEEDS),
        personal={v['patient_id']:dict(support=dense_support(v).tolist(),query=v['query_indices'].tolist(),
            data_hash=array_hash(v['x'],v['y'],v['start'])) for role in ('development','target') for v in cohort[role]},
        source={f.name:file_hash(f) for f in (Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('PROTOCOL.md'))},
        checkpoints={str(s):file_hash(a.population_root/f'se_last_seed{s}'/'best.pt') for s in SEEDS},device=a.device)
    a.output.mkdir(parents=True,exist_ok=True);mp=a.output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Changed run; use a new output')
    if not mp.exists():atomic_json(mp,manifest)
    mh=canonical_hash(manifest)
    models={s:train_population(s,originals[s],population,a.output/'population'/f'seed{s}',mh,a.device) for s in SEEDS}
    if a.phase=='population':return
    if a.phase in ('all','development'):
        locked,dev=personal_phase(cohort,models,originals,a.output,mh,a.device,'development')
        emit(dict(stage='development_done',selected=locked['selected']))
    else:locked=json.loads((a.output/'selection.json').read_text())
    if locked['manifest_hash']!=mh:raise ValueError('Wrong selection')
    if a.phase in ('all','test'):
        _,test=personal_phase(cohort,models,originals,a.output,mh,a.device,'test',locked)
        emit(dict(stage='complete',success=test['beats_both_controls_both_pressures'],
            selected=test['selected'],test_mae=test['methods'][test['selected']['method']]['macro']['mae']))
        atomic_json(log/'handoff.json',dict(status='complete',manifest_hash=mh,
            selected=locked['selected'],success=test['beats_both_controls_both_pressures']))
    (log/'classic-results.tsv').write_text('iteration\tmethod\tworst_development_ratio\n'+''.join(
        f"{i}\t{r['method']}\t{r['worst_ratio']:.8f}\n" for i,r in enumerate(locked['ranking'],1)))


if __name__=='__main__':main()
