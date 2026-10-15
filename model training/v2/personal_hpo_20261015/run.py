"""Frozen random-30, per-person 80/20 nested selection experiment."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import h5py
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from personal_hpo_20261015.core import (split_indices,encode,trajectory,predict_features,
                                      select_person,FIXED,LRS,STEPS)
from modality_se.personalize import restore
from shared.data import _read_ref, _read_string
from shared.metrics import summarize
from personalized_forecasting.run import atomic_json,file_hash,array_hash,canonical_hash,DEV_IDS

SEEDS=(42,43,44)
MODALITIES=('ppg','dual')
METHODS=('population','personal_mean','bias','pair_unadapted','pair_fixed','pair_personal_hpo')
METRICS=('mae','rmse','bias','error_sd','within_5','within_10','within_15')


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v


def read_all(path):
    patient='vitaldb:'+path.stem
    with h5py.File(path,'r') as f:
        g=f['Subj_Wins']
        if h5py.check_dtype(ref=g['T'].dtype) is None:
            raise ValueError('Single-window/non-reference record; not sufficient for nested split')
        n=g['T'].size
        x=np.empty((n,2,1250),np.float32);y=np.empty((n,2),np.float32)
        start=np.empty(n);end=np.empty(n);cases=set()
        for i in range(n):
            assert _read_string(f,g,'SubjectID',i)==path.stem,'Subject mismatch'
            cases.add(_read_string(f,g,'CaseID',i))
            t=_read_ref(f,g,'T',i)
            assert t.shape==(1250,) and np.isfinite(t).all()
            assert np.allclose(np.diff(t),.008,rtol=0,atol=1e-6),'Invalid sampling rate'
            start[i],end[i]=t[0],t[-1]+.008
            wave=np.asarray([_read_ref(f,g,k,i) for k in ('ECG_F','PPG_F')],np.float32)
            assert wave.shape==(2,1250) and np.isfinite(wave).all()
            assert (np.ptp(wave,axis=1)>1e-8).all(),'Flat waveform'
            x[i]=(wave-wave.min(1,keepdims=True))/np.ptp(wave,axis=1,keepdims=True)
            for j,k in enumerate(('SegSBP','SegDBP')):
                value=_read_ref(f,g,k,i);assert value.shape==(1,)
                y[i,j]=value[0]
    assert len(cases)==1 and np.all(np.diff(start)>0),'Mixed cases/unordered time'
    assert np.isfinite(y).all() and np.all(y[:,0]>y[:,1]),'Invalid BP labels'
    indices=split_indices(start,end)
    return dict(patient=patient,file=str(path.resolve()),x=x,y=y,start=start,end=end,
                case=next(iter(cases)),indices=indices)


def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'paired_se/results/manifest.json').read_text())['population']
    forbidden=set(DEV_IDS)
    for v in manifest['partitions'].values():forbidden.update(v['patients'])
    checkpoint_hashes={}
    hashes=json.loads((ROOT/'modality_se/models/manifest.json').read_text())
    for mode in MODALITIES:
        for seed in SEEDS:
            name=f'{mode}_pair_population_seed{seed}.pt'
            digest=file_hash(ROOT/'modality_se/models'/name)
            assert digest==hashes[name]
            checkpoint_hashes[name]=digest
    files=['personal_hpo_20261015/'+name for name in ('PROTOCOL.md','core.py','run.py','test_personal_hpo.py')]
    files+=['modality_se/model.py','modality_se/personalize.py','shared/models.py',
            'paired_se/model.py','shared/metrics.py','personalized_forecasting/run.py']
    spec=dict(seeds=SEEDS,modalities=MODALITIES,n_people=30,sampling_seed=20261015,
              split='80% ordered rows development;70/30 inner;10-second gap',
              lrs=LRS,steps=STEPS,scope='head',fixed=FIXED,
              checkpoints=checkpoint_hashes,files={n:file_hash(ROOT/n) for n in files})
    spec=json.loads(json.dumps(spec))
    lock_path=out/'lock.json'
    if lock_path.exists():
        lock=json.loads(lock_path.read_text());assert lock['spec']==spec,'Frozen experiment changed'
        people=[]
        for row in lock['cohort']['selected']:
            path=Path(row['file']);assert file_hash(path)==row['sha256']
            person=read_all(path)
            for k,ix in person['indices'].items():assert ix.tolist()==row[k]
            people.append(person)
        return people,lock
    candidates={}
    for folder in args.sources:
        for path in sorted(folder.glob('*.mat')):
            pid='vitaldb:'+path.stem
            if pid not in forbidden:
                if pid in candidates:assert file_hash(path)==file_hash(candidates[pid])
                candidates[pid]=path
    people={};audit=[]
    for pid,path in sorted(candidates.items()):
        row=dict(patient=pid,file=str(path.resolve()),sha256=file_hash(path))
        try:
            p=read_all(path);people[pid]=p
            row.update(status='eligible',n=len(p['y']),case=p['case'],
                       **{k:ix.tolist() for k,ix in p['indices'].items()})
        except (ValueError,AssertionError,IndexError,OSError,KeyError) as error:
            row.update(status='excluded',reason=str(error) or type(error).__name__)
        audit.append(row)
    eligible=sorted(people)
    assert len(eligible)>=30, f'Only {len(eligible)} eligible people'
    selected=sorted(np.random.default_rng(20261015).choice(eligible,30,replace=False).tolist())
    cohort=dict(frame='Local original personal and additional-participant MATLAB files',
                excluded_population_or_development_ids=sorted(forbidden),candidate_audit=audit,
                n_eligible=len(eligible),selected=[r for r in audit if r['patient'] in selected])
    lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),spec=spec,cohort=cohort)
    atomic_json(lock_path,lock)
    print(json.dumps(dict(stage='cohort_locked',eligible=len(eligible),selected=selected)),flush=True)
    return [people[p] for p in selected],lock


def score(y,pred):
    e=np.asarray(pred,float)-np.asarray(y,float)
    out=dict(mae=np.abs(e).mean(0),rmse=np.sqrt((e*e).mean(0)),bias=e.mean(0),
             error_sd=e.std(0,ddof=1))
    out.update({f'within_{t}':(np.abs(e)<=t).mean(0) for t in (5,10,15)})
    return clean(out)


def unit(person,mode,seed,out,lock_hash):
    path=out/'units'/mode/f'seed{seed}'/person['patient'].replace(':','_')
    path.mkdir(parents=True,exist_ok=True)
    if (path/'unit.json').exists():
        row=json.loads((path/'unit.json').read_text());assert row['lock_hash']==lock_hash
        for f,h in row['hashes'].items():assert file_hash(path/f)==h
        return row
    started=time.perf_counter()
    saved=torch.load(ROOT/'modality_se/models'/f'{mode}_pair_population_seed{seed}.pt',
                     map_location='cpu',weights_only=False)
    model=restore(saved,mode,'cpu')
    c,s=np.asarray(saved['center'],np.float32),np.asarray(saved['scale'],np.float32)
    indices=person['indices'];support,query=indices['support'],indices['query']
    # Only the development bank is exposed to the selection routine.
    x=person['x'][support];y=person['y'][support]
    start,end=person['start'][support],person['end'][support]
    h,p=encode(model,x)
    chosen_path=path/'selection.json'
    if chosen_path.exists():
        selected=json.loads(chosen_path.read_text());assert selected['lock_hash']==lock_hash
    else:
        selected=select_person(model,h,p,y,start,end,indices,c,s)
        selected.update(lock_hash=lock_hash,patient=person['patient'],seed=seed,modality=mode,
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        final_test_labels_used=False)
        atomic_json(chosen_path,selected)
    config=selected['selected']['config']
    fits={};audits={}
    required={FIXED['lr']:{FIXED['steps']}}
    if config is not None:required.setdefault(config['lr'],set()).add(config['steps'])
    for lr,steps in required.items():
        for count,fitted,audit in trajectory(model,h,p,y,start,end,c,s,lr,steps):
            if lr==FIXED['lr'] and count==FIXED['steps']:
                fits['pair_fixed']=copy.deepcopy(fitted);audits['pair_fixed']=audit
            if config is not None and lr==config['lr'] and count==config['steps']:
                fits['pair_personal_hpo']=copy.deepcopy(fitted);audits['pair_personal_hpo']=audit
    if config is None:
        fits['pair_personal_hpo']=copy.deepcopy(model)
        audits['pair_personal_hpo']=dict(steps=0,support_count=len(y),encoder_frozen=True)
    for method,fitted in fits.items():
        for k,v in model.state_dict().items():
            if not k.startswith('change.'):
                assert torch.equal(v,fitted.state_dict()[k]),'Encoder/buffer changed'
        torch.save(dict(state_dict=fitted.state_dict(),center=c,scale=s,modality=mode,
                        seed=seed,patient=person['patient'],config=config if method=='pair_personal_hpo' else FIXED,
                        lock_hash=lock_hash),path/(method+'.pt'))
    # All choices and final weights now fixed. Test inputs first, labels only to score.
    hq,pq=encode(model,person['x'][query])
    pop_support=p.numpy()*s+c;pop_query=pq.numpy()*s+c
    predictions=dict(population=pop_query,
        personal_mean=np.tile(y.astype(float).mean(0),(len(query),1)),
        bias=pop_query+(y.astype(float)-pop_support).mean(0),
        pair_unadapted=predict_features(model,hq,pq,h,p,y,c,s))
    predictions.update({m:predict_features(f,hq,pq,h,p,y,c,s) for m,f in fits.items()})
    reference=person['y'][query]
    metrics={method:score(reference,pred) for method,pred in predictions.items()}
    np.savez_compressed(path/'predictions.npz',y=reference,**predictions,
                        **{k:v for k,v in indices.items()})
    row=dict(patient=person['patient'],modality=mode,seed=seed,lock_hash=lock_hash,
             config=config,metrics=metrics,n_support=len(support),n_test=len(query),audits=audits,
             seconds=time.perf_counter()-started,
             hashes={f:file_hash(path/f) for f in ('predictions.npz','selection.json',
                          'pair_fixed.pt','pair_personal_hpo.pt')})
    atomic_json(path/'unit.json',clean(row))
    print(json.dumps(dict(stage='completed_unit',modality=mode,seed=seed,person=person['patient'],
                          n_support=len(support),n_test=len(query),selected=config,
                          seconds=round(row['seconds'],2))),flush=True)
    return row


def aggregate(out,rows,lock):
    result=dict(date='20261015',lock_sha256=canonical_hash(lock),n_people=30,
                calibration_fraction=.8,source='VitalDB-derived PulseDB, previously examined pool',
                modality_results={})
    ids=sorted({r['patient'] for r in rows});assert len(ids)==30 and len(rows)==180
    for mode in MODALITIES:
        results={}
        for method in METHODS:
            seeds={}
            for seed in SEEDS:
                ys=[];ps=[];subjects=[]
                for pid in ids:
                    path=out/'units'/mode/f'seed{seed}'/pid.replace(':','_')/'predictions.npz'
                    with np.load(path) as z:
                        ys.append(z['y']);ps.append(z[method]);subjects.extend([pid]*len(z['y']))
                summary=summarize(np.concatenate(ys),np.concatenate(ps),np.asarray(subjects))
                direct=score(np.concatenate(ys),np.concatenate(ps))
                for metric in METRICS:
                    np.testing.assert_allclose(summary['pooled'][metric],direct[metric],atol=1e-10,rtol=0)
                seeds[str(seed)]=summary
            avg={a:{m:np.asarray([s[a][m] for s in seeds.values()],float).mean(0)
                    for m in METRICS} for a in ('macro','pooled')}
            sd={a:{m:np.asarray([s[a][m] for s in seeds.values()],float).std(0,ddof=1)
                   for m in METRICS} for a in ('macro','pooled')}
            pm={pid:np.mean([s['per_patient'][pid]['mae'] for s in seeds.values()],0) for pid in ids}
            arr=np.asarray(list(pm.values()))
            results[method]=dict(**avg,seed_sd=sd,per_seed=seeds,participant_mae=pm,
                participant_mae_distribution=dict(min=arr.min(0),q25=np.quantile(arr,.25,axis=0),
                    median=np.median(arr,0),q75=np.quantile(arr,.75,axis=0),
                    q90=np.quantile(arr,.9,axis=0),max=arr.max(0)))
        comparisons={}
        hpo=results['pair_personal_hpo']
        for baseline in ('personal_mean','bias','pair_fixed','pair_unadapted'):
            difference=np.asarray([hpo['participant_mae'][p]-results[baseline]['participant_mae'][p] for p in ids])
            draws=np.random.default_rng(20261015).integers(0,30,(10000,30))
            boot=difference[draws].mean(1)
            comparisons[baseline]=dict(delta_macro_mae=difference.mean(0),
                participant_bootstrap_ci95=np.quantile(boot,[.025,.975],axis=0),
                improved_people_per_endpoint=(difference<0).sum(0),
                improved_people_both_endpoints=(difference<0).all(1).sum(),
                per_person_delta={p:difference[i] for i,p in enumerate(ids)})
        perperson={}
        for pid in ids:
            strict=True;avgwin=True
            for baseline in ('personal_mean','bias'):
                avgwin=avgwin and bool(np.all(hpo['participant_mae'][pid]<results[baseline]['participant_mae'][pid]))
                for seed in SEEDS:
                    a=np.asarray(hpo['per_seed'][str(seed)]['per_patient'][pid]['mae'])
                    b=np.asarray(results[baseline]['per_seed'][str(seed)]['per_patient'][pid]['mae'])
                    strict=strict and bool(np.all(a<b))
            perperson[pid]=dict(beats_both_all_three_seeds=strict,beats_both_seed_average=avgwin)
        result['modality_results'][mode]=dict(methods=results,comparisons=comparisons,
            gate=dict(each_person=perperson,strict_count=sum(r['beats_both_all_three_seeds'] for r in perperson.values()),
                      seed_average_count=sum(r['beats_both_seed_average'] for r in perperson.values()),
                      all_30_pass=all(r['beats_both_all_three_seeds'] for r in perperson.values())))
    atomic_json(out/'summary.json',clean(result))
    lines=['# Per-person hyperparameter selection — 20261015','',
        'Random 30 from the eligible LOCAL participant pool. Chronological 80% development / 20% test, with a 10-second gap.',
        'Two modalities; three independently pretrained seeds; individual inner validation, then full-development refit.',
        'Metrics below are equal-person means, then averaged across seeds (not an ensemble). Values are SBP / DBP.','']
    for mode,item in result['modality_results'].items():
        lines+=['## '+mode,'','| Method | MAE | RMSE | Bias | Sample error SD | Within 5 (%) | Within 10 (%) | Within 15 (%) |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method,r in item['methods'].items():
            cells=[]
            for m in METRICS:
                values=np.asarray(r['macro'][m])*(100 if m.startswith('within') else 1)
                cells.append(' / '.join(f'{v:.2f}' for v in values))
            lines.append('| '+method+' | '+' | '.join(cells)+' |')
        g=item['gate'];lines+=['',f"Strict individual wins over both controls: **{g['strict_count']}/30**; seed-averaged individual wins: **{g['seed_average_count']}/30**.",'']
        lines+=['| Seed | HPO MAE | Fixed MAE | Personal mean MAE | Bias MAE |','|---|---:|---:|---:|---:|']
        for seed in SEEDS:
            cells=[' / '.join(f'{v:.3f}' for v in item['methods'][m]['per_seed'][str(seed)]['macro']['mae'])
                   for m in ('pair_personal_hpo','pair_fixed','personal_mean','bias')]
            lines.append('| '+str(seed)+' | '+' | '.join(cells)+' |')
        lines+=['']
    lines+=['## Interpretation','',
        'All full metrics, per-person errors, chosen settings, and source/selection hashes are saved alongside this report.',
        'The strict gate requires lower SBP AND DBP MAE than BOTH controls for EVERY person in EVERY seed.',
        'This is a larger-calibration-budget, later-period test, not a like-for-like improvement over previous 20%-budget scores.',
        'These data have been explored previously. Serial dependence remains; this is neither unseen external validation nor a cuff-device certification.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return clean(result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sources',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    people,lock=prepare(args)
    if args.prepare_only:return
    rows=[];lh=canonical_hash(lock)
    for mode in MODALITIES:
        for seed in SEEDS:
            for person in people:
                rows.append(unit(person,mode,seed,args.output,lh))
                atomic_json(args.output/'progress.json',dict(completed=len(rows),total=180,
                    modality=mode,seed=seed,patient=person['patient'],updated_utc=datetime.now(timezone.utc).isoformat()))
    result=aggregate(args.output,rows,lock)
    atomic_json(args.output/'complete.json',dict(status='complete',units=len(rows),lock_hash=lh,
        completed_utc=datetime.now(timezone.utc).isoformat(),summary_sha256=file_hash(args.output/'summary.json'),
        gates={m:r['gate']['all_30_pass'] for m,r in result['modality_results'].items()}))
    print(json.dumps(dict(stage='complete',gates={m:r['gate'] for m,r in result['modality_results'].items()})),flush=True)


if __name__=='__main__':main()
