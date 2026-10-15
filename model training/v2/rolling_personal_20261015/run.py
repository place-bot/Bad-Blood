"""Train all nested selections before a separate final-test evaluation phase."""
import argparse
import copy
import json
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rolling_personal_20261015.core import (LRS,STEPS,FOLDS,rolling_folds,prefix,tail,
    trajectory,predict,choose,key)
from personal_hpo_20261015.run import read_all,clean,score
from modality_se.personalize import restore
from personalized_forecasting.run import atomic_json,file_hash,canonical_hash

BASE=Path('/Users/jguo9/Desktop/26fall/LLM/output/bad_blood_v2_runs/20261015_personal_hpo')
OUT=BASE.parent/'20261015_rolling_personal'
SEEDS=(42,43,44);MODES=('ppg','dual')


def utc():return datetime.now(timezone.utc).isoformat()


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    old=json.loads((BASE/'lock.json').read_text())
    assert json.loads((BASE/'verification.json').read_text())['status']=='passed'
    code=['rolling_personal_20261015/'+n for n in ('core.py','run.py','test_rolling.py','PROTOCOL.md')]
    code+=['personal_hpo_20261015/core.py','personal_hpo_20261015/run.py','shared/models.py',
           'modality_se/model.py','modality_se/personalize.py','paired_se/model.py','shared/metrics.py']
    models={f'{mode}_pair_population_seed{s}.pt':file_hash(ROOT/'modality_se/models'/f'{mode}_pair_population_seed{s}.pt')
            for mode in MODES for s in SEEDS}
    assert models==old['spec']['checkpoints']
    spec=clean(dict(old_lock_sha256=file_hash(BASE/'lock.json'),source_checkpoints=models,
        code={p:file_hash(ROOT/p) for p in code},lrs=LRS,steps=STEPS,folds=FOLDS,seeds=SEEDS,
        endpoint_losses='independent standardized MSE',selection='equal-fold endpoint MAE',
        backend='CPU for <256 support windows; MPS otherwise',previously_examined_test=True))
    if (out/'lock.json').exists():
        lock=json.loads((out/'lock.json').read_text());assert lock['spec']==spec,'Frozen spec changed'
    else:
        records=[]
        for row in old['cohort']['selected']:
            assert file_hash(Path(row['file']))==row['sha256']
            person=read_all(Path(row['file']))
            for k,v in person['indices'].items():assert v.tolist()==row[k]
            s=person['indices']['support']
            folds=rolling_folds(person['start'][s],person['end'][s])
            records.append(dict(**row,folds=clean(folds)))
        lock=dict(created_utc=utc(),spec=spec,people=records)
        atomic_json(out/'lock.json',lock)
    return lock


def folds_for_endpoint(model,z,y,start,end,c,s,folds,endpoint,folder,lh):
    results=[]
    for i,fold in enumerate(folds):
        path=folder/f'fold{i}_endpoint{endpoint}.json'
        if path.exists():
            record=json.loads(path.read_text());assert record['lock_hash']==lh
            assert file_hash(folder/record['predictions_file'])==record['predictions_sha256']
            results.append(record);continue
        t,v=np.asarray(fold['train']),np.asarray(fold['valid'])
        candidates=[];preds={}
        def add(config,fit):
            pred=predict(fit,z[v],z[t],y[t],c,s)[:,endpoint]
            mae=float(np.abs(pred.astype(float)-y[v,endpoint]).mean())
            candidates.append(dict(config=config,mae=mae));preds[key(config)]=pred
        add(None,model)
        for scope in ('head','tail'):
            for lr in LRS[scope]:
                for n,fitted,_ in trajectory(model,z[t],y[t],start[t],end[t],c,s,scope,lr,endpoint):
                    add(dict(scope=scope,lr=lr,steps=n),fitted)
        npz=folder/f'fold{i}_endpoint{endpoint}.npz'
        np.savez_compressed(npz,y=y[v,endpoint],train=t,valid=v,**preds)
        record=dict(lock_hash=lh,endpoint=endpoint,fold=i,candidates=candidates,
            train=t.tolist(),valid=v.tolist(),predictions_file=npz.name,predictions_sha256=file_hash(npz))
        atomic_json(path,record);results.append(record)
        print(json.dumps(dict(stage='fold_done',person=folder.name,endpoint=endpoint,fold=i,
                              modality=folder.parent.parent.name,seed=folder.parent.name)),flush=True)
    return results


def train_unit(person,entry,mode,seed,out,lock):
    folder=out/'units'/mode/f'seed{seed}'/person['patient'].replace(':','_');folder.mkdir(parents=True,exist_ok=True)
    lh=canonical_hash(lock)
    if (folder/'trained.json').exists():
        r=json.loads((folder/'trained.json').read_text());assert r['lock_hash']==lh
        for name,h in r['hashes'].items():assert file_hash(folder/name)==h
        return r
    tick=time.perf_counter();support=person['indices']['support']
    device='cpu' if len(support)<256 else 'mps'
    saved=torch.load(ROOT/'modality_se/models'/f'{mode}_pair_population_seed{seed}.pt',map_location='cpu',weights_only=False)
    model=restore(saved,mode,device);c,s=np.asarray(saved['center'],np.float32),np.asarray(saved['scale'],np.float32)
    x,y=person['x'][support],person['y'][support]
    start,end=person['start'][support],person['end'][support]
    z=prefix(model,x,device);folds=rolling_folds(start,end)
    assert clean(folds)==entry['folds']
    sels=[]
    for endpoint in (0,1):
        sels.append(choose(folds_for_endpoint(model,z,y,start,end,c,s,folds,endpoint,folder,lh),endpoint))
    selection=dict(lock_hash=lh,patient=person['patient'],mode=mode,seed=seed,selected_utc=utc(),
                   endpoint_selections=sels,final_test_used=False)
    atomic_json(folder/'selection.json',selection)
    audits={};files=['selection.json']
    for endpoint in (0,1):
        for scope in ('head','tail'):
            name=f'{scope}_endpoint{endpoint}.pt';files.append(name)
            cfg=sels[endpoint]['selected'][scope]['config']
            if cfg is None:fitted=copy.deepcopy(model);audit=dict(updates=0)
            else:
                _,fitted,audit=next(trajectory(model,z,y,start,end,c,s,cfg['scope'],cfg['lr'],endpoint,(cfg['steps'],)))
            state={k:v.detach().cpu().clone() for k,v in fitted.state_dict().items()}
            allowed=('change.',) if scope=='head' else ('change.','encoder.body.6.','encoder.gates.6.')
            for k,v in saved['state_dict'].items():
                if not k.startswith(allowed) or 'running_' in k or 'num_batches_tracked' in k:
                    assert torch.equal(state[k],v),'Unexpected unfrozen state '+k
            torch.save(dict(state_dict=state,center=c,scale=s,modality=mode,seed=seed,
                patient=person['patient'],endpoint=endpoint,scope=scope,config=cfg,lock_hash=lh),folder/name)
            audits[name]=audit
            del fitted
    r=dict(lock_hash=lh,patient=person['patient'],modality=mode,seed=seed,device=device,
        n_support=len(support),n_test=len(person['indices']['query']),audits=audits,
        hashes={f:file_hash(folder/f) for f in files},completed_utc=utc(),seconds=time.perf_counter()-tick)
    atomic_json(folder/'trained.json',clean(r))
    print(json.dumps(dict(stage='trained_unit',mode=mode,seed=seed,person=person['patient'],seconds=r['seconds'])),flush=True)
    del model,z
    if device=='mps':torch.mps.empty_cache()
    return r


def train(out,worker,workers):
    lock=prepare(out);lh=canonical_hash(lock);count=0
    # Fixed round-robin assignment; all workers use disjoint output paths.
    tasks=[(mode,seed,entry) for mode in MODES for seed in SEEDS for entry in lock['people']]
    people={}
    for task_id,(mode,seed,entry) in enumerate(tasks):
        if task_id%workers!=worker:continue
        pid=entry['patient']
        if pid not in people:
            path=Path(entry['file']);assert file_hash(path)==entry['sha256']
            people[pid]=read_all(path)
        train_unit(people[pid],entry,mode,seed,out,lock);count+=1
        atomic_json(out/f'progress_worker{worker}.json',dict(completed=count,total=len(tasks[worker::workers]),
            updated_utc=utc(),last_patient=pid,phase='training'))
    atomic_json(out/f'worker{worker}_complete.json',dict(lock_hash=lh,units=count,completed_utc=utc()))


def seal(out):
    lock=prepare(out);lh=canonical_hash(lock);records=[]
    for mode in MODES:
        for seed in SEEDS:
            for entry in lock['people']:
                folder=out/'units'/mode/f'seed{seed}'/entry['patient'].replace(':','_')
                r=json.loads((folder/'trained.json').read_text());assert r['lock_hash']==lh
                for name,h in r['hashes'].items():assert file_hash(folder/name)==h
                records.append(dict(path=str(folder.relative_to(out)),sha256=file_hash(folder/'trained.json')))
    assert len(records)==180
    path=out/'training_complete.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['records']==records and old['lock_hash']==lh
    else:atomic_json(path,dict(lock_hash=lh,records=records,completed_utc=utc(),units=180))


def evaluate(out):
    seal(out);lock=prepare(out);lh=canonical_hash(lock)
    for entry in lock['people']:
        person=read_all(Path(entry['file']));support,query=person['indices']['support'],person['indices']['query']
        for mode in MODES:
            for seed in SEEDS:
                folder=out/'units'/mode/f'seed{seed}'/person['patient'].replace(':','_')
                if (folder/'evaluation.json').exists():
                    old=json.loads((folder/'evaluation.json').read_text());assert old['lock_hash']==lh
                    assert file_hash(folder/'predictions.npz')==old['predictions_sha256'];continue
                selection=json.loads((folder/'selection.json').read_text())
                saved=torch.load(ROOT/'modality_se/models'/f'{mode}_pair_population_seed{seed}.pt',map_location='cpu',weights_only=False)
                model=restore(saved,mode);c,s=np.asarray(saved['center'],np.float32),np.asarray(saved['scale'],np.float32)
                zs=prefix(model,person['x'][support]);zq=prefix(model,person['x'][query]);preds={}
                for scope in ('head','tail'):
                    cols=[]
                    for endpoint in (0,1):
                        ckpt=torch.load(folder/f'{scope}_endpoint{endpoint}.pt',map_location='cpu',weights_only=False)
                        fitted=restore(ckpt,mode)
                        cols.append(predict(fitted,zq,zs,person['y'][support],c,s)[:,endpoint])
                    preds['rolling_'+scope]=np.stack(cols,1)
                chosen=[]
                for endpoint in (0,1):
                    cfg=selection['endpoint_selections'][endpoint]['selected']['combined']['config']
                    chosen.append(preds['rolling_'+('head' if cfg is None else cfg['scope'])][:,endpoint])
                preds['rolling_selected']=np.stack(chosen,1)
                oldfolder=BASE/'units'/mode/f'seed{seed}'/person['patient'].replace(':','_')
                oldrecord=json.loads((oldfolder/'unit.json').read_text())
                assert file_hash(oldfolder/'predictions.npz')==oldrecord['hashes']['predictions.npz']
                with np.load(oldfolder/'predictions.npz') as z:
                    for k in ('support','query'):np.testing.assert_array_equal(z[k],person['indices'][k])
                    np.testing.assert_array_equal(z['y'],person['y'][query])
                    for method in ('population','personal_mean','bias','pair_unadapted','pair_fixed','pair_personal_hpo'):
                        preds[method]=z[method].copy()
                np.savez_compressed(folder/'predictions.npz',y=person['y'][query],support=support,query=query,**preds)
                atomic_json(folder/'evaluation.json',dict(lock_hash=lh,metrics={m:score(person['y'][query],v) for m,v in preds.items()},
                    predictions_sha256=file_hash(folder/'predictions.npz'),evaluated_utc=utc()))
    from rolling_personal_20261015.report import report
    report(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate']);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads)
    if a.stage=='prepare':print(json.dumps(dict(locked=canonical_hash(prepare(a.output)))))
    elif a.stage=='train':train(a.output,a.worker,a.workers)
    else:evaluate(a.output)
