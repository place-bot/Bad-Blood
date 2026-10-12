"""Locked, outcome-blind additional-participant evaluation; no hyperparameter search."""
import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlencode
from urllib.request import urlopen
import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from paired_se.run import restore, fit_personal
from paired_se.model import chronological_pairs, predict_bank
from paired_se.predict import load_predict
from paired_se.report import equivalent
from se_personalization.run import summarize_phase
from shared.data import _read_ref, _read_string
from shared.metrics import summarize
from personalized_forecasting.run import atomic_json, file_hash, array_hash, canonical_hash
from personalized_forecasting.adaptation import predict

CONFIG = dict(scope='head', lr=.0003, steps=25)
SEEDS = (42, 43, 44)
METHODS = ('population', 'calibration_mean', 'bias', 'paired')
FOLDER = 'https://drive.google.com/drive/folders/1QZ-Z-C9MoLk1S9MovF3Tt6NjAEGbSJI3'

def freeze():
    manifest = json.loads((HERE.parent/'results/manifest.json').read_text())
    forbidden = set(manifest['population']['excluded_patients'])
    for part in manifest['population']['partitions'].values(): forbidden.update(part['patients'])
    page = urlopen(FOLDER, timeout=60).read().decode()
    decoded = page.replace('\\x22','"').replace('\\x5b','[').replace('\\x5d',']').replace('\\/','/')
    entries = dict((n, fid) for fid,n in re.findall(r'\["([\w-]+)",\["[\w-]+"\],"([^"]+)"',decoded) if re.fullmatch(r'p\d{6}\.mat',n))
    candidates = [dict(patient='vitaldb:'+n[:-4], file=n, drive_id=fid) for n,fid in sorted(entries.items()) if 'vitaldb:'+n[:-4] not in forbidden]
    lock = json.loads((HERE.parent/'results/selection.json').read_text())
    assert lock['selected']['config'] == CONFIG
    record = dict(date='20261012',created_utc=datetime.now(timezone.utc).isoformat(),folder=FOLDER,
        directory_page_unique_files=len(entries),candidates=candidates,excluded_identity_count=len(forbidden),
        excluded_identities=sorted(forbidden),config=CONFIG,seeds=list(SEEDS),
        prior_selection_sha256=file_hash(HERE.parent/'results/selection.json'),
        model_sha256={str(s):file_hash(HERE.parent/f'models/paired_population_seed{s}.pt') for s in SEEDS},
        code_sha256={p:file_hash(ROOT/p) for p in ('paired_se/run.py','paired_se/model.py','shared/models.py','shared/metrics.py')},
        protocol_sha256=file_hash(HERE/'PROTOCOL.md'))
    if (HERE/'candidate_manifest.json').exists(): raise ValueError('Already frozen')
    atomic_json(HERE/'candidate_manifest.json',record)
    print('Frozen',len(candidates),'unseen candidates',flush=True)

class Form(HTMLParser):
    def __init__(self): super().__init__(); self.fields={}; self.action=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':self.action=a['action']
        if tag=='input' and a.get('type')=='hidden':self.fields[a['name']]=a.get('value','')

def download(row,folder):
    path=folder/row['file'];folder.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not h5py.is_hdf5(path):raise ValueError('Existing source is not HDF5')
        return path
    url='https://drive.google.com/uc?'+urlencode(dict(export='download',id=row['drive_id']))
    response=urlopen(url,timeout=90)
    if 'text/html' in response.headers.get('Content-Type',''):
        form=Form();form.feed(response.read().decode());response.close()
        if form.action!='https://drive.usercontent.google.com/download':raise ValueError('Unexpected download form')
        response=urlopen(form.action+'?'+urlencode(form.fields),timeout=120)
    with response, path.with_suffix('.mat.part').open('wb') as out:shutil.copyfileobj(response,out,1024*1024)
    partial=path.with_suffix('.mat.part')
    if not h5py.is_hdf5(partial):raise ValueError('Download is not a MATLAB HDF5 file')
    partial.replace(path)
    return path

def read_person(path,patient):
    subject=patient.split(':')[1]
    with h5py.File(path,'r') as f:
        g=f['Subj_Wins']
        # MATLAB serializes a one-window struct directly, not as cell references.
        # It cannot provide both calibration and test windows under this protocol.
        if h5py.check_dtype(ref=g['T'].dtype) is None:
            assert g['T'].size!=1250, 'Single-window record: insufficient early/later windows'
            raise ValueError('Unsupported non-reference timestamp representation')
        n=g['T'].size
        x=np.empty((n,2,1250),np.float32);y=np.empty((n,2),np.float32)
        start=np.empty(n);end=np.empty(n);cases=set()
        for i in range(n):
            assert _read_string(f,g,'SubjectID',i)==subject, 'Subject identity mismatch'
            cases.add(_read_string(f,g,'CaseID',i))
            t=_read_ref(f,g,'T',i)
            assert t.shape==(1250,) and np.isfinite(t).all() and np.allclose(np.diff(t),.008,rtol=0,atol=1e-6), 'Invalid timestamps'
            start[i]=t[0];end[i]=t[-1]+.008
            wave=np.asarray([_read_ref(f,g,k,i) for k in ('ECG_F','PPG_F')],np.float32)
            assert wave.shape==(2,1250) and np.isfinite(wave).all() and (np.ptp(wave,axis=1)>1e-8).all(), 'Invalid signal'
            x[i]=(wave-wave.min(1,keepdims=True))/np.ptp(wave,axis=1,keepdims=True)
            for c,k in enumerate(('SegSBP','SegDBP')):
                v=_read_ref(f,g,k,i);assert v.shape==(1,), 'Non scalar BP';y[i,c]=v[0]
    assert len(cases)==1 and n>0 and (np.diff(start)>0).all(), 'Mixed cases or unordered time'
    assert np.isfinite(y).all() and (y[:,0]>y[:,1]).all(), 'Invalid BP label'
    boundary=start[0]+.2*(end[-1]-start[0])
    support=np.flatnonzero(end<=boundary+1e-6);query=np.flatnonzero(start>=boundary+10-1e-6)
    assert len(support)>=2 and len(query)>=2, 'Insufficient early/later windows'
    chronological_pairs(start[support],end[support])
    assert end[support].max()+10<=start[query].min()+1e-6
    return dict(patient=patient,x=x,y=y,start=start,end=end,support=support,query=query,boundary=float(boundary),case=next(iter(cases)))

def run(args):
    torch.set_num_threads(4)
    lock=json.loads((HERE/'candidate_manifest.json').read_text());lh=canonical_hash(lock)
    assert lock['config']==CONFIG and lock['seeds']==list(SEEDS)
    assert lock['protocol_sha256']==file_hash(HERE/'PROTOCOL.md')
    for p,h in lock['code_sha256'].items():assert file_hash(ROOT/p)==h, 'Frozen model code changed'
    for s,h in lock['model_sha256'].items():assert file_hash(HERE.parent/f'models/paired_population_seed{s}.pt')==h
    args.output.mkdir(parents=True,exist_ok=True)
    models={s:torch.load(HERE.parent/f'models/paired_population_seed{s}.pt',map_location='cpu',weights_only=False) for s in SEEDS}
    audit=[];records=[];replays=0;max_error=0.
    for candidate in lock['candidates']:
        person=candidate['patient'];print('LOAD',person,flush=True)
        assert person not in lock['excluded_identities']
        path=download(candidate,args.data)
        try:p=read_person(path,person)
        except (AssertionError,ValueError) as e:
            audit.append(dict(patient=person,status='ineligible',reason=str(e),file_sha256=file_hash(path)))
            atomic_json(args.output/'cohort_audit.json',audit);print('EXCLUDE',person,str(e),flush=True);continue
        s,q=p['support'],p['query'];x,y=p['x'],p['y']
        data_hash=array_hash(x,y,p['start']);source_hash=file_hash(path)
        audit.append(dict(patient=person,status='included',case=p['case'],n_total=len(y),n_support=len(s),n_query=len(q),
            boundary=p['boundary'],support_indices=s.tolist(),query_indices=q.tolist(),data_hash=data_hash,file_sha256=source_hash,
            support_hash=array_hash(s),query_hash=array_hash(q)))
        atomic_json(args.output/'cohort_audit.json',audit)
        for seed in SEEDS:
            torch.manual_seed(seed);saved=models[seed];pair=restore(saved,args.device);c,scale=saved['center'],saved['scale']
            sp,_=predict(pair.encoder,x[s],c,scale,args.device);qp,_=predict(pair.encoder,x[q],c,scale,args.device)
            controls=dict(population=qp,calibration_mean=np.tile(y[s].astype(float).mean(0),(len(q),1)),bias=qp+(y[s].astype(float)-sp).mean(0))
            for method in METHODS:
                folder=args.output/person/f'seed{seed}'/method;folder.mkdir(parents=True,exist_ok=True);marker=folder/'unit.json'
                config=dict(lock_hash=lh,patient=person,seed=seed,method=method,data_hash=data_hash,support_hash=array_hash(s),query_hash=array_hash(q),device=args.device)
                if marker.exists():
                    row=json.loads(marker.read_text());assert row['config']==config
                    for fn,h in row['hashes'].items():assert file_hash(folder/fn)==h
                else:
                    files=['predictions.npz']
                    if method=='paired':
                        fitted,fit_audit=fit_personal(pair,x[s],y[s],p['start'][s],p['end'][s],c,scale,CONFIG,args.device)
                        pred=predict_bank(fitted,x[q],x[s],y[s],c,scale,args.device)
                        torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in fitted.state_dict().items()},center=c,scale=scale,seed=seed,config=config),folder/'model.pt');files.append('model.pt')
                    else:pred=controls[method];fit_audit=dict(query_used=False,support_count=len(s))
                    np.savez_compressed(folder/'predictions.npz',y=y[q],pred=pred,support_indices=s,query_indices=q)
                    row=dict(config=config,patient=person,seed=seed,method=method,audit=fit_audit,
                        metrics=summarize(y[q],pred,np.repeat(person,len(q))),hashes={fn:file_hash(folder/fn) for fn in files})
                    atomic_json(marker,row)
                with np.load(folder/'predictions.npz') as z:
                    np.testing.assert_array_equal(z['y'],y[q]);np.testing.assert_array_equal(z['support_indices'],s);np.testing.assert_array_equal(z['query_indices'],q)
                    equivalent(summarize(z['y'],z['pred'],np.repeat(person,len(q))),row['metrics'])
                    if method=='paired':
                        again=load_predict(folder/'model.pt',x[q],x[s],y[s],args.device)
                        max_error=max(max_error,float(np.max(abs(again-z['pred']))));replays+=len(q)
                    else:again=controls[method]
                    np.testing.assert_allclose(z['pred'],again,rtol=0,atol=1e-4)
                row['_prediction_path']=str(folder/'predictions.npz');records.append(row)
                print(person,seed,method,row['metrics']['macro']['mae'],flush=True)
    methods=summarize_phase(records);n=sum(r['status']=='included' for r in audit)
    assert len(records)==n*len(SEEDS)*len(METHODS)
    result=dict(date='20261012',lock_hash=lh,config=CONFIG,methods=methods,n_candidates=len(audit),n_included=n,
        support_windows=sum(r.get('n_support',0) for r in audit),query_windows=sum(r.get('n_query',0) for r in audit),
        verification=dict(no_prior_identity_overlap=True,units=len(records),replayed_predictions=replays,max_replay_error=max_error),paired_differences={})
    for baseline in ('calibration_mean','bias'):
        ids=sorted(methods['paired']['per_patient_mae'])
        dif=np.array([np.array(methods['paired']['per_patient_mae'][p])-methods[baseline]['per_patient_mae'][p] for p in ids])
        boot=dif[np.random.default_rng(20261012).integers(0,n,(10000,n))].mean(1)
        result['paired_differences'][baseline]=dict(delta_mae=dif.mean(0).tolist(),ci95=np.quantile(boot,[.025,.975],axis=0).tolist(),people_improved=(dif<0).sum(0).tolist(),people=n)
    result['beats_both_controls_both_pressures']=bool(all(np.all(np.array(methods['paired']['macro']['mae'])<methods[m]['macro']['mae']) for m in ('calibration_mean','bias')))
    result['per_seed_mae']={str(seed):{m:np.mean([r['metrics']['macro']['mae'] for r in records if r['seed']==seed and r['method']==m],0).tolist() for m in METHODS} for seed in SEEDS}
    atomic_json(args.output/'summary.json',result)
    atomic_json(args.output/'per_person_seed_metrics.json',[{k:r[k] for k in ('patient','seed','method','metrics')} for r in records])
    print('COMPLETE',json.dumps({m:methods[m]['macro']['mae'] for m in METHODS}),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');a.add_argument('--data',type=Path);a.add_argument('--output',type=Path);a.add_argument('--device',default='mps');args=a.parse_args()
    if args.freeze:freeze()
    else:
        if args.data is None or args.output is None:a.error('--data and --output required')
        run(args)
