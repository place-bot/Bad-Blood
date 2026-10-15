"""Matched PairSE source training and fixed-budget personalization for two inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modality_se.model import ModalitySE
from paired_se import run as pair_run
from paired_se.model import PairSE,predict_bank,chronological_pairs
from paired_se.fresh_20261012.validate import read_person
from shared.data import load_population,load_personal_cohort
from shared.metrics import summarize
from se_personalization.run import load_se,dense_support,summarize_phase
from personalized_forecasting.adaptation import predict
from personalized_forecasting.run import atomic_json,file_hash,canonical_hash,array_hash

SEEDS=(42,43,44)

def restore(saved,modality,device='cpu'):
    model=PairSE(ModalitySE(modality)).to(device).eval();model.load_state_dict(saved['state_dict']);return model

def normalize_person(v):
    return dict(patient=v['patient_id'],x=v['x'],y=v['y'],start=v['start'],end=v['end_exclusive'],support=dense_support(v),query=v['query_indices'])

def load_original(path,seed,modality):
    if modality=='dual':return load_se(path,seed)
    done=json.loads((path.parent/'completion.json').read_text());saved=torch.load(path,map_location='cpu',weights_only=False)
    assert done['status']=='complete' and done['toy_test_only'] is False
    assert done['best_checkpoint_sha256']==file_hash(path) and saved['variant']=='se_ppg' and saved['seed']==seed
    assert saved['hashes']['manifest']==done['manifest_hash']
    return saved

def run_phase(people,modality,models,originals,output,manifest_hash,phase,lock=None):
    records=[]
    for seed in SEEDS:
        saved=models[seed];base=restore(saved,modality,'mps');c,scale=np.asarray(saved['center']),np.asarray(saved['scale'])
        original=ModalitySE(modality);original.load_state_dict(originals[seed]['state_dict'])
        for v in people:
            s,q=v['support'],v['query'];x,y=v['x'],v['y'];person=v['patient']
            assert not set(s)&set(q) and max(v['end'][s])+10<=min(v['start'][q])+1e-6
            chronological_pairs(v['start'][s],v['end'][s])
            sp,_=predict(original,x[s],c,scale,'mps');qp,_=predict(original,x[q],c,scale,'mps')
            controls=dict(population=qp,calibration_mean=np.tile(y[s].astype(float).mean(0),(len(q),1)),bias=qp+(y[s].astype(float)-sp).mean(0))
            configs=pair_run.configurations() if phase=='development' else [lock['selected']['config']]
            methods=[(k,'control') for k in controls]+[(pair_run.name(cfg),cfg) for cfg in configs]
            for method,cfg in methods:
                directory=output/modality/phase/person/f'seed{seed}'/method;directory.mkdir(parents=True,exist_ok=True)
                config=dict(modality=modality,seed=seed,patient=person,method=method,setting=cfg,phase=phase,manifest_hash=manifest_hash,
                    support_hash=array_hash(s),query_hash=array_hash(q),selection_hash=canonical_hash(lock) if lock else None)
                stamp=directory/'unit.json'
                if stamp.exists():
                    row=json.loads(stamp.read_text());assert row['config']==config
                    for f,h in row['hashes'].items():assert file_hash(directory/f)==h
                else:
                    files=['predictions.npz']
                    if cfg=='control':pred=controls[method];audit=dict(query_used=False)
                    else:
                        fitted,audit=pair_run.fit_personal(base,x[s],y[s],v['start'][s],v['end'][s],c,scale,cfg,'mps')
                        pred=predict_bank(fitted,x[q],x[s],y[s],c,scale,'mps')
                        torch.save(dict(modality=modality,state_dict={k:t.detach().cpu().clone() for k,t in fitted.state_dict().items()},center=c,scale=scale,config=config),directory/'model.pt');files.append('model.pt')
                        del fitted
                    np.savez_compressed(directory/'predictions.npz',y=y[q],pred=pred,support_indices=s,query_indices=q)
                    row=dict(patient=person,seed=seed,method=method,config=config,n_support=len(s),metrics=summarize(y[q],pred,np.repeat(person,len(q))),audit=audit,hashes={f:file_hash(directory/f) for f in files})
                    atomic_json(stamp,row)
                row['_prediction_path']=str(directory/'predictions.npz');records.append(row)
            print('PERSON DONE',modality,phase,seed,person,flush=True)
        del base,original
        torch.mps.empty_cache()
    summary=summarize_phase(records)
    atomic_json(output/modality/(phase+'_summary.json'),summary)
    return records,summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--personal',type=Path,required=True);p.add_argument('--fresh',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);population=load_population(a.data);originals={};models={}
    prior_manifest=json.loads((ROOT/'paired_se/results/manifest.json').read_text())
    assert prior_manifest['population']==population['manifest']
    manifest=dict(date='20261012',protocol_sha256=file_hash(ROOT/'modality_se/PROTOCOL.md'),source_hashes={f:file_hash(ROOT/f) for f in
        ('modality_se/model.py','modality_se/personalize.py','paired_se/run.py','paired_se/model.py','shared/metrics.py')},population_manifest_hash=population['manifest_hash'],seeds=list(SEEDS),checkpoints={})
    for modality in ('dual','ppg'):
        originals[modality]={}
        for seed in SEEDS:
            path=a.prior/f'se_last_seed{seed}'/'best.pt' if modality=='dual' else a.output/f'population/se_ppg_seed{seed}'/'best.pt'
            saved=load_original(path,seed,modality)
            assert saved['hashes']['manifest']==population['manifest_hash']
            originals[modality][seed]=saved;manifest['checkpoints'][f'{modality}_{seed}']=file_hash(path)
    mh=canonical_hash(manifest);mp=a.output/'personal_manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:atomic_json(mp,manifest)
    # Model constructor substitution is scoped to this dedicated worker process.
    for modality in ('dual','ppg'):
        pair_run.BPNet=lambda variant='se_last',m=modality:ModalitySE(m)
        models[modality]={}
        for seed in SEEDS:
            if modality=='dual':
                path=ROOT/f'paired_se/models/paired_population_seed{seed}.pt'
                expected=json.loads((ROOT/'paired_se/models/manifest.json').read_text())[path.name]
                assert file_hash(path)==expected
                saved=torch.load(path,map_location='cpu',weights_only=False)
                for k,v in originals[modality][seed]['state_dict'].items():assert torch.equal(saved['state_dict']['encoder.'+k],v)
            else:
                saved=pair_run.train_population(seed,originals[modality][seed],population,a.output/f'pair_population/ppg_seed{seed}',mh,'mps')
            models[modality][seed]=saved
            print('PAIR POPULATION DONE',modality,seed,flush=True)
    # Load development only before writing both independent modality selections.
    audit=json.loads((ROOT/'audit_snapshot.json').read_text());proto=audit['final_personalization_protocol'];proto['patients']=[v for v in proto['patients'] if v['role']=='development'];proto['target_subjects']=[]
    atomic_json(a.output/'development_only_audit.json',audit)
    development=[normalize_person(v) for v in load_personal_cohort(a.personal,a.output/'development_only_audit.json')['development']]
    locks={}
    for modality in ('dual','ppg'):
        rows,_=run_phase(development,modality,models[modality],originals[modality],a.output,mh,'development')
        lock=pair_run.selection(rows);lock['modality']=modality;lock['manifest_hash']=mh;locks[modality]=lock
        path=a.output/modality/'selection.json'
        if path.exists():assert json.loads(path.read_text())==lock
        else:atomic_json(path,lock)
        print('LOCKED',modality,json.dumps(lock['selected']),flush=True)
    # Only now open target label files. Previously examined cohorts, not new claims.
    targetaudit=json.loads((ROOT/'audit_snapshot.json').read_text());p=targetaudit['final_personalization_protocol'];p['patients']=[v for v in p['patients'] if v['role']=='target'];p['development_subjects']=[]
    atomic_json(a.output/'target_only_audit.json',targetaudit)
    historic=[normalize_person(v) for v in load_personal_cohort(a.personal,a.output/'target_only_audit.json')['target']]
    fixed=json.loads((ROOT/'paired_se/fresh_20261012/results/cohort_audit.json').read_text());recent=[]
    excluded=set(population['manifest']['excluded_patients']);source=set().union(*(set(v['patients']) for v in population['manifest']['partitions'].values()))
    for r in fixed:
        if r['status']!='included':continue
        path=a.fresh/(r['patient'].split(':')[1]+'.mat');assert file_hash(path)==r['file_sha256']
        v=read_person(path,r['patient']);assert r['patient'] not in excluded|source
        np.testing.assert_array_equal(v['support'],r['support_indices']);np.testing.assert_array_equal(v['query'],r['query_indices']);recent.append(v)
    assert len(historic)==6 and len(recent)==16
    for phase,people in [('historical_six',historic),('previous_sixteen',recent)]:
        for modality in ('dual','ppg'):run_phase(people,modality,models[modality],originals[modality],a.output,mh,phase,locks[modality])
    atomic_json(a.output/'personalization_complete.json',dict(status='complete',manifest_hash=mh,selections=locks,development_people=4,historical_people=6,previously_evaluated_people=16,independent_new_test=False))
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'modality_se/report.py'),'--run',str(a.output),'--prior',str(a.prior),'--personal',str(a.personal),'--fresh',str(a.fresh)],check=True)

if __name__=='__main__':main()
