"""Verify completed fixed inventory and build a complete modality comparison."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modality_se.predict import load_predict
from modality_se.personalize import normalize_person
from paired_se.run import selection
from paired_se.fresh_20261012.validate import read_person
from paired_se.report import equivalent,fmt
from shared.data import load_personal_cohort
from shared.metrics import summarize
from personalized_forecasting.run import atomic_json,file_hash

SEEDS=(42,43,44)
METRICS=('mae','rmse','bias','error_sd','within_5','within_10','within_15','q90_ae','q95_ae')

def aggregate(metrics):
    return {scope:{k:dict(mean=np.mean([m[scope][k] for m in metrics],0).tolist(),seed_sd=np.std([m[scope][k] for m in metrics],0,ddof=1).tolist()) for k in METRICS} for scope in ('macro','pooled')}

def differences(a,b):
    assert set(a)==set(b)
    ids=sorted(a);delta=np.array([np.array(a[p])-np.array(b[p]) for p in ids]);rng=np.random.default_rng(20261012)
    boot=delta[rng.integers(len(ids),size=(10000,len(ids)))].mean(1)
    return dict(delta_mae=delta.mean(0).tolist(),ci95=np.quantile(boot,[.025,.975],axis=0).tolist(),people_improved=(delta<0).sum(0).tolist(),n_people=len(ids),per_patient={p:d.tolist() for p,d in zip(ids,delta)},bootstrap_unit='person after averaging seeds')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--personal',type=Path,required=True);p.add_argument('--fresh',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    completion=json.loads((a.run/'personalization_complete.json').read_text());manifest=json.loads((a.run/'personal_manifest.json').read_text())
    assert completion['status']=='complete'
    for f,h in manifest['source_hashes'].items():assert file_hash(ROOT/f)==h
    population={};records={};maxreplay=0.;replays=0;units=0;support_identities={}
    people={v['patient_id']:normalize_person(v) for group in load_personal_cohort(a.personal).values() if isinstance(group,list) for v in group}
    for r in json.loads((ROOT/'paired_se/fresh_20261012/results/cohort_audit.json').read_text()):
        if r['status']=='included':people[r['patient']]=read_person(a.fresh/(r['patient'].split(':')[1]+'.mat'),r['patient'])
    for modality in ('dual','ppg'):
        runs=[]
        for seed in SEEDS:
            folder=a.prior/f'se_last_seed{seed}' if modality=='dual' else a.run/f'population/se_ppg_seed{seed}'
            done=json.loads((folder/'completion.json').read_text());assert done['status']=='complete' and done['toy_test_only'] is False
            assert file_hash(folder/'best.pt')==done['best_checkpoint_sha256']
            metrics=json.loads((folder/'metrics.json').read_text())
            for split in ('validation','test','aami'):
                path=folder/f'predictions_{split}.npz';assert file_hash(path)==done['prediction_sha256'][path.name]
                with np.load(path) as f:equivalent(summarize(f['y'],f['pred'],f['subjects']),metrics[split])
            runs.append(dict(seed=seed,completion=done,metrics=metrics))
        population[modality]=dict(runs=runs,summary={s:aggregate([r['metrics'][s] for r in runs]) for s in ('validation','test','aami')})
        records[modality]={}
        lock=json.loads((a.run/modality/'selection.json').read_text())
        for phase,n in [('development',4),('historical_six',6),('previous_sixteen',16)]:
            rows=[]
            for path in sorted((a.run/modality/phase).glob('*/*/*/unit.json')):
                r=json.loads(path.read_text());folder=path.parent;v=people[r['patient']]
                for fn,h in r['hashes'].items():assert file_hash(folder/fn)==h
                s,q=v['support'],v['query'];identity=(r['patient'],r['seed'])
                with np.load(folder/'predictions.npz') as f:
                    np.testing.assert_array_equal(s,f['support_indices']);np.testing.assert_array_equal(q,f['query_indices']);np.testing.assert_array_equal(v['y'][q],f['y'])
                    equivalent(summarize(f['y'],f['pred'],np.repeat(r['patient'],len(q))),r['metrics'])
                    e=f['pred'].astype(float)-f['y'].astype(float)
                    independent=dict(mae=abs(e).mean(0),bias=e.mean(0),error_sd=e.std(0,ddof=1),within_5=(abs(e)<=5).mean(0),within_10=(abs(e)<=10).mean(0),within_15=(abs(e)<=15).mean(0))
                    for key,val in independent.items():np.testing.assert_allclose(val,r['metrics']['macro'][key],atol=1e-9,rtol=0)
                    if r['method']=='calibration_mean':np.testing.assert_allclose(f['pred'],np.tile(v['y'][s].astype(float).mean(0),(len(q),1)),atol=1e-9,rtol=0)
                    if phase!='development' and r['method']==lock['selected']['method']:
                        pred=load_predict(folder/'model.pt',v['x'][q],v['x'][s],v['y'][s],'mps')
                        err=float(abs(pred-f['pred']).max());maxreplay=max(err,maxreplay);np.testing.assert_allclose(pred,f['pred'],atol=1e-3,rtol=0);replays+=len(q)
                budget=(r['config']['support_hash'],r['config']['query_hash'])
                if identity in support_identities:assert support_identities[identity]==budget
                else:support_identities[identity]=budget
                rows.append(r);units+=1
            methods=['population','calibration_mean','bias']+([r['method'] for r in lock['ranking']] if phase=='development' else [lock['selected']['method']])
            expected={(person,seed,method) for person in {r['patient'] for r in rows} for seed in SEEDS for method in methods}
            assert len({r['patient'] for r in rows})==n and len(rows)==len(expected) and {(r['patient'],r['seed'],r['method']) for r in rows}==expected
            if phase=='development':equivalent(selection(rows)['selected'],lock['selected'])
            records[modality][phase]=rows
    assert units==816
    results=dict(date='20261012',population=population,personalization={},selections=completion['selections'],independent_new_test=False)
    results['population_validation_selected_modality']=min(('dual','ppg'),key=lambda m:np.mean(population[m]['summary']['validation']['macro']['mae']['mean']))
    results['population_paired_differences']={}
    for split in ('validation','test','aami'):
        means={}
        for modality in ('dual','ppg'):
            per=[r['metrics'][split]['per_patient'] for r in population[modality]['runs']]
            means[modality]={pid:np.mean([d[pid]['mae'] for d in per],0) for pid in per[0]}
        results['population_paired_differences'][split]=differences(means['ppg'],means['dual'])
    for phase in ('development','historical_six','previous_sixteen'):
        arms={m:json.loads((a.run/m/(phase+'_summary.json')).read_text()) for m in ('dual','ppg')}
        paired={}
        for m in arms:
            selected=completion['selections'][m]['selected']['method'];model=arms[m][selected]
            paired[m]={k:differences(model['per_patient_mae'],arms[m][k]['per_patient_mae']) for k in ('calibration_mean','bias')}
            other='dual' if m=='ppg' else 'ppg'
            paired[m]['other_modality_bias']=differences(model['per_patient_mae'],arms[other]['bias']['per_patient_mae'])
            paired[m]['beats_both_controls_both_pressures']=bool(all(np.all(np.array(paired[m][k]['delta_mae'])<0) for k in ('calibration_mean','bias')))
            paired[m]['beats_mean_and_both_modality_biases']=bool(all(np.all(np.array(paired[m][k]['delta_mae'])<0) for k in ('calibration_mean','bias','other_modality_bias')))
        pm=completion['selections']['ppg']['selected']['method'];dm=completion['selections']['dual']['selected']['method']
        results['personalization'][phase]=dict(methods=arms,paired=paired,ppg_minus_dual=differences(arms['ppg'][pm]['per_patient_mae'],arms['dual'][dm]['per_patient_mae']))
        variability={}
        for m in ('dual','ppg'):
            variability[m]={}
            for method in arms[m]:
                seed_metrics=[]
                for seed in SEEDS:
                    ys=[];ps=[];ids=[]
                    for r in records[m][phase]:
                        if r['method']!=method or r['seed']!=seed:continue
                        f=a.run/m/phase/r['patient']/f'seed{seed}'/method/'predictions.npz'
                        with np.load(f) as z:ys.append(z['y']);ps.append(z['pred']);ids.extend([r['patient']]*len(z['y']))
                    seed_metrics.append(summarize(np.concatenate(ys),np.concatenate(ps),np.array(ids)))
                variability[m][method]=dict(by_seed={str(s):v for s,v in zip(SEEDS,seed_metrics)},summary=aggregate(seed_metrics))
        results['personalization'][phase]['seed_variability']=variability
    export=ROOT/'modality_se/results';export.mkdir(exist_ok=True)
    atomic_json(export/'comparison.json',results)
    atomic_json(export/'verification.json',dict(units=units,replayed_target_predictions=replays,max_replay_absolute_difference=maxreplay,
        exact_support_query_match_across_modalities=True,independent_metrics_recomputed=True,seed_count=3,previously_examined_targets=True))
    atomic_json(export/'per_person_seed_metrics.json',[dict(modality=m,phase=p,patient=r['patient'],seed=r['seed'],method=r['method'],metrics=r['metrics']) for m in records for p in records[m] for r in records[m][p]])
    shutil.copyfile(a.run/'personal_manifest.json',export/'manifest.json')
    weights=ROOT/'modality_se/models';weights.mkdir(exist_ok=True)
    for m in ('dual','ppg'):
        shutil.copyfile(a.run/m/'selection.json',export/(m+'_selection.json'))
        for seed in SEEDS:
            source=a.prior/f'se_last_seed{seed}/best.pt' if m=='dual' else a.run/f'population/se_ppg_seed{seed}/best.pt'
            ck=torch.load(source,map_location='cpu',weights_only=False)
            torch.save(dict(kind='population',modality=m,seed=seed,state_dict=ck['state_dict'],center=ck['center'],scale=ck['scale']),weights/f'{m}_population_seed{seed}.pt')
            source=ROOT/f'paired_se/models/paired_population_seed{seed}.pt' if m=='dual' else a.run/f'pair_population/ppg_seed{seed}/best.pt'
            ck=torch.load(source,map_location='cpu',weights_only=False);ck['modality']=m;ck['kind']='pair_population';torch.save(ck,weights/f'{m}_pair_population_seed{seed}.pt')
        for phase in ('historical_six','previous_sixteen'):
            method=completion['selections'][m]['selected']['method']
            for r in records[m][phase]:
                if r['method']!=method:continue
                source=a.run/m/phase/r['patient']/f"seed{r['seed']}"/method/'model.pt'
                target=weights/m/phase/f"{r['patient'].replace(':','_')}_seed{r['seed']}.pt";target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    atomic_json(weights/'manifest.json',{str(f.relative_to(weights)):file_hash(f) for f in sorted(weights.rglob('*.pt'))})
    lines=['# PPG-only versus dual-channel CNN+SE — 20261012 results','',
        'Three seeds (42/43/44), identical source splits and early20% personal budgets. Values are participant-macro SBP / DBP, averaged over seeds, not an ensemble. All test cohorts have been examined previously; do not interpret them as new independent acceptance.',
        '', '## Population models','', '| Split | Input | MAE | RMSE | Bias | Error SD | Within5 % | Within10 % | Within15 % |', '|---|---|---|---|---|---|---|---|---|']
    for split in ('validation','test','aami'):
        for m in ('dual','ppg'):
            stats=population[m]['summary'][split]['macro'];vals=[fmt(stats[k]['mean'],k.startswith('within')) for k in METRICS[:7]]
            lines.append('| '+split+' | '+m+' | '+' | '.join(vals)+' |')
    for phase in ('development','historical_six','previous_sixteen'):
        lines+=['',f'## Personalization — {phase}','', '| Input / method | MAE | RMSE | Bias | Error SD | Within5 % | Within10 % | Within15 % |','|---|---|---|---|---|---|---|---|']
        for m in ('dual','ppg'):
            method=completion['selections'][m]['selected']['method']
            for k in ('population','calibration_mean','bias',method):
                stats=results['personalization'][phase]['methods'][m][k]['macro'];vals=[fmt(stats[z],z.startswith('within')) for z in METRICS[:7]]
                lines.append('| '+m+' / '+k+' | '+' | '.join(vals)+' |')
            lines+=['',f"{m}: selected on development `{method}`; both-controls/both-endpoints success here = **{results['personalization'][phase]['paired'][m]['beats_both_controls_both_pressures']}**.",'']
    lines+=['','## Verification and interpretation',f'- Verified all {units} units and replayed {replays} target predictions; maximum difference {maxreplay:.8f} mmHg.',
        '- Complete per-person/seed results, seed variability, pressure ranges and paired person-bootstrap intervals are in the JSON files.',
        '- SE architecture and parameter count are matched. PPG-only zeroes the ECG input internally, including the reference encoder; it is not an inference-time deletion from a dual-trained model.',
        '- Pair heads learn from source participants; personal fitting only receives early references and timestamps. Every method has the same support/query IDs.',
        '- Threshold fractions and error statistics do not establish clinical blood-pressure-device certification.']
    (export/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic_json(a.run/'study_complete.json',dict(status='verified_complete',date='20261012',units=units,result=str(export/'comparison.json'),verification=str(export/'verification.json')))
    print('VERIFIED COMPLETE',units,'units',replays,'replayed predictions',flush=True)

if __name__=='__main__':main()
