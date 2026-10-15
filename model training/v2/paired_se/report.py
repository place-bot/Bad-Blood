"""Verify a completed pair study and publish metrics/code-checkpoint artifacts."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paired_se.run import configurations,name,selection,restore
from paired_se.predict import load_predict
from paired_se.model import chronological_pairs
from shared.data import load_personal_cohort
from shared.metrics import summarize
from se_personalization.run import dense_support,summarize_phase,SEEDS,DEV_IDS,TARGET_IDS
from se_personalization.run import load_se
from shared.models import BPNet
from personalized_forecasting.adaptation import predict
from personalized_forecasting.run import canonical_hash,file_hash,atomic_json


def equivalent(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Different summary keys')
        for k in a:equivalent(a[k],b[k])
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('Different list lengths')
        for x,y in zip(a,b):equivalent(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)
    elif a!=b:raise ValueError('Different metadata')


def fmt(values,percent=False):
    return ' / '.join('N/A' if v is None else f'{v*(100 if percent else 1):.2f}' for v in values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--individual-root',type=Path,required=True)
    p.add_argument('--population-root',type=Path,required=True)
    p.add_argument('--export',type=Path,required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(4)
    manifest=json.loads((a.run/'manifest.json').read_text());mh=canonical_hash(manifest)
    lock=json.loads((a.run/'selection.json').read_text())
    if lock['manifest_hash']!=mh:raise ValueError('Selection provenance')
    for fn,sha in manifest['source'].items():
        if file_hash(Path(__file__).with_name(fn))!=sha:raise ValueError('Frozen training source changed')
    cohort=load_personal_cohort(a.individual_root)
    patients={v['patient_id']:v for role in ('development','target') for v in cohort[role]}
    originals={}
    for seed in SEEDS:
        path=a.population_root/f'se_last_seed{seed}'/'best.pt'
        if file_hash(path)!=manifest['checkpoints'][str(seed)]:raise ValueError('Original checkpoint mismatch')
        saved=load_se(path,seed);model=BPNet(variant='se_last');model.load_state_dict(saved['state_dict'],strict=True)
        originals[seed]=(model,saved)
    summaries={};records={};replayed=0;max_error=0.;control_cache={}
    for phase,ids in [('development',DEV_IDS),('test',TARGET_IDS)]:
        saved=json.loads((a.run/f'{phase}_summary.json').read_text())
        methods=['population','calibration_mean','bias']+([name(c) for c in configurations()] if phase=='development' else [lock['selected']['method']])
        expected={(person,seed,m) for person in ids for seed in SEEDS for m in methods};rows=[]
        for f in sorted((a.run/phase).glob('*/*/*/unit.json')):
            r=json.loads(f.read_text());directory=f.parent;patient=patients[r['patient']]
            if r['config']['manifest_hash']!=mh:raise ValueError('Unit provenance')
            if phase=='test' and r['config']['selection_hash']!=canonical_hash(lock):raise ValueError('Wrong locked selection')
            for fn,sha in r['hashes'].items():
                if file_hash(directory/fn)!=sha:raise ValueError('Artifact checksum mismatch')
            support=dense_support(patient);q=patient['query_indices']
            chronological_pairs(patient['start'][support],patient['end_exclusive'][support])
            if not (patient['end_exclusive'][support].max()+10<=patient['start'][q].min()+1e-6):raise ValueError('Boundary gap violated')
            with np.load(directory/'predictions.npz') as z:
                np.testing.assert_array_equal(z['support_indices'],support);np.testing.assert_array_equal(z['query_indices'],q)
                np.testing.assert_array_equal(z['y'],patient['y'][q])
                equivalent(summarize(z['y'],z['pred'],np.repeat(r['patient'],len(q))),r['metrics'])
                if r['method']=='calibration_mean':
                    np.testing.assert_allclose(z['pred'],np.tile(patient['y'][support].astype(float).mean(0),(len(q),1)),atol=1e-6,rtol=0)
                if r['method'] in ('population','bias'):
                    cache_key=(r['patient'],r['seed'])
                    if cache_key not in control_cache:
                        original,meta=originals[r['seed']]
                        sp,_=predict(original,patient['x'][support],meta['center'],meta['scale'],a.device)
                        qp,_=predict(original,patient['x'][q],meta['center'],meta['scale'],a.device)
                        control_cache[cache_key]=(qp,qp+(patient['y'][support].astype(float)-sp).mean(0))
                    np.testing.assert_allclose(z['pred'],control_cache[cache_key][r['method']=='bias'],atol=1e-4,rtol=0)
                if phase=='test' and r['method']==lock['selected']['method']:
                    pred=load_predict(directory/'model.pt',patient['x'][q],patient['x'][support],patient['y'][support],a.device)
                    error=float(np.max(abs(pred-z['pred'])));max_error=max(max_error,error);replayed+=len(q)
                    np.testing.assert_allclose(pred,z['pred'],atol=1e-4,rtol=0)
            r['_prediction_path']=str(directory/'predictions.npz');rows.append(r)
        if len(rows)!=len(expected) or {(r['patient'],r['seed'],r['method']) for r in rows}!=expected:
            raise ValueError('Incomplete/duplicate inventory')
        equivalent(summarize_phase(rows),saved['methods'])
        records[phase]=rows;summaries[phase]=saved
    recomputed=selection(records['development'])
    equivalent(recomputed['selected'],lock['selected'])
    a.export.mkdir(parents=True,exist_ok=True)
    for fn in ('manifest.json','selection.json','development_summary.json','test_summary.json'):
        shutil.copyfile(a.run/fn,a.export/fn)
    checks=dict(development_units=len(records['development']),test_units=len(records['test']),
        replayed_query_predictions=replayed,max_replay_absolute_difference=max_error,
        checksum_metrics_and_temporal_guards_pass=True,selection_recomputed_from_development=True)
    atomic_json(a.export/'verification.json',checks)
    atomic_json(a.export/'per_seed_person_metrics.json',[
        dict(phase=phase,patient=r['patient'],seed=r['seed'],method=r['method'],
             n_query=r['metrics']['n_windows'],metrics=r['metrics']['macro'])
        for phase in ('development','test') for r in records[phase]])
    target=summaries['test'];m=lock['selected']['method']
    labels={'population':'Original SE','calibration_mean':'Personal mean','bias':'Original SE + personal bias',m:'Selected reference-conditioned SE'}
    lines=['# Reference-conditioned CNN+SE results','',
        'October 12, 2026. SBP / DBP; mmHg unless marked %. Three seeds averaged, not ensembled.',
        '',f"Joint MAE objective reached: **{target['beats_both_controls_both_pressures']}**.",
        f"Development-only choice: `{m}`. Six target people were previously examined; this is exploratory, not a new independent test.",
        '', '## Later-query participant-macro performance','',
        '| Model | MAE | RMSE | Signed bias | Error SD | Within 5 % | Within 10 % | Within 15 % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in ('population','calibration_mean','bias',m):
        z=target['methods'][k]['macro']
        values=[fmt(z[n]) for n in ('mae','rmse','bias','error_sd')]+[fmt(z[n],True) for n in ('within_5','within_10','within_15')]
        lines.append('| '+labels[k]+' | '+' | '.join(values)+' |')
    lines += ['', 'The SD above averages within-person sample error SDs. Full pooled metrics, tail errors, correlations and person-specific MAE are in the JSON results.',
        '', '## Per-person MAE','', '| Person | Early windows | Query windows | Mean | Bias correction | Pair model |',
        '|---|---:|---:|---:|---:|---:|']
    for person in TARGET_IDS:
        v=manifest['personal'][person]
        errors=[fmt(target['methods'][k]['per_patient_mae'][person]) for k in ('calibration_mean','bias',m)]
        lines.append(f"| {person} | {len(v['support'])} | {len(v['query'])} | "+' | '.join(errors)+' |')
    lines += ['', '## Paired differences','', 'Model minus control; negative is better. Descriptive 95% bootstrap intervals resample people after averaging seeds, not waveform windows.']
    for k,z in target['paired_differences'].items():
        lines.append(f"- Versus {labels[k]}: {fmt(z['delta_mae'])}; CI lower {fmt(z['ci95'][0])}, upper {fmt(z['ci95'][1])}; improved {z['people_improved'][0]}/{z['people']} SBP and {z['people_improved'][1]}/{z['people']} DBP.")
    lines += ['', '## Development screening','', '| Method | MAE | Worst ratio to either control |','|---|---:|---:|']
    for z in sorted(lock['ranking'],key=lambda v:v['worst_ratio']):
        lines.append(f"| {z['method']} | {fmt(z['mae'])} | {z['worst_ratio']:.4f} |")
    lines += ['', '## Population pair training','', '| Seed | Completed epochs | Selected epoch | Validation pair macro MAE |','|---|---:|---:|---:|']
    for seed in SEEDS:
        directory=a.run/'population'/f'seed{seed}'
        done=json.loads((directory/'complete.json').read_text());history=json.loads((directory/'history.json').read_text())
        if done['sha256']!=file_hash(directory/'best.pt') or done['manifest_hash']!=mh:raise ValueError('Population artifact mismatch')
        row=next(r for r in history if r['epoch']==done['best_epoch'])
        best=min(history,key=lambda r:np.mean(r['validation']['macro_mae']))
        if best['epoch']!=done['best_epoch']:raise ValueError('Population epoch not selected by validation')
        fitted=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
        for key,value in originals[seed][0].state_dict().items():
            torch.testing.assert_close(value.cpu(),fitted['state_dict']['encoder.'+key],atol=0,rtol=0)
        lines.append(f"| {seed} | {done['epochs']} | {done['best_epoch']} | {fmt(row['validation']['macro_mae'])} |")
        atomic_json(a.export/f'population_seed{seed}.json',dict(completion=done,history=history))
    lines += ['', '## Target seed stability','', '| Seed | Mean MAE | Bias MAE | Selected pair MAE |','|---|---:|---:|---:|']
    for seed in SEEDS:
        values=[]
        for method in ('calibration_mean','bias',m):
            subset=[r for r in records['test'] if r['seed']==seed and r['method']==method]
            values.append(fmt(np.mean([r['metrics']['macro']['mae'] for r in subset],0)))
        lines.append(f'| {seed} | '+' | '.join(values)+' |')
    lines += ['', '## Interpretation','',
        'The pair head was trained on exactly the existing population training participants; personal fitting uses all eligible early windows. Reference and current population rows are distinct but have no recoverable time ordering. Personal pairs and final reference/query banks have timestamp guards.',
        'This is an inspired ECG/PPG CNN+SE adaptation, not a reproduction of the cited PPG study. Expanded pairs are not new independent calibration measurements. No target-dependent retuning, discarded participants or extra reference labels were used.',
        f"Verified {checks['development_units']} development and {checks['test_units']} target units, and replayed {replayed} saved-model predictions; maximum difference {max_error:.8f} mmHg."]
    (a.export/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    weights=a.export.parent/'models';weights.mkdir(exist_ok=True);inventory={}
    for seed in SEEDS:
        src=a.run/'population'/f'seed{seed}'/'best.pt';dst=weights/f'paired_population_seed{seed}.pt'
        shutil.copyfile(src,dst);inventory[dst.name]=file_hash(dst)
    for row in records['test']:
        if row['method']!=m:continue
        src=Path(row['_prediction_path']).parent/'model.pt';dst=weights/f"{row['patient'].replace(':','_')}_seed{row['seed']}.pt"
        shutil.copyfile(src,dst);inventory[dst.name]=file_hash(dst)
    atomic_json(weights/'manifest.json',inventory)
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
