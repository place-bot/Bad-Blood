"""Audit completed dense-support experiments and export reports, not raw data."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shared.data import load_personal_cohort
from shared.metrics import summarize
from personalized_forecasting.run import file_hash, canonical_hash, atomic_json
from se_personalization.run import dense_support, summarize_phase, select, name, grid, SEEDS, DEV_IDS, TARGET_IDS
from se_personalization.predict import load_and_predict

def pair(values,digits=2): return ' / '.join('N/A' if v is None else f'{v:.{digits}f}' for v in values)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--individual-root',type=Path,required=True)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--population-root',type=Path,required=True)
    p.add_argument('--device',default='mps')
    a=p.parse_args();torch.set_num_threads(4)
    manifest=json.loads((a.run/'manifest.json').read_text())
    selection=json.loads((a.run/'selection.json').read_text())
    dev=json.loads((a.run/'development_summary.json').read_text())
    target=json.loads((a.run/'test_summary.json').read_text())
    cohort=load_personal_cohort(a.individual_root)
    patients={x['patient_id']:x for role in ('development','target') for x in cohort[role]}
    records={}; max_replay_error=0.;replayed=0
    for phase,expected_ids in [('development',DEV_IDS),('test',TARGET_IDS)]:
        rows=[]
        for path in sorted((a.run/phase).glob('*/*/*/unit.json')):
            r=json.loads(path.read_text());directory=path.parent
            if r['config']['manifest_hash']!=canonical_hash(manifest): raise ValueError('Unit manifest mismatch')
            if phase=='test' and r['config']['selection_hash']!=selection['selection_hash']: raise ValueError('Wrong test selection')
            for n,h in r['hashes'].items():
                if file_hash(directory/n)!=h: raise ValueError('Corrupted artifact')
            patient=patients[r['patient']]; support=dense_support(patient);q=patient['query_indices']
            with np.load(directory/'predictions.npz') as z:
                np.testing.assert_array_equal(z['support_indices'],support)
                np.testing.assert_array_equal(z['query_indices'],q)
                np.testing.assert_array_equal(z['support_y'],patient['y'][support])
                np.testing.assert_array_equal(z['y'],patient['y'][q])
                actual=summarize(z['y'],z['pred'],np.repeat(r['patient'],len(q)))
                if actual!=r['metrics']: raise ValueError('Metrics mismatch')
                if r['method']=='calibration_mean':
                    expected=np.tile(patient['y'][support].astype(float).mean(0),(len(q),1))
                    np.testing.assert_allclose(z['pred'],expected,atol=1e-5,rtol=0)
                if r['method']=='bias':
                    with np.load(directory.parent/'population'/'predictions.npz') as pop:
                        correction=(patient['y'][support].astype(float)-z['support_population_pred']).mean(0)
                        np.testing.assert_allclose(z['pred'],pop['pred']+correction,rtol=0,atol=1e-5)
                if phase=='test' and r['method']==target['selected_method']:
                    replay=load_and_predict(directory/'adapted.pt',patient['x'][q],a.device)
                    error=float(np.max(np.abs(replay-z['pred'])))
                    np.testing.assert_allclose(replay,z['pred'],rtol=0,atol=.001)
                    max_replay_error=max(max_replay_error,error);replayed+=len(q)
            r['_prediction_path']=str(directory/'predictions.npz');rows.append(r)
        expected_methods=['population','calibration_mean','bias']+([name(s) for s in grid()] if phase=='development' else [target['selected_method']])
        expected={(id,s,m) for id in expected_ids for s in SEEDS for m in expected_methods}
        if len(rows)!=len(expected) or {(r['patient'],r['seed'],r['method']) for r in rows}!=expected:
            raise ValueError('Incomplete or duplicated inventory')
        summary=summarize_phase(rows)
        saved=dev if phase=='development' else target
        if summary!=saved['methods']: raise ValueError('Summary mismatch')
        records[phase]=rows
    chosen=select(records['development'])
    # Filesystem traversal and original execution order differ. Floating-point
    # reductions can differ at the last bit; the selected configuration must
    # still match exactly and every numerical selection value is checked.
    if chosen['selected']['setting']!=selection['selected']['setting'] or chosen['selected']['method']!=selection['selected']['method']:
        raise ValueError('Selection differs from development')
    for key in ('mae','worst_ratio','mean_mae'):
        np.testing.assert_allclose(chosen['selected'][key],selection['selected'][key],rtol=0,atol=1e-12)
    a.export.mkdir(parents=True,exist_ok=True)
    for n in ('development_summary.json','test_summary.json','selection.json'):
        shutil.copyfile(a.run/n,a.export/n)
    public_manifest={k:v for k,v in manifest.items() if k!='source_manifest'}
    public_manifest['source_manifest_hash']=canonical_hash(manifest['source_manifest'])
    atomic_json(a.export/'provenance.json',public_manifest)
    checks=dict(development_units=len(records['development']),target_units=len(records['test']),
        exact_manifest_and_metrics=True,saved_model_replayed_query_predictions=replayed,
        max_replay_absolute_error_mmHg=max_replay_error,checks_passed=True)
    atomic_json(a.export/'verification.json',checks)
    lines=['# Full support CNN plus SE personalization results','',
        'October 12, 2026. SBP / DBP throughout; errors are mmHg. Seeds 42/43/44 are averaged, not ensembled.',
        '', '## Outcome','',
        f"Requested objective reached on the six previously examined target people: **{target['success_both_pressures_both_controls']}**.",
        f"Selected from development only: `{target['selected_method']}`.",
        'Target results are exploratory follow-up, not fresh independent validation. No target-based retuning was performed.',
        '', '## Patient macro target metrics','',
        '| Method | MAE | RMSE | Signed bias | Error SD | Within 5 % | Within 10 % | Within 15 % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels={'population':'Unadapted SE','calibration_mean':'Personal reference mean','bias':'Personal bias correction',target['selected_method']:'Selected SE fine tuning'}
    for m in ['population','calibration_mean','bias',target['selected_method']]:
        z=target['methods'][m]['macro']
        vals=[pair(z[k]) for k in ['mae','rmse','bias','error_sd']]+[pair([v*100 for v in z[k]]) for k in ['within_5','within_10','within_15']]
        lines.append('| '+labels[m]+' | '+' | '.join(vals)+' |')
    lines += ['', 'Patient-macro SD is the average of within-person sample error SDs, not a pooled clinical-validation SD. Pooled values and all other metrics are in `test_summary.json`.',
        '', '## Per person comparison','', '| Participant | Early training windows | Later query windows | Personal mean MAE | Bias correction MAE | SE fine tuning MAE |',
        '|---|---:|---:|---:|---:|---:|']
    for pid in TARGET_IDS:
        x=manifest['patients'][pid]
        vals=[pair(target['methods'][m]['per_patient_mae'][pid]) for m in ['calibration_mean','bias',target['selected_method']]]
        lines.append(f"| {pid} | {x['n_support']} | {len(x['query_source_indices'])} | "+' | '.join(vals)+' |')
    lines += ['', '## Paired differences','', 'Fine tuning minus control; negative favors fine tuning. Descriptive 95% patient-bootstrap intervals average seeds before resampling six people.']
    for m,z in target['paired_differences'].items():
        lines += ['',f"- Versus {labels[m]}: mean change **{pair(z['estimate'])}**, CI lower {pair(z['ci95'][0])}, upper {pair(z['ci95'][1])}; improved {z['improved_patients'][0]}/6 SBP and {z['improved_patients'][1]}/6 DBP."]
    lines += ['', '## Development selection','', '| Setting | SBP / DBP MAE | Worst ratio to either control |','|---|---:|---:|']
    for z in sorted(selection['ranking'],key=lambda v:(v['worst_ratio'],v['mean_mae'])):
        lines.append(f"| `{z['method']}` | {pair(z['mae'])} | {z['worst_ratio']:.4f} |")
    lines += ['',f"Development reference-mean MAE: {pair(selection['controls']['calibration_mean'])}; bias-correction MAE: {pair(selection['controls']['bias'])}.",
        '', '## Verification','',f"Audited {checks['development_units']} development units and {checks['target_units']} target units. Replayed {replayed} saved-model predictions; maximum absolute replay difference {max_replay_error:.8f} mmHg.",
        '', 'All eligible early windows were used in every optimizer update. The same early windows define both controls; later query labels do not enter personal fitting. No original sparse-reference result or V1 checkpoint was overwritten.',
        '', 'The dense early labels come from invasive ABP in public data, not from newly acquired cuffs. These same-record results do not establish cross-visit ND performance or clinical certification.']
    (a.export/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Only model state and inference normalization/offset; never publish raw waveforms or support labels.
    weights=a.export.parent/'models';weights.mkdir(exist_ok=True)
    weight_manifest={}
    for seed in SEEDS:
        src=a.population_root/f'se_last_seed{seed}'/'best.pt'
        saved=torch.load(src,map_location='cpu',weights_only=False)
        out=weights/f'pretrained_se_seed{seed}.pt'
        torch.save({k:saved[k] for k in ('variant','state_dict','center','scale','seed','epoch')},out)
        weight_manifest[out.name]={'sha256':file_hash(out),'source_checkpoint_sha256':file_hash(src),'role':'population'}
    for r in records['test']:
        if r['method']!=target['selected_method']:continue
        src=Path(r['_prediction_path']).parent/'adapted.pt'
        out=weights/f"{r['patient'].replace(':','_')}_seed{r['seed']}.pt"
        shutil.copyfile(src,out)
        weight_manifest[out.name]={'sha256':file_hash(out),'role':'personal','patient':r['patient'],'seed':r['seed']}
    atomic_json(weights/'manifest.json',weight_manifest)
    log=ROOT.parents[1]/'autoresearch'/'classic-261012-se-dense'
    log.mkdir(parents=True,exist_ok=True)
    rows=['iteration\tmethod\tworst_development_ratio\tstatus']
    for i,z in enumerate(selection['ranking'],1):
        rows.append(f"{i}\t{z['method']}\t{z['worst_ratio']:.8f}\t"+('keep' if z['method']==target['selected_method'] else 'not_selected'))
    (log/'classic-results.tsv').write_text('\n'.join(rows)+'\n')
    atomic_json(log/'handoff.json',{'mode':'classic','iterations':24,'selection':selection['selected'],
        'target_objective_met':target['success_both_pressures_both_controls'],'test_retuning':False,'verification':checks})
    print(json.dumps(checks))

if __name__=='__main__':main()
