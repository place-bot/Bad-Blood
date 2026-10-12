"""Build a dated, checksum-indexed V2 handoff from verified existing artifacts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from personalized_forecasting.run import file_hash,atomic_json

def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)

def fmt(v,percent=False):return ' / '.join(f'{x*(100 if percent else 1):.2f}' if x is not None else 'N/A' for x in v)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    fresh=json.loads((a.run/'summary.json').read_text())
    assert fresh['verification']['no_prior_identity_overlap'] and fresh['verification']['max_replay_error']<=1e-4
    target=ROOT/'releases/20261012';pop=target/'01_population_cnn_se';personal=target/'02_personalization'
    overview=json.loads((ROOT/'reports/results/metric_overview.json').read_text())
    for dest in (pop,personal):
        dest.mkdir(parents=True,exist_ok=True)
        # Preserve module paths so source snapshots are readable and executable.
        # Data stays external. This source tree deliberately excludes old outputs.
        for module in ('shared','attention_cnn','personalized_forecasting','se_personalization','paired_se'):
            for src in (ROOT/module).glob('*.py'):copy(src,dest/'source'/module/src.name)
            if (ROOT/module/'PROTOCOL.md').exists():copy(ROOT/module/'PROTOCOL.md',dest/'source'/module/'PROTOCOL.md')
        for name in ('PROTOCOL.md','audit_snapshot.json'):
            copy(ROOT/name,dest/'source'/name)
        # shared/data.py resolves this historical exclusion file next to its root.
        copy(ROOT.parent/'PulseDB-based Model-V1/reports/prior_exploratory_subjects.json',dest/'PulseDB-based Model-V1/reports/prior_exploratory_subjects.json')
    pop_metrics=dict(date='20261012',comparison='matched corrected V2 splits; historical V1 weights not used',
        selection=overview['validation_selections'],models={k:overview['attention'][k] for k in ('baseline','se_last','temporal_last','cbam_last','additive_last','se_all')})
    atomic_json(pop/'results/metrics.json',pop_metrics)
    copy(ROOT/'reports/results/attention_interpretation.json',pop/'results/attention_interpretation.json')
    for seed in (42,43,44):copy(ROOT/f'se_personalization/models/pretrained_se_seed{seed}.pt',pop/f'models/cnn_se_seed{seed}.pt')
    for src in (ROOT/'paired_se/models').glob('*.pt'):copy(src,personal/'models'/src.name)
    for src in (ROOT/'paired_se/results').glob('*.json'):copy(src,personal/'results/prior_six'/src.name)
    copy(ROOT/'paired_se/results/RESULTS.md',personal/'results/prior_six/RESULTS.md')
    for name in ('summary.json','cohort_audit.json','per_person_seed_metrics.json'):
        copy(a.run/name,ROOT/'paired_se/fresh_20261012/results'/name)
        copy(a.run/name,personal/'results/new_participants'/name)
    for name in ('PROTOCOL.md','candidate_manifest.json','validate.py'):
        copy(ROOT/'paired_se/fresh_20261012'/name,personal/'source/paired_se/fresh_20261012'/name)
    # Frozen runner's relative resources and original pair weights for reproducibility.
    for name in ('manifest.json','selection.json'):
        copy(ROOT/'paired_se/results'/name,personal/'source/paired_se/results'/name)
    for seed in (42,43,44):copy(ROOT/f'paired_se/models/paired_population_seed{seed}.pt',personal/f'source/paired_se/models/paired_population_seed{seed}.pt')
    for f in a.run.glob('vitaldb:*/seed*/paired/model.pt'):
        name=f'{f.parents[2].name.replace(":","_")}_{f.parents[1].name}.pt'
        copy(f,personal/'models/new_participants'/name)
    # All prediction rows are public-source numeric labels/predictions, not signals.
    # Keep these outside Git; publish complete per-person/per-seed metrics instead.
    lines=['# V2 fixed-method new-participant validation — 20261012','',
        f"{fresh['n_candidates']} prospectively locked candidates; {fresh['n_included']} eligible new participants; "
        f"{fresh['support_windows']} early calibration and {fresh['query_windows']} later query windows.",
        '', 'These people were absent from every prior population train/validation/test/AAMI partition and the 31 historical exploratory people. '
        'All eligible people from the first author-hosted folder page were included; this is a convenience sample, not a random or external cohort.',
        '', 'Fixed CNN+SE pair model, head-only Adam lr=0.0003, 25 updates, seeds 42/43/44. No hyperparameter or checkpoint selection on these results. '
        'All methods use the same early 20% of elapsed time and the same later windows. Numbers below are participant-macro, then averaged over seeds; SBP / DBP.',
        '', '| Method | MAE | RMSE | Bias | Within-person error SD | Within 5 (%) | Within 10 (%) | Within 15 (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels={'population':'Unadapted CNN+SE','calibration_mean':'Personal mean','bias':'SE + personal bias','paired':'Reference-conditioned SE'}
    for m in labels:
        v=fresh['methods'][m]['macro'];lines.append('| '+labels[m]+' | '+' | '.join(fmt(v[k],k.startswith('within')) for k in ('mae','rmse','bias','error_sd','within_5','within_10','within_15'))+' |')
    lines+=['','## Primary conclusion','',f"Lower SBP AND DBP MAE than BOTH controls: **{fresh['beats_both_controls_both_pressures']}**.",'',
        'The earlier six-person improvement is preserved separately. This follow-up is not pooled with those six to change the conclusion.']
    for m,d in fresh['paired_differences'].items():
        lines+=['',f"Against {labels[m]}: paired-model minus control MAE = {fmt(d['delta_mae'])} mmHg; "
            f"95% participant-bootstrap intervals: SBP [{d['ci95'][0][0]:.2f}, {d['ci95'][1][0]:.2f}], DBP [{d['ci95'][0][1]:.2f}, {d['ci95'][1][1]:.2f}]. "
            f"Improved people: {d['people_improved'][0]}/{d['people']} SBP, {d['people_improved'][1]}/{d['people']} DBP."]
    lines+=['','## Seed checks','','| Seed | Personal mean | Bias correction | Paired model |','|---|---:|---:|---:|']
    for seed,row in fresh['per_seed_mae'].items():lines.append('| '+seed+' | '+' | '.join(fmt(row[m]) for m in ('calibration_mean','bias','paired'))+' |')
    lines+=['','## Individual results','','| Participant | Early / later windows | Personal mean MAE | Bias MAE | Paired MAE |','|---|---:|---:|---:|---:|']
    audit=json.loads((a.run/'cohort_audit.json').read_text())
    for r in audit:
        if r['status']=='included':lines.append('| '+r['patient']+f" | {r['n_support']} / {r['n_query']} | "+' | '.join(fmt(fresh['methods'][m]['per_patient_mae'][r['patient']]) for m in ('calibration_mean','bias','paired'))+' |')
    lines+=['','## Technical exclusions','']+[f"- {r['patient']}: {r['reason']}" for r in audit if r['status']!='included']
    lines+=['','No exclusion used prediction performance. Singleton MATLAB records were identified as insufficient calibration/test data; the loader fix did not change model settings.',
        '',f"Verified {fresh['verification']['units']} completed units and replayed {fresh['verification']['replayed_predictions']} saved-model predictions; maximum replay difference {fresh['verification']['max_replay_error']:.8g} mmHg.",
        '', 'New records can be much shorter than the prior cohort. The identical 20%-of-time budget does not guarantee identical absolute numbers of calibration labels; counts are given above. '
        'All labels are ABP-derived. This within-record evaluation does not test real ND cuffs or persistence across visits.']
    text='\n'.join(lines)+'\n'
    (ROOT/'paired_se/fresh_20261012/results/RESULTS.md').write_text(text)
    (personal/'results/new_participants/RESULTS.md').write_text(text)
    manifests={}
    for dest in (pop,personal):
        manifests[dest.name]={str(f.relative_to(dest)):file_hash(f) for f in sorted(dest.rglob('*')) if f.is_file() and f.name!='SHA256SUMS.json' and '__pycache__' not in str(f)}
        atomic_json(dest/'SHA256SUMS.json',manifests[dest.name])
    atomic_json(target/'version.json',dict(version='v2-20261012',date='20261012',
        original_completed_experiment_commit='97aaeb30e7c7e60c148b191c41734eb2fe2e82fc',
        fresh_protocol_commit='981e15a',fresh_lock_hash=fresh['lock_hash'],
        population_model='CNN+SE se_last',personal_model='reference-conditioned CNN+SE, pair-head adaptation',
        fresh_primary_success=fresh['beats_both_controls_both_pressures'],fresh_people=fresh['n_included']))
    print('Packaged',target)

if __name__=='__main__':main()
