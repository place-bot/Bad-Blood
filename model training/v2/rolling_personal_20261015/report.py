"""Describe every predeclared method; no outcome-based selection."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shared.metrics import summarize
from personalized_forecasting.run import atomic_json,file_hash
from personal_hpo_20261015.run import clean

METHODS=('population','personal_mean','bias','pair_fixed','pair_personal_hpo',
         'rolling_head','rolling_tail','rolling_selected')
NEW=('rolling_head','rolling_tail','rolling_selected')
METRICS=('mae','rmse','bias','error_sd','within_5','within_10','within_15')


def report(out):
    lock=json.loads((out/'lock.json').read_text());ids=[r['patient'] for r in lock['people']]
    result=dict(people=30,seeds=[42,43,44],source='Previously examined VitalDB-derived PulseDB cohort',
                scope='Exploratory rolling personal 80/20 adaptation',modality_results={})
    for mode in ('ppg','dual'):
        methods={}
        for method in METHODS:
            seeds={}
            for seed in (42,43,44):
                yy=[];pp=[];subjects=[]
                for pid in ids:
                    with np.load(out/'units'/mode/f'seed{seed}'/pid.replace(':','_')/'predictions.npz') as z:
                        yy.append(z['y']);pp.append(z[method]);subjects.extend([pid]*len(z['y']))
                seeds[str(seed)]=summarize(np.concatenate(yy),np.concatenate(pp),np.asarray(subjects))
            pm={pid:np.mean([v['per_patient'][pid]['mae'] for v in seeds.values()],0) for pid in ids}
            methods[method]=dict(per_seed=seeds,participant_mae=pm,
                macro={m:np.mean([v['macro'][m] for v in seeds.values()],0) for m in METRICS},
                pooled={m:np.mean([v['pooled'][m] for v in seeds.values()],0) for m in METRICS},
                seed_sd={m:np.std([v['macro'][m] for v in seeds.values()],0,ddof=1) for m in METRICS},
                participant_mae_quantiles=np.quantile(list(pm.values()),[0,.25,.5,.75,.9,1],axis=0))
        comparisons={}
        draws=np.random.default_rng(20261015).integers(0,30,(10000,30))
        for method in NEW:
            a=methods[method];comp={};allwins=np.ones(30,bool);allseed=np.ones(30,bool)
            for control in ('personal_mean','bias','pair_fixed','pair_personal_hpo'):
                d=np.array([a['participant_mae'][pid]-methods[control]['participant_mae'][pid] for pid in ids])
                comp[control]=dict(delta_macro_mae=d.mean(0),win_count=(d<0).sum(0),joint_win_count=(d<0).all(1).sum(),
                    participant_bootstrap_ci95=np.quantile(d[draws].mean(1),[.025,.975],axis=0),
                    per_person_delta={pid:d[i] for i,pid in enumerate(ids)})
                if control in ('personal_mean','bias'):
                    allwins&=(d<0).all(1)
                    for seed in ('42','43','44'):
                        dd=np.array([np.asarray(a['per_seed'][seed]['per_patient'][pid]['mae'])-
                                     methods[control]['per_seed'][seed]['per_patient'][pid]['mae'] for pid in ids])
                        allseed&=(dd<0).all(1)
            comparisons[method]=dict(controls=comp,seed_average_all_four_count=allwins.sum(),
                strict_all_seeds_count=allseed.sum(),each_person={pid:dict(seed_average=allwins[i],all_seeds=allseed[i]) for i,pid in enumerate(ids)})
        result['modality_results'][mode]=dict(methods=methods,comparisons=comparisons)
    atomic_json(out/'summary.json',clean(result))
    lines=['# Rolling personal adaptation — 20261015','',
           'Same 30 people and guarded chronological 80/20 split as the previous experiment.',
           'Values are SBP / DBP; mmHg except percentages. Equal-person averages, then averaged across source seeds (not an ensemble).',
           'Selections and weights were locked for all people before evaluation. This previously examined cohort is not independent confirmation.','']
    def fmt(x):return ' / '.join(f'{v:.2f}' for v in x)
    for mode,item in result['modality_results'].items():
        lines+=['## '+mode,'','| Method | MAE | RMSE | Signed bias | Sample error SD | Within 5 (%) | Within 10 (%) | Within 15 (%) |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,r in item['methods'].items():
            cells=[fmt(np.array(r['macro'][m])*(100 if m.startswith('within') else 1)) for m in METRICS]
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
        lines+=['','| Method | Both endpoints beat both controls (seed average) | Same condition in all three seeds |',
                '|---|---:|---:|']
        for name,c in item['comparisons'].items():
            lines.append(f"| {name} | {c['seed_average_all_four_count']}/30 | {c['strict_all_seeds_count']}/30 |")
        lines+=['','### All individuals','', '| Person | Mean | Bias | Old personal HPO | Rolling head | Rolling tail | Selected scope |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for pid in ids:
            cells=[fmt(item['methods'][m]['participant_mae'][pid]) for m in
                   ('personal_mean','bias','pair_personal_hpo','rolling_head','rolling_tail','rolling_selected')]
            lines.append('| '+pid+' | '+' | '.join(cells)+' |')
        lines+=['','### Per-seed endpoint MAE','', '| Method | Seed 42 | Seed 43 | Seed 44 |','|---|---:|---:|---:|']
        for name,r in item['methods'].items():
            lines.append('| '+name+' | '+' | '.join(fmt(r['per_seed'][s]['macro']['mae']) for s in ('42','43','44'))+' |')
        lines+=['']
    lines+=['## Artifacts','',
        '- `lock.json`: participant/source/code hashes and each inner split.',
        '- `units/<modality>/seed<seed>/<person>/selection.json`: both endpoint-specific validation rankings.',
        '- `head_endpoint{0,1}.pt`, `tail_endpoint{0,1}.pt`: full source-compatible personal checkpoints.',
        '- `predictions.npz` and `evaluation.json`: final-test predictions and complete per-person metrics.',
        '- `summary.json`: pooled/equal-person metrics, three seeds, paired bootstrap intervals and win counts.',
        '- Clinical device certification is not implied by these research metrics.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    report(p.parse_args().output)
