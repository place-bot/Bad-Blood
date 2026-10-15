"""Human-readable person-by-person results, no model selection or retraining."""
import argparse
import json
from pathlib import Path


def pair(value):return ' / '.join(f'{v:.3f}' for v in value)


def build(out):
    summary=json.loads((out/'summary.json').read_text())
    verification=json.loads((out/'verification.json').read_text())
    assert verification['status']=='passed'
    lock=json.loads((out/'lock.json').read_text())
    rows={r['patient']:r for r in lock['cohort']['selected']}
    text=['# Individual results — 30-person 80/20 experiment','',
          'Values are SBP / DBP MAE in mmHg, averaged over three source seeds.',
          '`Strict pass` requires BOTH endpoints below BOTH controls in ALL three seeds.',
          'The personal choice is derived independently inside each person\'s development bank.',
          'No table below is used to choose configurations.','']
    for mode,item in summary['modality_results'].items():
        text+=['## '+mode,'',
            '| Person | Development / test segments | Personal mean | Bias correction | Fixed PairSE | Individual HPO | Strict pass |',
            '|---|---:|---:|---:|---:|---:|---|']
        for pid,row in rows.items():
            scores=[pair(item['methods'][m]['participant_mae'][pid]) for m in
                    ('personal_mean','bias','pair_fixed','pair_personal_hpo')]
            passed=item['gate']['each_person'][pid]['beats_both_all_three_seeds']
            text.append('| '+pid+' | '+str(len(row['support']))+' / '+str(len(row['query']))+' | '+
                        ' | '.join(scores)+' | '+('yes' if passed else 'no')+' |')
        text+=['','### Individual choices','',
               '| Person | Seed 42 | Seed 43 | Seed 44 |','|---|---|---|---|']
        for pid in rows:
            choices=[]
            for seed in (42,43,44):
                file=out/'units'/mode/f'seed{seed}'/pid.replace(':','_')/'selection.json'
                cfg=json.loads(file.read_text())['selected']['config']
                choices.append('no extra updates' if cfg is None else f"lr {cfg['lr']}, {cfg['steps']} updates")
            text.append('| '+pid+' | '+' | '.join(choices)+' |')
        text+=['','### Paired participant comparisons','',
               'Negative differences favor individual selection. Intervals resample people, not windows or seeds.','',
               '| Control | SBP mean MAE difference [95% interval] | DBP mean MAE difference [95% interval] |',
               '|---|---:|---:|']
        for control,result in item['comparisons'].items():
            values=[]
            for j in range(2):
                lo,hi=result['participant_bootstrap_ci95'][0][j],result['participant_bootstrap_ci95'][1][j]
                values.append(f"{result['delta_macro_mae'][j]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            text.append('| '+control+' | '+' | '.join(values)+' |')
        text+=['']
    text+=['## Verification','',
           f"Verified {verification['units']} units and {verification['model_replays']} saved-model inference replays.",
           f"Largest replay discrepancy: {verification['max_replay_difference_mmHg']:.7f} mmHg.",
           'All calibration/test rows match across methods/modalities; frozen encoders and source identity are checked.',
           'Full MAE, RMSE, bias, sample error SD and within-5/10/15 metrics are in RESULTS.md and summary.json.']
    (out/'INDIVIDUAL_RESULTS.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    build(p.parse_args().output)
