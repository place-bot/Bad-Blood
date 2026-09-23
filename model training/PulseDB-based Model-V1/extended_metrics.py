"""Summarize saved V1 predictions; no fitting or checkpoint selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def metrics(y, p, subjects):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError('Expected matching N x 2 labels and predictions')
    if len(subjects) != len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Invalid labels, predictions or IDs')
    e = p-y
    a = abs(e)
    people = []
    for sid in np.unique(subjects):
        mask = subjects == sid
        mae = a[mask].mean(0)
        people.append(dict(participant_id=str(sid), segments=int(mask.sum()), SBP_MAE=mae[0], DBP_MAE=mae[1]))
    pm = np.array([[r['SBP_MAE'], r['DBP_MAE']] for r in people])
    report = dict(MAE=a.mean(0).tolist(), Bias=e.mean(0).tolist(), error_SD=e.std(0,ddof=1).tolist(),
                  participant_macro_MAE=pm.mean(0).tolist(), participant_MAE_median=np.median(pm,axis=0).tolist(),
                  participant_MAE_p90=np.quantile(pm,.9,axis=0).tolist(), participants=len(people), segments=len(y))
    for t in (5,10,15):
        report[f'within_{t}_pct'] = (100*(a<=t).mean(0)).tolist()
    return report, people

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--saved-run',type=Path,required=True)
    args=parser.parse_args()
    out=Path(__file__).resolve().parent/'reports'
    split=json.loads((args.data/'split.json').read_text())
    prior=json.loads((out/'prior_exploratory_subjects.json').read_text())
    reports={}
    for name in ('validation','test','aami'):
        d=args.data/('train' if name=='validation' else name)
        s=np.load(d/'subjects.npy'); y=np.load(d/'y.npy'); valid=np.load(d/'valid.npy')
        if name=='validation':
            valid=valid & np.isin(s,split['validation'])
            p=np.load(args.saved_run/'models/cnn_full_validation.npy')
        else:
            p=np.load(args.saved_run/f'reports/{name}_predictions.npy')
        y,s=y[valid],s[valid]
        reports[name],people=metrics(y,p,s)
        with (out/f'{name}_participant_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(people[0]));w.writeheader();w.writerows(people)
        if name=='test':
            keep=~np.isin(s,prior)
            reports['test_excluding_prior'],_=metrics(y[keep],p[keep],s[keep])
    (out/'extended_metrics.json').write_text(json.dumps({'model':'V1 cnn_full','targets':['SBP','DBP'],'error_definition':'prediction minus reference; mmHg; SD uses ddof=1; thresholds inclusive','evaluations':reports},indent=2))
    lines=['# V1 complete error metrics','','Computed from the existing selected-model predictions; no retraining or test-based selection. Values are SBP / DBP. Error = prediction minus reference. SD uses N-1; threshold percentages include the boundary.','']
    for name,r in reports.items():
        lines += [f'## {name}', '', f"{r['participants']} participants; {r['segments']:,} segments.",'','| Metric | SBP | DBP |','|---|---:|---:|']
        for k in ('MAE','Bias','error_SD','within_5_pct','within_10_pct','within_15_pct','participant_macro_MAE','participant_MAE_median','participant_MAE_p90'):
            lines.append(f'| {k} | {r[k][0]:.3f} | {r[k][1]:.3f} |')
        lines += ['']
    lines += ['These are PulseDB source-data results, not ND cuff-reference results or a clinical device validation. All pressure errors are in mmHg; within-threshold values are percentages. Participant-macro MAE weights people equally. The AAMI-designated dataset name does not establish compliance with a clinical validation protocol.']
    (out/'METRICS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
