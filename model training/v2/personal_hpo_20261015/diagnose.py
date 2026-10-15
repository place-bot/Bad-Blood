"""Post-hoc diagnosis of a frozen run. Does not train, select, or change models."""
import json
from pathlib import Path
from collections import Counter

import h5py
import numpy as np


ROOT = Path('/Users/jguo9/Desktop/26fall/LLM/output/bad_blood_v2_runs/20261015_personal_hpo')
METHODS = ('personal_mean', 'bias', 'pair_unadapted', 'pair_fixed', 'pair_personal_hpo')


def stats(y):
    return dict(n=len(y), mean=y.mean(0).tolist(), sd=y.std(0, ddof=1).tolist(),
                minimum=y.min(0).tolist(), maximum=y.max(0).tolist(),
                p10_p90=np.percentile(y, [10, 90], axis=0).tolist())


def error_stats(y, p):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    e=p-y
    cor=[]
    for j in range(2):
        cor.append(float(np.corrcoef(y[:, j], p[:, j])[0, 1])
                   if np.std(p[:, j])>1e-6 and np.std(y[:, j])>1e-6 else None)
    return dict(mae=np.abs(e).mean(0).tolist(), bias=e.mean(0).tolist(),
                error_sd=e.std(0, ddof=1).tolist(), prediction_sd=p.std(0, ddof=1).tolist(),
                correlation=cor)


def build():
    lock=json.loads((ROOT/'lock.json').read_text())
    old=json.loads((ROOT/'summary.json').read_text())
    people=[]
    for record in lock['cohort']['selected']:
        pid=record['patient']
        with h5py.File(record['file'], 'r') as f:
            g=f['Subj_Wins']
            def ref(key, i): return np.asarray(f[g[key][0, i]]).reshape(-1)
            y=np.array([[ref('SegSBP', i)[0], ref('SegDBP', i)[0]] for i in range(record['n'])])
            start=np.array([ref('T', i)[0] for i in range(len(y))])
        splits={k:np.asarray(record[k]) for k in ('support','query','inner_train','inner_valid')}
        row=dict(patient=pid, counts={k:len(v) for k,v in splits.items()},
                 bp={k:stats(y[ix]) for k,ix in splits.items()}, modalities={})
        s,q=splits['support'],splits['query']
        row['test_minus_support_mean']=(y[q].mean(0)-y[s].mean(0)).tolist()
        row['test_minus_inner_validation_mean']=(y[q].mean(0)-y[splits['inner_valid']].mean(0)).tolist()
        row['test_outside_support_range_fraction']=((y[q]<y[s].min(0))|(y[q]>y[s].max(0))).mean(0).tolist()
        row['observed_time_minutes']={k:float((start[ix[-1]]-start[ix[0]]+10)/60) for k,ix in splits.items()}
        adjacent=np.flatnonzero(np.abs(np.diff(start)-10)<.01)
        row['adjacent_pairs']=len(adjacent)
        row['adjacent_bp_correlation']=[float(np.corrcoef(y[adjacent,j],y[adjacent+1,j])[0,1])
                                        if len(adjacent)>2 else None for j in range(2)]
        for mode in ('ppg','dual'):
            seedrows=[]
            for seed in (42,43,44):
                folder=ROOT/'units'/mode/f'seed{seed}'/pid.replace(':','_')
                arr=np.load(folder/'predictions.npz')
                np.testing.assert_allclose(arr['y'],y[q],rtol=1e-6,atol=1e-4)
                np.testing.assert_array_equal(arr['support'],s)
                sel=json.loads((folder/'selection.json').read_text())
                unit=json.loads((folder/'unit.json').read_text())
                seedrow=dict(seed=seed, selected=sel['selected'],
                             validation_best_ratio=sel['selected']['worst_ratio'],
                             metrics={m:error_stats(arr['y'],arr[m]) for m in METHODS})
                for method,audit in unit['audits'].items():
                    hist=audit.get('loss_history',[])
                    seedrow[method+'_training']=dict(initial_loss=hist[0] if hist else None,
                        last_loss=hist[-1] if hist else None, updates=len(hist),
                        pair_count=audit.get('pair_count'))
                seedrows.append(seedrow)
            avg={m:{k:np.mean([a['metrics'][m][k] for a in seedrows],0).tolist()
                    for k in ('mae','bias','error_sd','prediction_sd')} for m in METHODS}
            for m in METHODS:
                np.testing.assert_allclose(avg[m]['mae'],old['modality_results'][mode]['methods'][m]['participant_mae'][pid],atol=1e-6)
            win={m:(np.array(avg['pair_personal_hpo']['mae'])<avg[m]['mae']).tolist()
                 for m in ('personal_mean','bias','pair_fixed')}
            row['modalities'][mode]=dict(seeds=seedrows,average=avg,wins=win,
                all_four_win=bool(all(win['personal_mean']) and all(win['bias'])))
        people.append(row)
    result=dict(description='Post-hoc descriptive diagnosis; no new tuning or model selection.',
                people=people,aggregate={})
    for mode in ('ppg','dual'):
        v=[p['modalities'][mode] for p in people]
        allseed=[s for r in v for s in r['seeds']]
        a=dict(people=len(v),selected_configs=dict(Counter(str(s['selected']['config']) for s in allseed)),
               all_four_wins=sum(r['all_four_win'] for r in v), win_counts={},gain_loss={})
        vp=np.array([s['validation_best_ratio']<1 for s in allseed])
        tp=np.array([
            all(np.array(s['metrics']['pair_personal_hpo']['mae'])<s['metrics']['personal_mean']['mae']) and
            all(np.array(s['metrics']['pair_personal_hpo']['mae'])<s['metrics']['bias']['mae']) for s in allseed])
        a['seed_person_validation_test']=dict(total=len(allseed),validation_pass=int(vp.sum()),
            both_pass=int((vp&tp).sum()),validation_pass_test_fail=int((vp&~tp).sum()),
            validation_fail_test_pass=int((~vp&tp).sum()),both_fail=int((~vp&~tp).sum()))
        for control in ('personal_mean','bias','pair_fixed'):
            delta=np.array([r['average']['pair_personal_hpo']['mae'] for r in v])-np.array([r['average'][control]['mae'] for r in v])
            a['win_counts'][control]=dict(sbp=int((delta[:,0]<0).sum()),dbp=int((delta[:,1]<0).sum()),both=int((delta<0).all(1).sum()))
            a['gain_loss'][control]=[]
            for j in range(2):
                order=np.argsort(delta[:,j])
                a['gain_loss'][control].append(dict(mean_delta=float(delta[:,j].mean()),
                    win_mean_delta=float(delta[delta[:,j]<0,j].mean()),lose_mean_delta=float(delta[delta[:,j]>=0,j].mean()),
                    top_gain=[dict(patient=people[i]['patient'],delta=float(delta[i,j])) for i in order[:5]],
                    top_harm=[dict(patient=people[i]['patient'],delta=float(delta[i,j])) for i in order[-5:][::-1]],
                    leave_largest_gain_out=float(np.delete(delta[:,j],order[0]).mean())))
        result['aggregate'][mode]=a
    out=ROOT/'diagnostics';out.mkdir(exist_ok=True)
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Frozen 80/20 experiment: post-hoc individual diagnosis','',
        'SBP/DBP are systolic/diastolic blood pressure; all pressure values are mmHg.',
        'MAE values are averaged over source seeds 42/43/44. These observations do not change models or selection.',
        'Support is the first 80%; test is the later 20%, after the frozen boundary gap.','']
    def fmt(x):return ' / '.join(f'{a:.2f}' for a in x)
    for mode in ('ppg','dual'):
        lines+=['## '+mode,'', '| Person | Support/test N | Support BP SD | Test BP SD | Test minus support mean | Personal mean MAE | Bias MAE | HPO MAE | Beats all four |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for p in people:
            m=p['modalities'][mode]
            fields=[p['patient'],f"{p['counts']['support']}/{p['counts']['query']}",fmt(p['bp']['support']['sd']),
                    fmt(p['bp']['query']['sd']),fmt(p['test_minus_support_mean'])]
            fields += [fmt(m['average'][key]['mae']) for key in ('personal_mean','bias','pair_personal_hpo')]
            lines.append('| '+' | '.join(fields+['yes' if m['all_four_win'] else 'no'])+' |')
        lines+=['', 'Validation/test counts (90 person-seed fits):',
                '```json',json.dumps(result['aggregate'][mode]['seed_person_validation_test'],indent=2),'```','']
    (out/'DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['aggregate'],indent=2))


if __name__=='__main__':build()
