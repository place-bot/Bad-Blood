"""Freeze existing artifacts and regenerate report tables; never train/select models."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
MODEL_METHOD = 'paired_head_lr0.0003_s25'


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def best_mask(values, metric='mae'):
    """Compare unrounded values per pressure; preserve all numerical ties."""
    a = np.asarray(values, dtype=float)
    objective = np.abs(a) if metric == 'bias' else -a if metric.startswith('within') else a
    return np.isclose(objective, np.min(objective, axis=0), rtol=0, atol=1e-10)


def pair(value, percent=False, best=None):
    a = np.asarray(value) * (100 if percent else 1)
    mask = (False, False) if best is None else best
    return ' / '.join('\\textbf{' + f'{v:.2f}' + '}' if b else f'{v:.2f}'
                      for v, b in zip(a, mask))


def table(name, caption, headers, rows, wide=False):
    env = 'table*' if wide else 'table'
    start = '\\begin{'+env+'}[t]\n\\centering\\caption{'+caption+'}\\label{tab:'+name+'}\n'+('\\footnotesize\n' if wide else '\\scriptsize\n')
    start += '\\setlength{\\tabcolsep}{'+('4' if wide else '4')+'pt}\n'
    start += '\\begin{tabular}{l'+'r'*(len(headers)-1)+'}\\toprule\n'
    start += ' & '.join(headers)+'\\\\\\midrule\n'
    start += '\n'.join(' & '.join(map(str,row))+'\\\\' for row in rows)
    start += '\n\\bottomrule\\end{tabular}\n'
    if name == 'population':
        start += ('\\par\\smallskip\\begin{minipage}{\\textwidth}\\footnotesize '
                  'Table notation: bold marks the best value per SBP/DBP in the stated comparison group: '
                  'lower MAE, RMSE and SD, Bias closest to zero, and higher within-threshold percentages. '
                  'Comparisons use unrounded values; numerical ties are all bold. '
                  'Bold does not imply statistical significance or determine model selection.'
                  '\\end{minipage}\n')
    start += '\\end{'+env+'}\n'
    put(HERE/'report'/'tables'/f'{name}.tex',start)


def main():
    previous_summary = HERE/'metrics'/'release_summary.json'
    source_commit = (json.loads(previous_summary.read_text())['source_commit']
                     if previous_summary.exists() else
                     subprocess.check_output(['git','rev-parse','HEAD'], cwd=V2, text=True).strip())
    a = json.loads((V2/'modality_se/results/comparison.json').read_text())
    attention = json.loads(gzip.decompress((V2/'reports/results/attention.json.gz').read_bytes()))
    selection = json.loads((V2/'reports/results/attention_interpretation.json').read_text())
    for module in ('shared/models.py','modality_se/model.py','paired_se/model.py'):
        copy(V2/module,HERE/'implementation'/module)
    for f in (V2/'modality_se/models').rglob('*.pt'):
        rel = f.relative_to(V2/'modality_se/models')
        target = '01_pretrained' if '_population_' in f.name and '_pair_' not in f.name else '02_personalization'
        copy(f,HERE/target/'weights'/rel)
    for f in ('comparison.json','STABILITY.md','stability.json','per_person_seed_metrics.json','verification.json'):
        copy(V2/'modality_se/results'/f,HERE/'metrics'/f)
    for f in ('comparison.json','REVIEW.md','complete.json'):
        copy(V2/'residual_calibration/results'/f,HERE/'metrics'/'residual_followup'/f)
    for f in ('attention.json.gz','attention_interpretation.json'):
        copy(V2/'reports/results'/f,HERE/'metrics'/f)
    # Source snapshots preserve exact training implementation and its imports.
    packages = ('shared','attention_cnn','modality_se','paired_se','se_personalization','personalized_forecasting')
    for package in packages:
        for f in (V2/package).rglob('*.py'):
            if 'tests' not in f.parts:
                copy(f,HERE/'reproduction_source'/f.relative_to(V2))
    for f in ('audit_snapshot.json','requirements.txt'):
        copy(V2/f,HERE/'reproduction_source'/f)
    for f in ('modality_se/PROTOCOL.md','paired_se/PROTOCOL.md','paired_se/results/manifest.json',
              'paired_se/models/manifest.json','paired_se/fresh_20261012/results/cohort_audit.json'):
        copy(V2/f,HERE/'reproduction_source'/f)

    rows=[]
    for variant,label in [('baseline','Plain CNN'),('se_last','CNN+SE'),('temporal_last','CNN+temporal'),
                          ('cbam_last','CNN+CBAM'),('additive_last','CNN+additive'),('se_all','CNN+SE-all')]:
        runs=[attention['runs'][f'{variant}_seed{s}'] for s in (42,43,44)]
        val=np.mean([r['metrics']['validation']['macro']['mae'] for r in runs],axis=0)
        test=np.mean([r['metrics']['test']['macro']['mae'] for r in runs],axis=0)
        rows.append([label,runs[0]['completion']['parameters'],val,test])
    for column in (2,3):
        best = best_mask([row[column] for row in rows])
        for row,mask in zip(rows,best):
            row[column] = pair(row[column],best=mask)
    table('architecture','Dual-input architectures: three-seed participant-macro MAE (SBP / DBP), mmHg. Bold: lowest per pressure and split.',
          ['Model','Parameters','Validation','Main test'],rows)

    rows=[]
    metrics=('mae','rmse','bias','error_sd','within_5','within_10','within_15')
    for split in ('validation','test','aami'):
        modes=('dual','ppg')
        values=[a['population'][mode]['summary'][split]['pooled'] for mode in modes]
        best={k:best_mask([m[k]['mean'] for m in values],k) for k in metrics}
        for i,(mode,m) in enumerate(zip(modes,values)):
            rows.append([f'{split} / {mode}']+[pair(m[k]['mean'],k.startswith('within'),best[k][i]) for k in metrics])
    headers=['Split / input','MAE','RMSE','Bias','SD','Within 5 (\\%)','Within 10 (\\%)','Within 15 (\\%)']
    table('population','Population pooled metrics, averaged across three seeds. All pressure pairs are SBP / DBP. Thresholds are absolute errors in mmHg; SD uses N-1. Bold: best per pressure within each split.',headers,rows,True)

    for phase,title in [('historical_six','Original six people'),('previous_sixteen','Subsequent sixteen people')]:
        for aggregation in ('macro','pooled'):
            rows=[]
            for mode in ('dual','ppg'):
                for method,label in [('calibration_mean','Mean'),('bias','Bias correction'),(MODEL_METHOD,'PairSE')]:
                    m=a['personalization'][phase]['methods'][mode][method][aggregation]
                    rows.append([f'{mode} / {label}']+[m[k] for k in metrics])
            for column,k in enumerate(metrics,1):
                best=best_mask([row[column] for row in rows],k)
                for row,mask in zip(rows,best):
                    row[column]=pair(row[column],k.startswith('within'),mask)
            table(f'{phase}_{aggregation}',f'{title}: {aggregation} metrics, three-seed average. Pairs are SBP / DBP; errors in mmHg. '+('Macro SD averages within-person sample SD.' if aggregation=='macro' else 'Pooled SD includes between-person offsets.')+' Bold: best per pressure across all six rows.',headers,rows,True)
    rows=[]
    for phase,short in [('historical_six','Six'),('previous_sixteen','Sixteen')]:
        for mode in ('dual','ppg'):
            for s in ('42','43','44'):
                d=a['personalization'][phase]['seed_variability'][mode]
                values=[d[m]['by_seed'][s]['macro']['mae'] for m in ('calibration_mean','bias',MODEL_METHOD)]
                rows.append([short,mode,s]+[pair(v,best=b) for v,b in zip(values,best_mask(values))])
    table('seeds','Every personal seed: participant-macro MAE (SBP / DBP), mmHg. The reference-only mean is deterministic, not three independent replications. Bold: best method per pressure within each row.',
          ['Cohort','Input','Seed','Personal mean','Bias correction','PairSE'],rows,True)

    epochs=[]
    for mode in ('dual','ppg'):
        runs=a['population'][mode]['runs']
        best=best_mask([r['completion']['best_validation_macro_mae'] for r in runs])
        for r,b in zip(runs,best):
            c=r['completion']
            score=f"{c['best_validation_macro_mae']:.3f}"
            epochs.append([mode,c['seed'],c['actual_epochs'],c['best_epoch'],'\\textbf{'+score+'}' if b else score])
    table('epochs','Population fits and validation-selected checkpoints. Bold: lowest validation score per input configuration.',
          ['Input','Seed','Run epochs','Best epoch','Val. score'],epochs)
    summary=dict(date='20261013',source_commit=source_commit,
                 population_selection='dual within SE; plain CNN won prior architecture-wide validation',
                 learned_personalization='PairSE, both modalities retained; no universal bias-correction winner',
                 representative_seed=42,personal_config=a['selections'],
                 comparisons={p:a['personalization'][p]['methods'] for p in ('historical_six','previous_sixteen')})
    put(HERE/'metrics'/'release_summary.json',json.dumps(summary,indent=2)+'\n')
    entries=re.split(r'(?=^@)',(V2/'reports/references.bib').read_text(),flags=re.M)
    cited=('hu2018se','wang2023pulsedb','schlesinger2020','leitner2022')
    put(HERE/'report'/'references.bib','\n'.join(e for e in entries if any('{'+k+',' in e.split('\n')[0] for k in cited)))
    print('Packaged weights:',len(list(HERE.glob('0*/weights/**/*.pt'))))
    print('Generated tables and source snapshots. No training was performed.')


if __name__=='__main__':main()
