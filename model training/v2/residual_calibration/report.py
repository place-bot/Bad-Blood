"""Export frozen artifacts and a factual V2 architecture/method review."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from personalized_forecasting.run import file_hash,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    done=json.loads((a.run/'complete.json').read_text());assert done['status']=='verified_complete'
    data=json.loads((a.run/'comparison.json').read_text())
    assert data['verification']['units']==1260 and data['verification']['max_replay_error']<.001
    for f,h in data['manifest']['source_hashes'].items():assert file_hash(ROOT/f)==h
    dest=ROOT/'residual_calibration/results';dest.mkdir(exist_ok=True)
    for name in ('comparison.json','selection.json','manifest.json','complete.json'):
        shutil.copyfile(a.run/name,dest/name)
    models=ROOT/'residual_calibration/models'; models.mkdir(exist_ok=True)
    for path in a.run.glob('*/*/*/seed*/selected/model.pt'):
        relative=Path(*(part.replace(':','_') for part in path.relative_to(a.run).parts))
        target=models/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    atomic_json(models/'manifest.json',{str(p.relative_to(models)):file_hash(p) for p in sorted(models.rglob('*.pt'))})
    fmt=lambda a:f'{a[0]:.2f} / {a[1]:.2f}'
    names={'population':'Population','calibration_mean':'Personal mean','bias':'Same-input bias correction','prior_pair':'Previous PairSE','selected':'New development-selected pipeline'}
    lines=['# V2 architecture and personalization review — 20261012','',
        '## Outcome','',
        'The follow-up is complete and verified, but the desired joint superiority '
        'has not been achieved. Do not promote the new residual pipeline. Retain '
        'both pretrained input modalities, the prior PairSE research models, and '
        'bias/mean controls. A positive training result is not required to preserve '
        'a reproducible negative experiment.', '',
        '## Two retained population model families','',
        '| Stage | Shape / operation |', '|---|---|',
        '| Input | 10 seconds at 125 Hz; per-window/channel min-max normalization |',
        '| Modalities | ECG+PPG, or PPG with ECG internally masked to zero |',
        '| Stem | Conv1d 2→8, kernel 7, stride 2; BN; SiLU; length 625 |',
        '| Residual stages | 8→16→32→48, stride 2 per block; lengths 313→157→79 |',
        '| Block | Conv k7 + BN + SiLU + Conv k5 + BN + residual projection + SiLU |',
        '| Attention | Final-stage squeeze-and-excitation: temporal mean; 48→6→48; sigmoid channel gates |',
        '| Pooling and head | Temporal global average → 48; Dropout .15 → Linear 48→32 → SiLU → Linear 32→2 |',
        '| Output | Standardized SBP/DBP, converted to mmHg using source-training center/scale |',
        '| Parameters | 38,296 each; matched widths and parameter storage for the controlled comparison |', '',
        'The PPG model was trained from scratch without ECG, not obtained by masking '
        'ECG from a dual-trained model. SE weights feature channels, not an explicit '
        'choice between raw ECG and PPG. Neither model uses current ABP as input.', '',
        'Both use 1,085 source-training participants / 390,600 segments, 192 '
        'validation participants / 69,120 segments, seeds 42/43/44, standardized '
        'MSE and validation early stopping. Data are the corrected VitalDB-derived '
        'PulseDB partition, not the full combined PulseDB release.', '',
        'For a single representative artifact, seed42 has the lowest population '
        'validation score within each family. Keep all three seeds for reporting; '
        'do not choose the best test seed.', '',
        '- Dual representative: `../../modality_se/models/dual_population_seed42.pt`.',
        '- PPG representative: `../../modality_se/models/ppg_population_seed42.pt`.',
        '- Implementation: `../../shared/models.py` and `../../modality_se/model.py`.',
        '- Complete population metrics: `../../modality_se/results/comparison.json`.', '',
        '## Prior strongest learned personalization candidate','',
        'PairSE encodes the current and reference waveforms through a shared CNN+SE. '
        'It starts with the known reference pressure plus the difference between '
        'population predictions. A 100→64→32→2 MLP learns an additional '
        'reference-conditioned change from feature differences, reference features, '
        'reference BP and population-prediction differences. It subtracts the '
        'same-reference correction to anchor identical waveforms. Predictions '
        'average over the complete early calibration reference bank.', '',
        'A source-trained pair head is personalized using all eligible early '
        'support pairs. The previous development choice was head-only Adam, '
        'learning rate 0.0003, 25 steps; the head/full grid was evaluated before '
        'locking. No additional personal labels are introduced. This is a retained '
        'learned candidate, not a universal winner over bias correction.', '',
        '## This follow-up','',
        'Twenty centered residual-ridge/kernel configurations plus support-only '
        'forward validation were evaluated with the unchanged three seeds and '
        'calibration budget. Source-trained features stay fixed, and all personal '
        'reference labels fit the final correction. Separate SBP/DBP choices are '
        'made on four development participants, before opening target files.', '',
        'Dual development selection: RBF residual correction on 48-D features '
        '(width .25, penalty .01) for SBP; prediction-feature linear residual '
        'ridge (penalty .01) for DBP. PPG development selection: previous PairSE '
        'for both outputs. The candidate grid and choices were not changed after '
        'target evaluation.', '',
        '## Why DBP did not improve','',
        'Observed in the preceding PPG experiment: the extra personalized '
        'correction has DBP mean-square magnitude 19.72 mmHg², but its cross-term '
        'with the old error removes only 3.92 mmHg². Net pooled MSE increases '
        'from 68.08 to 83.88 mmHg². Thus the waveform-dependent correction adds '
        'more error variation than it removes, even though signed bias improves.', '',
        'This decomposition describes the failure, not its proven physiological '
        'cause. Plausible contributors are few independent early references, '
        'time-correlated windows, support-to-query drift, and model selection on '
        'only four development people. Source pair training also lacks recoverable '
        'timestamps, unlike the strictly chronological personal evaluation.', '',
        'The new dual endpoint-specific method reduces development DBP MAE to '
        '4.65, but yields 8.23 on the sixteen-person replication, versus 7.48 '
        'for bias correction. Separating outputs alone is therefore insufficient.']
    for phase,arms in data['targets'].items():
        lines+=['',f'## {phase}','',
            'Entries are SBP / DBP; macro gives each participant equal weight.', '',
            '| Input | Seed | Personal mean MAE | Bias MAE | Prior PairSE MAE | New selected MAE |','|---|---|---|---|---|---|']
        for m,z in arms.items():
            for seed,v in z['per_seed'].items():
                lines.append(f'| {m} | {seed} | '+' | '.join(fmt(v[k]['macro']['mae']) for k in ('calibration_mean','bias','prior_pair','selected'))+' |')
        for scope in ('macro','pooled'):
            lines+=['',f'### {scope} metrics, three-seed average','',
                '| Input / method | MAE | RMSE | Bias | Sample error SD | Within 5 % | Within 10 % | Within 15 % |',
                '|---|---|---|---|---|---|---|---|']
            for m,z in arms.items():
                for method in ('calibration_mean','bias','prior_pair','selected'):
                    stats=z['methods'][method][scope]
                    vals=[fmt(np.array(stats[k])*(100 if k.startswith('within') else 1)) for k in ('mae','rmse','bias','error_sd','within_5','within_10','within_15')]
                    lines.append(f'| {m} / {names[method]} | '+' | '.join(vals)+' |')
        lines+=['','### Paired comparison to same-input bias correction','']
        for m,z in arms.items():
            paired=z['paired']['bias'];ci=paired['ci95']
            wins=np.sum([np.array(v['selected']['macro']['mae']) < np.array(v['bias']['macro']['mae']) for v in z['per_seed'].values()],0)
            lines.append(f'- {m}: seeds better SBP/DBP {wins[0]}/3 / {wins[1]}/3; MAE change {fmt(paired["delta_mae"])}; '
                f'person-bootstrap 95% intervals SBP [{ci[0][0]:.2f},{ci[1][0]:.2f}], DBP [{ci[0][1]:.2f},{ci[1][1]:.2f}].')
    lines+=['','## Interpretation and next decision','',
        'No method currently establishes stable superiority to both controls on '
        'both pressures across the two target cohorts. The new residual method '
        'should remain an experiment, not replace the prior release. For V2 reporting, '
        'present the population architecture comparison as one result and the '
        'personalization study, including the strong bias baseline, as the other. '
        'If stable superiority remains a required milestone, V2 is not finished '
        'against that criterion. The next useful step is a larger, separately '
        'defined development cohort with a new untouched test cohort, not continued '
        'tuning against these same 22 previously evaluated people.', '',
        '## Verification','',
        f'- {data["verification"]["units"]} units; {data["verification"]["replayed_predictions"]} replayed target predictions; '
        f'maximum replay difference {data["verification"]["max_replay_error"]:.8f} mmHg.',
        '- Same support and query indices; both modality selections saved before target loading.',
        '- Both target cohorts were evaluated previously: exploratory replication, not independent acceptance.',
        '- Macro SD is the average within-person sample SD; pooled SD includes between-person offsets.',
        '- Threshold fractions are descriptive, not clinical-device certification.', '',
        '## Literature and reproducibility','',
        'The [protocol](../PROTOCOL.md) links the primary ridge-transfer, kernel-calibration '
        'and neural-personalization papers and explains where our implementation differs. '
        'Their published performance is not a target we can assume under this split.', '',
        '- [Ridge regression with pretrained CNN features](https://pubmed.ncbi.nlm.nih.gov/34784289/).',
        '- [Kernel/feedback calibration](https://pmc.ncbi.nlm.nih.gov/articles/PMC8914760/).',
        '- [Layer-selective neural personalization](https://pubmed.ncbi.nlm.nih.gov/34077378/).',
        '- Full comparison and per-seed metrics: `comparison.json`; all selected weights: `../models/`.',
        '- Predictor: `residual_calibration.run.predict_saved`; normalized N×2×1250 input, SBP/DBP output.',
        '- Reproduce: `python residual_calibration/run.py --output RUN --prior MODALITY_RUN --personal ORIGINAL_MAT --fresh PREVIOUS_SIXTEEN_MAT`.',
        '- This round is saved locally; no automatic push, email, deployment or replacement.']
    (dest/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    print('EXPORTED',len(list(models.rglob('*.pt'))),'models',dest/'REVIEW.md')


if __name__=='__main__':main()
