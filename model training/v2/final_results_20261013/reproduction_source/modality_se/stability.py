"""Summarize frozen results without training, selecting or changing models."""
import json
from pathlib import Path
import numpy as np


def summarize(data):
    out = {}
    for phase in ('historical_six', 'previous_sixteen'):
        phase_data = data['personalization'][phase]
        out[phase] = {}
        for modality in ('ppg', 'dual'):
            selected = data['selections'][modality]['selected']['method']
            by_method = phase_data['seed_variability'][modality]
            checks = {}
            for control in ('calibration_mean', 'bias'):
                diffs = np.array([
                    np.array(by_method[selected]['by_seed'][str(seed)]['macro']['mae'])
                    - by_method[control]['by_seed'][str(seed)]['macro']['mae']
                    for seed in (42, 43, 44)
                ])
                paired = phase_data['paired'][modality][control]
                checks[control] = dict(
                    seed_deltas=diffs.tolist(),
                    seeds_better_per_endpoint=(diffs < 0).sum(0).tolist(),
                    all_seeds_both_endpoints=bool((diffs < 0).all()),
                    participant_bootstrap_ci95=paired['ci95'],
                    ci_upper_below_zero_per_endpoint=(np.array(paired['ci95'][1]) < 0).tolist(),
                    people_improved=paired['people_improved'],
                    n_people=paired['n_people'],
                )
            out[phase][modality] = dict(selected=selected, controls=checks,
                all_seeds_beat_both_controls_both_pressures=all(
                    c['all_seeds_both_endpoints'] for c in checks.values()))
    return out


def main():
    folder = Path(__file__).resolve().parent / 'results'
    data = json.loads((folder / 'comparison.json').read_text())
    verification = json.loads((folder / 'verification.json').read_text())
    assert verification['units'] == 816 and verification['seed_count'] == 3
    result = summarize(data)
    (folder / 'stability.json').write_text(json.dumps(result, indent=2) + '\n')
    fmt = lambda x: f'{x[0]:.2f} / {x[1]:.2f}'
    labels = {'calibration_mean': 'Personal mean', 'bias': 'Same-input bias correction'}
    lines = ['# Matched modality and personalization stability — 20261012', '',
        'Both PPG-only and ECG+PPG models are retained; no published model is replaced.', '',
        '## Main finding', '',
        'PPG-only personalization beats the personal-mean control in both pressures '
        'for all three seeds on both target cohorts. It does **not** consistently beat '
        'same-input bias correction. On the 16-person cohort, SBP MAE improves in '
        'all three seeds but DBP MAE worsens in all three seeds. On the six-person '
        'cohort, the average SBP gain over bias correction is only 0.08 mmHg and '
        'occurs in one of three seeds.', '',
        'Dual-channel personalization also fails the both-pressures/both-controls '
        'criterion on the 16-person cohort. The PPG-versus-dual personalized MAE '
        'differences have participant-bootstrap intervals crossing zero for both '
        'pressures in both target cohorts; neither is a demonstrated stable winner.', '',
        '## Reading the metrics', '',
        'Pairs are SBP / DBP. Macro metrics first compute each participant\'s metric '
        'and then weight participants equally. Pooled metrics weight windows equally. '
        'Seed averages are averages of independently evaluated models, not an ensemble. '
        'Macro SD is the average within-person sample error SD, not the pooled error SD. '
        'These differ because people contribute unequal numbers of target windows.', '',
        'Both cohorts were evaluated in earlier research. This is fixed-protocol '
        'replication, not untouched independent confirmation. All methods use the same '
        'eligible early-20%-of-time calibration windows and later query windows. '
        'Settings were selected on four separate development participants, not these '
        'targets. No additional fitting or parameter selection is done by this report.']
    for phase in ('historical_six', 'previous_sixteen'):
        p = data['personalization'][phase]
        lines += ['', f'## {phase}: each seed', '',
            '| Input | Seed | Personal mean MAE | Bias-corrected MAE | Personalized MAE |',
            '|---|---|---|---|---|']
        for m in ('ppg', 'dual'):
            selected = result[phase][m]['selected']
            stats = p['seed_variability'][m]
            for s in ('42', '43', '44'):
                vals = [fmt(stats[k]['by_seed'][s]['macro']['mae'])
                        for k in ('calibration_mean', 'bias', selected)]
                lines.append(f'| {m} | {s} | ' + ' | '.join(vals) + ' |')
        for scope in ('macro', 'pooled'):
            lines += ['', f'### {scope}: all metrics, averaged over seeds', '',
                '| Input / method | MAE | RMSE | Bias | Sample error SD | Within 5 % | Within 10 % | Within 15 % |',
                '|---|---|---|---|---|---|---|---|']
            for m in ('ppg', 'dual'):
                selected = result[phase][m]['selected']
                for k in ('calibration_mean', 'bias', selected):
                    stats = p['methods'][m][k][scope]
                    vals = [fmt(np.array(stats[z]) * (100 if z.startswith('within') else 1))
                            for z in ('mae', 'rmse', 'bias', 'error_sd', 'within_5', 'within_10', 'within_15')]
                    lines.append(f'| {m} / {labels.get(k, "Personalized")} | ' + ' | '.join(vals) + ' |')
        lines += ['', '### Stability versus same-input bias correction', '']
        for m in ('ppg', 'dual'):
            c = result[phase][m]['controls']['bias']
            ci = c['participant_bootstrap_ci95']
            lines += [f'- {m}: seeds with lower SBP / DBP MAE: '
                f"{c['seeds_better_per_endpoint'][0]}/3 / {c['seeds_better_per_endpoint'][1]}/3; "
                f"participants improved after averaging seeds: {c['people_improved'][0]}/{c['n_people']} / "
                f"{c['people_improved'][1]}/{c['n_people']}. "
                f'Paired MAE-difference 95% intervals: SBP [{ci[0][0]:.2f}, {ci[1][0]:.2f}]; '
                f'DBP [{ci[0][1]:.2f}, {ci[1][1]:.2f}]. Negative means improvement.']
    lines += ['', '## Important trade-off in the 16-person PPG comparison', '',
        'Personalization reduces pooled signed bias from 2.72 / 1.92 to '
        '-0.25 / 0.37 mmHg versus bias correction, but pooled sample error SD rises '
        'from 14.49 / 8.03 to 15.28 / 9.15. Pooled MAE also worsens from '
        '12.03 / 6.58 to 12.34 / 7.30. Thus a lower aggregate bias is not evidence '
        'of better individual predictions. The small macro SBP improvement reverses '
        'under window-weighted aggregation.', '',
        '## Retained artifacts', '',
        '- `models/ppg_population_seed{42,43,44}.pt` and '
        '`models/dual_population_seed{42,43,44}.pt`: both population modalities.',
        '- `models/{ppg,dual}_pair_population_seed{42,43,44}.pt`: reference-conditioned source heads.',
        '- `models/{ppg,dual}/{historical_six,previous_sixteen}/`: per-person checkpoints.',
        '- `results/comparison.json`: full seed metrics, tails, pressure strata and paired intervals.',
        '- `results/per_person_seed_metrics.json`: all person/seed/method results.',
        '- `results/verification.json`: all 816 units verified; 39,564 replayed predictions match exactly.',
        '- Use `modality_se.predict.load_predict` to restore the correct input mask.', '',
        'Three seeds describe training variability, not independent participants. '
        'No claim of clinical validation is made.']
    (folder / 'STABILITY.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
