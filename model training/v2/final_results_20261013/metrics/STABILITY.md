# Matched modality and personalization stability — 20261012

Both PPG-only and ECG+PPG models are retained; no published model is replaced.

## Main finding

PPG-only personalization beats the personal-mean control in both pressures for all three seeds on both target cohorts. It does **not** consistently beat same-input bias correction. On the 16-person cohort, SBP MAE improves in all three seeds but DBP MAE worsens in all three seeds. On the six-person cohort, the average SBP gain over bias correction is only 0.08 mmHg and occurs in one of three seeds.

Dual-channel personalization also fails the both-pressures/both-controls criterion on the 16-person cohort. The PPG-versus-dual personalized MAE differences have participant-bootstrap intervals crossing zero for both pressures in both target cohorts; neither is a demonstrated stable winner.

## Reading the metrics

Pairs are SBP / DBP. Macro metrics first compute each participant's metric and then weight participants equally. Pooled metrics weight windows equally. Seed averages are averages of independently evaluated models, not an ensemble. Macro SD is the average within-person sample error SD, not the pooled error SD. These differ because people contribute unequal numbers of target windows.

Both cohorts were evaluated in earlier research. This is fixed-protocol replication, not untouched independent confirmation. All methods use the same eligible early-20%-of-time calibration windows and later query windows. Settings were selected on four separate development participants, not these targets. No additional fitting or parameter selection is done by this report.

## historical_six: each seed

| Input | Seed | Personal mean MAE | Bias-corrected MAE | Personalized MAE |
|---|---|---|---|---|
| ppg | 42 | 9.47 / 4.22 | 7.85 / 4.37 | 8.07 / 3.95 |
| ppg | 43 | 9.47 / 4.22 | 8.57 / 4.22 | 8.63 / 4.04 |
| ppg | 44 | 9.47 / 4.22 | 8.40 / 4.24 | 7.88 / 3.55 |
| dual | 42 | 9.47 / 4.22 | 9.00 / 4.35 | 7.59 / 3.73 |
| dual | 43 | 9.47 / 4.22 | 9.91 / 4.66 | 9.05 / 3.96 |
| dual | 44 | 9.47 / 4.22 | 8.58 / 4.20 | 8.32 / 3.58 |

### macro: all metrics, averaged over seeds

| Input / method | MAE | RMSE | Bias | Sample error SD | Within 5 % | Within 10 % | Within 15 % |
|---|---|---|---|---|---|---|---|
| ppg / Personal mean | 9.47 / 4.22 | 12.19 / 5.46 | 0.50 / -0.95 | 11.14 / 4.82 | 35.72 / 68.34 | 64.56 / 92.32 | 80.14 / 97.75 |
| ppg / Same-input bias correction | 8.27 / 4.28 | 10.31 / 5.34 | -1.49 / -1.83 | 8.49 / 4.15 | 39.01 / 63.42 | 68.40 / 93.20 | 85.32 / 99.11 |
| ppg / Personalized | 8.19 / 3.85 | 10.17 / 4.83 | -1.89 / -1.79 | 8.66 / 4.26 | 37.68 / 70.85 | 68.21 / 94.91 | 86.24 / 99.11 |
| dual / Personal mean | 9.47 / 4.22 | 12.19 / 5.46 | 0.50 / -0.95 | 11.14 / 4.82 | 35.72 / 68.34 | 64.56 / 92.32 | 80.14 / 97.75 |
| dual / Same-input bias correction | 9.16 / 4.40 | 11.23 / 5.27 | 0.60 / -0.46 | 9.35 / 4.25 | 33.26 / 62.42 | 63.41 / 93.89 | 81.90 / 99.53 |
| dual / Personalized | 8.32 / 3.76 | 10.27 / 4.69 | 0.98 / -0.53 | 9.07 / 4.11 | 36.65 / 71.51 | 68.06 / 95.53 | 86.12 / 99.28 |

### pooled: all metrics, averaged over seeds

| Input / method | MAE | RMSE | Bias | Sample error SD | Within 5 % | Within 10 % | Within 15 % |
|---|---|---|---|---|---|---|---|
| ppg / Personal mean | 9.86 / 4.17 | 12.98 / 5.57 | 1.55 / -0.70 | 12.89 / 5.53 | 35.46 / 68.81 | 62.28 / 92.46 | 77.64 / 97.74 |
| ppg / Same-input bias correction | 8.51 / 4.17 | 10.99 / 5.39 | 0.06 / -1.38 | 10.99 / 5.21 | 38.10 / 64.92 | 66.80 / 93.88 | 83.85 / 99.09 |
| ppg / Personalized | 8.47 / 3.94 | 10.80 / 5.16 | -1.14 / -1.73 | 10.73 / 4.86 | 36.35 / 70.15 | 66.69 / 94.49 | 85.08 / 98.93 |
| dual / Personal mean | 9.86 / 4.17 | 12.98 / 5.57 | 1.55 / -0.70 | 12.89 / 5.53 | 35.46 / 68.81 | 62.28 / 92.46 | 77.64 / 97.74 |
| dual / Same-input bias correction | 9.61 / 4.57 | 12.11 / 5.66 | 2.20 / 0.20 | 11.88 / 5.61 | 32.15 / 60.35 | 61.21 / 93.11 | 79.67 / 99.39 |
| dual / Personalized | 8.92 / 3.91 | 11.52 / 5.13 | 2.12 / -0.11 | 11.31 / 5.10 | 35.23 / 69.72 | 65.39 / 94.78 | 83.23 / 99.13 |

### Stability versus same-input bias correction

- ppg: seeds with lower SBP / DBP MAE: 1/3 / 3/3; participants improved after averaging seeds: 3/6 / 4/6. Paired MAE-difference 95% intervals: SBP [-1.61, 1.56]; DBP [-1.32, 0.19]. Negative means improvement.
- dual: seeds with lower SBP / DBP MAE: 3/3 / 3/3; participants improved after averaging seeds: 5/6 / 5/6. Paired MAE-difference 95% intervals: SBP [-1.46, -0.23]; DBP [-0.95, -0.29]. Negative means improvement.

## previous_sixteen: each seed

| Input | Seed | Personal mean MAE | Bias-corrected MAE | Personalized MAE |
|---|---|---|---|---|
| ppg | 42 | 17.41 / 9.71 | 12.68 / 6.85 | 12.40 / 7.29 |
| ppg | 43 | 17.41 / 9.71 | 12.91 / 6.94 | 12.42 / 7.68 |
| ppg | 44 | 17.41 / 9.71 | 12.62 / 6.87 | 12.58 / 7.57 |
| dual | 42 | 17.41 / 9.71 | 12.68 / 7.49 | 12.78 / 7.73 |
| dual | 43 | 17.41 / 9.71 | 12.78 / 7.57 | 12.62 / 7.50 |
| dual | 44 | 17.41 / 9.71 | 13.00 / 7.36 | 12.62 / 7.60 |

### macro: all metrics, averaged over seeds

| Input / method | MAE | RMSE | Bias | Sample error SD | Within 5 % | Within 10 % | Within 15 % |
|---|---|---|---|---|---|---|---|
| ppg / Personal mean | 17.41 / 9.71 | 19.56 / 10.87 | 3.23 / 3.02 | 11.18 / 5.73 | 15.06 / 23.73 | 28.68 / 57.18 | 42.71 / 81.01 |
| ppg / Same-input bias correction | 12.74 / 6.89 | 14.33 / 7.95 | 4.55 / 2.73 | 8.07 / 4.57 | 20.55 / 42.05 | 40.49 / 76.80 | 63.05 / 93.28 |
| ppg / Personalized | 12.47 / 7.51 | 14.07 / 8.60 | 2.63 / 1.61 | 8.78 / 4.81 | 26.39 / 38.33 | 46.45 / 70.90 | 63.59 / 88.89 |
| dual / Personal mean | 17.41 / 9.71 | 19.56 / 10.87 | 3.23 / 3.02 | 11.18 / 5.73 | 15.06 / 23.73 | 28.68 / 57.18 | 42.71 / 81.01 |
| dual / Same-input bias correction | 12.82 / 7.48 | 14.60 / 8.56 | 5.19 / 3.11 | 8.38 / 4.83 | 21.87 / 40.94 | 42.33 / 69.61 | 61.71 / 91.21 |
| dual / Personalized | 12.67 / 7.61 | 14.58 / 8.72 | 2.93 / 1.78 | 9.18 / 4.92 | 27.74 / 40.67 | 48.45 / 72.59 | 62.60 / 89.10 |

### pooled: all metrics, averaged over seeds

| Input / method | MAE | RMSE | Bias | Sample error SD | Within 5 % | Within 10 % | Within 15 % |
|---|---|---|---|---|---|---|---|
| ppg / Personal mean | 16.96 / 9.34 | 20.53 / 10.99 | 0.39 / 1.33 | 20.53 / 10.91 | 16.80 / 24.24 | 32.25 / 59.61 | 48.18 / 83.07 |
| ppg / Same-input bias correction | 12.03 / 6.58 | 14.74 / 8.25 | 2.72 / 1.92 | 14.49 / 8.03 | 26.22 / 46.35 | 47.40 / 76.83 | 65.08 / 94.13 |
| ppg / Personalized | 12.34 / 7.30 | 15.29 / 9.16 | -0.25 / 0.37 | 15.28 / 9.15 | 26.25 / 42.51 | 47.55 / 71.73 | 65.05 / 89.68 |
| dual / Personal mean | 16.96 / 9.34 | 20.53 / 10.99 | 0.39 / 1.33 | 20.53 / 10.91 | 16.80 / 24.24 | 32.25 / 59.61 | 48.18 / 83.07 |
| dual / Same-input bias correction | 11.77 / 6.83 | 14.68 / 8.79 | 3.00 / 1.64 | 14.36 / 8.63 | 27.50 / 45.63 | 48.69 / 75.11 | 67.60 / 92.73 |
| dual / Personalized | 13.01 / 7.66 | 16.57 / 9.73 | 0.58 / 0.67 | 16.57 / 9.71 | 28.04 / 41.79 | 47.62 / 70.39 | 61.95 / 87.94 |

### Stability versus same-input bias correction

- ppg: seeds with lower SBP / DBP MAE: 3/3 / 0/3; participants improved after averaging seeds: 8/16 / 4/16. Paired MAE-difference 95% intervals: SBP [-2.12, 1.53]; DBP [-0.04, 1.26]. Negative means improvement.
- dual: seeds with lower SBP / DBP MAE: 2/3 / 1/3; participants improved after averaging seeds: 8/16 / 6/16. Paired MAE-difference 95% intervals: SBP [-2.33, 2.09]; DBP [-0.87, 0.98]. Negative means improvement.

## Important trade-off in the 16-person PPG comparison

Personalization reduces pooled signed bias from 2.72 / 1.92 to -0.25 / 0.37 mmHg versus bias correction, but pooled sample error SD rises from 14.49 / 8.03 to 15.28 / 9.15. Pooled MAE also worsens from 12.03 / 6.58 to 12.34 / 7.30. Thus a lower aggregate bias is not evidence of better individual predictions. The small macro SBP improvement reverses under window-weighted aggregation.

## Retained artifacts

- `models/ppg_population_seed{42,43,44}.pt` and `models/dual_population_seed{42,43,44}.pt`: both population modalities.
- `models/{ppg,dual}_pair_population_seed{42,43,44}.pt`: reference-conditioned source heads.
- `models/{ppg,dual}/{historical_six,previous_sixteen}/`: per-person checkpoints.
- `results/comparison.json`: full seed metrics, tails, pressure strata and paired intervals.
- `results/per_person_seed_metrics.json`: all person/seed/method results.
- `results/verification.json`: all 816 units verified; 39,564 replayed predictions match exactly.
- Use `modality_se.predict.load_predict` to restore the correct input mask.

Three seeds describe training variability, not independent participants. No claim of clinical validation is made.
