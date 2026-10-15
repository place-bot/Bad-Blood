# Reference-conditioned CNN+SE results

October 12, 2026. SBP / DBP; mmHg unless marked %. Three seeds averaged, not ensembled.

Joint MAE objective reached: **True**.
Development-only choice: `paired_head_lr0.0003_s25`. Six target people were previously examined; this is exploratory, not a new independent test.

## Later-query participant-macro performance

| Model | MAE | RMSE | Signed bias | Error SD | Within 5 % | Within 10 % | Within 15 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original SE | 13.44 / 8.39 | 15.32 / 9.41 | 8.34 / 6.87 | 9.35 / 4.25 | 18.95 / 38.14 | 40.79 / 63.10 | 61.74 / 83.07 |
| Personal mean | 9.47 / 4.22 | 12.19 / 5.46 | 0.50 / -0.95 | 11.14 / 4.82 | 35.72 / 68.34 | 64.56 / 92.32 | 80.14 / 97.75 |
| Original SE + personal bias | 9.16 / 4.40 | 11.23 / 5.27 | 0.60 / -0.46 | 9.35 / 4.25 | 33.26 / 62.42 | 63.41 / 93.89 | 81.90 / 99.53 |
| Selected reference-conditioned SE | 8.32 / 3.76 | 10.27 / 4.69 | 0.98 / -0.53 | 9.07 / 4.11 | 36.65 / 71.51 | 68.06 / 95.53 | 86.12 / 99.28 |

The SD above averages within-person sample error SDs. Full pooled metrics, tail errors, correlations and person-specific MAE are in the JSON results.

## Per-person MAE

| Person | Early windows | Query windows | Mean | Bias correction | Pair model |
|---|---:|---:|---:|---:|---:|
| vitaldb:p005879 | 151 | 654 | 5.88 / 3.24 | 7.72 / 2.41 | 7.08 / 2.20 |
| vitaldb:p005885 | 228 | 1220 | 12.33 / 3.66 | 12.11 / 5.53 | 12.30 / 4.53 |
| vitaldb:p005977 | 152 | 569 | 8.82 / 3.72 | 6.72 / 3.74 | 5.76 / 2.73 |
| vitaldb:p005984 | 131 | 666 | 8.73 / 6.35 | 8.02 / 6.42 | 7.92 / 6.50 |
| vitaldb:p006009 | 59 | 761 | 11.00 / 4.32 | 10.73 / 4.13 | 8.77 / 3.39 |
| vitaldb:p006052 | 94 | 414 | 10.06 / 4.03 | 9.67 / 4.16 | 8.11 / 3.20 |

## Paired differences

Model minus control; negative is better. Descriptive 95% bootstrap intervals resample people after averaging seeds, not waveform windows.
- Versus Original SE + personal bias: -0.84 / -0.64; CI lower -1.46 / -0.95, upper -0.23 / -0.29; improved 5/6 SBP and 5/6 DBP.
- Versus Personal mean: -1.15 / -0.46; CI lower -2.23 / -0.96, upper 0.05 / 0.16; improved 5/6 SBP and 4/6 DBP.

## Development screening

| Method | MAE | Worst ratio to either control |
|---|---:|---:|
| paired_head_lr0.0003_s25 | 7.53 / 5.16 | 1.1145 |
| paired_full_lr0.001_s100 | 7.82 / 5.99 | 1.1567 |
| paired_head_lr0.0003_s100 | 8.22 / 5.76 | 1.2159 |
| paired_head_lr0.001_s25 | 8.28 / 5.31 | 1.2260 |
| paired_full_lr0.001_s25 | 8.38 / 6.50 | 1.2403 |
| paired_full_lr0.0003_s100 | 7.65 / 6.71 | 1.2742 |
| paired_head_lr0.001_s100 | 8.11 / 6.72 | 1.2761 |
| paired_unadapted | 8.88 / 5.90 | 1.3141 |
| paired_full_lr0.0003_s25 | 8.31 / 7.06 | 1.3406 |

## Population pair training

| Seed | Completed epochs | Selected epoch | Validation pair macro MAE |
|---|---:|---:|---:|
| 42 | 20 | 18 | 9.37 / 5.43 |
| 43 | 17 | 12 | 9.41 / 5.42 |
| 44 | 16 | 11 | 9.51 / 5.48 |

## Target seed stability

| Seed | Mean MAE | Bias MAE | Selected pair MAE |
|---|---:|---:|---:|
| 42 | 9.47 / 4.22 | 9.00 / 4.35 | 7.59 / 3.73 |
| 43 | 9.47 / 4.22 | 9.91 / 4.66 | 9.05 / 3.96 |
| 44 | 9.47 / 4.22 | 8.58 / 4.20 | 8.32 / 3.58 |

## Interpretation

The pair head was trained on exactly the existing population training participants; personal fitting uses all eligible early windows. Reference and current population rows are distinct but have no recoverable time ordering. Personal pairs and final reference/query banks have timestamp guards.
This is an inspired ECG/PPG CNN+SE adaptation, not a reproduction of the cited PPG study. Expanded pairs are not new independent calibration measurements. No target-dependent retuning, discarded participants or extra reference labels were used.
Verified 144 development and 72 target units, and replayed 12852 saved-model predictions; maximum difference 0.00000000 mmHg.
