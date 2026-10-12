# PPG-only versus dual-channel CNN+SE — 20261012 results

Three seeds (42/43/44), identical source splits and early20% personal budgets. Values are participant-macro SBP / DBP, averaged over seeds, not an ensemble. All test cohorts have been examined previously; do not interpret them as new independent acceptance.

## Population models

| Split | Input | MAE | RMSE | Bias | Error SD | Within5 % | Within10 % | Within15 % |
|---|---|---|---|---|---|---|---|---|
| validation | dual | 12.51 / 8.26 | 14.91 / 9.65 | -1.19 / -0.03 | 10.26 / 5.95 | 26.13 / 37.91 | 48.54 / 67.08 | 66.09 / 85.17 |
| validation | ppg | 12.50 / 8.41 | 14.82 / 9.73 | -1.43 / -0.72 | 10.23 / 5.80 | 25.39 / 37.28 | 48.35 / 66.37 | 66.64 / 84.30 |
| test | dual | 12.39 / 8.00 | 14.72 / 9.34 | -1.76 / -0.89 | 10.25 / 5.97 | 26.03 / 38.83 | 49.22 / 68.91 | 67.60 / 86.68 |
| test | ppg | 12.32 / 7.85 | 14.68 / 9.17 | -2.05 / -1.64 | 10.29 / 5.71 | 25.52 / 38.39 | 48.67 / 69.38 | 67.82 / 87.85 |
| aami | dual | 19.99 / 12.62 | 21.89 / 13.76 | -13.80 / -8.15 | 9.86 / 6.07 | 17.19 / 24.84 | 35.01 / 49.66 | 51.26 / 69.98 |
| aami | ppg | 20.80 / 13.19 | 22.66 / 14.20 | -14.70 / -9.04 | 9.83 / 5.82 | 18.54 / 22.60 | 35.02 / 46.11 | 49.81 / 67.42 |

## Personalization — development

| Input / method | MAE | RMSE | Bias | Error SD | Within5 % | Within10 % | Within15 % |
|---|---|---|---|---|---|---|---|
| dual / population | 13.86 / 9.42 | 15.74 / 11.03 | -8.33 / -1.21 | 8.26 / 6.25 | 14.70 / 28.02 | 33.90 / 54.83 | 57.03 / 84.57 |
| dual / calibration_mean | 12.53 / 5.26 | 15.40 / 7.66 | -2.72 / 0.60 | 11.38 / 7.22 | 25.31 / 64.58 | 44.53 / 85.84 | 63.48 / 93.89 |
| dual / bias | 6.76 / 5.88 | 8.95 / 7.91 | -1.41 / 0.81 | 8.26 / 6.25 | 48.86 / 57.67 | 78.85 / 83.52 | 90.49 / 93.95 |
| dual / paired_head_lr0.0003_s25 | 7.53 / 5.16 | 9.43 / 7.14 | -4.02 / -0.69 | 7.29 / 5.65 | 42.99 / 65.06 | 70.78 / 88.93 | 88.69 / 95.02 |
| ppg / population | 13.39 / 8.76 | 15.48 / 10.28 | -5.00 / -0.39 | 8.59 / 6.39 | 20.22 / 31.70 | 37.62 / 60.34 | 57.53 / 87.00 |
| ppg / calibration_mean | 12.53 / 5.26 | 15.40 / 7.66 | -2.72 / 0.60 | 11.38 / 7.22 | 25.31 / 64.58 | 44.53 / 85.84 | 63.48 / 93.89 |
| ppg / bias | 9.17 / 5.43 | 11.39 / 7.43 | -2.72 / 0.22 | 8.59 / 6.39 | 33.96 / 61.50 | 60.81 / 89.52 | 82.10 / 95.31 |
| ppg / paired_head_lr0.0003_s25 | 7.23 / 4.30 | 9.01 / 6.54 | -3.55 / -0.78 | 7.05 / 5.75 | 46.61 / 75.12 | 71.97 / 91.15 | 88.81 / 94.90 |

dual: selected on development `paired_head_lr0.0003_s25`; both-controls/both-endpoints success here = **False**.


ppg: selected on development `paired_head_lr0.0003_s25`; both-controls/both-endpoints success here = **True**.


## Personalization — historical_six

| Input / method | MAE | RMSE | Bias | Error SD | Within5 % | Within10 % | Within15 % |
|---|---|---|---|---|---|---|---|
| dual / population | 13.44 / 8.39 | 15.32 / 9.41 | 8.34 / 6.87 | 9.35 / 4.25 | 18.95 / 38.14 | 40.79 / 63.10 | 61.74 / 83.07 |
| dual / calibration_mean | 9.47 / 4.22 | 12.19 / 5.46 | 0.50 / -0.95 | 11.14 / 4.82 | 35.72 / 68.34 | 64.56 / 92.32 | 80.14 / 97.75 |
| dual / bias | 9.16 / 4.40 | 11.23 / 5.27 | 0.60 / -0.46 | 9.35 / 4.25 | 33.26 / 62.42 | 63.41 / 93.89 | 81.90 / 99.53 |
| dual / paired_head_lr0.0003_s25 | 8.32 / 3.76 | 10.27 / 4.69 | 0.98 / -0.53 | 9.07 / 4.11 | 36.65 / 71.51 | 68.06 / 95.53 | 86.12 / 99.28 |
| ppg / population | 12.50 / 8.30 | 14.42 / 9.18 | 5.65 / 4.68 | 8.49 / 4.15 | 24.89 / 34.00 | 45.66 / 67.54 | 63.15 / 84.21 |
| ppg / calibration_mean | 9.47 / 4.22 | 12.19 / 5.46 | 0.50 / -0.95 | 11.14 / 4.82 | 35.72 / 68.34 | 64.56 / 92.32 | 80.14 / 97.75 |
| ppg / bias | 8.27 / 4.28 | 10.31 / 5.34 | -1.49 / -1.83 | 8.49 / 4.15 | 39.01 / 63.42 | 68.40 / 93.20 | 85.32 / 99.11 |
| ppg / paired_head_lr0.0003_s25 | 8.19 / 3.85 | 10.17 / 4.83 | -1.89 / -1.79 | 8.66 / 4.26 | 37.68 / 70.85 | 68.21 / 94.91 | 86.24 / 99.11 |

dual: selected on development `paired_head_lr0.0003_s25`; both-controls/both-endpoints success here = **True**.


ppg: selected on development `paired_head_lr0.0003_s25`; both-controls/both-endpoints success here = **True**.


## Personalization — previous_sixteen

| Input / method | MAE | RMSE | Bias | Error SD | Within5 % | Within10 % | Within15 % |
|---|---|---|---|---|---|---|---|
| dual / population | 16.87 / 9.88 | 18.57 / 10.77 | 3.66 / 2.30 | 8.38 / 4.83 | 18.27 / 27.57 | 34.22 / 53.92 | 49.38 / 77.41 |
| dual / calibration_mean | 17.41 / 9.71 | 19.56 / 10.87 | 3.23 / 3.02 | 11.18 / 5.73 | 15.06 / 23.73 | 28.68 / 57.18 | 42.71 / 81.01 |
| dual / bias | 12.82 / 7.48 | 14.60 / 8.56 | 5.19 / 3.11 | 8.38 / 4.83 | 21.87 / 40.94 | 42.33 / 69.61 | 61.71 / 91.21 |
| dual / paired_head_lr0.0003_s25 | 12.67 / 7.61 | 14.58 / 8.72 | 2.93 / 1.78 | 9.18 / 4.92 | 27.74 / 40.67 | 48.45 / 72.59 | 62.60 / 89.10 |
| ppg / population | 17.93 / 10.49 | 19.56 / 11.25 | 3.07 / 1.81 | 8.07 / 4.57 | 17.47 / 27.49 | 29.93 / 50.84 | 43.88 / 73.97 |
| ppg / calibration_mean | 17.41 / 9.71 | 19.56 / 10.87 | 3.23 / 3.02 | 11.18 / 5.73 | 15.06 / 23.73 | 28.68 / 57.18 | 42.71 / 81.01 |
| ppg / bias | 12.74 / 6.89 | 14.33 / 7.95 | 4.55 / 2.73 | 8.07 / 4.57 | 20.55 / 42.05 | 40.49 / 76.80 | 63.05 / 93.28 |
| ppg / paired_head_lr0.0003_s25 | 12.47 / 7.51 | 14.07 / 8.60 | 2.63 / 1.61 | 8.78 / 4.81 | 26.39 / 38.33 | 46.45 / 70.90 | 63.59 / 88.89 |

dual: selected on development `paired_head_lr0.0003_s25`; both-controls/both-endpoints success here = **False**.


ppg: selected on development `paired_head_lr0.0003_s25`; both-controls/both-endpoints success here = **False**.


## Verification and interpretation
- Verified all 816 units and replayed 39564 target predictions; maximum difference 0.00000000 mmHg.
- Complete per-person/seed results, seed variability, pressure ranges and paired person-bootstrap intervals are in the JSON files.
- SE architecture and parameter count are matched. PPG-only zeroes the ECG input internally, including the reference encoder; it is not an inference-time deletion from a dual-trained model.
- Pair heads learn from source participants; personal fitting only receives early references and timestamps. Every method has the same support/query IDs.
- Threshold fractions and error statistics do not establish clinical blood-pressure-device certification.
