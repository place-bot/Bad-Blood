# Full support CNN plus SE personalization results

October 12, 2026. SBP / DBP throughout; errors are mmHg. Seeds 42/43/44 are averaged, not ensembled.

## Outcome

Requested objective reached on the six previously examined target people: **False**.
Selected from development only: `full_centered_lr0.001_s100`.
Target results are exploratory follow-up, not fresh independent validation. No target-based retuning was performed.

## Patient macro target metrics

| Method | MAE | RMSE | Signed bias | Error SD | Within 5 % | Within 10 % | Within 15 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unadapted SE | 13.44 / 8.39 | 15.32 / 9.41 | 8.34 / 6.87 | 9.35 / 4.25 | 18.95 / 38.14 | 40.79 / 63.10 | 61.74 / 83.07 |
| Personal reference mean | 9.47 / 4.22 | 12.19 / 5.46 | 0.50 / -0.95 | 11.14 / 4.82 | 35.72 / 68.34 | 64.56 / 92.32 | 80.14 / 97.75 |
| Personal bias correction | 9.16 / 4.40 | 11.23 / 5.27 | 0.60 / -0.46 | 9.35 / 4.25 | 33.26 / 62.42 | 63.41 / 93.89 | 81.90 / 99.53 |
| Selected SE fine tuning | 12.14 / 5.28 | 14.75 / 6.50 | -7.41 / -2.86 | 11.35 / 4.86 | 30.36 / 56.16 | 54.56 / 86.50 | 70.65 / 96.79 |

Patient-macro SD is the average of within-person sample error SDs, not a pooled clinical-validation SD. Pooled values and all other metrics are in `test_summary.json`.

## Per person comparison

| Participant | Early training windows | Later query windows | Personal mean MAE | Bias correction MAE | SE fine tuning MAE |
|---|---:|---:|---:|---:|---:|
| vitaldb:p005879 | 151 | 654 | 5.88 / 3.24 | 7.72 / 2.41 | 5.83 / 2.72 |
| vitaldb:p005885 | 228 | 1220 | 12.33 / 3.66 | 12.11 / 5.53 | 12.62 / 5.15 |
| vitaldb:p005977 | 152 | 569 | 8.82 / 3.72 | 6.72 / 3.74 | 7.57 / 4.19 |
| vitaldb:p005984 | 131 | 666 | 8.73 / 6.35 | 8.02 / 6.42 | 9.67 / 6.94 |
| vitaldb:p006009 | 59 | 761 | 11.00 / 4.32 | 10.73 / 4.13 | 12.87 / 5.78 |
| vitaldb:p006052 | 94 | 414 | 10.06 / 4.03 | 9.67 / 4.16 | 24.27 / 6.89 |

## Paired differences

Fine tuning minus control; negative favors fine tuning. Descriptive 95% patient-bootstrap intervals average seeds before resampling six people.

- Versus Personal bias correction: mean change **2.98 / 0.88**, CI lower -0.23 / 0.14, upper 7.82 / 1.76; improved 1/6 SBP and 1/6 DBP.

- Versus Personal reference mean: mean change **2.67 / 1.06**, CI lower -0.28 / 0.22, upper 7.47 / 1.94; improved 2/6 SBP and 1/6 DBP.

## Development selection

| Setting | SBP / DBP MAE | Worst ratio to either control |
|---|---:|---:|
| `full_centered_lr0.001_s100` | 6.68 / 5.20 | 0.9886 |
| `full_plain_lr0.0001_s20` | 7.23 / 5.13 | 1.0693 |
| `last_gate_centered_lr0.001_s100` | 7.00 / 5.65 | 1.0730 |
| `full_plain_lr0.001_s100` | 6.76 / 5.70 | 1.0833 |
| `head_gate_centered_lr0.001_s20` | 7.14 / 5.71 | 1.0850 |
| `head_gate_centered_lr0.0001_s100` | 6.85 / 5.78 | 1.0977 |
| `head_gate_centered_lr0.0001_s20` | 6.76 / 5.85 | 1.1119 |
| `last_gate_centered_lr0.0001_s20` | 6.32 / 6.00 | 1.1394 |
| `last_gate_plain_lr0.0001_s20` | 6.95 / 6.02 | 1.1434 |
| `full_centered_lr0.001_s20` | 7.90 / 5.38 | 1.1697 |
| `head_gate_plain_lr0.001_s100` | 7.92 / 5.82 | 1.1717 |
| `last_gate_plain_lr0.0001_s100` | 6.11 / 6.18 | 1.1743 |
| `last_gate_centered_lr0.001_s20` | 7.53 / 6.25 | 1.1880 |
| `full_plain_lr0.001_s20` | 8.43 / 5.82 | 1.2478 |
| `full_centered_lr0.0001_s100` | 6.87 / 6.63 | 1.2595 |
| `last_gate_plain_lr0.001_s20` | 8.64 / 5.27 | 1.2788 |
| `full_centered_lr0.0001_s20` | 6.80 / 6.75 | 1.2831 |
| `last_gate_plain_lr0.001_s100` | 6.72 / 7.00 | 1.3299 |
| `full_plain_lr0.0001_s100` | 7.22 / 7.57 | 1.4383 |
| `head_gate_plain_lr0.001_s20` | 10.26 / 6.99 | 1.5183 |
| `last_gate_centered_lr0.0001_s100` | 7.46 / 8.13 | 1.5438 |
| `head_gate_centered_lr0.001_s100` | 7.34 / 8.55 | 1.6237 |
| `head_gate_plain_lr0.0001_s100` | 11.87 / 8.18 | 1.7567 |
| `head_gate_plain_lr0.0001_s20` | 13.42 / 9.16 | 1.9867 |

Development reference-mean MAE: 12.53 / 5.26; bias-correction MAE: 6.76 / 5.88.

## Verification

Audited 324 development units and 72 target units. Replayed 12852 saved-model predictions; maximum absolute replay difference 0.00000000 mmHg.

All eligible early windows were used in every optimizer update. The same early windows define both controls; later query labels do not enter personal fitting. No original sparse-reference result or V1 checkpoint was overwritten.

The dense early labels come from invasive ABP in public data, not from newly acquired cuffs. These same-record results do not establish cross-visit ND performance or clinical certification.
