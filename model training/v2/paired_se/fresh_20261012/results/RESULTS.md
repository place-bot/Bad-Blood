# V2 fixed-method new-participant validation — 20261012

21 prospectively locked candidates; 16 eligible new participants; 652 early calibration and 2310 later query windows.

These people were absent from every prior population train/validation/test/AAMI partition and the 31 historical exploratory people. All eligible people from the first author-hosted folder page were included; this is a convenience sample, not a random or external cohort.

Fixed CNN+SE pair model, head-only Adam lr=0.0003, 25 updates, seeds 42/43/44. No hyperparameter or checkpoint selection on these results. All methods use the same early 20% of elapsed time and the same later windows. Numbers below are participant-macro, then averaged over seeds; SBP / DBP.

| Method | MAE | RMSE | Bias | Within-person error SD | Within 5 (%) | Within 10 (%) | Within 15 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unadapted CNN+SE | 16.87 / 9.88 | 18.57 / 10.77 | 3.66 / 2.30 | 8.38 / 4.83 | 18.27 / 27.57 | 34.22 / 53.92 | 49.38 / 77.41 |
| Personal mean | 17.41 / 9.71 | 19.56 / 10.87 | 3.23 / 3.02 | 11.18 / 5.73 | 15.06 / 23.73 | 28.68 / 57.18 | 42.71 / 81.01 |
| SE + personal bias | 12.82 / 7.48 | 14.60 / 8.56 | 5.19 / 3.11 | 8.38 / 4.83 | 21.87 / 40.94 | 42.33 / 69.61 | 61.71 / 91.21 |
| Reference-conditioned SE | 12.67 / 7.61 | 14.58 / 8.72 | 2.93 / 1.78 | 9.18 / 4.92 | 27.74 / 40.67 | 48.45 / 72.59 | 62.60 / 89.10 |

## Primary conclusion

Lower SBP AND DBP MAE than BOTH controls: **False**.

The earlier six-person improvement is preserved separately. This follow-up is not pooled with those six to change the conclusion.

Against SE + personal bias: paired-model minus control MAE = -0.15 / 0.13 mmHg; 95% participant-bootstrap intervals: SBP [-2.33, 2.09], DBP [-0.87, 0.98]. Improved people: 8/16 SBP, 6/16 DBP.

Against Personal mean: paired-model minus control MAE = -4.74 / -2.10 mmHg; 95% participant-bootstrap intervals: SBP [-8.35, -0.81], DBP [-3.62, -0.54]. Improved people: 13/16 SBP, 12/16 DBP.

## Seed checks

| Seed | Personal mean | Bias correction | Paired model |
|---|---:|---:|---:|
| 42 | 17.41 / 9.71 | 12.68 / 7.49 | 12.78 / 7.73 |
| 43 | 17.41 / 9.71 | 12.78 / 7.57 | 12.62 / 7.50 |
| 44 | 17.41 / 9.71 | 13.00 / 7.36 | 12.62 / 7.60 |

## Individual results

| Participant | Early / later windows | Personal mean MAE | Bias MAE | Paired MAE |
|---|---:|---:|---:|---:|
| vitaldb:p000006 | 3 / 117 | 19.33 / 18.06 | 23.50 / 19.13 | 28.61 / 20.44 |
| vitaldb:p000008 | 32 / 232 | 22.85 / 11.30 | 16.87 / 10.98 | 25.48 / 13.20 |
| vitaldb:p000022 | 9 / 28 | 5.88 / 4.50 | 9.51 / 4.44 | 5.44 / 3.95 |
| vitaldb:p000023 | 60 / 139 | 17.84 / 8.96 | 8.88 / 5.47 | 7.35 / 4.77 |
| vitaldb:p000027 | 3 / 5 | 18.03 / 14.78 | 12.42 / 9.48 | 4.14 / 6.56 |
| vitaldb:p000037 | 27 / 88 | 16.92 / 6.17 | 6.49 / 3.35 | 9.44 / 3.89 |
| vitaldb:p000038 | 81 / 181 | 13.94 / 7.37 | 9.66 / 5.25 | 10.24 / 5.38 |
| vitaldb:p000041 | 70 / 286 | 22.91 / 10.08 | 17.97 / 7.98 | 16.18 / 10.25 |
| vitaldb:p000056 | 35 / 68 | 6.05 / 4.45 | 15.72 / 6.30 | 18.97 / 8.75 |
| vitaldb:p000059 | 6 / 3 | 25.76 / 11.45 | 21.06 / 13.77 | 13.68 / 8.93 |
| vitaldb:p000061 | 35 / 202 | 19.19 / 9.44 | 9.26 / 5.43 | 14.96 / 6.72 |
| vitaldb:p000066 | 19 / 116 | 30.58 / 14.27 | 14.55 / 8.44 | 15.38 / 9.14 |
| vitaldb:p000071 | 76 / 303 | 8.24 / 6.76 | 4.54 / 3.20 | 5.07 / 4.59 |
| vitaldb:p000079 | 28 / 60 | 22.50 / 10.42 | 15.59 / 5.59 | 11.17 / 5.02 |
| vitaldb:p000085 | 96 / 186 | 17.26 / 9.75 | 10.56 / 4.79 | 9.04 / 5.05 |
| vitaldb:p000107 | 72 / 296 | 11.32 / 7.58 | 8.59 / 6.02 | 7.65 / 5.12 |

## Technical exclusions

- vitaldb:p000010: Mixed cases or unordered time
- vitaldb:p000047: Every support row must participate in a separated pair
- vitaldb:p000058: Insufficient early/later windows
- vitaldb:p000073: Single-window record: insufficient early/later windows
- vitaldb:p000110: Every support row must participate in a separated pair

No exclusion used prediction performance. Singleton MATLAB records were identified as insufficient calibration/test data; the loader fix did not change model settings.

Verified 192 completed units and replayed 6930 saved-model predictions; maximum replay difference 0 mmHg.

New records can be much shorter than the prior cohort. The identical 20%-of-time budget does not guarantee identical absolute numbers of calibration labels; counts are given above. All labels are ABP-derived. This within-record evaluation does not test real ND cuffs or persistence across visits.
