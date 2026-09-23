# V1 complete error metrics

Computed from the existing selected-model predictions; no retraining or test-based selection. Values are SBP / DBP. Error = prediction minus reference. SD uses N-1; threshold percentages include the boundary.

## validation

194 participants; 69,840 segments.

| Metric | SBP | DBP |
|---|---:|---:|
| MAE | 12.384 | 8.335 |
| Bias | -1.115 | -0.927 |
| error_SD | 15.645 | 10.583 |
| within_5_pct | 25.577 | 37.427 |
| within_10_pct | 48.256 | 66.539 |
| within_15_pct | 66.777 | 84.943 |
| participant_macro_MAE | 12.384 | 8.335 |
| participant_MAE_median | 11.583 | 7.148 |
| participant_MAE_p90 | 18.938 | 14.159 |

## test

144 participants; 57,600 segments.

| Metric | SBP | DBP |
|---|---:|---:|
| MAE | 12.459 | 8.457 |
| Bias | -1.916 | -1.645 |
| error_SD | 15.895 | 10.660 |
| within_5_pct | 26.059 | 37.240 |
| within_10_pct | 48.800 | 66.111 |
| within_15_pct | 67.550 | 84.623 |
| participant_macro_MAE | 12.459 | 8.457 |
| participant_MAE_median | 11.024 | 7.316 |
| participant_MAE_p90 | 20.796 | 14.353 |

## test_excluding_prior

141 participants; 56,400 segments.

| Metric | SBP | DBP |
|---|---:|---:|
| MAE | 12.442 | 8.387 |
| Bias | -1.971 | -1.893 |
| error_SD | 15.900 | 10.543 |
| within_5_pct | 26.239 | 37.521 |
| within_10_pct | 49.039 | 66.615 |
| within_15_pct | 67.754 | 85.062 |
| participant_macro_MAE | 12.442 | 8.387 |
| participant_MAE_median | 11.002 | 7.314 |
| participant_MAE_p90 | 21.012 | 13.708 |

## aami

116 participants; 666 segments.

| Metric | SBP | DBP |
|---|---:|---:|
| MAE | 19.339 | 13.169 |
| Bias | -12.743 | -9.307 |
| error_SD | 22.719 | 13.935 |
| within_5_pct | 19.219 | 24.775 |
| within_10_pct | 36.186 | 45.195 |
| within_15_pct | 53.003 | 67.267 |
| participant_macro_MAE | 19.580 | 13.095 |
| participant_MAE_median | 15.948 | 11.465 |
| participant_MAE_p90 | 36.757 | 23.723 |

These are PulseDB source-data results, not ND cuff-reference results or a clinical device validation. All pressure errors are in mmHg; within-threshold values are percentages. Participant-macro MAE weights people equally. The AAMI-designated dataset name does not establish compliance with a clinical validation protocol.
