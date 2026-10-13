# V2 architecture and personalization review — 20261012

## Outcome

The follow-up is complete and verified, but the desired joint superiority has not been achieved. Do not promote the new residual pipeline. Retain both pretrained input modalities, the prior PairSE research models, and bias/mean controls. A positive training result is not required to preserve a reproducible negative experiment.

## Two retained population model families

| Stage | Shape / operation |
|---|---|
| Input | 10 seconds at 125 Hz; per-window/channel min-max normalization |
| Modalities | ECG+PPG, or PPG with ECG internally masked to zero |
| Stem | Conv1d 2→8, kernel 7, stride 2; BN; SiLU; length 625 |
| Residual stages | 8→16→32→48, stride 2 per block; lengths 313→157→79 |
| Block | Conv k7 + BN + SiLU + Conv k5 + BN + residual projection + SiLU |
| Attention | Final-stage squeeze-and-excitation: temporal mean; 48→6→48; sigmoid channel gates |
| Pooling and head | Temporal global average → 48; Dropout .15 → Linear 48→32 → SiLU → Linear 32→2 |
| Output | Standardized SBP/DBP, converted to mmHg using source-training center/scale |
| Parameters | 38,296 each; matched widths and parameter storage for the controlled comparison |

The PPG model was trained from scratch without ECG, not obtained by masking ECG from a dual-trained model. SE weights feature channels, not an explicit choice between raw ECG and PPG. Neither model uses current ABP as input.

Both use 1,085 source-training participants / 390,600 segments, 192 validation participants / 69,120 segments, seeds 42/43/44, standardized MSE and validation early stopping. Data are the corrected VitalDB-derived PulseDB partition, not the full combined PulseDB release.

For a single representative artifact, seed42 has the lowest population validation score within each family. Keep all three seeds for reporting; do not choose the best test seed.

- Dual representative: `../../modality_se/models/dual_population_seed42.pt`.
- PPG representative: `../../modality_se/models/ppg_population_seed42.pt`.
- Implementation: `../../shared/models.py` and `../../modality_se/model.py`.
- Complete population metrics: `../../modality_se/results/comparison.json`.

## Prior strongest learned personalization candidate

PairSE encodes the current and reference waveforms through a shared CNN+SE. It starts with the known reference pressure plus the difference between population predictions. A 100→64→32→2 MLP learns an additional reference-conditioned change from feature differences, reference features, reference BP and population-prediction differences. It subtracts the same-reference correction to anchor identical waveforms. Predictions average over the complete early calibration reference bank.

A source-trained pair head is personalized using all eligible early support pairs. The previous development choice was head-only Adam, learning rate 0.0003, 25 steps; the head/full grid was evaluated before locking. No additional personal labels are introduced. This is a retained learned candidate, not a universal winner over bias correction.

## This follow-up

Twenty centered residual-ridge/kernel configurations plus support-only forward validation were evaluated with the unchanged three seeds and calibration budget. Source-trained features stay fixed, and all personal reference labels fit the final correction. Separate SBP/DBP choices are made on four development participants, before opening target files.

Dual development selection: RBF residual correction on 48-D features (width .25, penalty .01) for SBP; prediction-feature linear residual ridge (penalty .01) for DBP. PPG development selection: previous PairSE for both outputs. The candidate grid and choices were not changed after target evaluation.

## Why DBP did not improve

Observed in the preceding PPG experiment: the extra personalized correction has DBP mean-square magnitude 19.72 mmHg², but its cross-term with the old error removes only 3.92 mmHg². Net pooled MSE increases from 68.08 to 83.88 mmHg². Thus the waveform-dependent correction adds more error variation than it removes, even though signed bias improves.

This decomposition describes the failure, not its proven physiological cause. Plausible contributors are few independent early references, time-correlated windows, support-to-query drift, and model selection on only four development people. Source pair training also lacks recoverable timestamps, unlike the strictly chronological personal evaluation.

The new dual endpoint-specific method reduces development DBP MAE to 4.65, but yields 8.23 on the sixteen-person replication, versus 7.48 for bias correction. Separating outputs alone is therefore insufficient.

## historical_six

Entries are SBP / DBP; macro gives each participant equal weight.

| Input | Seed | Personal mean MAE | Bias MAE | Prior PairSE MAE | New selected MAE |
|---|---|---|---|---|---|
| dual | 42 | 9.47 / 4.22 | 9.00 / 4.35 | 7.59 / 3.73 | 9.68 / 4.79 |
| dual | 43 | 9.47 / 4.22 | 9.91 / 4.66 | 9.05 / 3.96 | 10.69 / 4.21 |
| dual | 44 | 9.47 / 4.22 | 8.58 / 4.20 | 8.32 / 3.58 | 9.24 / 4.29 |
| ppg | 42 | 9.47 / 4.22 | 7.85 / 4.37 | 8.07 / 3.95 | 8.07 / 3.95 |
| ppg | 43 | 9.47 / 4.22 | 8.57 / 4.22 | 8.63 / 4.04 | 8.63 / 4.04 |
| ppg | 44 | 9.47 / 4.22 | 8.40 / 4.24 | 7.88 / 3.55 | 7.88 / 3.55 |

### macro metrics, three-seed average

| Input / method | MAE | RMSE | Bias | Sample error SD | Within 5 % | Within 10 % | Within 15 % |
|---|---|---|---|---|---|---|---|
| dual / Personal mean | 9.47 / 4.22 | 12.19 / 5.46 | 0.50 / -0.95 | 11.14 / 4.82 | 35.72 / 68.34 | 64.56 / 92.32 | 80.14 / 97.75 |
| dual / Same-input bias correction | 9.16 / 4.40 | 11.23 / 5.27 | 0.60 / -0.46 | 9.35 / 4.25 | 33.26 / 62.42 | 63.41 / 93.89 | 81.90 / 99.53 |
| dual / Previous PairSE | 8.32 / 3.76 | 10.27 / 4.69 | 0.98 / -0.53 | 9.07 / 4.11 | 36.65 / 71.51 | 68.06 / 95.53 | 86.12 / 99.28 |
| dual / New development-selected pipeline | 9.87 / 4.43 | 11.99 / 5.39 | 1.24 / -0.90 | 9.73 / 4.45 | 33.40 / 63.11 | 61.05 / 92.73 | 79.39 / 99.07 |
| ppg / Personal mean | 9.47 / 4.22 | 12.19 / 5.46 | 0.50 / -0.95 | 11.14 / 4.82 | 35.72 / 68.34 | 64.56 / 92.32 | 80.14 / 97.75 |
| ppg / Same-input bias correction | 8.27 / 4.28 | 10.31 / 5.34 | -1.49 / -1.83 | 8.49 / 4.15 | 39.01 / 63.42 | 68.40 / 93.20 | 85.32 / 99.11 |
| ppg / Previous PairSE | 8.19 / 3.85 | 10.17 / 4.83 | -1.89 / -1.79 | 8.66 / 4.26 | 37.68 / 70.85 | 68.21 / 94.91 | 86.24 / 99.11 |
| ppg / New development-selected pipeline | 8.19 / 3.85 | 10.17 / 4.83 | -1.89 / -1.79 | 8.66 / 4.26 | 37.68 / 70.85 | 68.21 / 94.91 | 86.24 / 99.11 |

### pooled metrics, three-seed average

| Input / method | MAE | RMSE | Bias | Sample error SD | Within 5 % | Within 10 % | Within 15 % |
|---|---|---|---|---|---|---|---|
| dual / Personal mean | 9.86 / 4.17 | 12.98 / 5.57 | 1.55 / -0.70 | 12.89 / 5.53 | 35.46 / 68.81 | 62.28 / 92.46 | 77.64 / 97.74 |
| dual / Same-input bias correction | 9.61 / 4.57 | 12.11 / 5.66 | 2.20 / 0.20 | 11.88 / 5.61 | 32.15 / 60.35 | 61.21 / 93.11 | 79.67 / 99.39 |
| dual / Previous PairSE | 8.92 / 3.91 | 11.52 / 5.13 | 2.12 / -0.11 | 11.31 / 5.10 | 35.23 / 69.72 | 65.39 / 94.78 | 83.23 / 99.13 |
| dual / New development-selected pipeline | 10.92 / 4.53 | 14.28 / 5.72 | 3.54 / -0.23 | 13.74 / 5.67 | 30.49 / 62.08 | 56.77 / 92.30 | 75.26 / 98.93 |
| ppg / Personal mean | 9.86 / 4.17 | 12.98 / 5.57 | 1.55 / -0.70 | 12.89 / 5.53 | 35.46 / 68.81 | 62.28 / 92.46 | 77.64 / 97.74 |
| ppg / Same-input bias correction | 8.51 / 4.17 | 10.99 / 5.39 | 0.06 / -1.38 | 10.99 / 5.21 | 38.10 / 64.92 | 66.80 / 93.88 | 83.85 / 99.09 |
| ppg / Previous PairSE | 8.47 / 3.94 | 10.80 / 5.16 | -1.14 / -1.73 | 10.73 / 4.86 | 36.35 / 70.15 | 66.69 / 94.49 | 85.08 / 98.93 |
| ppg / New development-selected pipeline | 8.47 / 3.94 | 10.80 / 5.16 | -1.14 / -1.73 | 10.73 / 4.86 | 36.35 / 70.15 | 66.69 / 94.49 | 85.08 / 98.93 |

### Paired comparison to same-input bias correction

- dual: seeds better SBP/DBP 0/3 / 1/3; MAE change 0.71 / 0.03; person-bootstrap 95% intervals SBP [-0.75,2.81], DBP [-0.22,0.26].
- ppg: seeds better SBP/DBP 1/3 / 3/3; MAE change -0.08 / -0.43; person-bootstrap 95% intervals SBP [-1.61,1.56], DBP [-1.32,0.19].

## previous_sixteen

Entries are SBP / DBP; macro gives each participant equal weight.

| Input | Seed | Personal mean MAE | Bias MAE | Prior PairSE MAE | New selected MAE |
|---|---|---|---|---|---|
| dual | 42 | 17.41 / 9.71 | 12.68 / 7.49 | 12.78 / 7.73 | 12.39 / 8.23 |
| dual | 43 | 17.41 / 9.71 | 12.78 / 7.57 | 12.62 / 7.50 | 12.02 / 8.63 |
| dual | 44 | 17.41 / 9.71 | 13.00 / 7.36 | 12.62 / 7.60 | 12.61 / 7.82 |
| ppg | 42 | 17.41 / 9.71 | 12.68 / 6.85 | 12.40 / 7.29 | 12.40 / 7.29 |
| ppg | 43 | 17.41 / 9.71 | 12.91 / 6.94 | 12.42 / 7.68 | 12.42 / 7.68 |
| ppg | 44 | 17.41 / 9.71 | 12.62 / 6.87 | 12.58 / 7.57 | 12.58 / 7.57 |

### macro metrics, three-seed average

| Input / method | MAE | RMSE | Bias | Sample error SD | Within 5 % | Within 10 % | Within 15 % |
|---|---|---|---|---|---|---|---|
| dual / Personal mean | 17.41 / 9.71 | 19.56 / 10.87 | 3.23 / 3.02 | 11.18 / 5.73 | 15.06 / 23.73 | 28.68 / 57.18 | 42.71 / 81.01 |
| dual / Same-input bias correction | 12.82 / 7.48 | 14.60 / 8.56 | 5.19 / 3.11 | 8.38 / 4.83 | 21.87 / 40.94 | 42.33 / 69.61 | 61.71 / 91.21 |
| dual / Previous PairSE | 12.67 / 7.61 | 14.58 / 8.72 | 2.93 / 1.78 | 9.18 / 4.92 | 27.74 / 40.67 | 48.45 / 72.59 | 62.60 / 89.10 |
| dual / New development-selected pipeline | 12.34 / 8.23 | 14.30 / 9.49 | 4.84 / 2.83 | 8.81 / 5.75 | 24.78 / 38.00 | 44.97 / 64.99 | 63.32 / 86.39 |
| ppg / Personal mean | 17.41 / 9.71 | 19.56 / 10.87 | 3.23 / 3.02 | 11.18 / 5.73 | 15.06 / 23.73 | 28.68 / 57.18 | 42.71 / 81.01 |
| ppg / Same-input bias correction | 12.74 / 6.89 | 14.33 / 7.95 | 4.55 / 2.73 | 8.07 / 4.57 | 20.55 / 42.05 | 40.49 / 76.80 | 63.05 / 93.28 |
| ppg / Previous PairSE | 12.47 / 7.51 | 14.07 / 8.60 | 2.63 / 1.61 | 8.78 / 4.81 | 26.39 / 38.33 | 46.45 / 70.90 | 63.59 / 88.89 |
| ppg / New development-selected pipeline | 12.47 / 7.51 | 14.07 / 8.60 | 2.63 / 1.61 | 8.78 / 4.81 | 26.39 / 38.33 | 46.45 / 70.90 | 63.59 / 88.89 |

### pooled metrics, three-seed average

| Input / method | MAE | RMSE | Bias | Sample error SD | Within 5 % | Within 10 % | Within 15 % |
|---|---|---|---|---|---|---|---|
| dual / Personal mean | 16.96 / 9.34 | 20.53 / 10.99 | 0.39 / 1.33 | 20.53 / 10.91 | 16.80 / 24.24 | 32.25 / 59.61 | 48.18 / 83.07 |
| dual / Same-input bias correction | 11.77 / 6.83 | 14.68 / 8.79 | 3.00 / 1.64 | 14.36 / 8.63 | 27.50 / 45.63 | 48.69 / 75.11 | 67.60 / 92.73 |
| dual / Previous PairSE | 13.01 / 7.66 | 16.57 / 9.73 | 0.58 / 0.67 | 16.57 / 9.71 | 28.04 / 41.79 | 47.62 / 70.39 | 61.95 / 87.94 |
| dual / New development-selected pipeline | 11.39 / 7.34 | 14.53 / 9.23 | 2.66 / 1.14 | 14.27 / 9.15 | 30.29 / 41.88 | 51.93 / 72.53 | 68.73 / 90.53 |
| ppg / Personal mean | 16.96 / 9.34 | 20.53 / 10.99 | 0.39 / 1.33 | 20.53 / 10.91 | 16.80 / 24.24 | 32.25 / 59.61 | 48.18 / 83.07 |
| ppg / Same-input bias correction | 12.03 / 6.58 | 14.74 / 8.25 | 2.72 / 1.92 | 14.49 / 8.03 | 26.22 / 46.35 | 47.40 / 76.83 | 65.08 / 94.13 |
| ppg / Previous PairSE | 12.34 / 7.30 | 15.29 / 9.16 | -0.25 / 0.37 | 15.28 / 9.15 | 26.25 / 42.51 | 47.55 / 71.73 | 65.05 / 89.68 |
| ppg / New development-selected pipeline | 12.34 / 7.30 | 15.29 / 9.16 | -0.25 / 0.37 | 15.28 / 9.15 | 26.25 / 42.51 | 47.55 / 71.73 | 65.05 / 89.68 |

### Paired comparison to same-input bias correction

- dual: seeds better SBP/DBP 3/3 / 0/3; MAE change -0.48 / 0.75; person-bootstrap 95% intervals SBP [-1.04,0.08], DBP [-0.08,1.58].
- ppg: seeds better SBP/DBP 3/3 / 0/3; MAE change -0.27 / 0.63; person-bootstrap 95% intervals SBP [-2.12,1.53], DBP [-0.04,1.26].

## Interpretation and next decision

No method currently establishes stable superiority to both controls on both pressures across the two target cohorts. The new residual method should remain an experiment, not replace the prior release. For V2 reporting, present the population architecture comparison as one result and the personalization study, including the strong bias baseline, as the other. If stable superiority remains a required milestone, V2 is not finished against that criterion. The next useful step is a larger, separately defined development cohort with a new untouched test cohort, not continued tuning against these same 22 previously evaluated people.

## Verification

- 1260 units; 39564 replayed target predictions; maximum replay difference 0.00001526 mmHg.
- Same support and query indices; both modality selections saved before target loading.
- Both target cohorts were evaluated previously: exploratory replication, not independent acceptance.
- Macro SD is the average within-person sample SD; pooled SD includes between-person offsets.
- Threshold fractions are descriptive, not clinical-device certification.

## Literature and reproducibility

The [protocol](../PROTOCOL.md) links the primary ridge-transfer, kernel-calibration and neural-personalization papers and explains where our implementation differs. Their published performance is not a target we can assume under this split.

- [Ridge regression with pretrained CNN features](https://pubmed.ncbi.nlm.nih.gov/34784289/).
- [Kernel/feedback calibration](https://pmc.ncbi.nlm.nih.gov/articles/PMC8914760/).
- [Layer-selective neural personalization](https://pubmed.ncbi.nlm.nih.gov/34077378/).
- Full comparison and per-seed metrics: `comparison.json`; all selected weights: `../models/`.
- Predictor: `residual_calibration.run.predict_saved`; normalized N×2×1250 input, SBP/DBP output.
- Reproduce: `python residual_calibration/run.py --output RUN --prior MODALITY_RUN --personal ORIGINAL_MAT --fresh PREVIOUS_SIXTEEN_MAT`.
- This round is saved locally; no automatic push, email, deployment or replacement.
