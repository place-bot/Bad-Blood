# PulseDB-based Model — V1

V1 establishes the four-model comparison and selects the higher-resolution
CNN. [V2](../PulseDB-based%20Model-V2/README.md) retains that architecture and
investigates a revised training objective with a broader error analysis.

## Run predictions

Python 3.12 was used. From this directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python predict.py "../../data/PulseDB/examples/real_20_segments.csv" --format filtered --output predictions.csv
```

`models/selection.json` selects `cnn_full.pt`. The selected network needs
NumPy, SciPy and PyTorch; XGBoost and the other packages support reproduction
and comparison. CPU inference is supported. Raspberry Pi installation and
latency still require testing on the target OS/hardware; the current execution
verification is macOS ARM64, not a claim of measured Pi performance.

The input is a headered wide CSV. It needs `fs_hz=125`, `ecg_0000` through
`ecg_1249`, and `ppg_0000` through `ppg_1249`. IDs are copied into the output
but do not affect predictions. SBP/DBP reference columns are not read by the
predictor and can be blank for inference. `filtered` means the versioned
filtered/min-max representation. `raw125` applies the documented raw adapter
first. Do not filter a converted ND file twice.

Output columns are `participant_id,segment_id,pred_SBP_mmHg,pred_DBP_mmHg,status,detail`.
`ok` means the numeric input/output checks passed, not medical validity or a
signal-quality certificate. Nonfinite/flat/wrong-length inputs are rejected.
Existing output files are not overwritten.

## Models and training

| Candidate | Validation SBP MAE | Validation DBP MAE |
|---|---:|---:|
| Small CNN (5x initial pooling) | 12.86 | 8.44 |
| Higher-resolution CNN (selected) | 12.38 | 8.33 |
| XGBoost, depth 3 | 13.22 | 8.53 |
| XGBoost, depth 5 | 13.07 | 8.51 |
| Training-mean reference | 14.82 | 9.44 |

Units are mmHg. Both CNNs use residual 1D convolution blocks, global average
pooling and a two-output head (37,666 trainable parameters). Training used
395,640 segments from 1,099 people; validation used 69,840 segments from 194
different people. Target standardization is fitted on training only.
AdamW, batch 256, initial learning rate 0.001, weight decay 0.001, normalized
MSE, small waveform noise augmentation and validation-based LR reduction were
used. Both runs stopped after epoch 11; their best checkpoint was epoch 3.
The full history is preserved. The selection criterion is the average of
participant-macro SBP/DBP MAE. No ensemble is used.

XGBoost fits a separate regressor per target from 122 ECG/PPG-only features.
The source files contain all hyperparameters. The requested comparison is
limited to these four candidates; depth 7 is excluded.

## Locked test results

| Evaluation | People / segments | SBP MAE | DBP MAE |
|---|---|---:|---:|
| Official calibration-free | 144 / 57,600 | 12.46 | 8.46 |
| Excluding earlier exploratory participants | 141 / 56,400 | 12.44 | 8.39 |
| AAMI-designated subset | 116 / 666 | 19.34 | 13.17 |

Three official-test participants had appeared in earlier exploratory work.
They were not used to fit this run, and an exposure-excluded result is provided
for transparency. `reports/final_evaluation.json` includes subject-bootstrap
95% confidence intervals, bias, RMSE, error SD, R² and mean-reference results.
The AAMI subset result is not a clinical certification. Performance at broader
BP extremes is substantially worse and is retained in the report.

## Reproduce

Download and prepare data as described under `data/PulseDB`. Then:

```sh
python train_models.py cnn --epochs 40
python train_models.py cnn_full --epochs 40
sh run.sh train_models.py trees
```

The launcher supplies PyTorch's packaged OpenMP library path on macOS for
XGBoost. Set `BP_PYTHON` if the desired interpreter is not `python3`.
Training overwrites candidate checkpoints: reproduce in a separate copy if
preserving this release. `evaluate.py` refuses to overwrite the saved final
evaluation. It selects from validation first, then evaluates the fixed test
sets. Test results must not be used to select or retune this release.

`model_checksums.json` contains checkpoint hashes. `test_contract.py` checks
input shape, invalid inputs, normalization and the raw adapter. The included
example predictions are format/integration fixtures, not test-set results.

## ND adaptation

Keep the PulseDB source model frozen initially. Use 40 ND development people
for any residual model, preprocessing decisions and participant-grouped
validation; reserve 10 people for final comparison. Compare source-only,
ND-only and adapted predictions on the same final people. No ND adaptation
weights are claimed or supplied before ND data exist.

## References

- Wang et al. (2023), [PulseDB](https://doi.org/10.3389/fdgth.2022.1090854).
- Huang et al. (2024), [subject-independent model validation](https://www.oaepublish.com/articles/chatmed.2023.23).
- Moulaeifard et al. (2025), [generalization benchmark](https://pubmed.ncbi.nlm.nih.gov/40959521/).

Our compact model is not a reproduction of any of these networks. The
references inform the signal interface and subject-independent evaluation.
