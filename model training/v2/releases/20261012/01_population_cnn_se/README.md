# V2 outcome 1 — population CNN+SE — 20261012

Current version: **high-resolution residual 1D CNN + SE at the final residual
block**, implementation variant `se_last`, 38,296 parameters. Input is a filtered,
per-window/channel min-max normalized 10-second ECG/PPG pair at 125 Hz
(`N × 2 × 1250`); output is contemporaneous SBP/DBP (`N × 2`, mmHg).

## Everything in this folder

- [Architecture](source/shared/models.py): `BPNet`, `Block`, `ChannelAttention`.
- [Training implementation](source/attention_cnn/train.py) and [protocol](source/PROTOCOL.md).
- [Runnable inference](source/se_personalization/predict.py).
- [Trained weights](models/): `cnn_se_seed42.pt`, `cnn_se_seed43.pt`, `cnn_se_seed44.pt`.
- [Full comparison metrics](results/metrics.json) and [selection/uncertainty analysis](results/attention_interpretation.json).
- [Checksums](SHA256SUMS.json): source and artifact integrity.

## Architecture and training

Two input channels → stride-2 Conv1D (8 channels) → residual blocks
16→32→48 → SE gate (48→6→48, sigmoid) → global average pooling →
48→32→2 regression head. SE adds 630 parameters to the 37,666-parameter CNN.

Population training uses 1,085 people / 390,600 windows; validation uses 192 /
69,120. AdamW, lr=0.001, weight decay=0.001, batch 256, normalized-target MSE,
gradient clipping 5, maximum 40 epochs, validation early stopping patience 8.
The three SE fits ran nine epochs each and selected epoch 1 independently by
validation score. The checkpoints are already trained; no retraining is needed
for inference. Three seeds quantify run variation, not an ensemble.

## Observed primary-test comparison

141 people, 56,400 windows. Participant-macro metrics, averaged over three seeds.
SBP / DBP; mmHg except percentages.

| Architecture | MAE | Macro RMSE | Bias | Within-person error SD | Within 5 (%) | Within 10 (%) | Within 15 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unchanged CNN, matched retraining | 12.59 / 8.36 | 14.96 / 9.79 | −1.81 / −1.29 | 10.58 / 6.31 | 25.60 / 37.59 | 48.11 / 66.88 | 66.60 / 84.90 |
| CNN+SE | **12.39 / 8.00** | **14.72 / 9.34** | −1.76 / −0.89 | **10.25 / 5.97** | **26.03 / 38.83** | **49.22 / 68.91** | **67.60 / 86.68** |

This matched comparison isolates adding attention more cleanly than comparing
the historical V1 checkpoint trained with a different exclusion list. The SE
test ranking does not overturn the original validation selection, which slightly
favored the unchanged CNN. Gains are modest; uncertainty and all comparisons
remain available rather than claiming universal superiority.

## Run prediction from this folder

Use the project's Python environment with NumPy, PyTorch and h5py. Only load
trusted project checkpoints. `waves.npy` must already meet the input contract.

```sh
python source/se_personalization/predict.py \
  --checkpoint models/cnn_se_seed42.pt \
  --waves /path/to/waves.npy --output /path/to/predictions.csv --device cpu
```

The input contains ECG first and PPG second; no reference BP is required by this
population model. For individual reference-conditioned estimation use the
[separate personalization outcome](../02_personalization/).

## Reproduce population training

```sh
python source/attention_cnn/train.py --variant se_last --seed 42 \
  --data-root /path/to/bp_model_v2/data \
  --output-dir /path/to/new_se_seed42 --device mps
```

Repeat for seeds 43/44; CPU/CUDA are also supported. Source data stays external.
The historical exclusion file is included at the path expected by the source
snapshot. [Dataset provenance and license](../../../../../data/PulseDB/LICENSE.txt).
