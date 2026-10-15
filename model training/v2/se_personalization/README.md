# CNN plus SE full support personalization

This October 12 extension uses V2's **best observed three-seed primary-test MAE
architecture, CNN+SE**, as requested. Its 38,296-parameter model adds a 48→6→48
squeeze-and-excitation gate after the final residual block. This label does not
mean it won the original validation selection or has the lowest metric in every
comparison. PPG-only was a separate, lower-error single-seed ablation.

## Current outcome

**The requested personalization advantage has not been demonstrated.** The
development-selected full-network tune had target MAE **12.14 / 5.28 mmHg**,
versus **9.47 / 4.22** for personal mean and **9.16 / 4.40** for personal bias
correction. These personal checkpoints are retained for reproducibility, not
recommended in place of the stronger controls. See [all metrics](results/RESULTS.md).

A subsequent 24-setting frozen-feature regularized-head screen also failed its
development gate: selected MAE **6.91 / 5.56**, versus **12.53 / 5.26** (mean)
and **6.76 / 5.88** (bias). It was **not evaluated on the target people**.
The development and target tables concern different people and are not directly
comparable. [Literature and next design](LITERATURE.md) explain reference-conditioned
change learning, parameter-efficient adaptation and pretrained-weight anchoring.

The regularized screen is reproducible with:

```sh
python se_personalization/regularized_head.py \
  --population-root /path/to/20261009/attention \
  --individual-root /path/to/bp_model/data/vital \
  --output /path/to/new_regularized_development_run --device mps
```

This command intentionally exposes no target-evaluation phase. It fits on all
early support rows, compares 32- versus 48-dimensional SE features, two anchors,
and six regularization penalties. Prediction requires current waveforms; labels
are accepted only by the support-fitting function.

## Experiment

Use every eligible ten-second ECG/PPG segment ending in the first 20% of each
person's elapsed recording time. The subsequent query windows and ten-second
boundary guard are unchanged from the original V2 experiment. Input remains
two filtered, per-window-normalized 125-Hz channels, shape `N × 2 × 1250`;
output remains contemporaneous scalar SBP/DBP, not future waveform prediction.

Development uses four people and three SE population seeds. A fixed grid of 24
settings compares head+gate, final block+head+gate, and full-network tuning, with
plain or support-centered MSE. All support rows enter every gradient update.
Settings are selected only from development performance against both personal
mean and additive-bias controls. Only the selected setting reaches the six
previously examined target participants; results are exploratory. See the
[frozen protocol](PROTOCOL.md) for the precise selection and success rules.

Dense support contains 97–215 windows per development person and 59–228 per
target person. It is ABP-derived supervision; it is not equivalent to ten cuffs,
and results cannot establish cross-visit performance on ND students.

## Reproduce

Run from `model training/v2` with the existing V2 Python environment:

```sh
python se_personalization/run.py --phase development \
  --population-root /path/to/20261009/attention \
  --individual-root /path/to/bp_model/data/vital \
  --output /path/to/new_se_dense_run --device mps

python se_personalization/run.py --phase test \
  --population-root /path/to/20261009/attention \
  --individual-root /path/to/bp_model/data/vital \
  --output /path/to/new_se_dense_run --device mps
```

CPU and CUDA are also supported. The second command requires the frozen
development selection. Repeating an interrupted command verifies and reuses
completed units; source/configuration/checksum differences cause an error.
Do not point this runner at prior sparse-reference output directories.

Original population checkpoints and source data access are described in
[DATA_ACCESS.md](../DATA_ACCESS.md). No raw ND data enter this experiment.

## Results and saved models

The completed run is audited and exported with:

```sh
python se_personalization/report.py \
  --run /path/to/new_se_dense_run \
  --individual-root /path/to/bp_model/data/vital \
  --population-root /path/to/20261009/attention \
  --export se_personalization/results --device mps
```

This checks all support/query indices, checksums, metric calculations and the
development-only selection, then replays each saved personal model. It exports
[result tables](results/RESULTS.md), full JSON metrics, and inference weights.

`models/pretrained_se_seed42.pt` (also seeds 43/44) is a population model.
`models/vitaldb_pXXXXXX_seed42.pt` is fitted specifically to that public-data
person's early records, not a universal personalized model for a new person.
For a new person, the pretrained model still needs calibration/training with
that person's observed early waveform/reference pairs.

```sh
python se_personalization/predict.py \
  --checkpoint se_personalization/models/pretrained_se_seed42.pt \
  --waves /path/to/normalized_filtered_ecg_ppg.npy \
  --output /path/to/new_predictions.csv --device cpu
```

The array must be `N × 2 × 1250`, filtered ECG then PPG, independently min-max
normalized within each ten-second window/channel to [0,1]. No reference BP is
read at prediction time. Personal checkpoints include any support-derived output
offset; do not omit it when deploying. Existing outputs are not overwritten.
Only load trusted project checkpoints, since PyTorch checkpoint files can use
pickle. CPU/MPS execution does not establish Raspberry Pi performance.

Data provenance remains PulseDB's VitalDB-derived material, with existing
[CC BY-NC-SA 4.0 data attribution and terms](../../../data/PulseDB/LICENSE.txt).
The published models and results contain no raw ND records or newly uploaded
source waveforms. Consult the original dataset terms before reuse.
