# Reference-conditioned CNN+SE

## 20261012 update — previously unused participants

[Fixed-method follow-up results](fresh_20261012/results/RESULTS.md): 16 eligible
new people out of 21 prospectively locked candidates, no overlap with prior
train/validation/test/AAMI or exploratory identities. The same 20%-of-time
calibration rule, model, checkpoints and adaptation settings were retained.

MAE (SBP / DBP): paired **12.67 / 7.61**, personal mean **17.41 / 9.71**, bias
correction **12.82 / 7.48** mmHg. **The model did not stably beat both controls
on new participants.** Its advantage over personal mean remained, while both
MAE-difference intervals versus bias include zero. The earlier six-person
result below is a separate exploratory experiment.

The [dated personalization package](../releases/20261012/02_personalization/)
keeps architecture, source, trained weights and both result sets together.

This experiment implements a paired-measurement approach inspired by
[Mekonnen et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11467377/).
It keeps our pretrained ECG/PPG CNN+SE backbone, rather than reproducing that
paper's PPG model. The [protocol](PROTOCOL.md) was saved before the experiment.

**Status: completed and verified.** On the six previously examined target
participants, the development-selected pair-head adaptation achieved lower
SBP and DBP participant-macro MAE than both controls, in each of three seeds.

| Method | SBP MAE | DBP MAE |
|---|---:|---:|
| Personal mean | 9.47 | 4.22 |
| Original SE + personal bias | 9.16 | 4.40 |
| Reference-conditioned SE | **8.32** | **3.76** |

The selected setting updates the 8,610-parameter pair head for 25 steps at
lr=0.0003, with the original SE encoder fixed. Full-network settings were also
tested on development, without L2-SP or weight decay, but were not selected.
This is genuine waveform-dependent personal adaptation; it does not require
updating every encoder weight to become a personal model.

RMSE, within-person error SD and within-5/10 proportions also improved over both
controls. Signed bias was not best, and the DBP within-15 fraction was slightly
below bias correction. Five of six people improved for SBP versus each control;
DBP improved for four versus personal mean and five versus bias correction.
The descriptive interval versus personal mean still includes zero. Moreover,
development did not beat both controls, and the targets had been examined
previously: treat this as promising exploratory evidence, not independent proof.
See [all metrics and individual results](results/RESULTS.md).

Verification audited all 144 development and 72 target units and replayed
12,852 saved-model predictions with zero numerical difference. All 113 project
tests passed. Raw waveforms and reference banks are not published.

## Prediction

Inputs are a current ten-second ECG/PPG segment, an earlier reference segment,
and the observed SBP/DBP belonging to that reference. Both waveforms pass through
the same SE encoder. A new nonlinear head predicts a change relative to the
known reference BP. At inference we average over every early reference window.
Current/reference signals are `N x 2 x 1250` float32, ECG then PPG, filtered and
independently min-max normalized per window/channel, 125 Hz. Reference pressures
are `N x 2`, SBP then DBP, in mmHg. Current BP is never an input.

The added head has 8,610 parameters; the combined model has 46,906 parameters.
At initialization it equals the original SE model with personal bias correction.
Training learns waveform-dependent departures from this starting point. Equal
current and reference waveforms always return the known reference BP.

## Training and controls

- Population training: the same 1,085 people / 390,600 windows; validation:
  192 people / 69,120 windows. Three original SE seeds: 42, 43, 44.
- Train the new paired head on distinct same-person window pairs with the SE
  encoder fixed. Population arrays lack timestamps; no chronological claim is
  made for these population pairs.
- Personal adaptation: all eligible early-20%-of-time windows, with timestamp-
  separated earlier/current pairs. Compare head-only and full-network adaptation
  without weight-decay or pretrained-weight anchoring. All reference labels are
  confined to the early period; no later labels enter adaptation.
- Select among nine alternatives (eight adaptations plus no adaptation) on four
  development people. Evaluate only the selected alternative on six targets.
- Personal mean and simple SE bias correction use the identical reference bank,
  later query windows and pretrained seed. These six targets have been examined
  in earlier studies; this is exploratory follow-up, not fresh confirmation.

## Run

From `model training/v2` using the project's existing Python environment:

```sh
python paired_se/run.py --data-root /path/to/bp_model_v2/data \
  --individual-root /path/to/bp_model/data/vital \
  --population-root /path/to/20261009/attention \
  --output /path/to/new_paired_se_run --device mps --phase all
```

CPU/CUDA are also supported. Completed population fits and personal units are
checksum-verified on resume. Interrupted population fits restart that seed's
pair-head fitting using the verified feature cache. Original checkpoints and V1
are never overwritten. Use a new output directory when changing the protocol.

## Predict with a completed checkpoint

```sh
python paired_se/predict.py --checkpoint /path/to/model.pt \
  --current /path/to/current_ecg_ppg.npy \
  --reference /path/to/early_reference_ecg_ppg.npy \
  --reference-bp /path/to/observed_early_sbp_dbp.npy \
  --output /path/to/new_predictions.csv --device cpu
```

Only use a person's own earlier reference bank and corresponding personal model.
The reference bank is not stored in the model or published with its weights.
Only load trusted PyTorch checkpoints. Public-data labels here are ABP-derived;
they are not newly collected ND cuffs. Same-record evaluation does not establish
cross-visit performance or medical-device certification.

## Rebuild and verify results

```sh
python paired_se/report.py --run /path/to/completed_paired_se_run \
  --individual-root /path/to/bp_model/data/vital \
  --population-root /path/to/20261009/attention \
  --export paired_se/results --device mps
```

`models/paired_population_seed42.pt` (also 43/44) contains a fitted population
pair model. `models/vitaldb_pXXXXXX_seed42.pt` is specifically adapted to that
public-data person's early recordings. A new person still needs their own
reference bank and adaptation; do not apply another person's fitted model as a
universal personal model. Checksums are in `models/manifest.json`.

Source data and reuse terms: [PulseDB attribution / CC BY-NC-SA 4.0](../../../data/PulseDB/LICENSE.txt).
