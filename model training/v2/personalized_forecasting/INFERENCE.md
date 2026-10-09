# Inference from a saved personal calibration

`predict.py` applies an **existing completed Track B unit** to new, already-observed ECG/PPG windows from the same person. It never fits a calibration, adapts weights, chooses a strategy, or reads later BP labels. The result estimates **same-window SBP and DBP**, not future BP before the ECG/PPG is observed. The historical `personalized_forecasting` directory name is not a forecasting claim.

This is research-only packaging of the completed models. It does not establish clinical accuracy, cuff-label validity, cross-visit transfer, streaming causality, or deployment safety. The experiments used sparse **ABP-derived scalar references**, not actual cuff events. No published checkpoint/download bundle is assumed; use your completed local reproduction artifacts.

## Required local files

- One `adapted.pt` (for `head`, `last_block`, or `full`) **or** `calibration.json` (for `population`, `bias`, `affine`, `last_reference`, or `calibration_mean`).
- Its original adjacent `unit.json`, written on unit completion.
- The original personalization `run_manifest.json`. It is discovered in the artifact's ancestor directories, or supplied with `--run-manifest` after copying a minimal package.
- The exact population `best.pt` used by the unit, normally `attention/baseline_seed{seed}/best.pt`.
- This repository's matching frozen V2 source and the Python dependencies described in the [V2 reproduction README](../README.md).

`predictions.npz`, the source waveform corpus, support labels, later labels, and target demographic features are **not needed**. The saved coefficients/weights already encode the calibration. `--patient-id` is only a guard against accidentally using another person's artifact; it is not a predictor input.

The loader checks the unit configuration and artifact hashes, exact population-checkpoint SHA256, original run-manifest hash, population normalization, source hashes, and calibration/strategy consistency. Adapted neural states must preserve all parameters outside their adaptation scope and all frozen buffers. An adjacent population data `manifest.json`, if present, is also checked by the shared population loader. Copied bundles need not retain the original absolute data paths: inference does not reopen those paths.

These checks establish package consistency, not authenticity. Use files whose origin you personally trust. The population checkpoint contains optimizer/RNG objects and requires `torch.load(weights_only=False)`, which can execute malicious pickle code. `--trust-checkpoint` explicitly permits this load; an SHA256 check does not make an untrusted checkpoint safe. The smaller `adapted.pt` is loaded with `weights_only=True`. This inference check is not a substitute for the full scientific inventory verifier.

## Input and output contracts

Input is a nonempty, real numeric `.npy` array of shape `N × 2 × 1250`: **ECG first, PPG second**, sampled at **125 Hz**, ten seconds per window. Signals must already be filtered through the same source-compatible preprocessing pathway used for training. This CLI does not implement new filtering, resampling, segmentation, signal alignment, or artifact rejection. NPY does not store signal metadata: channel order, sampling rate, filtering compatibility, identity, and acquisition chronology remain caller responsibilities. Declaring different channel order or sampling rate is rejected, not converted.

By default each observed window/channel is independently min-max normalized, matching the frozen personal loader. If it is already normalized, pass `--already-normalized`; every sample must lie in `[0, 1]`. Nonfinite samples, empty/mis-shaped inputs, unrepresentable amplitudes, and constant channels are rejected. Do not substitute ABP waveforms, BP labels, handcrafted ABP-derived features, or future signals for these two channels.

Output is a new `.npy` array of shape `N × 2`, columns **SBP, DBP**, in **mmHg**. It is `float64` to preserve the original Track B de-normalization/calibration arithmetic; this differs intentionally from the standalone population CLI's float32 output. No physiological clipping is applied. The command prints a compact JSON record containing the patient, strategy, support budget, seed, and provenance hashes. Inference uses CPU evaluation mode without dropout, BN updates, or gradients; default batch size is 256, matching Track B. Exact replay is environment/batch dependent; bitwise identity is not promised across different PyTorch versions or hardware.

The two reference-only controls (`last_reference`, `calibration_mean`) validate input shape/metadata but deliberately ignore waveform values and return the saved constant estimate. The other methods use the ECG/PPG. `affine` exists only at K ≥ 3; no missing method/budget combination is fabricated. The command applies exactly the supplied unit, with no automatic "best method" selection or ensembling.

## Exact invocation

Set paths for your checkout, completed run, prepared input, and a separate output directory. The example uses a fixed, existing `head` unit for the first frozen target/seed/K10; it is an interface example, **not a recommendation selected from target performance**.

```bash
export BB_REPO="/path/to/Bad-Blood"
export BB_RUNS="/path/to/bad_blood_v2_runs/20261009"
export BB_INPUT="/path/to/observed_ecg_ppg.npy"
export BB_INFERENCE="/path/to/new-inference-output"
mkdir -p "$BB_INFERENCE"

python "$BB_REPO/model training/v2/personalized_forecasting/predict.py" \
  --artifact "$BB_RUNS/personalization/test/vitaldb:p005879/seed42/K10/head/lr0.001_steps100/adapted.pt" \
  --population-checkpoint "$BB_RUNS/attention/baseline_seed42/best.pt" \
  --patient-id "vitaldb:p005879" \
  --input "$BB_INPUT" \
  --output "$BB_INFERENCE/head_sbp_dbp.npy" \
  --trust-checkpoint
```

Add `--already-normalized` only when the input already satisfies the normalization contract. The explicit metadata options are `--fs-hz 125 --channel-order ECG PPG`; they validate the declared contract. `--batch-size 256` is the replay default.

For an existing scalar affine calibration, substitute the artifact and choose a new output path:

```bash
python "$BB_REPO/model training/v2/personalized_forecasting/predict.py" \
  --artifact "$BB_RUNS/personalization/test/vitaldb:p005879/seed42/K10/affine/lambda1/calibration.json" \
  --population-checkpoint "$BB_RUNS/attention/baseline_seed42/best.pt" \
  --patient-id "vitaldb:p005879" \
  --input "$BB_INPUT" \
  --output "$BB_INFERENCE/affine_sbp_dbp.npy" \
  --trust-checkpoint
```

For a detached portable bundle, keep `unit.json` next to its artifact and add `--run-manifest "/path/to/original/run_manifest.json"`. Preserve file bytes; do not rewrite provenance to make mismatched files load. A changed population checkpoint, patient identity, normalization, or frozen source is an error. To inspect the saved strategy/settings/identity before choosing an artifact, read its `unit.json`; do not use later labels to retune it.

Existing outputs are never overwritten. The output must also be outside the saved unit directory, population-checkpoint directory, and run-manifest directory tree. No experiment artifact is edited. Keep raw arrays, personal outputs, and checkpoints outside Git, as described in the [data access guide](../DATA_ACCESS.md).

## Python API and tests

With the V2 directory on `PYTHONPATH`:

```python
import numpy as np
from personalized_forecasting.predict import load_personal_predictor, predict_personal

predictor = load_personal_predictor(
    "/path/to/unit/adapted.pt",
    "/path/to/baseline_seed42/best.pt",
    expected_patient_id="vitaldb:p005879",
    trusted_checkpoint=True,
    run_manifest_path="/path/to/personalization/run_manifest.json",
)
observed = np.load("/path/to/observed_ecg_ppg.npy", allow_pickle=False)
sbp_dbp_mmhg = predict_personal(predictor, observed)
```

There is no label argument. The API does not write files. Use the CLI's fixed CPU/thread settings and batch size for the recorded replay conditions.

```bash
python -m unittest discover \
  -s "$BB_REPO/model training/v2/tests" \
  -p test_personal_prediction.py -v
```

The unit tests use synthetic fixtures and deterministic parameter edits, not fitted research runs. They cover all eight strategies, prediction arithmetic, input/QC checks, explicit trust before pickle loading, checkpoint/artifact/config/source identity, copied bundles, patient mismatch, frozen-state violations, reference-only controls, and no-overwrite/no-experiment-write behavior.

## Recorded real-artifact replay check

On 2026-10-09, the new API replayed all **654 held-out query windows** (zero-based source rows 152–805) for the fixed first target `vitaldb:p005879`, case `c5552`, seed 42, K10. Inputs came from the frozen personal loader and were passed as ECG/PPG only, with `already_normalized=True`. All eight completed unit outputs were **bitwise equal** (`numpy.array_equal`) to their existing `predictions.npz`; repeated inference was also identical. The `head` unit additionally passed an end-to-end CLI NPY replay in a temporary directory. All existing unit-file hashes were unchanged afterward. No fitting, score-based method choice, or target-performance claim was part of this check.

The checked artifact suffixes below are relative to `personalization/test/vitaldb:p005879/seed42/K10/` in the local run above:

| Strategy | Saved setting | Artifact SHA256 |
| --- | --- | --- |
| population | `fixed/calibration.json` | `9814438923d0952bf567ed1932b3b3940343e924e0000f7092aa6e892f64bf12` |
| bias | `fixed/calibration.json` | `f493df2ca901c20d4266fe99a434c55242ef69ed5fbc7b54d0bc71fcfe5bd93c` |
| affine | `lambda1/calibration.json` | `9ae27a34044af4888ddf1f69db358b39be9e2ea8d57f7f1aa9723f3128cddc3f` |
| last_reference | `fixed/calibration.json` | `a00b5ea5ec596602f99eee2aed17741adbdf7328d4311dbccac37af6548f3a02` |
| calibration_mean | `fixed/calibration.json` | `26b6de0c05ea1e814fe07c0efa1c582720e3b296125f2fdba0aeab013657d7f7` |
| head | `lr0.001_steps100/adapted.pt` | `019fde1dd05b6429bfdee5c7338006a7fa06469f4badf733747821e2fcefaeb6` |
| last_block | `lr0.0001_steps100/adapted.pt` | `beba0a5bb7b5fba18ea9805c5039b09ca88052564fd0f9094e3a3840cfc21470` |
| full | `lr0.0001_steps100/adapted.pt` | `4281a861d92cf04e2624eb2cdccff2c74aa851af9d7b68d54d8ee0e03cdc73d3` |

Population checkpoint SHA256: `972125f69da1b01d827e2d8162228311987bb781da5640b72309d2b36cf15f35`. Personal source-manifest hash: `687a571b42e6431ce5f33da35882d1f863c38930b16154b3fe1a1f1c70009f73`. Personal run-provenance hash: `baaa1d3ba470b8167062b037c2edcfd977ccddf68c60bdf779b55281ca7eb031`. The original CPU/4-thread/batch-256 execution contract was used. These identifiers document one integrity check, not a new evaluation or independently released artifact bundle.
