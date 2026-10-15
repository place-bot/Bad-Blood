# Reproduce the adopted tail policy

The population CNN+SE and PairSE source checkpoints are unchanged. The dated
experiment used the same 30 people, chronological splits and reference budget
for all methods. Its model/hyperparameter rules are recorded in
`metrics/lock.json`. Do not change those rules in a reproduction comparison.

## 1. Recover public data and indices

Follow the repository's [data access guide](../DATA_ACCESS.md) and the
[official PulseDB repository](https://github.com/pulselabteam/PulseDB) to obtain
the individual VitalDB-derived MATLAB files listed in `metrics/lock.json`.
The manifest's original absolute paths record provenance; substitute your own
data location, keeping the recorded SHA256, participant ID and indices.
The archive does not include raw waveforms or any Notre Dame student data.

The frozen loader in `../personal_hpo_20261015/run.py::read_all` reads
`Subj_Wins/ECG_F`, `PPG_F`, `SegSBP`, `SegDBP`, `T`, `SubjectID`, `CaseID`.
It validates 1,250 samples at 125 Hz, a single ordered recording and finite
signals/labels; it independently min-max scales the two signal channels.
The saved `support` and `query` indices must match the guarded 80/20 split.
For each person, create a development-only NPZ with `x`, `y`, `start`, `end`.
Keep test labels in a different evaluation input, never the fit input.

## 2. Fit the adopted personal method

Use `implementation/personalize.py fit` as shown in
[the personal-method guide](02_personalization/README.md). For the full
experiment repeat it for 30 people, two input modes and source seeds 42/43/44.
The portable entrypoint searches only the adopted tail candidate group, with
the same zero-update candidate, fold boundaries, losses and tie-breaking as
the frozen study. It does not search head-only adaptation as a deployed option.

The exact frozen full-comparison code remains in
`../rolling_personal_20261015/`, with its dependency `../personal_hpo_20261015/`.
Those original scripts retain their recorded experiment paths to preserve code
hashes. The portable entrypoint avoids that machine-specific directory layout.
Reference model definitions and `implementation/rolling_core.py` are byte-for-byte
copies of the corresponding frozen source files; `build_release.py` packages
existing evidence and does not train or revise predictions.

## 3. Evaluate after selecting/refitting

Call `predict` with the two endpoint checkpoints, this person's saved
development reference bank and their query signal windows. Concatenate SBP
from endpoint 0 and DBP from endpoint 1. Compare with separately held query
labels. Compute per-person metrics first, then average the 30 people equally;
also report the segment-pooled panel. Average metrics across source seeds,
not predictions. `metrics/summary.json` contains both panels and every seed.

The same development bank supplies the personal-mean baseline. Bias correction
uses this modality/seed's unadapted population model and the mean development
residual. All compared methods must use exactly the same query windows.

## 4. Verify saved evidence without retraining

`metrics/unit_evidence.tar.gz` contains all per-unit selections, fold predictions,
test predictions and evaluation records. `personal_weights/` archives contain
only the adopted tail endpoint weights. The original independent verifier
checked 180 units, 1,080 folds and 720 endpoint models; its passed record and
the completion hashes are preserved. The release verification loads all 360
published personal checkpoints and six source initializers, then replays 96
real held-out predictions through the portable endpoint interface. Maximum
replay difference is 0.0000153 mmHg, with exact PPG invariance to the ECG slot.

The four frozen adaptation tests and two input-mode fit/inference smoke tests
pass. Smoke-test data are synthetic and are used only for software testing;
they are not included in any reported performance metric.

The original test people were previously evaluated. Reproducing these scores
is a consistency check, not a new independent evaluation cohort.
