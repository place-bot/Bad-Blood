# Bad-Blood V2: attention and personal BP calibration

## Current personal method - 20261015

**[Current V2 folder: tail adaptation, both signal modes](final_results_20261015/)**
contains the updated IEEE report/Beamer, runnable fitting and inference, six
source initializers, 360 independently selected endpoint checkpoints, and all
metrics for the same 30 people's chronological 80/20 comparison.

Only **tail adaptation** is adopted: pressure-change head + last residual CNN
block + SE. Learning rate and update count are selected independently for each
person and each pressure in three internal temporal validation periods.
PPG-only MAE is **6.89 / 4.29**; dual MAE is **7.14 / 4.31** mmHg. Compared
with matched bias correction, reductions are **19.0% / 20.3%** and
**21.5% / 24.4%**. Both endpoints beat both controls in the overall mean for
each of three seeds; not every individual improves. Head-only and automatic
scope selection are comparison policies, not adopted models. This previously
examined cohort is exploratory; earlier results remain below as dated history.

## Archived final results package - 20261013

**[Archived consolidated V2 models, IEEE report and full comparisons](final_results_20261013/)**
groups both pretrained CNN+SE input modes and their learned PairSE personalizers.
It includes 144 weight files, portable inference/adaptation, complete metrics,
three-seed and per-person results, source snapshots and checksums. All models
load successfully; the 136-test suite passes. Earlier dated results below are
preserved as the research history, not a replacement for this final index.

## Dated outcomes: V2 — 20261012

Start at the [20261012 version record](releases/20261012/README.md):

- [Outcome 1 — population CNN+SE](releases/20261012/01_population_cnn_se/):
  architecture, implementation, comparison metrics and three trained weights.
- [Outcome 2 — personalization](releases/20261012/02_personalization/): paired
  architecture, adaptation implementation, population/personal weights, old-six
  results and the new fixed-method validation on 16 previously unused people.

**New validation:** paired MAE 12.67 / 7.61 vs personal mean 17.41 / 9.71 and
bias correction 12.82 / 7.48 mmHg (SBP / DBP). Methods/parameters were unchanged.
The advantage over personal mean replicated; stable superiority over bias
correction did not. Full metrics and exclusions are in the dated record.

**Development follow-up (20261012):** [three alternative reference-set
architectures and nine personal-gradient settings](context_se/README.md) have
been evaluated on the four development participants. None passed the joint
comparison against personal mean and bias correction. Their implementations,
experimental weights and full results are preserved separately; they do not
replace the frozen `v2-20261012.1` outcomes or claim new-test superiority.

The subsequent [chronological source-training comparison](temporal_pretraining/README.md)
uses recovered timestamps (22 source-training people, two source-validation
people) and the four-person development cohort. Source-validation MAE improved,
but matched random training was competitive and personal post-training did not
beat both controls. This is a frozen-encoder conditioning-head experiment,
not full-PulseDB encoder retraining; the released model is unchanged.

This directory implements two controlled studies using the existing VitalDB-derived data. Track A compares attention mechanisms in the unchanged high-resolution V1 CNN architecture. Track B uses early personal ECG/PPG and scalar BP references to adapt that population model, then estimates **same-window SBP/DBP from later observed ECG/PPG**. The directory name `personalized_forecasting/` is retained from the original assignment; this is not open-loop waveform forecasting.

V1 is preserved in `../PulseDB-based Model-V1/`. The unchanged reference is `BPNet(pool=1)`, with 37,666 trainable parameters, two 125-Hz input channels, 1,250 samples per channel, and two scalar outputs. “Higher resolution” means retaining the original ten-second input resolution rather than applying the alternate model's initial fivefold pooling. New V2 baseline weights are trained from scratch on the corrected population split; the historical V1 checkpoint is not a clean personal-cohort initializer.

Dataset expansion, MIMIC training, and the teammate's larger CRC experiments are outside this study.

## Latest: reference-conditioned CNN+SE

The [paired-measurement extension](paired_se/README.md) is complete. It uses
known reference BP and reference/current ECG/PPG with the same support/query
data as the dense SE study. Selected pair-head adaptation achieved **8.32/3.76**
SBP/DBP target MAE versus **9.47/4.22** (personal mean) and **9.16/4.40** (bias).
All three seeds improved over both controls on these previously examined six
targets, although development did not meet the joint objective. This is an
exploratory result, not new independent validation. All metrics, uncertainty,
saved checkpoints and verification are linked in the extension.

## October 12 extension — best test MAE pretrained CNN plus SE

**CNN+SE (`se_last`) is the strongest pretrained model by primary-test MAE
among our six required three-seed V2 architecture comparisons** (SBP 12.386,
DBP 8.002 mmHg; 38,296 parameters). This describes the observed test ranking,
not a change to the original validation selection. The later matched
three-seed PPG-only CNN+SE comparison is now complete in the final package;
it is separate from the earlier one-seed plain-CNN PPG ablation.

At the user's request, the [new SE personalization experiment](se_personalization/README.md)
starts from these SE checkpoints and uses each person's **complete early 20%
support**, rather than only ten sparse references. It preserves the original
later query windows and compares against support-mean and additive-bias controls
using exactly the same support. This is a separately versioned exploratory
extension; the October 9 sparse-reference results below are not overwritten.
The dense fine-tune did not beat both controls on the target participants.
A regularized frozen-feature follow-up also failed its development gate and
was not taken to target testing. See the extension for full results and sources.

## Verified status and authoritative documents

At the documentation snapshot **2026-10-09 22:49 America/Chicago**, all **20 required population fits, 1,068 personal development units, and 684 final-target units are complete**. The strict report builder verified the complete inventory and generated the [combined report](reports/final_report.pdf) and [machine-readable results](reports/results/inventory.json). All 25 report pages pass visual inspection, and **98 automated tests pass**. The saved personal inference package exactly replays all eight strategies on 654 real query windows. Supplementary ECA fits were not run and are not required for completion. [PR #1](https://github.com/place-bot/Bad-Blood/pull/1) contains the dedicated research branch; V1 and main have not been changed by this work. See [final verification](reports/FINAL_VERIFICATION.md) for checks and PDF identities. The full local checkpoint/prediction bundle is not published; see [DATA_ACCESS.md](DATA_ACCESS.md) for reproduction requirements.

The review/protocol gate was committed at [1b93cf13249931af650cdfa91e19343564032451](https://github.com/place-bot/Bad-Blood/commit/1b93cf13249931af650cdfa91e19343564032451), before implementation or training. The repository research base is `7553248168295bcad593739fdc733dbefff0c3b1`; the V1 network's source-changing commit is `1f33cdcf090d51133c91dd50af2e575f5eaa0053`.

| Document | Purpose |
|---|---|
| [RESEARCH_PROMPT.md](RESEARCH_PROMPT.md) | User-confirmed task revision and completion requirements |
| [PROTOCOL.md](PROTOCOL.md) | Frozen methods, optimization, cohorts, selection, statistics |
| [experiment_matrix.json](experiment_matrix.json) | Machine-readable required and supplementary inventory |
| [AUDIT.md](AUDIT.md) | Architecture, local data, provenance, eligibility, limitations |
| [audit_snapshot.json](audit_snapshot.json) | Original proposal and frozen final support/query manifests |
| [DATA_DIAGNOSTICS.md](DATA_DIAGNOSTICS.md) | Observed upstream timestamp anomalies |
| [DATA_ACCESS.md](DATA_ACCESS.md) | Fresh-machine source preparation, ten personal downloads/checksums, and artifact availability |
| [REVIEW_GATE.md](REVIEW_GATE.md) | Both literature PDFs compiled and visually checked |
| [Track A review](attention_cnn/literature_review.pdf) | Attention evidence and prospective protocol |
| [Track B review](personalized_forecasting/literature_review.pdf) | Calibration evidence and prospective protocol |
| [Combined report PDF](reports/final_report.pdf) / [LaTeX source](reports/final_report.tex) | Actual completed-run results; all 25 pages visually inspected |
| [Final verification](reports/FINAL_VERIFICATION.md) | Inventory, numeric audit, 98 tests, visual checks and PDF SHA-256 identities |
| [Verified inventory](reports/results/inventory.json) / [Metric overview](reports/results/metric_overview.json) | Completed counts and browsable pooled/patient-macro metrics |
| [Attention results](reports/results/attention.json.gz) / [Personal results](reports/results/personalization.json.gz) | Lossless detailed metric archives, not raw waveforms or model weights |
| [STATUS.md](STATUS.md) | Dated workflow notes; runtime artifacts supply live run status |
| [Runtime integrity check](reports/RUNTIME_INTEGRITY_CHECK_20261009.md) | Eight actual development units, exact saved-model replay, and frozen-parameter checks |
| [Personal inference guide](personalized_forecasting/INFERENCE.md) | Apply completed personal weights or calibration coefficients without later labels; exact real-artifact replay evidence |

## Main findings from the completed runs

MAEs below are patient-macro values in mmHg, averaged across the three separately evaluated seeds 42/43/44, not an ensemble. SBP/DBP are kept separate. The paired 95% confidence intervals are descriptive patient-bootstrap intervals after averaging matched-seed differences; they are conditional on the fitted seeds and not adjusted for multiple comparisons.

- **Attention:** validation selects the unchanged CNN. On the 141-patient primary test, CNN SBP/DBP MAE is **12.59/8.36**, versus **12.39/8.00** for SE-last. SE-minus-CNN changes are **−0.20 [−0.48, +0.07]** for SBP and **−0.36 [−0.53, −0.18]** for DBP. Thus SE has lower observed errors, with the DBP interval below zero and the SBP direction still uncertain; these test comparisons do not override validation selection. SE-last has the lowest observed MAEs among the six required three-seed variants, not proof of universal superiority. [Detailed attention interpretation](reports/results/attention_interpretation.json).
- **Modality result:** at seed 42, PPG-only MAE is **12.296/7.771**, below matched-seed CNN **12.603/8.235** and SE-last **12.366/7.949**. This is a genuine observed advantage in the one completed run. ECG-only and PPG-only are retrained single-seed ablations, not part of the three-seed architecture ranking; they do not establish seed-robust sensor importance or show that ECG is universally unnecessary.
- **Personal calibration, K=10:** across six target patients, population, full-model adaptation, and reference-mean SBP/DBP MAEs are **12.73/7.80**, **9.24/4.66**, and **9.01/4.04**, respectively. Full adaptation lowers the observed population-model error, but reference mean is lower still: full-minus-reference-mean changes are **+0.23 [−1.11, +1.37]** for SBP and **+0.62 [+0.13, +1.07]** for DBP. This six-person, ABP-derived-reference pilot does not establish a universal calibration count, cross-visit durability, or clinical deployment readiness. [Detailed personal interpretation](reports/results/personal_interpretation.json).

## Data contract and provenance

These are VitalDB-derived **PulseDB** data, not native unprocessed VitalDB recordings. Provenance was checked using decoded source suffixes, the official PulseDB generator/field dictionary, and all 31 case-to-subject mappings in the 30 complete individual files against the public VitalDB clinical metadata.

Source-qualified canonical IDs are `vitaldb:pXXXXXX`. The `_1` in official-subset IDs means VitalDB source, not visit number. Distinct `CaseID` values identify recordings of a patient. The balanced population subsets do not retain timestamps or CaseIDs, so their row order cannot establish chronology.

References and terms:

- [PulseDB repository](https://github.com/pulselabteam/PulseDB), [dataset paper](https://doi.org/10.3389/fdgth.2022.1090854), and [field dictionary](https://github.com/pulselabteam/PulseDB/blob/main/Supplementary%20Materials.pdf).
- [VitalDB dataset and parameter documentation](https://vitaldb.net/dataset/) and [clinical mapping API](https://api.vitaldb.net/cases).
- VitalDB-derived PulseDB material carries CC BY-NC-SA 4.0 terms; retain the existing dataset attribution and license. Raw source access and storage remain separate from publishing this code.

Both tracks use filtered `ECG_F` and `PPG_F`, normalized independently within each observed window/channel to [0,1], with no additional raw-input filter. Labels are invasive-ABP-derived `SBP/DBP` or `SegSBP/SegDBP` in mmHg. ABP samples, reference-derived landmarks, and demographic/identity fields are not predictors. Population target mean/SD are fitted only on eligible population training rows.

**There are no actual cuff/NIBP events in the local files.** Track B is a sparse-reference simulation using distinct ABP-labelled windows. Sixty-second spacing does not make them independently acquired cuff readings or statistically independent physiological observations.

Upstream offline filtering, undocumented exact resampling support, and reference-dependent quality selection limit online/deployment interpretation. The study does not establish cross-visit durability, clinical-standard compliance, or Raspberry Pi latency.

### Population files

Point `--data-root` at the existing prepared `bp_model_v2/data` directory. It must contain:

```text
split.json
train/{waves.npy,y.npy,subjects.npy,valid.npy,complete.json}
test/{waves.npy,y.npy,subjects.npy,valid.npy,complete.json}
aami/{waves.npy,y.npy,subjects.npy,valid.npy,complete.json}
```

Waveforms must be float32 with shape N × 2 × 1250; labels must have shape N × 2 in SBP/DBP order. These are the prepared arrays from the V1 source pathway. If rebuilding source arrays is necessary, follow the preserved [PulseDB preparation documentation](../../data/PulseDB/README.md); do not retrain or overwrite V1 release weights.

All 31 historically explored IDs in V1's `reports/prior_exploratory_subjects.json` are excluded from every new population partition:

| Partition | Patients | Windows |
|---|---:|---:|
| Training | 1,085 | 390,600 |
| Validation | 192 | 69,120 |
| Primary calibration-free test | 141 | 56,400 |
| Separate AAMI-named stress subset | 116 | 666 |

The original source files contain 1,293/465,480 training-subset patients/windows, 144/57,600 calibration-free test patients/windows, and 116/666 stress-subset patients/windows. The corrected counts above are asserted at load time. All retained rows are used. AAMI-named results are reported separately and do not imply certification. Historical test exposure is disclosed; these are controlled comparisons, not newly collected external validation.

The loader computes full SHA-256 values for waveform arrays, labels, IDs, validity masks, split files, and completion metadata. Large waveform hashes have a process-local cache keyed by resolved path, byte size, and modification time in nanoseconds; small identity files are rehashed. Train/validation share memory-mapped source waveforms and differ by frozen row indices.

### Personal records and frozen windows

Point `--individual-root` at the existing directory containing individual PulseDB v2 MATLAB files. Thirty complete files contain 30 patients, 31 records, and 15,486 retained windows/43.0167 observed hours. The original manifest has one absent file and one truncated file; neither enters the final cohort.

| Role | Patients | All retained windows | Final query windows |
|---|---|---:|---:|
| Development | p005548, p005647, p005968, p006090 | 2,409 | 1,855 |
| Final targets | p005879, p005885, p005977, p005984, p006009, p006052 | 5,103 | 4,284 |

The final allocation, original proposal, and availability-only replacement p005983 → p005977 are preserved in `audit_snapshot.json`. No replacement was based on model performance.

Within each single case, the calibration boundary is the first retained timestamp plus 20% of the elapsed span through the final window's exclusive endpoint. It is not 20% of shuffled rows. A greedy early-window pool enforces start-to-start spacing of at least 60 seconds. Ten entries are selected using rounded evenly spaced pool positions; budgets K=1,2,3,5,10 use nested chronological prefixes. All query windows start at least ten seconds after the boundary, and the query set is fixed across budgets and methods.

Each patient manifest stores source-row indices and exact intervals. The loader verifies these against original `T`, `SubjectID`, and `CaseID`, checks full raw-file hashes, and aborts on mismatches or invalid inputs. Gaps are retained. Two late query-to-query pairs have an upstream one-sample/8-ms timestamp overlap; they are preserved, diagnosed in metadata, and do not overlap support windows.

Primary drift measures time from the **actual final reference endpoint to query endpoint**, with bins [0,15), [15,30), [30,60), and [60,infinity) minutes. Boundary-relative time is secondary. Reference age differs by budget, and empty bins remain unavailable rather than zero error.

## Required experiment inventory

Track A has **20 required population fits**:

| Variant | Parameters | Seeds | Purpose |
|---|---:|---|---|
| `baseline` | 37,666 | 42,43,44 | Exact full-resolution V1 architecture |
| `se_last` | 38,296 | 42,43,44 | Final-block feature-channel gate |
| `temporal_last` | 37,680 | 42,43,44 | Final-block temporal gate |
| `cbam_last` | 38,310 | 42,43,44 | Channel then temporal gates |
| `additive_last` | 38,296 | 42,43,44 | SE-matched parameter-count control |
| `se_all` | 38,670 | 42,43,44 | Placement ablation over all residual blocks |
| `baseline_ecg` | 37,666 | 42 | ECG-only, other input channel zeroed |
| `baseline_ppg` | 37,666 | 42 | PPG-only, other input channel zeroed |
| `eca_last` | 37,669 | 42,43,44 | Supplementary; three additional fits |

Attention acts on learned features, not directly on sensor modality identity. The principal comparisons require three seeds; the two modality ablations are explicitly single-seed. ECA does not replace any required run.

Every population fit uses the frozen optimizer/augmentation settings, at most 40 epochs, and eight-epoch early stopping. Selection is the mean of patient-macro SBP/DBP validation MAE. Final predictions always load `best.pt`. There are no CLI controls to silently reduce data, training duration, or tuning budgets.

Track B requires **1,068 development units and 684 final-target units**. A unit is a patient × seed × budget × strategy × adaptation-setting evaluation; repeated units do not add independent patients.

- Four reference/control strategies: population, additive bias, last reference, calibration mean.
- Three neural strategies: head-only, last block plus head, full model.
- Affine calibration only at K=3,5,10.
- Development searches two learning rates × two update counts for each neural strategy and three affine penalties. Settings are chosen across all four development patients and seeds 42/43/44, then saved before final-target evaluation.
- Every final adaptation starts afresh from the corresponding newly trained unchanged baseline. BatchNorm running statistics and dropout remain frozen during adaptation. Final query labels never guide adaptation or setting selection.

Development arithmetic is 12 patient-seed pairs × [5 budgets × (4 controls + 3 neural methods × 4 settings) + 3 affine budgets × 3 penalties] = 1,068. Final arithmetic is 18 patient-seed pairs × [5 × (4 + 3) + 3] = 684.

## Environment and setup

The recorded local environment is Python 3.12.13, Apple M5, 24 GiB memory, ten logical CPUs, macOS ARM64, and PyTorch 2.14.0. Version pins are in [requirements.txt](requirements.txt). MPS is available in an unrestricted local process; the restricted sandbox can report it unavailable. CPU and CUDA are also accepted. Accelerator execution is best-effort deterministic; exact epoch-resume equality is tested on CPU.

For a portable reproduction, change the following paths to your checkout and existing datasets. Keep the runtime directory **outside the Git repository** and use a new directory for a scientifically different run.

```sh
export BB_REPO="/path/to/Bad-Blood"
export BB_POP="/path/to/existing/bp_model_v2/data"
export BB_PERSONAL="/path/to/existing/bp_model/data/vital"
export BB_RUN_ROOT="/path/to/bad_blood_v2_runs/reproduction"
export BB_DEVICE="mps"
export BB_PERSONAL_DEVICE="cpu"

mkdir -p "$BB_RUN_ROOT"
python3.12 -m venv "$BB_RUN_ROOT/venv"
export BB_PY="$BB_RUN_ROOT/venv/bin/python"
cd "$BB_REPO/model training/v2"
"$BB_PY" -m pip install -r requirements.txt
export MPLCONFIGDIR="$BB_RUN_ROOT/mpl-cache"
mkdir -p "$MPLCONFIGDIR"
"$BB_PY" -m unittest discover -s tests -v
```

Choose `BB_DEVICE=cpu` when no accelerator is available, or `cuda` for an appropriately configured CUDA host. Keep the selected device and source/configuration unchanged within a resumable run. Unit tests use small fixtures to test failure modes; their outputs are not research results.

The current local paths are:

```text
Repository: /Users/jguo9/Desktop/26fall/LLM/output/Bad-Blood
Population: /Users/jguo9/Desktop/26fall/LLM/output/bp_model_v2/data
Individual: /Users/jguo9/Desktop/26fall/LLM/output/bp_model/data/vital
Python:     /Users/jguo9/Desktop/26fall/LLM/output/bp_model/.venv/bin/python
Artifacts:  /Users/jguo9/Desktop/26fall/LLM/output/bad_blood_v2_runs/20261009
```

## Execute or resume the full suite

From `model training/v2`, one command runs the required population fits sequentially, then personal development, then final personal evaluation:

```sh
"$BB_PY" run_experiments.py \
  --stage all \
  --data-root "$BB_POP" \
  --individual-root "$BB_PERSONAL" \
  --artifact-root "$BB_RUN_ROOT" \
  --device "$BB_DEVICE" \
  --personal-device "$BB_PERSONAL_DEVICE"
```

For an existing suite, repeat the command with `--resume`. Add `--include-eca` only to request the three supplementary ECA fits. The orchestrator runs one population fit at a time; avoid concurrent writers to the same artifact root.

To execute stages separately, use the same command with `--stage population` first and `--stage personalization` afterward. Supply both data-root arguments in either mode because the parser requires them. The personal stage requires completed real baseline runs at all three seeds and executes development before test.

For a single population run:

```sh
"$BB_PY" attention_cnn/train.py \
  --variant baseline --seed 42 \
  --data-root "$BB_POP" \
  --output-dir "$BB_RUN_ROOT/attention/baseline_seed42" \
  --device "$BB_DEVICE"
```

Append `--resume` only when that individual run directory already exists. Interrupted population runs restore the last complete epoch, optimizer, scheduler, RNG states, and accumulated time. An interrupted partial epoch is replayed. Changed data, split, source, protocol, or configuration hashes cause a resume error rather than silent continuation. A completed run can be verified/skipped with `--resume`.

Track B can also be run directly after all three baseline checkpoints exist:

```sh
"$BB_PY" personalized_forecasting/run.py \
  --phase development \
  --population-root "$BB_RUN_ROOT/attention" \
  --individual-root "$BB_PERSONAL" \
  --output-dir "$BB_RUN_ROOT/personalization" \
  --device "$BB_PERSONAL_DEVICE" --threads 4

"$BB_PY" personalized_forecasting/run.py \
  --phase test \
  --population-root "$BB_RUN_ROOT/attention" \
  --individual-root "$BB_PERSONAL" \
  --output-dir "$BB_RUN_ROOT/personalization" \
  --device "$BB_PERSONAL_DEVICE" --threads 4
```

Add `--resume` when continuing either phase. Completed units are checked against their configuration and artifact hashes; an incomplete personal unit restarts from its original population baseline, not partially adapted weights. The final phase requires the existing development provenance and frozen `selection.json`. A mismatch is a failed reproducibility check, not a reason to edit hashes manually.

## Runtime outputs and checking completion

Raw data, large checkpoints, full predictions, caches, and logs remain outside Git. The artifact root contains:

```text
suite_population.json / suite_personalization.json / suite_all.json
logs/
attention/
  baseline_seed42/
    config.json, manifest.json, environment.json
    history.json, metrics.json, costs.json
    best.pt, last.pt
    predictions_validation.npz, predictions_test.npz, predictions_aami.npz
    completion.json
  ...other variant/seed directories...
personalization/
  run_manifest.json, selection.json
  development_index.json, test_index.json
  development/<patient>/seed<seed>/K<budget>/<method>/<setting>/
  test/<patient>/seed<seed>/K<budget>/<method>/<setting>/
    predictions.npz, unit.json
    adapted.pt OR calibration.json
```

Suite status is invocation-level: `completed_this_invocation` is not the cumulative research inventory. Individual `completion.json` and personal `unit.json` files are written after their outputs. A log line or launched process alone does not mean completion. Known runner caveat: preflight errors such as an existing run without `--resume`, a missing required baseline, or a log-open failure can leave the suite JSON at `running` after the process exits; check the process/log and verified individual artifacts rather than interpreting that field as a heartbeat.

Population predictions retain references, estimates, source-qualified patient IDs, source-row indices, selected epoch and manifest identity. They do not fabricate unavailable case IDs/timestamps. Personal predictions retain support/query indices, timestamps, case IDs, labels, estimates, reference ages, and checkpoint/configuration identities. Neural units also record trainable masks and measured parameter changes.

The personal phase indexes must report exactly 1,068 development and 684 test units. These checks, all 20 required population completion markers, corresponding checkpoints/prediction hashes, and the report builder's validation establish computational completion. A final report and inspected PDF are additional deliverables.

## Offline prediction from a completed population checkpoint

The reusable CLI accepts a NumPy `.npy` array shaped N × 2 × 1250: already-filtered ECG followed by PPG, at 125 Hz. It requires no labels, demographics, IDs, or raw ABP. It applies per-window/channel min-max normalization by default and performs no additional filtering or resampling. If the array already uses the normalized source representation, add `--already-normalized`; the values must then lie in [0,1]. Nonfinite values, constant channels, incompatible sampling/channel declarations, and wrong shapes are rejected. NPY has no inherent channel/filter metadata, so these declarations remain the caller's responsibility.

```sh
cd "$BB_REPO/model training/v2"
"$BB_PY" shared/predict.py \
  --checkpoint "$BB_RUN_ROOT/attention/baseline_seed42/best.pt" \
  --input "$BB_RUN_ROOT/new_filtered_windows.npy" \
  --output "$BB_RUN_ROOT/new_predictions.npy" \
  --fs-hz 125 --channel-order ECG PPG \
  --trust-checkpoint
```

Output is a new float32 N × 2 `.npy` array in SBP/DBP order, in mmHg. Existing output files are not overwritten. Inference runs deterministically on CPU in evaluation mode without augmentation or adaptation. Other completed V2 population variants use the same CLI; the variant is read from the checkpoint.

`--trust-checkpoint` is required because full training checkpoints contain optimizer/RNG Python objects and require `torch.load(weights_only=False)`. Only use a locally controlled checkpoint you trust; deserializing an untrusted pickle can execute code. The loader subsequently verifies pool=1, configuration consistency, the frozen model/data/protocol source contract, and finite target center/scale. An adjacent `manifest.json`, when present, must also match. A standalone copied `best.pt` is supported with the matching V2 source checkout; historical V1 or per-patient `adapted.pt` formats are not accepted by this CLI.

## Inference from a completed personal calibration

Completed Track B units also have a label-free inference entry point in
`personalized_forecasting/predict.py`. It supports all eight saved strategies,
checks the exact patient and population-checkpoint identity, and never fits or
selects a method. See the [personal inference guide](personalized_forecasting/INFERENCE.md)
for the minimal portable package and exact commands. Real replay of all eight
K10/seed42 units for the first frozen target reproduced all 654 stored query
predictions bitwise without modifying the existing artifacts.

## Build the result-derived report and PDFs

Only after the full required inventory is complete:

```sh
cd "$BB_REPO/model training/v2"
"$BB_PY" reports/build_report.py --artifact-root "$BB_RUN_ROOT" \
  --data-root "$BB_POP" --individual-root "$BB_PERSONAL"
```

The builder rejects missing required runs, toy runs, changed prediction/checkpoint artifacts, incomplete personal indexes, and inconsistent frozen selections. It generates `reports/final_report.tex`, a combined bibliography, figures, and machine-readable results under `reports/results/`. It does not train models or compile LaTeX.

`metric_overview.json` provides browsable seed-mean pooled and patient-macro metrics, with the number of defined seeds for each endpoint. Complete per-patient, pressure-range, drift and provenance summaries remain losslessly available in `attention.json.gz` and `personalization.json.gz`. Deterministic gzip keeps these detailed results compact without dropping metrics or rounding values. Small interpretation and inventory JSON files stay uncompressed. To read a detailed archive with Python:

```python
import gzip
import json

with gzip.open("reports/results/personalization.json.gz", "rt", encoding="utf-8") as stream:
    results = json.load(stream)
```

With a LaTeX installation providing `pdflatex` and `bibtex`, compile each document from its own directory:

```sh
cd "$BB_REPO/model training/v2/attention_cnn"
pdflatex -interaction=nonstopmode -halt-on-error literature_review.tex
bibtex literature_review
pdflatex -interaction=nonstopmode -halt-on-error literature_review.tex
pdflatex -interaction=nonstopmode -halt-on-error literature_review.tex

cd "$BB_REPO/model training/v2/personalized_forecasting"
pdflatex -interaction=nonstopmode -halt-on-error literature_review.tex
bibtex literature_review
pdflatex -interaction=nonstopmode -halt-on-error literature_review.tex
pdflatex -interaction=nonstopmode -halt-on-error literature_review.tex

cd "$BB_REPO/model training/v2/reports"
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
bibtex final_report
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
```

Inspect every page of the resulting PDFs; successful compilation alone does not verify table widths, figure legibility, or citation layout. The unmodified official ACL `acl.sty`/`acl_natbib.bst` are vendored from [acl-style-files](https://github.com/acl-org/acl-style-files), commit `d5adc823ff0f80f98c80405ca0ab66c68e684409`. Documents use the nonanonymous page-numbered preprint format; this does not imply conference submission or acceptance.

## Metrics, interpretation, and publication

Report SBP and DBP separately in mmHg: MAE, RMSE, signed bias, sample error SD, median and tail absolute errors, fractions within 5/10/15 mmHg, correlation, and R-squared when defined. Window-pooled and equally weighted patient-macro values answer different questions. Paired uncertainty bootstraps patients after averaging matched seed differences, rather than treating correlated windows as independent. Pressure strata, all six personal targets, negative results, and empty drift bins remain visible.

The current pilot cannot establish a universal cuff-calibration count or persistence across visits. Reference-only controls test whether personal neural adaptation adds information beyond the early pressures themselves. Desktop timing is not Raspberry Pi timing.

Commit code, versioned configuration/manifests, documentation, LaTeX sources, PDFs, and appropriately sized result summaries. Keep the raw recordings and detailed runtime tree external. Preserve V1 and unrelated team work; use the dedicated `research/v2-vitaldb-attention-personalization` branch without force-pushing or merging. Computational completion, PDF inspection, and successful GitHub delivery must each be verified rather than inferred from the presence of this README.
