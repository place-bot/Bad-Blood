# Workflow status

## 2026-10-09 16:12 America/Chicago: implementation and real execution

- The review gate was committed as `1b93cf1`, followed by the population
  implementation (`13e42ee`) and personal-calibration implementation (`57c6059`).
- Sixty-two automated tests pass, including exact V1 architectural equality,
  CPU checkpoint/resume equivalence, support/query and split isolation,
  parameter-freezing checks, metric arithmetic, full-inventory corruption
  checks, portable artifact verification, report aggregation, and prediction-CLI parity.
- The required 20-fit population suite is running on local MPS. Baseline seed
  42 completed nine epochs and selected epoch 1; seed 43 completed seventeen
  epochs and selected epoch 9; seed 44 completed sixteen epochs and selected
  epoch 8 by the frozen validation rule. SE-last seed 42 is running.
  These are three completed fits, not a completed experiment suite.
- Personal development has started on CPU using the three completed baseline
  seeds. Final-target evaluation will follow the frozen development selection.
  CPU personalization overlaps later MPS population fits; timing results must
  acknowledge shared-host load rather than claim isolated hardware benchmarks.
- Run artifacts and resumable state are outside Git under
  `/Users/jguo9/Desktop/26fall/LLM/output/bad_blood_v2_runs/20261009`.
- The result builder requires the full hash-verified inventory before writing
  the combined report. No synthetic or partial-result final report is provided.

## 2026-10-09: review gate (earlier snapshot)

- Dedicated branch: `research/v2-vitaldb-attention-personalization`.
- V1 release retained. Latest team changes fetched; no unrelated files edited.
- Audit and both literature reviews are complete. The main agent inspected
  all seven attention-review pages and all six personalization-review pages
  after compilation; both have resolved citations and readable layouts.
  The protocol is frozen before implementation. No new model-training
  experiment has started at this gate.
- The project owner confirmed the revised task: early ECG/PPG and BP personal
  calibration followed by same-window BP estimation from later observed
  ECG/PPG. RESEARCH_PROMPT.md supersedes the original waveform-forecasting
  scope; PROTOCOL.md specifies the experiments.
- Local MPS access works outside the restricted process sandbox: Apple M5,
  24 GiB RAM, 10 logical CPUs, PyTorch 2.14.0. No CRC allocation is used here.
- Temporal analysis must use individual-record files with timestamps and
  CaseID. Balanced subset row order cannot establish chronology.
