# Workflow status

## 2026-10-09 22:49 America/Chicago: complete experiment and report verification

- All 20 required population fits are complete, including the ECG-only and
  PPG-only controls. The original orchestrator exited successfully. No test
  outcome changed a training setting or the frozen inventory.
- All 1,068 personal development units and 684 final-target units are complete.
  The strict builder verified source/configuration/checkpoint/prediction hashes
  and generated the complete actual-result report and metric archives.
- Independent numeric checks reproduced all 20 population metric sets, all
  684 personal units, 114 personal summary groups, the paired comparisons,
  and all six patients' K10 outcomes. The full suite passes 98 tests.
- SE-last has the lowest observed test SBP/DBP MAE among the six required
  three-seed dual-channel models, while CNN has the lowest mean validation
  score. PPG-only seed 42 is lower still than matched-seed CNN and SE-last;
  this single-run evidence is explicitly separated from the three-seed results.
- Personal full adaptation at K10 improves on the population model but does
  not outperform reference mean on the observed average errors. The report
  includes the paired uncertainty, per-patient failures and pressure strata.
- The two literature reviews (7 and 6 pages) were compiled and inspected
  before implementation. The completed combined ACL report has 25 pages;
  every page has been visually checked. No missing references or overfull
  boxes remain. The appendix uses full-width tables.
- All changes remain under `model training/v2/`; V1 and unrelated files are
  unchanged relative to the research base. Final artifacts are prepared for
  the dedicated branch and PR, without merging. PDF hashes and verification
  details are in [FINAL_VERIFICATION.md](reports/FINAL_VERIFICATION.md).
- No required experiment is outstanding. Supplementary ECA was not run;
  actual cuff calibration, cross-visit durability, and Raspberry Pi benchmarks
  are unsupported questions, not completed claims. Full runtime weights and
  predictions remain local and outside ordinary Git history.

## 2026-10-09 22:17 America/Chicago: all required three-seed comparisons complete

- Eighteen of twenty population fits are complete: all six required dual-channel
  architectures have finished seeds 42/43/44. The original orchestrator is
  running the ECG-only control, with the PPG-only control next.
- SE-last retains the lowest observed primary-test SBP and DBP MAEs among
  these six fixed three-seed models (12.386/8.002 mmHg). The CNN retains the
  lowest mean validation score. Report generation now explicitly includes
  both findings, paired uncertainty, and the difference between a planned
  selection rule and observed test performance. No test outcome changed a
  training setting or the remaining inventory.
- Ninety-six automated tests pass, including target-specific test findings,
  differing validation and test rankings, and avoiding unsupported comparisons
  of an observed test minimum with every competitor.
- The full required inventory, final report compilation, visual inspection,
  and final GitHub handoff are still pending. V1 and unrelated files remain
  unchanged relative to the research base.

## 2026-10-09 21:52 America/Chicago: sixteen fits and explicit split reporting

- Sixteen of twenty required population fits are complete. All three seeds
  for the four main architectures and the additive capacity control are done;
  SE-all seed 42 is complete and seed 43 is running. Both modality ablations
  and SE-all seed 44 remain. The original orchestrator continues without a
  duplicate job or any change to the frozen training configuration.
- Ninety-three automated tests pass. Report generation now includes a
  separate validation table with SBP/DBP means, the three per-seed selection
  scores, and their architecture-level average. Primary-test captions state
  the split and aggregation explicitly. Three-seed results average separately
  evaluated runs; they are not an ensemble of predictions.
- In response to an interim user question, the sixteen saved validation/test
  prediction hashes and pooled metrics were checked. Among the completed
  three-seed models, SE-last has test MAE 12.39/8.00 mmHg versus 12.59/8.36
  for the retrained CNN. The CNN's mean validation selection score is 10.371
  versus SE-last's 10.390. A validation-selected model and an observed
  test-metric leader are different statements; neither set of results changes
  the remaining experiment configurations.
- SE-last reduces primary-test MAE at all three matched seeds. Descriptive
  paired patient-bootstrap intervals for SE-minus-CNN are [-0.480, 0.068]
  mmHg for SBP and [-0.534, -0.184] for DBP. These are interim comparisons
  of fixed fits, not a new model-selection rule or a completed final report.
- The final report remains gated on all twenty required fits, strict artifact
  verification, compilation, and all-page visual review. V1 remains unchanged.

## 2026-10-09 20:16 America/Chicago: connection restored, existing run retained

- The user reported restored connectivity. The original population orchestrator
  and its CBAM seed 43 child were still alive; new epoch logs and checkpoints
  confirmed active computation. No replacement or duplicate job was started.
- Ten of twenty required population fits are complete. CBAM seed 42 completed
  nine epochs and selected epoch 1. CBAM seed 43 is now running. The full
  personal-development and personal-test inventories remain complete.
- Remote branch access was verified at `e356030`; the working tree was clean.
  Eighty-eight tests and exact label-free replay of all eight personal strategies
  were verified before this connection check. These are not a claim that the
  remaining population fits or final PDF are finished.
- Run timers, process elapsed time, and calendar elapsed time are different
  quantities. Shared-host load and interruptions prevent treating these
  observations as isolated throughput comparisons.

## 2026-10-09 17:03 America/Chicago: draft pull request and continuing population suite

- [Draft PR #1](https://github.com/place-bot/Bad-Blood/pull/1) is open. The
  branch was verified remotely at `d21b23a`; nothing was merged into main.
- Eight of twenty required population fits have completed. Temporal seed 44
  is now running. The completed personal inventory is unchanged; the final
  combined report remains gated on the full population inventory.
- Independent source/protocol and report-method reviews found no new
  consequential inconsistency. Seventy-eight automated tests pass again.
- Report typography checks use isolated, explicitly synthetic layout fixtures
  only. Actual result PDFs will be compiled and visually checked after all
  required fits finish. Core figures now have numbered cross-references, and
  mixed pooled/patient-macro summary columns are explicitly distinguished.
- Other local compute jobs were observed alongside population training.
  Timing measurements are shared-host observations, not isolated benchmarks.
  No other task's process or resource settings were changed.

## 2026-10-09 16:46 America/Chicago: personal inventory verified

- Track B completed all 1,068 development and 684 final-target units. The
  authoritative verifier checked source/data/checkpoint/configuration hashes,
  all saved outputs, and the development-only selection. The final inventory
  contains six targets, 4,284 common queries, 114 method/budget/seed groups and
  33 paired comparisons. A runtime-only verified analysis snapshot is stored
  at the artifact root under `analysis/personal_verified.json`.
- At K=10, seed-mean patient-macro SBP/DBP MAE is 12.73/7.80 for population,
  9.24/4.66 for full adaptation, 9.02/4.84 for bias correction and 9.01/4.04
  for reference mean. Full-minus-population paired changes have descriptive
  95% patient intervals of [-7.65,-0.87] and [-7.03,0.10] mmHg. These six-patient
  results do not establish a uniform advantage for deeper adaptation.
- Track A has six completed fits and is running Temporal seed 42. Its full
  required inventory and the combined final report remain incomplete.
- Seventy-eight automated tests pass. Complete metric summaries will be
  stored losslessly as deterministic gzip JSON, with an uncompressed browsable
  overview; compression does not drop metrics or round values.
- The research branch was backed up to GitHub at `d024ec9`; main/V1 remain
  unchanged. The final result/report commit and pull request are still pending.

## 2026-10-09 16:30 America/Chicago: personal test phase

- All 1,068 development units completed; the development-selected settings
  were saved before the final-target phase began. The 684-unit final phase
  is still running. No target outcome changes adaptation settings.
- The population suite is still running: three baseline fits and SE-last
  seed 42 have completed. Neither track is represented as a complete study.
- Seventy automated tests pass. The report now distinguishes the prospective
  four-model attention ranking from the six-required-model validation ranking,
  adds paired capacity/placement/modality contrasts, and reports early/later
  reference distributions as descriptive failure context rather than causes.
- Eight real development units passed direct checkpoint replay, exact
  source/reference/timestamp alignment, and frozen-parameter/BN-buffer checks;
  see `reports/RUNTIME_INTEGRITY_CHECK_20261009.md`.

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
