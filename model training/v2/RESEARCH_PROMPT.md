# Bad-Blood v2: revised research and engineering assignment

Updated October 9, 2026 following the project owner's clarification. This is
the working instruction for this assignment. It replaces the original Track B
requirement for open-loop future-waveform prediction. The directory name
`personalized_forecasting/` is retained for compatibility, not as a description
of the statistical task.

You are the research and engineering owner for version 2 of Bad-Blood.
Complete the full workflow: audit, literature reviews, English ACL-format
reports and compiled PDFs, implementation, real experiments, analysis, and
GitHub delivery. Repository: https://github.com/place-bot/Bad-Blood. Work in
`model training/v2/`, preserving V1 and unrelated team changes.

## 1. Scope and ordered workflow

Use the existing VitalDB-derived data only. Dataset expansion and full
PulseDB/CRC training are handled by a teammate and are outside this assignment.
Use the actual existing high-resolution CNN as the architectural baseline;
inspect its implementation rather than substituting a smaller network.

All added documentation, reviews, reports, figures, captions and tables must
be in English. Maintain `attention_cnn/`, `personalized_forecasting/`, `shared/`
and `reports/` under the new directory.

Complete these gates in order:

1. Audit the repository, baseline, checkpoints, data, provenance and resources.
2. Research both tracks and freeze their experimental protocols.
3. Compile and visually inspect both literature-review PDFs.
4. Commit the reviews and protocols before implementation or training.
5. Implement and test the complete experimental pipelines.
6. Execute and verify the principal experiments on real data.
7. Produce the combined result-derived ACL research report and inspect its PDF.
8. Commit and push a dedicated branch and open a pull request when available.

Before the review gate, only inspection, metadata/numeric data auditing,
document preparation and protocol design are allowed. A plan, review, scaffold
or smoke test is not completion. Continue through the authorized workflow.

## 2. Audit

Record the baseline source commit, architecture, parameter count, inputs,
outputs, sample rate, duration, preprocessing, loss, checkpoint-selection rule
and training settings. Identify whether it outputs scalar pressure or a
waveform. Verify the VitalDB origin independently of filenames.

Audit patient and case identity, existing splits, prior exploratory exposure,
temporal overlap, raw versus filtered signals, labels, and available cuff/NIBP
events. Report actual hardware, runtime and software. Preserve historical
results as historical: if a split is corrected, retrain the unchanged baseline
and every competitor on that same corrected protocol. Never claim an
architecture gain by comparing incompatible historical and new scores.

## 3. Track A: attention-enhanced CNNs

Question: can attention improve the existing ECG/PPG-to-SBP/DBP task under a
controlled comparison? Retain its current input/output definition.

Review SE, ECA, temporal attention, 1D CBAM, ECG-PPG fusion, and relevant
multiscale/CNN-attention methods. Distinguish attention over learned feature
channels from attention over sensor modalities. For important papers verify
citation, link, publication status, task, dataset, inputs/outputs, mechanism,
placement, patient split, metrics, calibration, code and limitations. Prefer
original papers and official repositories. Do not invent or overinterpret
quantitative claims.

Produce `attention_cnn/literature_review.tex`, bibliography and assets, and
`literature_review.pdf` using the official ACL style, with a comparison table
and a concrete experimental protocol.

Compare at least the unchanged high-resolution CNN, channel attention,
temporal attention and combined channel-temporal attention. Add another
channel mechanism where resources permit. Document real tensor operations,
insertion locations, trainable parameters and compute. Keep preprocessing,
cohorts, targets, optimizer budgets and selection criteria matched. Include
capacity control, attention-placement ablation and ECG/PPG modality ablations.

## 4. Track B: personal calibration and later BP estimation

### Exact task

Start from a population model. Use an individual's EARLY ECG/PPG windows and
their corresponding reference SBP/DBP to calibrate or adapt it. Later, receive
that individual's NEWLY OBSERVED ECG/PPG and estimate SBP/DBP for the same
observed window. Later reference BP is available only to the evaluator.

Formally, for early personal support set C_i and a later observed window X_it:

    phi_i = Adapt(theta, C_i)
    predicted_SBP_DBP_it = f(theta, phi_i, X_it)

Alternatively, a reference-conditioned model may consume the early reference
pairs directly as context; distinguish that mechanism from weight fine-tuning.

This is personalized contemporaneous estimation with chronological evaluation.
It is NOT predicting future ECG/PPG or ABP without observing later ECG/PPG,
and NOT generating a dense BP waveform from sparse cuff readings. Strict
future-waveform forecasting and waveform reconstruction are outside the
current implementation scope. The early 20%/later 80% split is a chronological
calibration/evaluation split, not an instruction to forecast 80% of a record
from the first 20% alone.

The application is an initial small reference-measurement session followed by
ECG/PPG-based monitoring. Research questions are how much personalization
helps, which adaptation depth works under each budget, how many references
are needed, how benefit changes over elapsed time, and what cross-visit
durability evidence exists. These are questions, not promised benefits.

### Literature and protocol

Review population pretraining, few-shot BP personalization, reference-pair
conditioning, bias/affine calibration, head-only, last-block/head, full-model
and parameter-efficient adaptation. Examine calibration counts, reference
source, chronological versus shuffled tests, deterioration and cross-session
evidence. Briefly distinguish waveform forecasting to prevent task confusion;
do not use forecasting accuracy as evidence for the current task.

Produce `personalized_forecasting/literature_review.tex`, bibliography/assets
and compiled PDF using official ACL style. Include verified evidence, a
comparative table and a frozen, feasible protocol.

Choose a fixed small personal cohort using data availability and prespecified
quality criteria, before performance inspection. Keep final targets and
adaptation-development patients outside ALL population training, validation
and preprocessing-statistic fitting. Use separate development patients for
adaptation hyperparameters. Report patients, cases, observed duration, elapsed
duration, retained windows and exclusion reasons. Repeated windows/seeds do
not increase independent patient counts.

Use early chronological support and later disjoint queries within a case.
Define exactly whether 20% means elapsed time or retained-window count, and
protect boundaries including preprocessing support. Retain gaps and case
boundaries; row number in a shuffled subset is not time. Match the later query
set across all support budgets. Never tune or early-stop a final patient's
adaptation on their later query labels.

Required matched-budget comparisons:

1. Population model without adaptation.
2. Personal additive bias correction.
3. Head-only fine-tuning.
4. Last residual block plus head fine-tuning.
5. Full-model fine-tuning.

Also include last-reference and calibration-mean predictors. Include a
regularized affine correction when references provide enough support. A
parameter-efficient method is supplementary if justified. Explicitly freeze
or update batch-normalization state; record trainable parameters and verify
the intended changes. Reuse identical population initialization per seed.

Evaluate a prespecified calibration ladder such as 1, 2, 3, 5 and 10 distinct
reference events, with their timestamps and spacing. Distinguish true cuff
events, invasive ABP-derived references, and sparse-reference simulations.
The available local data have no actual cuff events: experiments based on
selected ABP-labelled windows must be called ABP-derived sparse-reference
simulations, not real cuff validation. Do not call adjacent windows independent
cuff measurements or copy one held cuff value as multiple reference events.

Measure error and paired improvement by elapsed time since calibration, with
window duration and target definition fixed. Assess cross-visit transfer only
if actual patient linkage and separate-session metadata support it. A later
part of the same case is not another visit. Explain unsupported questions
through both the audit and the literature.

## 5. Leakage prevention and metrics

Prevent patient overlap, overlapping support/query windows, target-dependent
input features, query-informed normalization, future-dependent preprocessing,
test-based tuning and accidental batch-normalization adaptation. Normalize
using only available inputs or training/support statistics. Full-record
filtering must not be misrepresented as a strictly online implementation;
document upstream data processing and deployment gaps honestly. Report
reference-dependent QC separately from input-only QC available in use.

Mandatory separate SBP and DBP endpoints: MAE and RMSE in mmHg. Also report
signed bias, sample error SD, median absolute error, error quantiles, fractions
within 5/10/15 mmHg, Pearson correlation and R-squared where defined. Include
per-patient metrics and paired improvements, pressure-range counts, and
Bland-Altman analysis acknowledging repeated measurements.

Report both pooled-window and equal-patient aggregation. Use patient-level
paired uncertainty rather than treating correlated windows independently.
Separate seed variability from patient variability. Use at least three seeds
for principal trainable comparisons where resources support them. Report all
personal targets, including failures. Accuracy thresholds do not alone imply
clinical-standard certification.

## 6. Engineering and reproducibility

Implement maintainable data loaders, model modules, versioned configurations,
deterministic split manifests, synchronization/QC rules, checkpointing, resume,
validation-based selection, explicit parameter-freezing, and reusable metrics.
Save predictions, references, patient/case IDs, available times/source indices,
experiment IDs, checkpoint/config/split hashes and environment information.
Measure training time, inference latency, parameters and memory where possible.

Test split leakage, boundary indexing, label alignment, shapes, metric
calculations and actual parameter updates. Freeze an experiment matrix before
training, distinguishing required/supplementary runs, seeds, models, personal
budgets, adaptation settings, drift intervals and outputs. Any amendment must
be dated with its reason and cannot be motivated by final-test performance.

No synthetic results, placeholder outputs or reduced toy studies as final
deliverables. If a genuine compute/access blocker prevents a required run,
save resumable state and exact commands, report completed and pending runs,
and do not present the study as complete.

## 7. Final report and GitHub delivery

Create `reports/final_report.tex` and `final_report.pdf` with official ACL style.
Cover research questions, related work, audit, exact tasks, protocol, leakage
controls, implementation, methods, complete actual results, ablations, cost,
patient failures, uncertainty, limitations and reproduction. Generate tables
and figures from saved outputs. Include calibration-budget curves, error and
benefit versus time since calibration, per-patient paired changes, pressure
range results and representative observed-waveform/prediction examples chosen
by stated rules. Do not fabricate waveform targets for this scalar task.

Compile and inspect every PDF for citations, typography and layout. Push the
dedicated branch, preserve V1, do not force-push or merge without authorization.
Keep raw data, credentials, caches and large checkpoints out of ordinary Git;
document external artifact paths and reproduction. Include code, configurations,
small result summaries, sources and PDFs. Final handoff must provide branch,
commit/PR links, both review PDFs, final PDF, experiment inventory, quantitative
findings and uncertainty, reproduction commands and unfinished items if any.

Completion requires actual principal runs, analyzed results, verified PDFs and
successful push. Start from the completed audit and finish the review gate
before implementation.
