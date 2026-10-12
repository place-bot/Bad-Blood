# Frozen experimental protocol

Version 1, October 9, 2026. Frozen before model implementation and training.
The revised task is defined in [RESEARCH_PROMPT.md](RESEARCH_PROMPT.md).
All subsequent amendments must give date, reason and whether any affected
final-test outputs had been inspected. Negative results do not justify changes.

## Shared task, provenance and split

Input: float32 ECG_F and PPG_F, 2 x 1250 samples, 125 Hz, ten seconds.
Output: SBP and DBP for that SAME interval, mmHg. Labels are invasive-ABP
derived. No actual cuff/NIBP events exist in the local data. No waveform
decoder, future-waveform task, external dataset or external checkpoint is used.

Population source is the existing official VitalDB-specific PulseDB subsets
in `output/bp_model_v2/data`. Keep the V1 deterministic patient assignments but
exclude all 31 IDs in V1 `reports/prior_exploratory_subjects.json` from every
new population partition. Expected train: 1,085 patients / 390,600 windows;
validation: 192 / 69,120; primary test: 141 / 56,400. The AAMI-named subset
retains all 116 patients / 666 windows after exclusions. It is a separately
reported stress evaluation, never combined with the primary test.
All existing quality-valid rows within retained patients are used. No reduced
toy dataset is substituted. Test data were historically inspected; this is a
controlled new comparison, not a previously untouched external validation.

Inputs are normalized separately within each observed window/channel, using
the original processed array pathway (already filtered PulseDB inputs). Target
mean/SD are fitted on population training only. Record upstream filtering and
reference-dependent signal selection; this offline experiment does not verify
strict native-device streaming causality. Do not apply an additional raw-input
filter to already filtered inputs. Retain source indices when times/cases are
unavailable; never fabricate population timestamps or infer chronology by row.

## Track A matrix

Required model runs:

| Model | Seeds | Role |
| --- | --- | --- |
| baseline | 42, 43, 44 | Exact BPNet(pool=1), 37,666 parameters |
| se_last | 42, 43, 44 | Feature-channel attention |
| temporal_last | 42, 43, 44 | Temporal gate |
| cbam_last | 42, 43, 44 | Channel then temporal gates |
| additive_last | 42, 43, 44 | Same 630 added parameters as SE, additive rather than gated |
| se_all | 42, 43, 44 | Placement ablation |
| baseline_ecg | 42 | ECG-only modality ablation |
| baseline_ppg | 42 | PPG-only modality ablation |

Total: 20 required population runs. ECA-last at seeds 42/43/44 is supplementary
if runtime permits. The two modality ablations keep the missing channel zero,
apply noise only to present channels, and are interpreted as single-seed
ablations, not seed-robust effect estimates. Each other model retains V1's
stem, residual widths, pooling and head. Gates and counts are specified in the
attention review. All attention is over learned features, not direct modality
importance. No architecture-specific hyperparameter search is permitted.

Training: MSE on train-standardized pressure targets, AdamW lr=0.001,
weight_decay=0.001, batch=256, noise SD=0.003, clip norm=5. Maximum 40 epochs;
early stopping after 8 consecutive epochs without improvement. Scheduler:
ReduceLROnPlateau factor=0.5, patience=3. Selection is the mean of validation
patient-macro SBP and DBP MAE. Save best and latest resumable state, optimizer,
scheduler, RNG state and actual epoch timing. Epoch-boundary resume is required.
Use seeded initialization, ordering and noise; record backend nondeterminism.
All final evaluation loads the selected BEST checkpoint, not the last epoch.

## Track B cohort and timing

Use timestamped individual files in `output/bp_model/data/vital`. Start with
audited complete, single-case subjects having >=400 windows, >=300 adjacent
pairs, >=100 windows starting at least an hour after first retained time, and
finite/nonflat inputs. Sort and shuffle with NumPy default_rng(20261009).
Six targets and four development subjects were allocated before performance
inspection. After checking support availability, replace target p005983
(only six >=60s-spaced early references) with the next unused eligible subject,
p005977. This is an availability-only pretraining amendment, not an outcome-
based exclusion. Preserve this audit trail rather than retrospectively hiding it.

- Final targets: p005879, p005885, p005977, p005984, p006009, p006052.
- Adaptation development: p005548, p005647, p005968, p006090.
- All ten are excluded from population training, validation and testing.
- All were historically explored in earlier project work; disclose this. The
  new population fits exclude them, and no final query is used to tune v2.

Within each case, let b = first retained sample time + 0.20 * elapsed span
through the exclusive end of the final retained window. Support windows must
end at/before b; common later query windows must begin at/after b+10 seconds.
Do not compress gaps, join cases, or include a boundary-crossing window.
The split is 20% of ELAPSED case span, not 20% of shuffled/retained rows.

Build a chronological candidate support pool by greedily retaining starts
at least 60 seconds apart. Select ten approximately evenly spaced entries
using round(linspace(0,n_candidates-1,10)); require unique indices. Budgets
K={1,2,3,5,10} use nested prefixes of these ten references. These are distinct
ABP-labelled windows, not real cuff measurements and not statistically
independent physiological observations. Record all source indices and times.
The query set is identical across K and methods. Final query labels are never
used for adaptation, early stopping, parameter choice or reference selection.

Primary inputs use provided ECG_F/PPG_F and the same within-window scaling as
population training. One ten-second observation can use its own samples; no
new operation reads the later query window during support fitting. Store and
disclose the upstream offline-filtering limitation separately.

## Track B methods and model selection

Use the newly trained unchanged population baseline at seeds 42/43/44 for
every B strategy. This isolates personalization from attention. Do not pick
an attention architecture using final personal-target results.

For every seed, K and target, evaluate:

- population (no adaptation);
- bias: mean support residual, separately for SBP/DBP;
- last_reference and calibration_mean: reference-only controls;
- affine (K>=3): slope regularized toward 1 with unpenalized intercept;
- head: train prediction head only;
- last_block: train final residual block plus head;
- full: train all parameters, but freeze BatchNorm running statistics.

All neural strategies start afresh from the same seed checkpoint for every
patient/budget. Use standardized-target MSE, AdamW weight_decay=0.001, gradient
clip=5, full support batch, and no augmentation. Keep dropout inactive and BN
running statistics frozen in all three strategies; BN affine parameters only
train when their containing block is enabled. Record exact parameter masks.

For each strategy and K choose lr in {0.0001,0.001} and fixed update steps in
{20,100}, using average patient-macro MAE across the four development patients
and the three seeds. No future-query-based early stopping on final targets.
Resolve ties by fewer steps, then smaller lr. Affine penalty grid {1,10,100}
uses standardized predictions and labels, minimizing squared residuals plus
lambda*(slope-1)^2; ties choose stronger shrinkage. Its K<3 entries are N/A.
Do not choose a personal method separately for each final patient using their
query performance. Save development grid scores and selected settings before
final-target evaluation. No further final-test-driven search.

## Drift, metrics and uncertainty

For each K record actual final support-reference end time. Primary drift bins
are [0,15),[15,30),[30,60),[60,infinity) minutes from that final reference to
query END. B-relative elapsed time is secondary and labelled time since the
common adaptation boundary, not time since last reference. Report patient/
window counts and empty bins; do not imply all targets have every interval.
Reference age varies by K and is a real design distinction, not hidden.
No cross-visit durability claim: the only multi-case file is excluded from
the main cohort and its absolute session interval is unavailable.

SBP and DBP separately: MAE, RMSE, signed bias, sample error SD (ddof=1),
median absolute error, q90/q95 absolute error, fractions <=5/10/15 mmHg,
Pearson r and R2 (undefined flagged). Report per-patient and pooled metrics;
patient-macro MAE/RMSE average patient-specific values with equal weight.
Selection scalar never replaces the two target-specific outcomes.

Pressure strata SBP<100,100-139.999,>=140; DBP<60,60-89.999,>=90 mmHg,
with window/patient counts. These are descriptive strata, not diagnoses.
For paired gains, average each patient's paired difference across matched
seeds, then bootstrap patients with replacement (2,000 resamples, seed20261009)
for95%percentile intervals. Also show seed-specific scores/SD separately.
Six targets are a pilot; intervals are descriptive and all individual effects
must be shown. No window-level independence assumption or clinical compliance
claim. Bland-Altman uses patient mean errors plus repeated-error variance
decomposition; plot pooled errors only with dependence explicitly stated.

## Required saved artifacts and tests

Save configs, frozen manifest/hash, source/data/checkpoint hashes, environment,
history, exact selected epoch, weights outside Git, predictions/references/IDs/
times or original indices, metrics, adaptation masks and actual changes.
Save measured CPU and accelerator latency, parameters, memory when available,
and elapsed training cost. Do not label desktop latency Raspberry Pi latency.
One run directory cannot be overwritten by a different configuration silently.

Tests must cover V1 numerical architecture equality, all attention shapes and
counts, patient split disjointness, support/query nonoverlap and index-label
alignment, exact hand-computed metrics, constant-correlation behavior, neural
freeze masks including BN statistics, adaptation using only support, and resume.
No final report tables until actual saved predictions exist. Examples use fixed
rules (median/worst patient under the population baseline), not visual appeal.

## Execution and completion

Required A:20fits plus all held-out/stress predictions. Required B:3seeds x
5budgets x6targets for population,bias,head,last_block,full and reference-only
controls; affine forK>=3. Development search is part of the experiment, not a
test shortcut. ECA is explicitly supplementary; no low-rank adapter or decoder
is required because the existing network is small and primary comparisons
already isolate trainable depth.

Use local MPS when available, one population fit at a time, with resumable
state. Model implementations start only after both compiled reviews have been
visually checked and this protocol has been committed. Do not stop at smoke
tests or call launched-but-unfinished experiments complete.
