# Chronological reference-conditioned source training — 20261012

Frozen before source-record downloads or candidate training. This is a bounded
development experiment, not a replacement of the 20261012.1 release.

## Question and comparison

Does learning from early calibration references to later observed ECG/PPG
improve current-window scalar SBP/DBP estimation, relative to random same-person
source episodes and the existing pretrained reference-conditioned models?
This does not predict future unobserved waveforms. Reference-conditioned MAE
is reported separately from zero-calibration population MAE.

The existing three-seed CNN+SE representation is preserved. Initial screening
uses seed42. We train the reference-conditioning heads on timestamped source
records, starting from existing source-trained weights: PairSE and the
context-conditioned residual network. Each has chronological and matched
random-episode training, four candidates total. The waveform encoder is frozen;
therefore zero-calibration population predictions are unchanged. This isolates
the source episode design rather than confounding it with an encoder change.

## Cohorts and temporal rules

Download all files on the observed first public PulseDB Vital directory page
whose IDs belong to the original population TRAIN or VALIDATION partitions.
Freeze identities/roles before downloading any signal/label values. Exclude
all personal-development, historical-target and newly evaluated target people.
This is a convenience source subset, not full-PulseDB temporal retraining.
The original balanced array files lack timestamps and cannot support a genuine
temporal training claim. Preserve their original participant assignments.

Use the existing strict raw reader: one case, ordered real timestamps, finite
ECG/PPG and labels, >=2 eligible early/later windows, and all early rows able to
participate in separated pairs. Exclusions are technical, never error-based.
Early support ends by 20% of elapsed time; later queries start at least ten
seconds after that boundary. Per-window/channel min-max scaling remains fixed.
Features and target standardization come exclusively from previous TRAIN data.

## Training and selection

Use all eligible source participants with equal-person episode sampling.
Each update selects one person, <=32 references and <=32 queries. Chronological
episodes draw references only from early20%, queries only from guarded later80%.
Random control draws the same counts, disjoint without replacement, from that
same person's full record. Evaluation always uses all early references and all
guarded later queries. No validation/target BP is used in source gradients.

Adam learning rate 0.0003, no weight decay or L2-SP, normalized MSE, gradient
clip5, 64 updates/epoch, max20 epochs, patience5. Record initialized epoch0.
Select epoch on source-validation participant-macro mean SBP/DBP MAE, including
epoch0 to expose negative transfer. Select the chronological family on that
same validation score; random controls are diagnostic, not candidates to rename.
Report MAE, signed Bias, sample error SD, within5/10/15 and per-person metrics.

Then evaluate the selected source checkpoint on the four existing development
people only. Do not reread the old six targets or newly examined sixteen.
Only if both source-validation SBP and DBP improve over its own initialized
head, run personal head-only adaptation (lr0.0001/0.0003; 10/25/100 updates)
plus no-adaptation on these four people. All early support participates; the
query BP never enters training. Compare same-budget personal mean, SE bias
correction and old PairSE's fixed 25-step/lr0.0003 personalization.

A development gain is not independent-test confirmation. A candidate must lower
both endpoints against both controls to be a prospective winner; if it passes,
replicate source fits over seeds43/44 with each seed's own initializer before
planning another untouched-person evaluation. Otherwise record NOT_MET and
preserve the old release. Do not tune on any old target results.

## Scope

This screen answers whether temporal conditioning helps with currently
recoverable source records. It cannot by itself establish that rebuilding the
population waveform encoder on all PulseDB is better. No automatic deployment,
email sending, or published-release replacement is part of this experiment.
