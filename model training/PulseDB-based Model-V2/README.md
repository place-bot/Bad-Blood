# PulseDB-based Model — V2

Status: retraining in progress; no V2 inference release yet.

## From selection to refinement

[V1](../PulseDB-based%20Model-V1/README.md) compared four models and selected
the higher-resolution residual CNN. V2 develops that selected architecture,
rather than repeating the four-model search.

| Component | V1 selected model | V2 |
|---|---|---|
| Architecture | Higher-resolution 1D residual CNN, 37,666 parameters | Unchanged |
| Input | Two synchronized ECG/PPG channels, 1,250 samples each | Unchanged |
| Output | SBP and DBP in mmHg | Unchanged |
| Objective | Standardized-target MSE | Huber plus batch-bias penalty |
| Data split | Participant-disjoint train/validation | Same participants and segments |
| Selection | Mean SBP/DBP participant-macro validation MAE | Unchanged |

Training starts from a fresh initialization using the same seed, optimizer,
batch size, augmentation, learning-rate schedule and early-stopping settings
as V1. This isolates the training-objective change instead of conflating it
with extra fine-tuning of the existing checkpoint.

For error `e = prediction - reference` in mmHg, the objective is the mean
Huber loss with delta 5 mmHg divided by the training target variance, plus
0.1 times the squared batch-mean standardized error, averaged over SBP/DBP.
Delta and the penalty coefficient were fixed before this run. These are
optimization choices, not a claim of compliance with a clinical standard.

## Evaluation

Both versions will be evaluated using identical definitions:

- Segment MAE and signed mean error (bias).
- Sample SD of signed errors.
- Percentage with absolute error at most 5, 10 and 15 mmHg.
- Each participant's MAE, plus participant-macro mean, median and 90th percentile.

Validation selects the checkpoint. Previously evaluated official test sets
are reused for the version comparison; they are not a new untouched test.
Results will show the changes for SBP and DBP separately, including metrics
that worsen. V1 weights remain intact.
