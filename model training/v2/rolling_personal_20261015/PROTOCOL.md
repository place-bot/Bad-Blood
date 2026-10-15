# Rolling personal adaptation — 20261015

## Fixed scope

This is one bounded, exploratory comparison, not a search against the final test set. Reuse exactly the 30 participants, source checkpoints (seeds 42/43/44), input normalization, first-80% development indices and guarded final-20% test indices from `personal_hpo_20261015/lock.json`. Both PPG-only and ECG+PPG remain. These people and test outcomes have been examined previously; this is not an independent external confirmation. Existing published models are untouched.

## Within-person selection

Three expanding-window folds within each person's development bank:

- Train first 50%; validate next 15% (to 65%).
- Train first 65%; validate next 15% (to 80%).
- Train first 80%; validate final 20% (to 100% of development).

Integer cut points use floor. Validation begins at least 10 seconds after the end of the last training segment; reject insufficient folds rather than alter membership after observing outcomes. Every fold is restricted to this person's first 80%; final-test inputs and labels are not used in selection. The same folds and full reference budgets apply to both modalities and all source seeds.

Train separate SBP and DBP endpoint models. Each candidate uses MSE on its one standardized target, with both known reference BP labels available as reference inputs. Select independently per person, endpoint, modality and source seed by equal-fold mean validation MAE in mmHg. Ties prefer fewer steps, head-only, then smaller learning rate. No cohort-wide selection, no final-test scoring, no fallback to controls in the candidate set.

Two scope families, each with a zero-extra-update candidate and 10/25/50/100 Adam updates:

- `head`: train only `change.*`, learning rate 0.0003 or 0.001.
- `tail`: train `change.*`, final residual CNN block `encoder.body.6.*`, and its squeeze-and-excitation gate `encoder.gates.6.*`, learning rate 0.0001 or 0.0003. Earlier blocks, the original population regression head, feature/target normalizers and all BatchNorm running buffers stay fixed. Last-block BatchNorm affine parameters can train. Dropout is disabled in both scopes.

Use all valid temporally separated within-person pairs for every update, exactly as the previous experiment; gradient norm cap 5, no weight decay. Pair chunking and caching frozen prefix activations are computational optimizations, not sampling. The common source checkpoint initializes every fold and every final fit; no carryover fold weights. Reference-conditioned predictions average over all available calibration references.

After selection, refit the chosen configuration for each scope/endpoint on the full first 80%. Save four endpoint checkpoints, then form the combined method using the scope selected for each endpoint by its own validation score. This may choose different scopes for SBP and DBP. Freeze all 180 person/modality/seed selections and weights before the evaluation phase.

## Comparators and reporting

Evaluate the identical final-test rows against population model, personal mean, additive bias correction, old fixed head (lr0.0003/25 updates), and old single-block joint-endpoint HPO. Report rolling head, rolling tail and validation-selected scope separately. Report per-person and equal-person/pooled MAE, RMSE, signed bias, sample error SD (ddof=1), absolute error <=5/10/15 mmHg fractions, participant MAE distribution, all three source seeds, paired participant differences/intervals and per-endpoint/joint wins. Primary descriptive criterion: more people with BOTH endpoints strictly below BOTH controls; also report all-three-seed robustness. Do not infer benefit for everyone from an aggregate mean. No clinical validation claim.

## Verification and release

Before the experiment, test split boundaries, cached/direct gradients, frozen tensors, single-channel masking, and endpoint selection. Hash the code, previous experiment lock, source checkpoints and dataset files. Resume only matching units. Independently recompute metrics and replay saved endpoint checkpoints after evaluation. No automatic publication, GitHub replacement, email or Beamer changes.
