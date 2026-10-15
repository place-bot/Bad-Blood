# Individual hyperparameter selection, 80/20 — 20261015

## Locked question

Does selecting the PairSE adaptation settings separately for each person improve
on fixed-setting PairSE, personal mean, and additive bias correction? This is a
new within-person experiment; it does not invalidate or overwrite earlier results.

- Randomly select 30 eligible people from the locally available VitalDB-derived
  PulseDB participant files. The sampling frame, exclusions, seed (20261015), and
  selected IDs are saved before model evaluation. This is not a random sample of
  the entire PulseDB population. Exclude all population train/validation/test IDs
  and the four people previously used to select the fixed adaptation settings.
- Some candidate people and their later records have been evaluated in earlier
  exploratory experiments. This is not a wholly unseen external validation.
- Use the earliest floor(0.8 N) valid, time-ordered ten-second segments for personal
  development, and the remaining segments only for final testing. Remove test
  segments starting less than ten seconds after the last development segment ends.
- Split each person's development bank chronologically into the first 70% for
  inner training and the last 30% for inner validation, with the same ten-second
  gap. Require at least two inner validation and two final test segments and valid
  separated training pairs. Eligibility depends on identity/timing/quality, never
  on prediction accuracy. All methods use exactly the same final test segments.
- Repeat both ECG+PPG and PPG-only with source seeds 42, 43, 44. Reuse the saved
  population CNN+SE and PairSE checkpoints; do not retrain the population model.
- Independently for each person, modality and source seed, search head-only Adam
  learning rates 0.0003 and 0.001 and update counts 10, 25, 50, 100, plus zero
  updates. The encoder stays frozen; each person has a separate 8,610-parameter
  change head. Batch-normalization buffers stay frozen, weight decay is zero,
  and gradient norm clipping is 5. All valid chronological training pairs are
  used at every update. There is no sampling of easier training pairs.
- Select by this person's inner-validation worst MAE ratio across SBP/DBP and
  the two controls (personal mean and bias correction); denominator floor 0.5
  mmHg. Tie-break by mean SBP/DBP MAE and then candidate order. No other person's
  validation or final-test labels enter this choice. The grid and selection rule
  are common; selected values and fitted weights are person-specific.
- After selection is locked, reset to the population PairSE checkpoint and fit
  on the person's complete first-80% bank, including former inner-validation rows.
  Predict the final 20% using its ECG/PPG and the earlier reference bank only.
- Matched controls: frozen population model; mean of all first-80% BP labels;
  population prediction plus the mean first-80% residual; unadapted PairSE;
  original fixed PairSE (head, learning rate 0.0003, 25 updates), refit on the same
  first-80% bank. Do not compare with the earlier 20%-budget test scores as if
  they used the same budget or test period.
- Report MAE, RMSE, signed bias (prediction minus reference), sample error SD
  (ddof=1), fractions within 5/10/15 mmHg, participant MAE/distribution, all seeds,
  pooled and participant-macro summaries, and participant-paired bootstrap intervals.
- Publication gate: for a modality, each of the 30 people must have lower SBP
  AND DBP test MAE than BOTH personal mean and bias correction in ALL three seeds.
  Also report the less strict seed-averaged counts, without replacing this gate.
  Do not publish a success claim or replace V2 personalization if the gate fails.
- Freeze code, checkpoints, cohort, and settings before evaluation. Never tune
  on final-test results, replace a poorly performing participant, or select a
  favorable seed. No assumption of independent adjacent windows is made.

The labels are supplied arterial-pressure-derived SBP/DBP, not ND cuff readings.
The experiment measures within-record adaptation, not clinical device validation.
