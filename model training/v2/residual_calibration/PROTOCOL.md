# Endpoint-specific residual calibration — 20261012

This is one bounded, literature-informed follow-up. Existing population and
personal models are retained. No results are assumed and no deployment is planned.

## Fixed inputs and comparisons

- PPG-only SE-last and dual ECG/PPG SE-last; seeds 42, 43, 44.
- Use the exact verified population weights from `modality_se/models`.
- Same four development, six historical target and sixteen previously examined
  target people; unchanged eligible early-20%-of-time support and guarded queries.
- Use all support records for the final personal fit, never query BP.
- Baselines: identical personal mean, same-modality additive bias correction;
  retain the prior selected reference-conditioned PairSE for comparison.
- Tests are exploratory replication, NOT new independent acceptance. No target
  results select candidates or trigger another parameter search in this run.

## New methods

Keep the CNN+SE encoder fixed. Fit an intercept and centered residual correction
from the individual's observed early data. The intercept is the mean residual
and the correction is trained against residual deviations, so a constant
residual exactly reproduces ordinary bias correction.

1. Linear residual ridge using the two population predictions as features.
2. Linear residual ridge using the pretrained 48-dimensional representation.
3. Centered RBF-kernel residual ridge on the same representation (a kernel/GP
   posterior-mean approach, without claiming a fully optimized probabilistic GP).

Ridge/noise penalties: 0.01, 0.1, 1, 10. RBF squared-distance denominators:
0.25, 1, 4. Kernels use mean squared feature distance. Training kernels are
centered using support only; regularization is n_support times the penalty.
Feature normalization comes from source training, not target query statistics.
No arbitrary BP clipping, amplitude cap, or test-based tuning is applied.

One additional candidate chooses a method separately for SBP and DBP using
two forward validation blocks inside the support period (50–75%, 75–100% of
support time), with a ten-second separation from earlier fitting windows.
Candidate selection minimizes validation absolute error; ties prefer bias.
If no valid forward split exists, fall back to bias correction. After selection,
refit on ALL support windows. This uses no extra cuff labels.

## Selection and stopping

Before target evaluation, select a method separately for SBP and DBP using
the mean participant MAE over the four development people and all three seeds.
Candidates include the 20 fixed residual fits, support-forward-CV, both controls
and the previous locked PairSE. Thus a baseline may legitimately win: that does
not count as outperforming itself. The two modalities receive the same search.

Freeze both selections before opening target labels. Evaluate each selected
pipeline and the controls on both target cohorts once. No follow-on search
based on these targets. Report all development candidates, all per-seed and
per-person target metrics, macro and pooled metrics, paired person-bootstrap
intervals, and counts of seeds/people improved. Joint success requires lower
SBP AND DBP MAE than BOTH controls; investigate Bias, sample error SD, RMSE,
within 5/10/15 mmHg and tails, not MAE alone.

Verify support/query identity, constant-residual behavior, query-label isolation,
input-mask invariance and saved-predictor replay before declaring completion.

## Literature basis

- Wang et al., *Cuff-Less Blood Pressure Estimation From Photoplethysmography
  via Visibility Graph and Transfer Learning*, JBHI (2022),
  https://pubmed.ncbi.nlm.nih.gov/34784289/ . The work couples pretrained CNN
  features with ridge regression. Here we retain our 1-D SE representation;
  this is an adaptation of the feature-transfer idea, not a reproduction of
  their visibility-graph pipeline or their reported accuracy.
- Chen et al., *A Data-Driven Model with Feedback Calibration Embedded Blood
  Pressure Estimator Using Reflective Photoplethysmography*, Sensors (2022),
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8914760/ . This supports investigating
  kernel-based individual calibration. Their reflective-PPG hardware, personal
  covariates and feedback design differ from our fixed early-budget experiment.
- Leitner et al., *Personalized Blood Pressure Estimation Using
  Photoplethysmography: A Transfer Learning Approach*, JBHI (2022),
  https://pubmed.ncbi.nlm.nih.gov/34077378/ . Layer-selective adaptation is a
  relevant alternative, already represented by our head/full fitting ablations.
- Mekonnen et al., mixed deduction learning (2024),
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11467377/ . The preceding PairSE
  experiment was inspired by reference-conditioned estimation, not an exact
  reproduction. Here we test whether direct personal residual fitting is more
  reliable under our unchanged chronological calibration budget.

Published numerical results are not comparable to ours without matching the
data, participants, chronology and calibration budget.
