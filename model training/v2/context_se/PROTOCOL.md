# Context-conditioned residual architectures — started 20261012

Status: development experiment, separate from the completed v2-20261012.1 release.
Goal: improve participant-macro SBP and DBP MAE over both simple personal bias
correction and personal mean, without adding calibration labels. No success is
claimed from model complexity alone.

## Why change the architecture?

The prior pair model did not consistently beat bias correction on 16 unused
people. Those 16 are now an examined cohort, not a fresh final test. Their labels
must not enter this experiment's optimizer or selection. The historical six are
also not an independent acceptance cohort. Develop on the original four personal
development people, then require another unused cohort for confirmation if a
candidate passes. Do not run new candidates on the old six or the new sixteen.

## Three bounded alternatives

Keep the pretrained SE backbone fixed, but replace the pairwise MLP with a
reference-set-conditioned residual estimator. For reference i, residual is
r_i = observed standardized BP_i - pretrained prediction_i. Every model starts
from current pretrained prediction + mean reference residual (bias correction).

1. Learned-metric residual attention: learn which reference waveforms are most
   relevant to the current waveform; correct using their residuals instead of
   a constant average. An unconstrained learned output gain controls the learned
   departure; it is not clipped and is initialized at zero.
2. Context-conditioned residual network: encode the entire reference set with
   a permutation-invariant network, condition a nonlinear current-waveform
   correction on that context, and subtract its mean correction over references.
   This enforces the observed reference mean without shrinking parameters toward
   pretrained weights.
3. Learned-feature kernel residual regression: learn an SE-feature projection
   and RBF kernel scale/noise from population episodes, then solve an individual
   residual regression on all references. This combines a learned representation
   with a person-specific nonlinear fitted function.

Inspiration (not exact reproductions):
- Attentive Neural Processes: https://arxiv.org/abs/1901.05761
- Deep Kernel Learning: https://proceedings.mlr.press/v51/wilson16.html
- Paired BP calibration: https://pmc.ncbi.nlm.nih.gov/articles/PMC11467377/

## Fixed training and selection

Reuse verified source features for 1,085 training and 192 validation people;
feature normalization uses training-only statistics. No personal-development,
historical-target or 20261012-fresh person is a population training participant.
Per population episode, sample 32 support and 16 disjoint query windows from
one person. Batch 16 people. A source epoch has 128 episodes-batches. Adam
lr=0.001, normalized-target MSE, no weight decay/L2-SP, gradient clip 5, maximum
8 epochs, patience 3. Use 192 fixed validation episodes, with the same disjoint
support/query indices across models and seeds. Select epoch by participant-macro
mean SBP/DBP MAE. Source timestamps are unavailable: do not claim these are
chronological source episodes.

Bounded first screen: all three architectures with SE seed42. If none improves
both SBP and DBP over both controls on the four development people, stop this
screen and report failure; do not test them on examined targets. If any passes,
replicate that candidate using seeds43/44 before a new held-out evaluation.
No per-person gradient updates in this first screen: adaptation is inference
conditioned on all early reference pairs (kernel variant fits its coefficients).
This is a new personalization architecture, not the old 25-step fine-tuning.

Personal evaluation uses ALL original early-20%-of-time support windows and the
unchanged later-query rows/guard. No current query label is an input. Baselines
use identical rows and the matched original SE seed. Report all metrics, including
negative results. Primary development score is the maximum of SBP/DBP MAE ratios
against both controls; lower is better, success requires <1. Fresh held-out
confirmation is mandatory before calling this a replacement model.
