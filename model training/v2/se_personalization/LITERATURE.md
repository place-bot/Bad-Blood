# Literature-guided personalization

Reviewed October 12, 2026. This is a design note, not a claim of reproduced paper results.

## What the completed SE experiment establishes

The development-selected full-network centered fine-tune did not transfer to the
six previously examined target participants. Patient-macro SBP/DBP MAE was
12.14/5.28 mmHg, versus 9.47/4.22 for personal mean and 9.16/4.40 for
population-model bias correction. More early labels and more trainable weights
did not by themselves establish an advantage. Preserve this negative result.

## Relevant primary sources

1. Mekonnen et al. (2024), *Accurate and 30-plus days reliable cuffless blood
   pressure measurements with 9-minutes personal photoplethysmograph data and
   mixed deduction learning*.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11467377/
   The study pairs preceding and current PPG measurements and learns pressure
   change, combining personal and population examples. It evaluates later
   measurement rounds, not just a random window split. Its cohort, PPG inputs,
   and measurement schedule differ from ours. The useful principle is to learn
   changes relative to a known reference, not to copy its reported accuracy.

2. Li, Grandvalet and Davoine (ICML 2018), *Explicit Inductive Bias for Transfer
   Learning with Convolutional Networks*.
   https://proceedings.mlr.press/v80/li18a.html
   L2-SP regularizes adapted parameters toward their pretrained values rather
   than toward zero. This is general transfer-learning evidence, not a BP trial.
   It motivates retaining the SE representation and penalizing personal changes
   to the prediction head. A frozen-feature linear head permits a deterministic
   closed-form version of this objective.

3. Li et al. (2026), *Few-Shot Personalized Blood Pressure Estimation from
   Photoplethysmography and Physiological Priors via Low-Rank Adaptation*.
   https://doi.org/10.1109/JBHI.2026.3665810
   Author-institution abstract:
   https://research.cuhk.edu.hk/en/publications/few-shot-personalized-blood-pressure-estimation-from-photoplethys/
   The Transformer-based work combines personalized learning, low-rank
   adaptation designed for sampling-rate differences, and a pulse-pressure
   penalty. Only the abstract was available in this review; the exact adaptation
   modules, split timing and hyperparameters have not been audited. A low-rank
   adapter on our CNN would be an inspired variant, not its reproduction.

4. Tang et al. (2026), *Enhancing few-shot personalized cuffless blood pressure
   estimation with self-supervised learning*.
   https://doi.org/10.1088/1361-6579/ae52a1
   Author-institution abstract:
   https://pureportal.coventry.ac.uk/en/publications/enhancing-few-shot-personalized-cuffless-blood-pressure-estimatio/
   This PulseDB study learns a shared representation with self-supervision,
   followed by few-shot personalization. This motivates changing population
   training if existing supervised features cannot capture within-person change.
   It is a larger retraining step; the accessible abstract is insufficient to
   reproduce its entire protocol.

## Implementable next step

Keep the existing pretrained CNN+SE encoder. Fit a small regularized personal
head using **all** eligible first-20%-of-time windows. Compare both:

- Population-anchored residual head: predict the bias-corrected population output
  plus a regularized waveform-dependent correction.
- Mean-anchored head: predict the personal reference mean plus a regularized
  waveform-dependent change.

With centered frozen features H and residual targets R, solve
`D = (H.T H / n + lambda I)^-1 H.T R / n`.
The residual head predicts `base(x) + (h(x)-mean(h_support)) D`.
For the penultimate features and population anchor, this is last-linear-layer
adaptation penalized toward the pretrained weights, with a free personal offset.
It uses the original signals at inference and does not reduce to a constant
unless the learned change is zero. A zero-penalty head is not preferred by fiat.

Use a bounded development-only grid: feature dimension 32 (penultimate) or 48
(SE-pooled), population or mean anchor, six penalties from 0.01 to 1000.
These 24 settings are engineering choices, not hyperparameters quoted from a
paper. Choose against both controls on four development people and three seeds.
Report all metrics, participant-level differences and stability across people.
Do not open the six target outcomes again during this development screen.

The completed development screen selected the 32-dimensional population-anchored
head with penalty 100. Its MAE was 6.91/5.56, versus 12.53/5.26 for personal mean
and 6.76/5.88 for bias correction. It failed the joint improvement criterion;
no subsequent target evaluation was performed. This result does not test the
paired-input architecture or the published LoRA method.

If stronger modeling is necessary, the next architecture is reference-conditioned
change prediction: a shared SE encoder processes a known-reference ECG/PPG
segment and the current ECG/PPG segment, and a population-trained difference
head estimates the change from the known reference BP. Training pairs must be
within population-training participants only. It needs population retraining;
merely subtracting features without learning the relation is not sufficient.

## Evaluation safeguards

- Every control uses exactly the same support labels and later query windows.
- Overlapping or adjacent windows must not straddle the time boundary.
- Personal means and bias corrections remain mandatory controls.
- Development selection uses participant-macro errors, not pooled window counts.
- Previously examined target people cannot become a new independent test set.
  Any later evaluation on them is explicitly exploratory.
- Papers with different datasets or random personal splits do not establish an
  achievable error for our chronological experiment or a clinical certification.
