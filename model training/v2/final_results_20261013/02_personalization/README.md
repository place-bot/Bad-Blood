# Reference-conditioned PairSE

**Retained learned personalization:** a shared CNN+SE encoder plus a
100 -> 64 -> 32 -> 2 pressure-change MLP. The model uses a known early reference
BP and reference waveform, together with the currently observed waveform.
It predicts current scalar SBP/DBP, not an unobserved future waveform.

- [Architecture](../implementation/paired_se/model.py)
- [Portable personal adaptation](../implementation/adapt.py)
- [Original source/personal training](../reproduction_source/paired_se/run.py)
- [Three-seed full metrics](../metrics/comparison.json)
- [Cohort/seed stability](../metrics/STABILITY.md)

The six `*_pair_population_seed*.pt` files are **initializers for a new person**.
The 132 checkpoints below `weights/{dual,ppg}/{historical_six,previous_sixteen}/`
are already fitted to the public-data participant in their file name; do not
apply one person's fitted checkpoint to another person.

For a new participant, use all eligible early20%-of-time reference windows,
then hold later windows aside. The fixed selected adaptation is Adam lr0.0003,
25 steps, change-head-only, frozen encoder/BatchNorm/dropout, no weight decay.
All support rows must occur in time-separated reference/current pairs. The
ten-second separation guard is also respected at the support/query boundary.
The early20% refers to elapsed recording time, not a fixed ten-reference budget.

```sh
python implementation/adapt.py \
  --checkpoint 02_personalization/weights/dual_pair_population_seed42.pt \
  --support early_support.npz --output my_personal_model.pt
python implementation/predict.py \
  --checkpoint my_personal_model.pt --input later_normalized_windows.npy \
  --references early_normalized_windows.npy --reference-bp early_bp_mmHg.npy \
  --output personal_estimates.npy
```

Run from the final-package root. `early_support.npz` contains `x` (Kx2x1250
normalized signal), `y` (Kx2 SBP/DBP mmHg), `start` and `end` (seconds in a
shared recording clock). It contains **early support only**, not query labels.

PairSE beats personal mean on both pressures in every seed on both cohorts.
Dual PairSE beats both controls in all three seeds on the original six people.
On sixteen people, neither modality consistently beats bias correction on DBP.
The PPG mode has lower development MAE and slightly lower sixteen-person
average PairSE MAE; dual has stronger six-person evidence against both controls.
No single learned modality is established as universally best. Both are retained
with their own matching population baseline, not a post-hoc mixture of endpoints.
