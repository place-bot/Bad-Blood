# Adopted V2 personal method: tail adaptation (20261015)

One policy, two inputs: PPG-only or ECG+PPG. The pair model predicts current
SBP/DBP from current signals plus this person's earlier reference signals and
known scalar BP. It does not need current true BP or an ABP waveform.

## Exactly what is fitted

- Trainable: `change.*`, `encoder.body.6.*`, `encoder.gates.6.*` (33,336 parameters).
  These are the pressure-change head, final residual CNN block, and last SE gate.
- Frozen: earlier encoder blocks, population readout weights, source label/feature
  normalization and BatchNorm running statistics. Dropout stays disabled.
  Trainable BatchNorm affine parameters inside the last block are retained.
- Adam; no weight decay; gradient norm capped at 5. Each update uses **all valid
  within-person chronological pairs**, with reference end + 10 seconds <= query start.
- Each person's first 80% of ordered eligible segments is development data.
  Last 20% is test, after a 10-second guard. Do not randomly shuffle this split.
- Within development, three expanding training/validation periods use
  0-50% -> 50-65%, 0-65% -> 65-80%, 0-80% -> 80-100%, with guards.
  These percentages refer to the development bank, not the full recording.
- Independently for SBP and DBP, choose learning rate {0.0001, 0.0003} and
  optimizer updates {10,25,50,100} by average validation MAE across the three periods.
  A zero-extra-update source checkpoint is also evaluated. Ties prefer fewer
  updates, then smaller learning rate. No other person's labels select these settings.
- Restart from the source checkpoint and refit on the entire personal development
  bank. Save **two endpoint models** per person/source seed. At inference, take SBP
  from endpoint 0 and DBP from endpoint 1, each averaged over all known references.

The `rolling_selected` experiment additionally allowed head-only versus tail
selection; that is a comparator, **not the adopted deployment policy**.
Population seed variation changes pretrained initialization/order/pair sampling,
not the 30 people, personal split or calibration budget. Results average three
separate runs; no ensemble is deployed by this package.

## Portable use

Run from `final_results_20261015/` (Python 3 + NumPy + PyTorch):

```sh
python -m pip install -r implementation/requirements.txt
python implementation/personalize.py fit \
  --checkpoint 02_personalization/source_weights/ppg_pair_population_seed42.pt \
  --support person_development_only.npz --output my_personal_model
python implementation/personalize.py predict \
  --models my_personal_model --input later_windows.npy --output predictions.npy
```

`person_development_only.npz` contains `x` (K x 2 x 1250, ECG then PPG), `y`
(K x 2, SBP then DBP in mmHg), `start` and `end` (seconds in the same recording
clock). PPG-only also accepts K x 1 x 1250. Supply filtered, synchronized 125-Hz
signals, min-max normalized separately per channel/window. Reserve and guard
the final test period **before** creating this file. `fit` accepts no test data.
Its `reference_bank.npz` is required for later predictions; store it with the
person's models in controlled storage. Source seed 42 remains the representative
source choice from population validation, not a personal-test winner.

Six archives in `personal_weights/` contain the 30 people's two selected tail
endpoint checkpoints for each modality/seed (360 checkpoints total). Extract
the desired archive into a separate directory. These archives do not contain
raw reference waveforms. To replay an archived person, reconstruct their exact
reference bank using `metrics/lock.json` and supply `--references bank.npz`.
Never use one participant's personal model for a different person.
