# Per-person hyperparameter selection — 20261015

**Historical comparator, not the adopted personal model.** This directory
preserves the earlier single-period, head-only selection experiment and supplies
its data loader to the matched follow-up. The current adopted method is
[tail adaptation with personal endpoint selection](../final_results_20261015/).
Statements below about the publication gate describe this earlier experiment.

This follow-up uses a **random 30-person local cohort**, with **80% of each
person's ordered segments for training and validation, and 20% for testing**.
Hyperparameters are selected separately for each person, modality and source seed.
The complete protocol is in [PROTOCOL.md](PROTOCOL.md).

## What changes

The previous PairSE experiment fitted separate personal weights but used one
globally selected learning rate/update count. This experiment also personalizes
those settings, using only an inner split of that person's first-80% bank.
After selection the model is reset to the pretrained source checkpoint and
refitted on the complete first-80% bank. The encoder remains frozen; the
8,610-parameter change head is personal. A common candidate grid is not the same
as a common selected hyperparameter configuration.

The single-signal and two-signal models are both retained. Controls include
personal mean, additive bias correction, unadapted PairSE, and original
fixed-setting PairSE refitted with the same expanded calibration budget.

## Run and verify

From the repository's V2 directory, with the existing project Python environment:

```sh
python personal_hpo_20261015/test_personal_hpo.py
python personal_hpo_20261015/run.py \
  --sources /path/to/bp_model/data/vital /path/to/bp_model/data/fresh_20261012 \
  --output /path/to/20261015_personal_hpo
python personal_hpo_20261015/verify.py --output /path/to/20261015_personal_hpo
```

The run freezes the code and checkpoint hashes, sampling frame, random selection,
chronological split indices and candidate grid before evaluation. Resume checks
those identities and never repeats completed units. Each unit contains the
selection trace, two runnable model checkpoints, predictions and all metrics.

Output: `lock.json`, `progress.json`, `units/`, `summary.json`, `RESULTS.md`,
`complete.json`, and `verification.json`. No claim of success should be made
before checking both completion and verification.

Publication is conditional on the predeclared **every-person** gate, not just a
lower mean error. Prior released models and presentation files are unchanged.

## Completed local result

All 180 units completed and passed independent verification, including 360
saved-model replays (maximum difference 0.0000153 mmHg). The random sample was
drawn from 39 eligible local people. It contains 8,882 personal development
segments and 2,211 test segments; some people were previously evaluated.

Three-seed, equal-person mean MAE, SBP / DBP in mmHg:

| Method | PPG only | ECG + PPG |
|---|---:|---:|
| Personal mean | 12.240 / 6.018 | 12.240 / 6.018 |
| Additive bias correction | 8.509 / 5.382 | 9.094 / 5.700 |
| Fixed PairSE settings | 7.752 / 4.797 | 8.776 / 5.050 |
| Individual hyperparameter selection | 7.720 / 4.675 | 8.166 / 4.783 |

The every-person gate **did not pass**. Both-endpoint wins over BOTH controls
occurred for 6/30 PPG and 7/30 dual people after averaging seeds; requiring every
seed reduces these counts to 4/30 and 5/30. Seed 43 also did not improve the
macro MAE over fixed settings. No existing V2 release was replaced or pushed.

Dual-signal macro metrics all improved relative to fixed settings. PPG bias and
within-5/10 fractions improved, but sample error SD rose slightly and within-15
fractions declined. This is not uniform dominance on every metric or person.

The local execution folder is `output/bad_blood_v2_runs/20261015_personal_hpo`
relative to the parent workspace, not this Git repository. It contains the
complete `RESULTS.md`, `INDIVIDUAL_RESULTS.md`, `summary.json`, and
`verification.json`. Individual models are in `units/<modality>/seed<seed>/<ID>/`.
Inference also requires that person's earlier waveform/BP reference bank, not
just the saved weights; see the existing `modality_se/predict.py` interface.

Generate the individual report after verification with:

```sh
python personal_hpo_20261015/individual_report.py --output /path/to/20261015_personal_hpo
```
