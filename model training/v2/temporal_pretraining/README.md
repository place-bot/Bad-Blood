# Chronological source conditioning and personal adaptation — 20261012

**Complete development experiment; no replacement model promoted.**

We tested whether source learning should imitate deployment: use known early
ECG/PPG + BP references to estimate scalar SBP/DBP from later observed ECG/PPG.
The CNN+SE encoder remained frozen. These experiments retrain the population
reference-conditioning heads, not the entire waveform encoder; zero-calibration
population predictions are unchanged. The existing `v2-20261012.1` stays intact.

## Data and controlled design

The official balanced population arrays do not contain timestamps. We retrieved
26 raw files from the original author-hosted VitalDB-derived PulseDB directory,
with roles frozen before download. Two files had insufficient early/later
windows and were excluded without using model errors. The resulting subset is:

- Source training: **22 people, 19,926 ten-second windows**.
- Source validation: **2 people, 1,107 windows**; used for checkpoint selection.
- Personal development: the **same four original development people**.
- No historical six-person or newly examined sixteen-person target was read.

This is a convenience subset of the source cohort, not full-PulseDB retraining.
Two validation people are insufficient for a generalization claim. All results
below use seed42 and equal-participant macro aggregation. They are not the
metrics from the earlier 141-person population test.

Input: ECG and PPG, two channels at 125 Hz, 1,250 values each. Reference BP is
provided only for known calibration windows. The label order is SBP, DBP, mmHg.
All early-20%-of-elapsed-time references are available at evaluation. Later
queries start at least ten seconds after the 20% boundary. No query BP enters
training, feature normalization, or reference selection.

Two head architectures were tested: PairSE and a whole-reference-set conditional
residual network. Each was trained with chronological episodes and a matched
random same-person control. Per epoch the same source people were selected in
the same order for both regimes. Source updates use <=32 references and <=32
queries per episode, 64 updates/epoch, Adam 0.0003, standardized MSE, clip5,
max20 epochs, patience5. Initialized epoch0 is eligible to expose negative
transfer. No L2-SP or weight decay was added. See the [frozen protocol](PROTOCOL.md)
and [matched-RNG correction note](IMPLEMENTATION_NOTE.md).

## Source-validation results

MAE in mmHg; all source evaluations use the same early/later split and all
available calibration references, even for the random-training controls.

| Architecture / training | Selected epoch | SBP MAE | DBP MAE |
|---|---:|---:|---:|
| Existing PairSE initializer | 0 | 11.46 | 9.67 |
| PairSE / random | 1 | 9.58 | 8.70 |
| PairSE / chronological | 2 | 9.77 | 9.07 |
| Existing context initializer | 0 | 10.22 | 6.65 |
| Context / random | 1 | 8.67 | **5.93** |
| Context / chronological | 7 | **8.50** | 6.37 |

Chronological context training improved both endpoints over its initializer,
but **did not dominate matched random training**. Thus the observed improvement
cannot be attributed specifically to chronology. Signed Bias also worsened.

| Context model | MAE | Signed Bias | Error SD | Within5 | Within10 | Within15 |
|---|---|---|---|---|---|---|
| Initial | 10.22 / 6.65 | -2.78 / -2.38 | 8.04 / 5.65 | 37.50% / 51.88% | 54.43% / 69.39% | 66.06% / 92.23% |
| Random-trained | 8.67 / 5.93 | -2.79 / -2.38 | 8.15 / 5.90 | 40.84% / 55.73% | 59.89% / 76.93% | 75.90% / 93.00% |
| Chronological-trained | 8.50 / 6.37 | -4.70 / -3.30 | 6.98 / 5.18 | 44.95% / 54.79% | 63.57% / 71.56% | 75.19% / 90.79% |

All pairs are SBP / DBP; SD uses sample ddof=1 within each participant before
macro aggregation. Complete pooled, participant-level, tail and range metrics
are in [results/summary.json](results/summary.json). These are engineering
evaluation metrics, not cuff-device clinical certification.

## Personal post-training

Since the selected chronological source checkpoint improved both source
validation endpoints over its initializer, six predeclared personal settings
were evaluated: Adam lr0.0001/0.0003, 10/25/100 updates, head parameters only.
Every eligible early window participates as a training query or reference;
references must precede their training query with a ten-second guard. Later
query labels are excluded from adaptation. All methods receive the same budget.

| Method on four development people | SBP MAE | DBP MAE |
|---|---:|---:|
| Personal mean | 12.53 | **5.26** |
| Simple SE bias correction | **6.84** | 6.18 |
| Old PairSE + its fixed personal adaptation | 7.56 | 5.27 |
| New chronological context, without personal gradient updates | 7.28 | 5.69 |
| New context, lr0.0001 / 10 updates | 7.19 | 5.72 |
| New context, lr0.0001 / 25 updates | 7.16 | 5.91 |
| New context, lr0.0001 / 100 updates | 8.11 | 7.81 |
| New context, lr0.0003 / 10 updates | 7.21 | 6.01 |
| New context, lr0.0003 / 25 updates | 7.08 | 6.67 |
| New context, lr0.0003 / 100 updates | 8.49 | 10.45 |

**No setting passed the joint development criterion.** Without adaptation was
best by the predefined worst-control ratio. Some gradient updates reduced SBP
but increased DBP. The earlier unadapted context model scored 6.97 / 5.57 on
these same four people, so source-validation gains did not transfer here.
No seed replication or new final-target evaluation was triggered.

## Artifacts and reproduction

- [Source identities/roles](sources.json), [download/eligibility audit](results/source_cohort_audit.json).
- [Raw-record preparation](prepare.py), [training and personal adaptation](run.py).
- [Full metrics and all histories](results/), [experimental weights](models/), [weight hashes](models/manifest.json).
- [Verification](results/verification.json): 40 prediction files independently
  checked for MAE, Bias, SD and within5/10/15, then replayed from saved weights
  with maximum absolute difference **0.0**. The encoder was verified unchanged.

```sh
python temporal_pretraining/prepare.py --raw /path/to/raw --output /path/to/features
python temporal_pretraining/run.py --features /path/to/features \
  --individual-root /path/to/original_personal_mat --output /path/to/matched_run
python temporal_pretraining/verify_and_package.py --run /path/to/matched_run \
  --features /path/to/features --individual-root /path/to/original_personal_mat
```

Use the committed `sources.json`; do not refreeze a different folder listing.
The runner imports the preserved PairSE and context modules and their published
initializers from sibling folders. Context weights require the original SE
encoder/normalization and the person's known reference bank; they are not a
standalone zero-calibration waveform predictor. Raw ECG/PPG files and temporary
feature/prediction arrays remain local, not in Git.

The present evidence supports expanding timestamped source coverage and
examining representation quality before another end-to-end encoder experiment.
It does not establish that chronological training is better, or that more
personal gradient updates reliably outperform a simple offset.
