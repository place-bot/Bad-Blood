# V2 outcome 2 — reference-conditioned personalization — 20261012

Current model: **paired/reference-conditioned CNN+SE**, 46,906 parameters;
38,296 in the shared CNN+SE encoder/predictor and 8,610 in the pair head.

## Everything in this folder

- [Pair architecture](source/paired_se/model.py), [CNN+SE backbone](source/shared/models.py).
- [Population pair training and personal adaptation](source/paired_se/run.py).
- [Original frozen protocol](source/paired_se/PROTOCOL.md).
- [New-participant fixed-method validation](source/paired_se/fresh_20261012/validate.py), [protocol](source/paired_se/fresh_20261012/PROTOCOL.md), [candidate manifest](source/paired_se/fresh_20261012/candidate_manifest.json).
- [Models](models/): three population-pair initializers, prior six-person models
  and [48 new personal checkpoints](models/new_participants/) (16 people × 3 seeds).
- [Prior six-person results](results/prior_six/RESULTS.md).
- [New-participant results](results/new_participants/RESULTS.md), including every
  metric, person-level comparisons, seed variation and exclusions.
- [Checksums](SHA256SUMS.json).

## Method

Shared CNN+SE encodes reference and current ECG/PPG into 48 features each.
A 100→64→32→2 SiLU head uses feature difference, reference features, reference
SBP/DBP and the original predictor's pressure difference. The predicted BP is
reference BP + original predicted change + a learned nonlinear correction.
An identity subtraction makes identical current/reference signals return the
reference pressure; zero initialization equals simple bias correction when
averaged over all references.

The pair head is first trained on same-person public-data pairs (1,085 training
people; 192 validation people), with the SE encoder fixed. Then each person's
head is adapted using all eligible first-20%-of-time records: Adam lr=0.0003,
25 full-pair gradient updates, standardized-target MSE, no L2-SP or weight
decay. This setting was selected on four development people before either
reported target evaluation; full-network alternatives were tested there too.

Prediction pairs each current waveform with every known early reference and
averages uniformly. Current query BP never enters prediction or adaptation.
This predicts current scalar SBP/DBP from observed current signals, not future
waveforms. Reference banks must be retained locally alongside personal models.

## Results and current conclusion

| Participant-macro MAE, SBP / DBP | Personal mean | SE + bias | Paired model |
|---|---:|---:|---:|
| Prior six (exploratory) | 9.47 / 4.22 | 9.16 / 4.40 | **8.32 / 3.76** |
| 16 new people, fixed method | 17.41 / 9.71 | 12.82 / **7.48** | **12.67** / 7.61 |

**The new cohort supports improvement over personal mean, but does not establish
a stable advantage over simple bias correction.** Against bias, SBP improved by
only 0.15 mmHg and DBP worsened by 0.13; both paired bootstrap intervals include
zero. Eight of 16 people improved in SBP and six in DBP versus bias, compared
with 13 and 12 versus personal mean. No post-result tuning was performed.

The new cohort has 652 calibration windows and 2,310 later queries. All methods
use identical rows and the same 20%-of-time rule, but records are shorter than
the original six, with 3–96 reference windows each. This convenience sample
was chosen by directory availability and prior-identity exclusion before outcome
inspection, not by model errors. Five of 21 locked candidates were excluded by
predefined data/time eligibility. See the complete audit rather than interpreting
the old and new scores as results on interchangeable populations.

All labels in these experiments are ABP-derived, not new ND cuff observations.
The personal weights correspond to the named public-data individuals; they are
not universal personal calibrations for other people.

## Run inference from this folder

Inputs: normalized filtered ECG/PPG arrays of shape `N × 2 × 1250`, float32,
125 Hz; reference BP array `K × 2`, SBP/DBP in mmHg. Use the matching person's
own earlier reference bank. Only load trusted checkpoints.

```sh
python source/paired_se/predict.py \
  --checkpoint models/new_participants/vitaldb_p000006_seed42.pt \
  --current /path/to/current.npy --reference /path/to/early_reference.npy \
  --reference-bp /path/to/early_bp.npy \
  --output /path/to/predictions.csv --device cpu
```

## Reproduce the locked new-participant evaluation

```sh
python source/paired_se/fresh_20261012/validate.py \
  --data /path/to/public_mat_files \
  --output /path/to/new_validation_run --device mps
```

The fixed manifest downloads missing author-hosted MATLAB files and excludes
all previous identities; it does not search new hyperparameters. Saved units
are checksum-checked and replayed on resume. Raw waves and prediction arrays
are external; complete metrics, models and checksums are included here.
[PulseDB attribution and license](../../../../../data/PulseDB/LICENSE.txt).
