# Bad Blood V2 - Final results package, 20261013

This is the consolidated **V2 final research snapshot**. It preserves the
completed experiments and distinguishes a retained model from a universal
winner. V1 and the October 12 snapshot remain unchanged.

## Two outcomes

1. **Population estimation: high-resolution residual CNN + final-stage SE.**
   Retain ECG+PPG as the primary population configuration within the SE family:
   it has the better three-seed validation score and AAMI-subset result.
   Retain PPG-only as the single-sensor alternative: its main-test MAE is
   slightly lower. Both have 38,296 parameters. The earlier architecture-wide
   validation comparison favored the plain CNN; SE led its observed test MAE.
   These are different rankings, not contradictory selection claims.
2. **Learned personalization: reference-conditioned PairSE.**
   A shared CNN+SE encodes the current and known reference signals. A learned
   pressure-change head uses reference BP and is adapted on all eligible early
   personal data. Both input modes outperform personal mean in SBP and DBP in
   all three seeds on both reported cohorts. The dual-input model also beats
   bias correction in both endpoints in all three seeds on the original six
   people. This joint advantage does not hold on the subsequent sixteen.

### Main comparison

Participant-macro MAE in mmHg, SBP / DBP; average of three separately evaluated
seeds, **not an ensemble**. The same support and query windows are used for all
personal methods. Both cohorts were previously evaluated.

| Cohort / method | ECG+PPG | PPG-only |
|---|---:|---:|
| Population main test (141 people) | 12.39 / 8.00 | 12.32 / 7.85 |
| Six people: personal mean | 9.47 / 4.22 | 9.47 / 4.22 |
| Six people: own-modality bias correction | 9.16 / 4.40 | 8.27 / 4.28 |
| Six people: PairSE | **8.32 / 3.76** | **8.19 / 3.85** |
| Sixteen people: personal mean | 17.41 / 9.71 | 17.41 / 9.71 |
| Sixteen people: own-modality bias correction | 12.82 / **7.48** | 12.74 / **6.89** |
| Sixteen people: PairSE | **12.67** / 7.61 | **12.47** / 7.51 |

The learned methods beat personal mean; neither establishes stable superiority
to bias correction across cohorts and endpoints. See the report for pooled
metrics, participant variation, seed results, and the negative residual follow-up.

## Everything in one place

- [IEEE report (PDF)](report/Bad_Blood_V2_Final_Report_20261013.pdf)
- [Editable LaTeX report](report/main.tex)
- [01 - Population models](01_pretrained/README.md): six trained checkpoints,
  architecture source, training configuration and full evaluation records.
- [02 - Personalization](02_personalization/README.md): six source PairSE
  checkpoints, 132 personal checkpoints, architecture and calibration code.
- [Runnable inference and adaptation](implementation/): local portable entrypoints.
- [Complete comparisons](metrics/): population, all personal participants/seeds,
  macro/pooled errors, threshold fractions and uncertainty.
- [Reproduction guide](REPRODUCE.md), [artifact checksums](manifest.json),
  [package verification](verification.json).

The representative population checkpoint for each modality is **seed 42**, chosen
by validation, not by test performance. A saved personal checkpoint belongs to
its original participant; for a new person start from a source PairSE checkpoint
and calibrate using only that person's early reference data.

## Data and interface

These models use the **VitalDB-derived partition of PulseDB**, not the complete
MIMIC-III + VitalDB collection and not ND participant data. Source training used
1,085 people / 390,600 ten-second segments; validation used 192 / 69,120.

Input: filtered, synchronized ECG/PPG at 125 Hz, shape `N x 2 x 1250`, with
per-segment, per-channel min-max scaling. Channel order is ECG then PPG.
PPG-only also accepts `N x 1 x 1250`; the loader inserts a zero ECG channel.
Output: `N x 2`, columns **SBP, DBP**, in mmHg. Current reference ABP is never an
inference input. Calibration additionally requires known early scalar BP labels.

```sh
python -m pip install -r implementation/requirements.txt
python implementation/predict.py \
  --checkpoint 01_pretrained/weights/dual_population_seed42.pt \
  --input normalized_windows.npy --output estimates.npy
```

Full raw signals are obtained from the official source, not copied into this
release. File names retain public dataset subject identifiers for reproducibility;
there are no ND participant records here. Reported error statistics and the
dataset's AAMI-subset name do not establish clinical-device certification.
