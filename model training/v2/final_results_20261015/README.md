# Bad Blood V2 - adopted personalization, 20261015

**The adopted personal method is PairSE tail adaptation with per-person,
per-endpoint hyperparameter selection.** Both PPG-only and ECG+PPG are retained.
The architecture is shared CNN + Squeeze-and-Excitation (SE) channel attention,
plus the reference-conditioned pressure-change head. Personal adaptation updates
the change head, final residual CNN block and its SE gate. Earlier blocks stay
frozen. SBP and DBP each use their own selected endpoint checkpoint.

Head-only adaptation and automatic head/tail scope selection were tested as
comparators. They are **not additional adopted models**. The experiment includes
a validation-selected zero-update source candidate; we preserve it as evaluated.

## Results on each person's later 20%

Thirty people; 8,882 personal development windows and 2,211 test windows.
Participant-macro MAE (mmHg), averaged across seeds 42, 43, 44, not an ensemble.
All methods use identical calibration and test records within each person.

| Input | Personal mean | Bias correction | Adopted tail adaptation | Reduction vs bias (SBP / DBP) |
|---|---:|---:|---:|---:|
| PPG only | 12.24 / 6.02 | 8.51 / 5.38 | **6.89 / 4.29** | 19.00 / 20.34% |
| ECG + PPG | 12.24 / 6.02 | 9.09 / 5.70 | **7.14 / 4.31** | 21.54 / 24.37% |

Both endpoints improve over both controls in the **overall 30-person mean in
every seed**. Tail adaptation also has lower mean MAE than head-only adaptation.
This does not mean every individual improves: seed-averaged simultaneous wins
on both pressures against both controls are 10/30 (PPG) and 13/30 (dual).
This cohort was evaluated previously; this is an exploratory comparison, not
new independent external validation. The 20261013 results used different cohorts
and a smaller support budget; do not interpret their score difference as a
matched-method improvement. The current comparison re-evaluates the old fixed
method on the same 80/20 data as the new methods.

## Read and run

- [Current IEEE report](report/Bad_Blood_V2_Final_Report_20261015.pdf) and [LaTeX](report/main.tex).
- [Updated teaching Beamer](beamer/Bad_Blood_V2_Models_Explained.pdf) and [source](beamer/Bad_Blood_V2_Models_Explained.tex).
- [Population models](01_pretrained/README.md): unchanged source CNN+SE models and training records.
- [Adopted personal method](02_personalization/README.md): six source initializers,
  360 endpoint-specific personal checkpoints, and usage.
- [All methods, metrics, seeds and people](metrics/RESULTS.md), [machine-readable results](metrics/summary.json).
- [Portable fitting and inference](implementation/personalize.py).
- [Frozen full experiment code](../rolling_personal_20261015/), including head and scope-selection controls.
- [Independent experiment verification](metrics/verification.json) and [release checks](release_verification.json).

Only publicly sourced participant identifiers and derived experiment artifacts
are included. Raw PulseDB signals must be obtained from the official source;
there are no Notre Dame student records in this release. Saved personal weights
belong to the named participant; fit a fresh source copy for a new person.
The [20261013 snapshot](../final_results_20261013/) and V1 remain preserved.
