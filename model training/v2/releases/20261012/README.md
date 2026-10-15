# Bad-Blood V2 research snapshot — 20261012

Version: **v2-20261012.1** (complete source, results and weights). This dated snapshot packages two V2 research outcomes.
V1, previous experiments and their negative results remain unchanged. Each
folder includes architecture/training source, results and trained weights;
`SHA256SUMS.json` identifies the packaged files. `version.json` records provenance.

| V2 outcome | Current implementation | Folder |
|---|---|---|
| Population architecture upgrade | High-resolution residual 1D CNN + final-block SE channel attention (`se_last`), 38,296 parameters | [01_population_cnn_se](01_population_cnn_se/) |
| Personalization research | Reference-conditioned CNN+SE; known reference ECG/PPG + BP and current ECG/PPG → current SBP/DBP; 46,906 parameters | [02_personalization](02_personalization/) |

## Result 1: population CNN+SE

On the matched V2 primary test partition (141 people, 56,400 windows), three-seed
mean participant-macro MAE was **12.39 / 8.00 mmHg** (SBP / DBP), versus
**12.59 / 8.36** for the unchanged high-resolution CNN architecture retrained on
the same corrected split. SE had the lowest observed primary-test MAE among the
six required three-seed architecture comparisons; the original validation-only
selection still slightly favored the unchanged CNN. These are different claims.
The PPG-only single-seed ablation is not that three-seed comparison.

## Result 2: reference-conditioned personalization

The model learns waveform-dependent pressure changes relative to known personal
reference measurements. Its fixed individual adaptation updates only the pair
head, using Adam lr=0.0003 for 25 steps. No hyperparameters changed for the new
participant follow-up.

| Cohort; participant-macro MAE, SBP / DBP | Personal mean | Bias correction | Paired CNN+SE |
|---|---:|---:|---:|
| Prior six, exploratory | 9.47 / 4.22 | 9.16 / 4.40 | **8.32 / 3.76** |
| 16 additional previously unused people | 17.41 / 9.71 | 12.82 / **7.48** | **12.67** / 7.61 |

**New-participant conclusion: the improvement over personal mean replicated,
but a stable advantage over simple bias correction did not.** Against bias,
the mean MAE differences were −0.15 / +0.13 mmHg; both paired participant
bootstrap intervals include zero. All three seeds beat personal mean, but none
beat bias correction on both SBP and DBP together. Do not describe the paired
model as consistently outperforming both controls.

The follow-up locked 21 candidates before reading their signals/results; 16
were eligible and five failed published temporal/data conditions. Their IDs
were absent from all prior train/validation/test/AAMI and exploratory lists.
All three methods used the identical early-20%-of-time calibration rule and
later-query rows: 652 reference windows and 2,310 later windows in total.
These are shorter recordings (3–96 reference windows per person), not a fixed
absolute label budget or a random sample of all PulseDB. Full per-person
results, exclusions, every metric and uncertainty are included.

Data source throughout: **PulseDB's VitalDB-derived recordings**, not the entire
PulseDB dataset. Pressure labels are derived from ABP. This snapshot does not
claim new ND cuff results or clinical-device certification.

## Version history

- `97aaeb3`: original completed and verified paired-SE six-person experiment.
- `981e15a`: new-participant protocol, candidate manifest and fixed settings
  committed before downloading/evaluating new signals.
- `v2-20261012`: initial documentation/results snapshot.
- `v2-20261012.1`: complete outcome packages including all published model weights and
  reproduction instructions. A singleton-MATLAB loader fix only identifies
  one-window files as ineligible; it does not change the model or calibration rule.

No raw waveforms or private participant records are included in this snapshot.
See [PulseDB attribution and reuse terms](../../../../data/PulseDB/LICENSE.txt).
