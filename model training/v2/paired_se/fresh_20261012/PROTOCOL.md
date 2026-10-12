# V2 additional-participant validation — 20261012

Freeze this protocol before downloading or evaluating new participant signals.
Reuse the completed population pair checkpoints and the existing selected
configuration: CNN+SE, pair-head-only Adam, lr 0.0003, 25 full-pair updates,
MSE, no weight decay/L2-SP, frozen encoder/BatchNorm and disabled dropout.
Seeds 42/43/44 are repeated fits, not an ensemble. No model selection or tuning.

## Cohort selected without outcomes

Use every unique eligible subject listed in the initial public author-hosted
PulseDB_Vital folder page on 2026-10-12, excluding the union of ALL historical
population train/validation/test/AAMI subjects and all 31 prior exploratory
subjects. This is a convenience sample of that directory page, not a random
sample of all PulseDB or an external dataset. Freeze the IDs and download URLs
in candidate_manifest.json before reading signals or pressure labels. No
replacement based on prediction performance; report all technical exclusions.

## Identical calibration rule and controls

Load the original timestamped per-person MATLAB files. Same filtered ECG_F/PPG_F,
125 Hz, 10 seconds, independent per-channel/window min-max normalization.
Use ALL windows ending within the first 20% of elapsed recording time for each
method. Queries start at least 10 seconds after that boundary. Within-support
pairs require reference end + 10 seconds <= current start. Require every support
row to participate in a pair and at least two query rows. Check finite non-flat
signals, scalar valid BP labels, increasing timestamps and consistent identity.
Do not exclude by BP range, model error, calibration drift or inferred benefit.
The time-fraction budget is identical, not a fixed number of BP observations;
new shorter recordings can have fewer calibration observations. Report counts.

Compare the fixed personalized pair model with personal mean, original-SE bias
correction and unadapted SE, using identical support/query rows for each seed.
No source retraining. Reuse fit_personal and predict_bank without modification.
Current query BP is used only after prediction for scoring.

## Analysis and release

Primary: participant-macro SBP AND DBP MAE lower than BOTH controls. Report all
MAE, RMSE, signed bias, sample error SD, within-5/10/15 fractions and per-person
MAE; include seed variation and paired participant bootstrap intervals (10,000
draws, seed 20261012, average matched seeds before resampling participants).
Report failure as well as success; never pool this new cohort with the prior six
to obscure a failed replication. Replay saved personal models, independently
recalculate metrics and controls, verify no identity overlap and hash inputs,
checkpoints, configuration and predictions. Publish reproducible source, model
weights and summaries, not participant waveforms. These are public ABP-derived
labels, not ND cuff data or clinical device validation.
