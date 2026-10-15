# PPG-only versus dual-channel CNN+SE — frozen 20261012

Three matched seeds: 42/43/44. Compare two inputs using the same SE-last
architecture, 38,296 parameters, identical initial weights at each seed,
preprocessing, optimizer and source partitions. PPG-only masks ECG to zero
inside the model, including the feature path used by PairSE. Keep two input
slots so the only experimental change is available information, not width.
This is a zeroed-channel ablation, not a smaller one-channel implementation.

Reuse verified dual-channel runs from the frozen V2 study; train three PPG+SE
runs from scratch. Full corrected source train =1085 people/390600 windows;
validation =192/69120; previously examined test =141/56400; AAMI subset=116/666.
These remain VitalDB-derived PulseDB subsets, not full combined PulseDB.
Use original AdamW .001, decay .001, batch256, noise .003, MSE standardized by
training labels, clip5, max40 epochs, patience8, plateau scheduler .5/patience3.
Epoch selection uses validation participant-macro average SBP/DBP MAE only.
Evaluate test/AAMI only after each checkpoint is fixed; no tuning from them.

Personalization uses the SAME PairSE reference-conditioned architecture for
both modalities. Fit a population pair head using original training windows
and validation pairs: original 20-epoch/5-patience Adam .001, batch2048 setup.
Reuse verified completed dual pair-head fits; independently fit PPG pair heads.
Do not use the tiny chronological-source subset from the preceding experiment.

Use every eligible window from the first20% of elapsed personal recording time
as the calibration bank. All later query windows and ten-second guard are
unchanged. Compare population, personal mean, own-modality additive bias,
unadapted PairSE and personal PairSE. Both arms get identical labels/window IDs.
Eight gradient settings: head-only/full-network × lr .0003/.001 ×25/100 steps,
Adam without weight decay/L2-SP; BN running stats and dropout held in eval mode.
All chronological early-window pairs enter each update; no later query labels.

Select a configuration SEPARATELY per modality on the same four development
people across all three seeds: minimize worst SBP/DBP MAE ratio against personal
mean and own-modality bias. Ties: mean MAE, then fixed original configuration
order. Save selections BEFORE evaluation of locked settings on the original
six targets and the previously evaluated sixteen (reported separately).
These target groups are already examined, so this is exploratory replication,
not untouched independent acceptance. Never revise settings from target scores.

Report all pooled/macro/per-person MAE, RMSE, signed Bias, sample error SD,
within5/10/15, tail errors and pressure strata, seed means/SD, paired
participant-bootstrap intervals and fraction of people improved. A win requires
both SBP and DBP macro MAE below BOTH controls, not merely lower mean of endpoints.
Also compare each personalized arm against the stronger of the two bias baselines.
Do not assume improved population MAE guarantees improved personalization.

Bounded inventory: 3 new waveform fits, 3 new population pair fits,
2 modalities ×3 seeds ×4 development people ×12 methods (3 controls+9 pair
settings), then 2×3×(6+16)×4 fixed target methods. Retain all failures and
old release. No new hyperparameter search outside this matrix; no email sending.
