# Reference-conditioned CNN+SE experiment — October 12, 2026

Frozen before fitting. Inspired by Mekonnen et al. (2024),
https://pmc.ncbi.nlm.nih.gov/articles/PMC11467377/ . This is an ECG/PPG adaptation
of the paired-measurement idea, not a reproduction of their PPG architecture,
cohort or reported performance.

## Task and fixed data

Predict current scalar SBP/DBP from current ECG/PPG, a known earlier reference
ECG/PPG and its BP. A shared pretrained V2 SE encoder processes both waveforms.
No current/query BP enters prediction. Average predictions over every eligible
early reference; do not select references by closeness to the unknown BP.

Use exactly the existing population training partition (1,085 people, 390,600
windows), population validation partition (192 people, 69,120 windows), four
personal-development people and six previously examined target people. Use the
same three pretrained SE seeds 42/43/44. No new people or new data sources.
Every personal window ending in the first 20% of elapsed recording time is in
the reference/training bank. Keep the original later-query indices and guard.
Personal-mean and original-SE bias-correction controls use this identical bank.

Population arrays have no recoverable timestamps. Population pair training uses
two distinct windows from the same person; it does NOT pretend these pairs are
chronological or prove temporal separation. The personal experiment uses the
original timestamps: training pairs have a reference ending at least 10 seconds
before the current window begins; all final test references precede all queries.
All early rows remain in the bank; each must participate in at least one eligible
training pair. Pair expansion adds no independent BP observations.

## Model and population fitting

Let h be the 48-dimensional SE pooled feature and p the original normalized BP
prediction. A 100→64→32→2 SiLU MLP g receives [h_current-h_ref, h_ref, y_ref,
p_current-p_ref]. Features use population-training-only mean/SD, floor 0.01.
The prediction in standardized BP units is

`y_ref + p_current-p_ref + g(current,ref,y_ref) - g(ref,ref,y_ref)`.

The identity subtraction makes identical waveforms predict exactly the reference
BP. Zero-initializing the last MLP layer starts from pairwise bias correction.
This is a reference-conditioned nonlinear model, not a constant offset.

Freeze the original SE encoder while training the new population pair head from
cached waveform features. Each epoch visits every training current window once,
with a newly sampled distinct same-person reference. Adam lr=0.001, no weight
decay or L2-SP, batch 2,048, maximum 20 epochs, patience 5, gradient clip 5.
Select the epoch by mean SBP/DBP participant-macro MAE on fixed validation pairs;
each validation reference is also a distinct window from that validation person.
Both are unseen in population training. Three matched seeds run independently.

## Personalization and selection

Compare no additional personal fitting and eight settings:
pair-head-only or full-network; lr {0.0003,0.001}; steps {25,100}.
Adam with no weight decay/L2-SP. Keep BatchNorm running statistics fixed and
dropout disabled; full-network still updates all encoder weights and SE gates.
Every eligible earlier/current support pair enters every optimizer update.
Targets are the observed current early BP, not the later evaluation labels.
No gradient from the held-out later windows or their labels is permitted.

Select only on four development people, equal person and seed weights.
Primary score is the worst of four MAE ratios: SBP and DBP versus each control.
Tie-break by average MAE then fixed method order. Freeze the selection before
evaluating exactly one setting on the six targets (plus the two controls and
unadapted original SE). Do not retune on these outcomes. These targets were
examined in prior experiments, so the comparison remains exploratory.

## Reporting and verification

Report MAE, RMSE, signed bias, sample error SD, fractions within 5/10/15 mmHg,
person-level errors, paired differences and seed variation. Success requires
lower participant-macro SBP AND DBP MAE than BOTH controls. Also report how many
people improve; a marginal aggregate win is not called a decisive improvement.
Preserve negative results. Report labels are ABP-derived, not real cuff data.
Verify patient exclusion, distinct pair identity, chronological personal pairs,
reference-only labels at inference, fixed support/query hashes, saved-model
replay, complete experiment inventory and numerical metrics.
