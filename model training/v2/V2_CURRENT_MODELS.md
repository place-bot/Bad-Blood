# V2 current model inventory — 20261012

Both input modalities are retained. The population comparison is complete;
personalization does not yet satisfy stable superiority over both controls.

| Artifact | Location | Selection / interpretation |
|---|---|---|
| PPG-only CNN+SE | [population seed42](modality_se/models/ppg_population_seed42.pt) | Representative chosen by population validation; all three seeds retained |
| ECG+PPG CNN+SE | [population seed42](modality_se/models/dual_population_seed42.pt) | Representative chosen by population validation; all three seeds retained |
| Architecture and population inference | [implementation](modality_se/model.py), [loader](modality_se/predict.py) | Shared residual 1-D CNN, final SE, 38,296 parameters; only input availability differs |
| Prior reference-conditioned personalization | [PairSE](paired_se/model.py), [weights](modality_se/models/) | Retain as the learned personalization research candidate, not a universal baseline winner |
| Same-budget modality comparison | [stability report](modality_se/results/STABILITY.md) | Three seeds; original six and previous sixteen people reported separately |
| New ridge/kernel personal calibration | [experiment](residual_calibration/README.md) | Verified negative follow-up; does not replace previous models |
| Complete architecture/method review | [V2 review](residual_calibration/results/REVIEW.md) | Literature, selection, both cohorts, every seed, macro/pooled metrics and uncertainty |

Population structure: 10-second input at 125 Hz → Conv1d stem (8 channels) →
residual blocks (16, 32, 48 channels) → SE (48→6→48) → global average →
48→32→2 regressor → SBP/DBP in mmHg. PPG-only is trained without ECG;
the two-channel storage shape with an internal ECG mask preserves the matched
architecture in this experiment. It does not consume ECG information.

On the 16-person replication, three-seed participant-macro MAE (SBP/DBP) is:

| Method | PPG-only | ECG+PPG |
|---|---|---|
| Personal mean | 17.41 / 9.71 | 17.41 / 9.71 |
| Same-input bias correction | 12.74 / 6.89 | 12.82 / 7.48 |
| Prior PairSE | 12.47 / 7.51 | 12.67 / 7.61 |
| New development-selected calibration | 12.47 / 7.51 (retains prior) | 12.34 / 8.23 |

The new dual method improves SBP but worsens DBP. PPG development selection
retains the prior method. There is no single personalized method that wins
both endpoints against both controls across cohorts. Selecting a new SBP/DBP
hybrid after reading this target table would be post-hoc selection, not a new
validated model. All target cohorts here were evaluated previously.

For a scientific V2 wrap-up, report both the architecture result and this
personalization finding. If beating both controls remains mandatory, that
milestone is still open; do not label it achieved. No current release is replaced.
