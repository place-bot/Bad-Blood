# Population models

**Retained architecture:** high-resolution residual 1-D CNN + final-stage
squeeze-and-excitation (SE). Both modalities have 38,296 stored parameters.

- Primary within this family: `weights/dual_population_seed42.pt` (ECG+PPG).
- Single-sensor alternative: `weights/ppg_population_seed42.pt`.
- Seeds 43 and 44 are included for reproducible three-seed comparisons.
- Architecture: [backbone](../implementation/shared/models.py),
  [modality mask](../implementation/modality_se/model.py).
- Training: [frozen source](../reproduction_source/attention_cnn/train.py),
  [PPG trainer](../reproduction_source/modality_se/train.py).
- Results: [full metrics](../metrics/comparison.json), [IEEE report](../report/Bad_Blood_V2_Final_Report_20261013.pdf).

Seed42 is the best validation checkpoint within each input family. This does
not mean it is the best test seed. The source model learns standardized scalar
SBP/DBP using MSE, AdamW, batch256, lr0.001, weight decay0.001 and validation
early stopping. All six fits ran nine epochs and retained epoch1. Neither
this selection nor the released model was retrained on test labels.

Inputs are normalized filtered ten-second signals at125Hz. SE is a feature
channel gate (48 -> 6 -> 48), not a gate choosing which raw sensor to trust.
The PPG model is trained without ECG and internally masks that channel in all
population and reference-encoding paths.
