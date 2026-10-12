# Endpoint-specific personal residual calibration

See [architecture, results and V2 review](results/REVIEW.md).
The bounded experiment is complete. It does not establish joint superiority to
personal mean and additive bias correction, so no existing model is replaced.
Both PPG-only and ECG+PPG population models remain in `../modality_se/models`.

- `PROTOCOL.md`: fixed candidate grid, selection rules and primary literature.
- `model.py`: personal linear/kernel residual fitting and forward support CV.
- `run.py`: three-seed development selection, locked target comparison, replay.
- `report.py`: reproducible results and architecture review export.
- `results/comparison.json`: complete development candidates and target metrics.
- `models/`: self-contained selected pipelines for the evaluated participants.

The saved predictor consumes normalized N×2×1250 windows. PPG-only ignores the
ECG slot. It returns SBP and DBP in mmHg. New users require fitting on their own
reference data: evaluated participant checkpoints are not generic personal models.

Do not retune against the two previously examined target cohorts or interpret
this exploratory replication as new independent confirmation. A known weaker
new method remains archived as an experiment, not promoted as the best model.
