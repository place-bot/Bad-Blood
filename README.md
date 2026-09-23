# Bad Blood

ECG/PPG blood-pressure prediction and a PulseDB-aligned collection workflow.

## Start here

- [English collection manual](data/Data%20Collection%20Manual/Data_Collection_Manual_EN.pdf)
- [中文采集手册](data/Data%20Collection%20Manual/Data_Collection_Manual_ZH.pdf)
- [Model and prediction commands](model%20training/PulseDB-based%20Model/README.md)
- [Public-data processing and examples](data/PulseDB/README.md)
- [ND data interface and blank templates](data/nd/README.md)

The manuals include a printable visit record and walk through twenty real
public-data segments and twenty sampling points. Paper records supply cuff
values and IDs; ECG/PPG waveforms are saved electronically.

## Current result

Four candidates were compared on a participant-disjoint validation set.
The selected model is a compact two-channel residual CNN without the initial
5x average-pooling stage. Its official calibration-free test MAE is
**12.46 mmHg SBP / 8.46 mmHg DBP** (144 participants, 57,600 segments).
The AAMI-designated subset MAE is **19.34 / 13.17 mmHg**. This is a research
prediction model, not a clinically validated blood-pressure monitor.

The training run used the complete author-provided **VitalDB training subset**,
not all 5.2 million segments in the full PulseDB release. ND adaptation and
Raspberry Pi execution measurements are subsequent work.

Actual ND participant records and photos belong in restricted research
storage. This public repository contains no ND participant measurements.
The 17.3 GiB source archive is downloaded from its official host rather than
duplicated in Git. PulseDB-derived examples retain CC BY-NC-SA 4.0 attribution;
see [data terms](data/PulseDB/LICENSE.txt).
