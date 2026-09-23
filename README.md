# Bad Blood

ECG/PPG blood-pressure prediction and a PulseDB-aligned collection workflow.

## Embedded system and deployment plan

Our goal is a compact, locally running blood-pressure estimation system.
We train the model on a computer, then deploy the trained weights to the
Raspberry Pi for on-device inference. Model training does not need to run
on the Pi.

The planned live pipeline is:

**ECG/PPG sensors → Raspberry Pi acquisition and filtering → local model inference → browser dashboard**

The Pi will handle signal acquisition, preprocessing and prediction. The
dashboard will show live waveforms and estimated SBP/DBP, using the latest
10 seconds of signals to update the estimate every 5 seconds.

A computer on the same local network can open the dashboard in its browser.
The computer serves only as the display and control terminal; inference stays
on the Pi. This setup does not require a separate screen, an internet connection
or a cloud server. If the display computer disconnects, the Pi is intended to
continue acquisition and inference, with results visible again on reconnection.

Local processing keeps physiological signals on the device during inference
and removes dependence on a separate computer's processing power. The hardware
setup includes the Pi, sensors, acquisition interfaces, safe connections and
power. The upper-arm cuff supplies reference measurements for training,
evaluation and any calibration, rather than an input required for every
prediction.

Model training and source-data evaluation are complete. The next steps are
sensor integration, Raspberry Pi deployment, the browser dashboard and a stable
live demonstration. We will measure prediction error, inference latency and
memory use, then compare transfer-learning approaches when ND data are available.

## Start here

- [English collection manual](data/Data%20Collection%20Manual/Data_Collection_Manual_EN.pdf)
- [中文采集手册](data/Data%20Collection%20Manual/Data_Collection_Manual_ZH.pdf)
- [V1: model comparison and prediction commands](model%20training/PulseDB-based%20Model-V1/README.md)
- [Public-data processing and examples](data/PulseDB/README.md)
- [ND data interface and blank templates](data/nd/README.md)

The manuals include a printable visit record and walk through twenty real
public-data segments and twenty sampling points. Paper records supply cuff
values and IDs; ECG/PPG waveforms are saved electronically.

## V1 result

Four candidates were compared on a participant-disjoint validation set.
The selected model is a compact two-channel residual CNN without the initial
5x average-pooling stage. Its official calibration-free test MAE is
**12.46 mmHg SBP / 8.46 mmHg DBP** (144 participants, 57,600 segments).
The AAMI-designated subset MAE is **19.34 / 13.17 mmHg**.

The training run used the complete author-provided **VitalDB training subset**,
not all 5.2 million segments in the full PulseDB release. ND adaptation and
Raspberry Pi execution measurements are subsequent work.

Actual ND participant records and photos belong in restricted research
storage. This public repository contains no ND participant measurements.
The 17.3 GiB source archive is downloaded from its official host rather than
duplicated in Git. PulseDB-derived examples retain CC BY-NC-SA 4.0 attribution;
see [data terms](data/PulseDB/LICENSE.txt).
