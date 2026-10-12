# Bad Blood

ECG/PPG blood-pressure prediction and a PulseDB-aligned collection workflow.

## V2 results snapshot — 20261012

The dated [V2 release record](model%20training/v2/releases/20261012/README.md)
groups two research outcomes, with implementation, metrics and trained weights:

1. [Population CNN+SE architecture](model%20training/v2/releases/20261012/01_population_cnn_se/):
   final-block SE attention; observed matched-test MAE **12.39 / 8.00** mmHg.
2. [Reference-conditioned personalization](model%20training/v2/releases/20261012/02_personalization/):
   known reference waveform/BP plus current waveform. On **16 new people**, with
   unchanged settings and the same early-20%-of-time calibration rule, MAE was
   **12.67 / 7.61**, versus **17.41 / 9.71** for personal mean and **12.82 / 7.48**
   for bias correction. The new validation **did not confirm stable superiority
   over bias correction**. The earlier six-person improvement below is retained
   separately, not pooled with the new cohort.

All values above are SBP / DBP. The snapshot records the `20261012` version date,
provenance, checksums, full metrics and reproduction instructions. V1 is preserved.

## Reference-conditioned personalization — October 12

The new [paired CNN+SE experiment](model%20training/v2/paired_se/README.md)
uses known reference BP plus reference/current ECG/PPG. With identical early
calibration data and later query windows, it achieved **8.32 / 3.76 mmHg**
SBP/DBP MAE versus **9.47 / 4.22** for personal mean and **9.16 / 4.40** for
simple personal bias correction. Three seeds each improved over both controls
on these six previously examined target people. This is exploratory evidence;
the full report includes development results, uncertainty, individual outcomes
and metrics that did not improve. V1 and all earlier results remain preserved.

## V2 pretrained model highlight

**CNN + SE is V2's strongest pretrained model by observed primary-test MAE
among the six required three-seed architecture comparisons:** 12.39 mmHg SBP /
8.00 mmHg DBP, with 38,296 parameters. It adds a channel-attention gate to the
last residual block of the V1 high-resolution CNN. This is a test-result ranking;
the original validation-selected architecture was the unchanged CNN. PPG-only
had lower errors in a separate single-seed ablation, not a three-seed comparison.

The [full-support SE personalization experiment](model%20training/v2/se_personalization/README.md)
uses **all eligible windows in each person's first 20% of recording time** and
compares later predictions with both that person's reference mean and a simple
personal bias correction. Earlier V1 and V2 results are preserved.
The completed dense fine-tune did not beat those controls; the linked report
includes the negative result and literature-guided follow-up screening.

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

- [Computer-to-Pi deployment: prompt for Codex](deployment/OMEN-Codex-Prompt/README.md)
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
