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

### 中文说明

我们的目标是做一个基于树莓派、完全本地推理的小型化血压估计系统。
模型在电脑上训练，之后部署到树莓派。树莓派连接真实 ECG／PPG 传感器，
完成信号采集、滤波和模型推理，计划利用最近 10 秒的信号，每 5 秒更新一次预测。

不必另外购买屏幕：我的电脑可以通过同一局域网，用浏览器查看树莓派提供的
波形和预测结果。电脑只负责显示和操作，计算仍在树莓派上，不需要互联网或
云端服务器。显示端断开后，树莓派按设计继续采集和推理，重新连接后查看结果。

这样可以减少设备对外部电脑算力和网络的依赖，让推理过程中的生理数据留在
本地。袖带血压计用于提供训练、验证及可能的校准参考，不是每次预测所需的输入。
目前模型训练和公共数据评估已完成，接下来重点是传感器接入、树莓派部署、
浏览器界面和稳定的现场演示，并记录准确性、推理延迟和内存占用。

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
The AAMI-designated subset MAE is **19.34 / 13.17 mmHg**.

The training run used the complete author-provided **VitalDB training subset**,
not all 5.2 million segments in the full PulseDB release. ND adaptation and
Raspberry Pi execution measurements are subsequent work.

Actual ND participant records and photos belong in restricted research
storage. This public repository contains no ND participant measurements.
The 17.3 GiB source archive is downloaded from its official host rather than
duplicated in Git. PulseDB-derived examples retain CC BY-NC-SA 4.0 attribution;
see [data terms](data/PulseDB/LICENSE.txt).
