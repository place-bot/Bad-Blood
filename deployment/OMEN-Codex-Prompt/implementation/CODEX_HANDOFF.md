# 可复制给另一台 Codex 的任务

请用这个**完整仓库**复现 Bad Blood V1 的树莓派部署。先读
`deployment/OMEN-Codex-Prompt/PROMPT_ZH.md` 和同目录 `implementation/` 中的文件，
重点看 `REPRODUCE.md`、`SOURCE_MAP.md` 和 `STATUS.md`。实际运行代码只有
`deployment/raspberry-pi/` 一份；不要再复制出另一套实现。

先检查**眼前这台**树莓派的型号、系统、架构、Python、网络和磁盘空间，不要沿用原
OMEN 的 IP、用户名或环境假设。保留 microSD 现有内容，除非我另行允许清空。
按照锁定依赖安装 Pi 环境，在**树莓派上**运行部署测试和 `run_pi_checks.py`，
生成新的 Pi 结果，不能拿原机或电脑上的数字充数。启动只绑定 Pi 本机地址的服务，
从电脑建立 SSH 隧道，核对持续更新的预测值、ECG/PPG 输入波形、真实的数据来源
标识，以及断连/过期时清空旧数值。电脑只负责显示，V1 模型必须在 Pi 上推理。

当前来源是记录好的 PulseDB 样本。未来设备必须提供同一经过验证的
`SignalWindow` 格式；目前没有真实设备测试或临床验证。最后分别说明已完成项、
实际测量结果和剩余问题。不要把密码、SSH 密钥、含本机信息的原始结果或私人
数据上传 GitHub。根据原 prompt 的要求，未经我明确同意，不要启用开机自启或
重启树莓派。
