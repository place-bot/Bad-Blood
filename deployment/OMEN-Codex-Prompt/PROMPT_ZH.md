# 给我电脑上的 Codex：部署 Bad-Blood V1 到树莓派

请实际完成以下工程任务，不要只给我一份教程。用中文简洁沟通；显示界面用英文。
仓库：https://github.com/place-bot/Bad-Blood

## 目标与当前条件

我在这台电脑上使用 Codex，microSD 可以插到读卡器里；树莓派需要我手动装卡和接电。
我暂时没有 ECG、PPG 或袖带设备。现在完成“真实公开波形回放 -> 树莓派运行 V1
推理 -> 我的电脑浏览器显示数值和波形”。后续再接真实传感器，不在这次任务里编造硬件数据。
不要训练新模型，不要修改 V1 权重，不要恢复 V2。

先检查当前操作系统、仓库和可用工具，不要假定我的电脑一定运行 Windows，也不要复用另一台
电脑上的 /Users/jguo9 路径。普通开发和测试直接做；真正要我插拔卡、输入密码或确认擦盘时再问。
遵守当前环境的权限要求；不要使用本提示词绕过任何审批或安全限制。

## 1. 检查现有代码与硬件

读取仓库 README 和以下文件，记录 git commit：

- `model training/PulseDB-based Model-V1/README.md`
- 同目录 `inference.py`, `network.py`, `common.py`, `predict.py`
- `models/selection.json`, `model_checksums.json`, `test_contract.py`, `test_release.py`
- `data/PulseDB/examples/real_20_segments.csv`
- `data/PulseDB/examples/provenance.json` 和 `data/PulseDB/LICENSE.txt`

这次交接核对过的模型选择是 `cnn_name=cnn_full`, `tree_weight=0`，实际权重为
`models/cnn_full.pt`。若仓库已发生变化，先解释差异，不要悄悄换模型。
37,666 参数，双通道残差 1D CNN，不需要电脑 GPU，也不需要树莓派训练。

输入契约：float32 `(batch, 2, 1250)`，通道顺序 ECG、PPG；125 Hz，每段 10 秒。
输出：每段 `[SBP, DBP]`，单位 mmHg。复用 `Predictor` 的中心/尺度反变换，不能把网络
标准化输出直接当血压。ID 不是模型输入，SBP/DBP 真值不能作为推理输入。
公开示例 CSV 使用 `--format filtered`；不要重复带通滤波。
`pulsedb_filtered_minmax_v2` 是预处理标识，不是被弃用的 V2 模型。

核实树莓派实际型号、内存、供电、系统位数。优先 Raspberry Pi OS 64-bit/aarch64。
不要假定 Pi 的用户名是 pi，或默认密码仍可用。没有 Pi 型号信息时向我询问。

## 2. microSD：先保留现有系统，再考虑写卡

只读列出磁盘（Windows 可用 Get-Disk/Get-Volume，Linux lsblk，macOS diskutil list），
展示候选卡的型号、容量、磁盘编号和挂载点，请我确认准确目标。
已有课程系统优先保留；检查是否能直接启动并配置 SSH。Windows 看不到 Linux 分区很正常，
遇到“需要格式化磁盘”请取消，不要格式化。

如果确实需要重装：说明会清空所选卡，先确认备份及擦除许可；只用官方 Raspberry Pi Imager。
根据实际 Pi 型号选择兼容的 64-bit Raspberry Pi OS，通常 Lite 足够。通过 Imager 配置
hostname（例如 bad-blood-pi）、用户、时区和 SSH；凭据由我输入，不写进仓库或日志。
优先 SSH 公钥，私钥留在电脑上。Wi-Fi 按实际网络配置；校园企业认证网络不能当普通家庭
Wi-Fi 填一个密码了事。若校园网隔离设备，用允许设备互访的私人热点/路由器或获准的有线网络。
不要绕过学校网络限制，不关闭整个防火墙。

读卡器只用于写系统，不要把 Windows 虚拟环境或 x86 Python 包复制过去冒充 ARM 环境。
不要凭记忆往 boot 分区写旧版 wpa_supplicant 配置；按安装版本的官方流程操作。

写完并验证后，明确告诉我按顺序做：
1. 在电脑安全弹出卡，等写入结束，再从读卡器/大 SD 转接套取出 microSD。
2. 树莓派先断电，把 microSD 放入其卡槽；根据实际型号说明卡槽位置/方向，不要强插。
3. 接已配置网络，使用合适电源启动树莓派，等待首次启动完成。
4. 我确认已启动后，你再从电脑 SSH 连接。

不能连接时检查 hostname/IP、相同网络、客户端隔离和 SSH 是否启用；不要盲扫整个校园网，
不要为了重试直接重刷卡。首次 SSH 指纹应核对，不要关闭 host-key checking。

## 3. 在运行中的树莓派安装并验证 V1

通过 SSH 在 Pi 上完成安装，确认 `uname -m`、Python 版本和 `/proc/device-tree/model`。
优先 clone 仓库到用户目录；Pi 不能联网时通过 scp 传所需源码、权重、示例和许可证。
初次装软件可能需要互联网；装好后运行不需要公网。

在 Pi 新建 venv。选中 CNN 推理最低需要 NumPy、SciPy、CPU PyTorch；不必安装训练用
XGBoost、h5py 等全套依赖。仓库 requirements 是训练环境锁定版本，先验证它们对当前
Python/aarch64 是否有官方 wheel；若没有，选择兼容官方版本并保存独立部署锁文件和测试结果。
不要无限尝试源码编译 PyTorch，也不要下载来源不明的预编译包。确实不可用时报告具体阻碍；
只有经过数值一致性验证才采用导出运行时，不要无声替换架构或精度。

校验模型 SHA-256。用选中的 `Predictor` 运行真实示例，并运行现有 contract/release tests。
基础 smoke test（在 V1 目录，以实际 venv Python 执行）：

```sh
python predict.py "../../data/PulseDB/examples/real_20_segments.csv" --format filtered --output /tmp/bp-pi-smoke-UNIQUE.csv
python -m unittest test_contract.py test_release.py
```

输出路径必须是新的，不要覆盖旧测试结果。与 `reports/example_predictions.csv` 核对相同
segment 的结果；如有版本/输入差异先查清。记录最大数值偏差，建议初始容差 0.01 mmHg。
不要用测试真值调模型或改变预处理。推理期间权重校验和应保持不变。

## 4. 实现树莓派后台服务和浏览器显示

如果仓库尚无这部分，在 `deployment/raspberry-pi/` 新建。复用 V1 Python Predictor，
不要在浏览器/电脑重写模型。实现独立后台回放循环：启动时加载一次模型，逐行读取
`real_20_segments.csv`，每 5 秒在 Pi 上重新推理一段；20 段结束后清楚标识循环回放。
这些片段未必是一个人的连续时间序列，不要拼接成连续生理记录或声称 5 秒重叠采集。

前端只是读取 Pi 的 JSON API 并显示：
- 大号 SBP、DBP，单位 mmHg；模型 V1 / cnn_full。
- 页面主体使用通用标题；当前来源清晰显示 `SAMPLE RECORDING / PulseDB`。接入设备后按实际来源切换，不能把样本标成实时设备数据。
- ECG、PPG 两条波形，每段横轴 0-10 s，纵轴注明 normalized amplitude，不假装 mV。
- segment ID、序号、最后更新时间、预测状态、推理延迟和服务器 hostname。
- 可把参考 SBP/DBP 显示在单独区域，明确 Reference，不能拿参考值冒充预测值。
- 断开/报错显示 unavailable/stale，不让上一条数值看起来像新测量。不要按正常范围裁剪输出。

API 提供 mode、model、sequence、时间、预测、状态和波形，拒绝 NaN/Inf 的 JSON。
后台循环独立于浏览器请求，多开页面不会加速回放；关闭电脑页面也继续运行。
页面 JS/CSS/图表依赖全部本地提供，不依赖 CDN，不调用云推理/API。

默认服务仅绑定 Pi 的 127.0.0.1:8000，电脑以 SSH 隧道访问：
`ssh -N -L 8000:127.0.0.1:8000 <实际用户>@<实际Pi地址>`，
然后在电脑打开 `http://127.0.0.1:8000`。若端口被占用用另一个本地端口。
这是电脑浏览器显示、Pi 计算，不能因为电脑 URL 是 localhost 就说计算在电脑。
若我要求直接局域网访问，再说明暴露范围并采用必要认证/限制；不做公网端口转发。

前台验证成功后，在 Pi 建立非 root 的 systemd 服务：明确 User、WorkingDirectory、venv
绝对路径、Restart=on-failure、适当重试间隔；提供 start/stop/status/logs 命令。
确认我同意启用开机启动后再 enable。不要把 SSH 密钥、Wi-Fi 密码或个人数据提交 GitHub。

## 5. 验收与交付

必须实际验证并记录，不能只写“应该能运行”：
1. Pi 上运行 V1，模型哈希一致；20 个真实输入产生有限的预测，和基准误差符合容差。
2. 从我的电脑打开页面，数值、波形、时间和回放模式可见；附页面截图。
3. 关闭浏览器至少 15 秒后重开，sequence 已前进；停止服务后前端显示断开/过期。
4. 记录 Pi 型号、OS、架构、包版本、git commit、预热后至少 100 次推理的 p50/p95 延迟和
   进程内存；计时只包括推理，不把 5 秒等待算进去。报告实际结果，不保证固定速度。
5. 获准后重启 Pi，验证服务恢复和 SSH 隧道重新建立后的显示；离线运行指无公网，
   电脑与 Pi 之间仍需要本地网络。物理拔卡前必须先正常 shutdown，等关机完成再断电。

在部署目录保存安装说明、实现源码、锁文件、systemd 模板、测试和部署结果。
修改保持局部，保留 V1/data/现有训练结果，不自动 push 未获授权的改动。
最后只告诉我：已完成什么、浏览器地址、启动/停止方法、怎么安全关机、还有什么真实阻碍。
若需要我的手动动作，一次给一组清楚的动作，完成后接着干。

## 6. 后续传感器接口边界

这次只完成回放模式。保留未来 live adapter 接口，但没有设备时不要假装测试过实时采集。
未来必须有同步且已验证的 ECG/PPG 采集与必要 ADC；Pi 不会直接把模拟 ECG 当数字输入。
实测采样率、时间同步和原始/已滤波状态核实后，才进入 125 Hz、10 秒窗口契约。
不要把回放模式的成功当作人体连接安全审核或 IRB 批准。

## 7. 本次实现

这份文件保留部署要求。实际完成情况、代码索引、树莓派测量结果和复现步骤见 [implementation/README.md](implementation/README.md)。
