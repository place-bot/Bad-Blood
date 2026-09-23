# Reproduce the Pi V1 dashboard

This procedure starts from a Raspberry Pi 4B with a working **64-bit**
Raspberry Pi OS, network access, SSH login and an already bootable microSD
card. It does not re-image or erase the card. Use the Pi's own username and
current IP address; the original operator's credentials and hotspot details
are intentionally absent. Clone the **whole** repository because the released
model weight and public input CSV are outside this folder.

## 1. Check the destination Pi

On the Pi over SSH:

```sh
cat /proc/device-tree/model
uname -m
python3 --version
df -h .
```

The tested machine reported Raspberry Pi 4 Model B Rev 1.5, `aarch64` and
Python 3.13.5. Package availability may differ on another OS/Python version;
do not substitute benchmarks from a laptop. Check that the Pi has enough disk
space and can reach the package index before installation. If `python3 -m venv`
is unavailable, install the OS `python3-venv` package using the Pi's
normal package manager.

## 2. Get code and install the Pi environment

On a Pi with GitHub/package access:

```sh
git clone https://github.com/place-bot/Bad-Blood.git
cd Bad-Blood
git rev-parse HEAD
test -f 'model training/PulseDB-based Model-V1/models/cnn_full.pt'
test -f data/PulseDB/examples/real_20_segments.csv
python3 -m venv .venv
.venv/bin/python -m pip install -r deployment/raspberry-pi/requirements-pi.lock
.venv/bin/python -m pip freeze
```

The lock file records the tested Pi versions, including transitive packages.
`requirements-pi.txt` lists just the direct inference dependencies. If the
Pi has no Internet but the computer does, an agent can transfer the **entire
repository** and compatible ARM64 wheel files over SSH; the copied code must
retain the model, manifest and example data. Do not use the computer's
Windows/AMD64 Python environment on the Pi.

## 3. Test on the Pi and keep local evidence

From the repository root on the Pi:

```sh
.venv/bin/python deployment/raspberry-pi/test_deployment.py
.venv/bin/python deployment/raspberry-pi/run_pi_checks.py
```

The first command has six deployment/contract tests. The second runs the
released V1 contract and release suites, predicts all 20 tracked public
windows, compares them with the released output within **0.01 mmHg**, then
times 10 warmup and 100 predictor calls. It writes timestamped JSON and CSV
under `deployment/raspberry-pi/results/` **on the Pi**. That folder is ignored
by Git because raw CSVs contain public-source participant identifiers and
machine-local evidence. The sanitized result from the original Pi is in
[EVIDENCE_SUMMARY.json](EVIDENCE_SUMMARY.json); a new Pi must produce its own
results. Latency can vary; do not require the same p50/p95 values.

## 4. Run and view

Keep one Pi SSH session open with:

```sh
.venv/bin/python -u deployment/raspberry-pi/app.py
```

The app checks the model SHA-256, serves only on the Pi's loopback address
`127.0.0.1:8000`, and processes a new recorded window about every five
seconds. In a **separate computer terminal**, replace the placeholders and
open the local SSH tunnel:

```sh
ssh -N -L 8000:127.0.0.1:8000 <pi-user>@<current-pi-ip>
```

Open `http://127.0.0.1:8000/` on that computer. `localhost` is the computer
end of the tunnel; the Pi still computes every estimate. The ECG/PPG traces
are **model input**, while SBP/DBP are Pi model output. Confirm that the
source reads `SAMPLE RECORDING / PulseDB`, the sequence and update time
advance, and the displayed waveforms change. Close the browser for at least
15 seconds and reopen it: sequence should have advanced without the browser.
Press `Ctrl+C` in the Pi app terminal to stop it; the page should clear old
readings and show unavailable. Restart the app to recover.

On the Pi, `curl -fsS http://127.0.0.1:8000/api/health` returns a fresh status
only while the service is producing estimates. If the computer page is
unavailable, check Pi power/network, its current IP, the app process and the
SSH tunnel before changing code. No public router port forwarding is needed.

## 5. Optional persistent service, after manual validation

The tracked [`bad-blood-v1.service.template`](../../raspberry-pi/bad-blood-v1.service.template)
is a **system** unit that runs the process as an ordinary Pi user. Render a
copy with `__PI_USER__` replaced by the actual Pi user, `__REPO_PATH__` by
the absolute repository directory, and `__VENV_PYTHON__` by its absolute
`.venv/bin/python` path. Review that copy and stop the manually launched app
before installing it:

```sh
sudo install -m 0644 <reviewed-unit-file> /etc/systemd/system/bad-blood-v1.service
sudo systemctl daemon-reload
sudo systemctl start bad-blood-v1.service
systemctl status bad-blood-v1.service --no-pager
journalctl -u bad-blood-v1.service -n 50 --no-pager
```

Normal controls are `sudo systemctl stop bad-blood-v1.service`, `sudo
systemctl start bad-blood-v1.service`, and `systemctl status
bad-blood-v1.service`. **Starting** is different from **enabling at boot**.
The original prompt requires the operator's explicit agreement before
`sudo systemctl enable bad-blood-v1.service` or a Pi reboot. This deployment
has not installed or enabled the unit, and boot recovery has not been tested.
Never ask the operator to paste a password into a chat or repository file.

For card removal, run `sudo systemctl poweroff` on the Pi and wait for the
activity light to stop before disconnecting power; only then remove the card.

## 6. Future acquisition device

The current `PulseDBReplay` is only a source adapter. A future device adapter
must produce synchronized ECG/PPG in the exact `SignalWindow` contract in
[SOURCE_MAP.md](SOURCE_MAP.md), with validated collection, filtering, sample
rate, timing and normalization. Then `app.py` can select that adapter; the
same `InferenceState` and front end can display its input waveforms and V1
estimates with the `device_live` source label. This has **not** been tested
with a real device and is not medical or clinical validation.
