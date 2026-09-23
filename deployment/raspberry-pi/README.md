# Raspberry Pi V1 replay deployment

This prototype runs the repository's unchanged V1 `cnn_full` model on a Raspberry Pi 4B. It replays 20 **public, recorded** PulseDB windows. Each five-second update is a separate ten-second ECG/PPG segment; segments may be from different participants. The numbers are model estimates from recordings, not live measurements or clinical readings. The bundled PulseDB source and its terms are under `data/PulseDB/`.

## One input contract for replay and a future device

`sources.py` defines `WaveformSource.next_window() -> SignalWindow`. The currently implemented `PulseDBReplay` and any later device adapter must emit this identical validated window:

| Field | Required value |
| --- | --- |
| `waves` | NumPy `float32` array shaped `(2, 1250)`; channel 0 ECG, channel 1 PPG |
| `fs_hz` | 125 Hz |
| `preprocessing` | `pulsedb_filtered_minmax_v2`; filtered and independently min-max normalized to `[0, 1]` |
| `segment_id` | Nonempty source window identifier (may repeat on later replay cycles) |

`InferenceState` calls `validate_window()` and then **the same** released `Predictor.predict(waves[None], input_format="filtered")` for every source. The added batch axis makes the exact V1 `(1, 2, 1250)` input. The replay adapter discards the CSV's reference SBP/DBP and participant metadata; they never reach the model. The original V1 normalization is retained, without another bandpass filter.

A future medical-device adapter belongs at this source boundary. It must collect or resample synchronized ECG and PPG to 125 Hz, construct ten-second windows, apply and **validate** the same filtering/min-max convention, and emit `SignalWindow` with `mode="device_live"`. Changing `source = PulseDBReplay(...)` to a verified adapter would not change the V1 model entry. The browser uses the mode to label recorded versus connected-device input accurately while reusing the same two waveform plots and prediction display. No real device adapter exists yet, and we have not established sensor compatibility, calibration, or clinical suitability. `test_deployment.py` uses a device stub to enforce the shared entry, and rejects incorrect order, shape, dtype, nonfinite data and preprocessing IDs.

## Files

- `app.py`: checks released model SHA-256 and selection, loads `Predictor` once, starts the source worker and loopback HTTP server.
- `sources.py`: canonical input contract and public-recording replay adapter.
- `service.py`: inference worker, state freshness, local API and static assets.
- `web/`: locally hosted source-neutral dashboard, with truthful input-source labeling and stale-state clearing.
- `test_deployment.py`: source/API contract tests.
- `run_pi_checks.py`: **Pi-only** release tests, all 20 original predictions and reference comparison, and 10 warmup/100 timed predictor calls. It records Pi hardware, software, commit, weight hash, p50/p95, and peak RSS to timestamped JSON/CSV under `results/`.

## Deployment sequence

1. Keep the existing Pi OS card. Eject it from the OMEN, insert it into the unpowered Pi, power the Pi, and confirm its current local IP. Avoid writing directly to Windows' `bootfs` partition.
2. Verify the Pi's actual model, `aarch64` architecture, Raspberry Pi OS, Python version, package availability, Wi-Fi/hotspot connectivity, and free space. Create a project-only venv on the Pi and install `deployment/raspberry-pi/requirements-pi.txt`. These are the verified **ARM64 CPU** NumPy, SciPy and PyTorch versions; do not install training-only dependencies. Capture a full `pip freeze` as environment evidence.
3. Copy this repository to the Pi over SSH and record the source commit. From the repository root, run `python deployment/raspberry-pi/test_deployment.py`, then `python deployment/raspberry-pi/run_pi_checks.py` **on the Pi**. The latter must match all 20 `reports/example_predictions.csv` results within 0.01 mmHg. Keep its `results/pi_predictions_*.csv` and `results/pi_evidence_*.json` as Pi evidence; OMEN runs do not substitute.
4. Start `python deployment/raspberry-pi/app.py` on the Pi. The server binds only `127.0.0.1:8000`. On the OMEN, open an SSH tunnel with `ssh -N -L 8000:127.0.0.1:8000 <user>@<verified-Pi-IP>` and browse `http://127.0.0.1:8000/`. The page has no CDN/cloud assets. Check an advancing sequence over at least 15 seconds, reopen the page, and confirm it advanced independently; stop the service and verify the dashboard clears the old readings.
5. Configure a non-root systemd service after runtime validation. Starting it manually is distinct from enabling automatic startup. Obtain user approval before `systemctl enable` or reboot, then test startup recovery if approved.

The local endpoint returns mode, model, source segment/sequence/time, predictions, inference latency, waveforms and status. `/api/health` is 200 only when there is a fresh estimate. On error or when the worker has not updated for 12 seconds, stale blood-pressure values and waveforms are removed. The browser also clears them when the Pi service/tunnel becomes unreachable.

No credentials, private ND data, or reference BP labels are placed into API output. The service binds to Pi loopback and is reached only through the SSH tunnel. Do not expose this prototype to the public internet.
