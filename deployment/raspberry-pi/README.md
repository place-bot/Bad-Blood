# Raspberry Pi V1 deployment and acquisition foundation

This prototype runs the repository's unchanged V1 `cnn_full` model on a Raspberry Pi 4B. It replays 20 **public, recorded** PulseDB windows. Each five-second update is a separate ten-second ECG/PPG segment; segments may be from different participants. The numbers are model estimates from recordings, not live measurements or clinical readings. The bundled PulseDB source and its terms are under `data/PulseDB/`.

The new software foundation also supports paced continuous **raw-recording replay**, bounded acquisition buffering, acquisition timestamps, experimental raw quality checks and failure recovery. See [ACQUISITION.md](ACQUISITION.md) for the raw CSV/sidecar format, driver contract, tests and remaining hardware work. No physical sensor driver is implemented, and the new raw pipeline has not yet been measured on the Pi.

## One input contract for replay and a future device

`sources.py` defines the canonical `SignalWindow`. `PulseDBReplay`, the raw replay source and any later device adapter must emit this identical model input:

| Field | Required value |
| --- | --- |
| `waves` | NumPy `float32` array shaped `(2, 1250)`; channel 0 ECG, channel 1 PPG |
| `fs_hz` | 125 Hz |
| `preprocessing` | `pulsedb_filtered_minmax_v2`; filtered and independently min-max normalized to `[0, 1]` |
| `segment_id` | Nonempty source window identifier (may repeat on later replay cycles) |

`InferenceState` calls `validate_window()` and then **the same** released `Predictor.predict(waves[None], input_format="filtered")` for every source. The added batch axis makes the exact V1 `(1, 2, 1250)` input. The replay adapter discards the CSV's reference SBP/DBP and participant metadata; they never reach the model. The original V1 normalization is retained, without another bandpass filter.

A future device reader belongs behind `BufferedSource` in `acquisition.py`. It must collect verified synchronized ECG and PPG at 125 Hz and provide sample counters, monotonic acquisition timestamps and quality flags. Unsupported native rates require a separately validated resampler. Streaming windows must include acquisition provenance and raw-screening metadata; a mode label alone is insufficient. The browser distinguishes public replay, raw replay and connected-device input. No real device adapter exists yet, and sensor compatibility, calibration and clinical suitability remain unestablished. `test_deployment.py` uses a device stub to enforce the shared model entry; `test_acquisition.py` injects data gaps, resets, duplicates, clock faults and bad quality.

## Files

- `app.py`: verifies the deployment bundle and released model SHA-256/selection, loads `Predictor` once, starts the source worker and loopback HTTP server. Select raw replay with `--raw-input` plus `--session`, and profile inference threads with `--threads 1|2|4`.
- `bundle.py` / `bundle.json`: bind the released predictor, preprocessing, selection and weights to the deployed input contract.
- `acquisition.py`: bounded raw receiver, sample-based windows, explicit quality configuration and raw CSV replay.
- `sources.py`: canonical input contract and public-recording replay adapter.
- `service.py`: inference worker, state freshness, local API and static assets.
- `web/`: locally hosted source-neutral dashboard, with truthful input-source labeling and stale-state clearing.
- `test_deployment.py`, `test_acquisition.py`, `test_system.py`: source/API contracts, fault injection and a timed real-process synthetic replay test.
- `test_dashboard.js`: dashboard failure, freshness and polling tests using Node's built-in test runner.
- `run_pi_checks.py`: **Pi-only** release tests, all 20 original predictions and reference comparison, and 10 warmup/100 timed predictor calls. It records Pi hardware, software, commit, weight hash, p50/p95, and peak RSS to timestamped JSON/CSV under `results/`.
- `requirements-pi.lock`: exact versions from the tested Pi virtual environment; `requirements-pi.txt` lists the direct inference dependencies.

## Deployment sequence

1. Keep the existing Pi OS card. Eject it from the OMEN, insert it into the unpowered Pi, power the Pi, and confirm its current local IP. Avoid writing directly to Windows' `bootfs` partition.
2. Verify the Pi's actual model, `aarch64` architecture, Raspberry Pi OS, Python version, package availability, Wi-Fi/hotspot connectivity, and free space. Create a project-only venv on the Pi and install `deployment/raspberry-pi/requirements-pi.txt`. These are the verified **ARM64 CPU** NumPy, SciPy and PyTorch versions; do not install training-only dependencies. Capture a full `pip freeze` as environment evidence.
3. Copy this repository to the Pi over SSH and record the source commit. From the repository root, run `python -m unittest discover -s deployment/raspberry-pi -p 'test_*.py' -v`, then `python deployment/raspberry-pi/run_pi_checks.py` **on the Pi**. The latter must match all 20 `reports/example_predictions.csv` results within 0.01 mmHg. Keep its `results/pi_predictions_*.csv` and `results/pi_evidence_*.json` as Pi evidence; workstation runs do not substitute.
4. Start `python deployment/raspberry-pi/app.py` on the Pi. The server binds only `127.0.0.1:8000`. On the OMEN, open an SSH tunnel with `ssh -N -L 8000:127.0.0.1:8000 <user>@<verified-Pi-IP>` and browse `http://127.0.0.1:8000/`. The page has no CDN/cloud assets. Check an advancing sequence over at least 15 seconds, reopen the page, and confirm it advanced independently; stop the service and verify the dashboard clears the old readings.
5. Configure a non-root systemd service after runtime validation. Starting it manually is distinct from enabling automatic startup. Obtain user approval before `systemctl enable` or reboot, then test startup recovery if approved.

The local endpoint returns mode, model, source provenance, predictions, stage timing, waveforms and status. `/api/health` is 200 only when there is a fresh valid estimate; `/api/live` separately reports HTTP process liveness. For streaming inputs the 12-second freshness limit is based on acquisition time, not publication. Packet faults or a three-second packet watchdog can clear readings earlier. The browser also clears them when the Pi service/tunnel becomes unreachable, a response is delayed beyond freshness, or a source ends.

No credentials, private ND data, or reference BP labels are placed into API output. The service binds to Pi loopback and is reached only through the SSH tunnel. Do not expose this prototype to the public internet.
