# Source and data map

Paths are relative to the repository root. Links below point to the **single
runnable copy**; copying these files into this folder would make two versions
that could silently diverge.

| File | Role |
| --- | --- |
| [`deployment/raspberry-pi/app.py`](../../raspberry-pi/app.py) | Pi entry point; checks released weight hash/model selection, loads `Predictor`, starts input source and local HTTP service |
| [`deployment/raspberry-pi/sources.py`](../../raspberry-pi/sources.py) | `SignalWindow`, validation, `WaveformSource` protocol and current 20-segment PulseDB source |
| [`deployment/raspberry-pi/service.py`](../../raspberry-pi/service.py) | Source-independent inference loop, fresh/stale state, loopback JSON API and static-file server |
| [`deployment/raspberry-pi/web/index.html`](../../raspberry-pi/web/index.html) | Source-neutral page structure |
| [`deployment/raspberry-pi/web/app.js`](../../raspberry-pi/web/app.js) | Fetches Pi state, draws input waveforms, displays predictions and truthful source labels; clears stale values |
| [`deployment/raspberry-pi/web/style.css`](../../raspberry-pi/web/style.css) | Responsive dashboard styling |
| [`deployment/raspberry-pi/test_deployment.py`](../../raspberry-pi/test_deployment.py) | Contract, source-mode, real predictor and local API tests |
| [`deployment/raspberry-pi/run_pi_checks.py`](../../raspberry-pi/run_pi_checks.py) | Pi-only release checks, 20-row comparison, 10 warmups/100 predictor timings, JSON/CSV evidence |
| [`deployment/raspberry-pi/requirements-pi.txt`](../../raspberry-pi/requirements-pi.txt) | Direct inference dependencies |
| [`deployment/raspberry-pi/requirements-pi.lock`](../../raspberry-pi/requirements-pi.lock) | Exact package versions from the tested Pi environment |
| [`deployment/raspberry-pi/bad-blood-v1.service.template`](../../raspberry-pi/bad-blood-v1.service.template) | Non-root systemd unit template; installation/enablement are separate actions |
| [`model training/PulseDB-based Model-V1/inference.py`](<../../../model training/PulseDB-based Model-V1/inference.py>) | **Unmodified** released V1 predictor used on the Pi |
| [`model training/PulseDB-based Model-V1/models/cnn_full.pt`](<../../../model training/PulseDB-based Model-V1/models/cnn_full.pt>) | Released CNN weight file, hash checked at startup |
| [`model training/PulseDB-based Model-V1/model_checksums.json`](<../../../model training/PulseDB-based Model-V1/model_checksums.json>) | Release hash manifest |
| [`model training/PulseDB-based Model-V1/models/selection.json`](<../../../model training/PulseDB-based Model-V1/models/selection.json>) | Release selection; `cnn_full`, tree weight zero |
| [`model training/PulseDB-based Model-V1/reports/example_predictions.csv`](<../../../model training/PulseDB-based Model-V1/reports/example_predictions.csv>) | Released 20-row output for comparison; **not** fed to the model |
| [`data/PulseDB/examples/real_20_segments.csv`](../../../data/PulseDB/examples/real_20_segments.csv) | Tracked public, already filtered sample windows used by the current source |
| [`data/PulseDB/LICENSE.txt`](../../../data/PulseDB/LICENSE.txt) | Source data terms |

The model input contract is `float32` `(2, 1250)`, ECG then PPG, sampled at
125 Hz over ten seconds, filtered and independently min-max normalized to
`[0, 1]`, preprocessing ID `pulsedb_filtered_minmax_v2`. The service adds a
batch axis and calls the same released `Predictor.predict(...,
input_format="filtered")` for either source mode. The current source mode is
`recorded_pulsedb_replay`; `device_live` is reserved for a future verified
adapter. Mode changes page labeling, not predictor input.

The local API is `GET /api/state` (prediction, waveforms, mode, sequence,
timestamps and status) and `GET /api/health` (200 only while fresh). It binds
to `127.0.0.1:8000` on the Pi. The computer reaches it through an SSH tunnel.
There is no cloud inference or CDN dependency.
