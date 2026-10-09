# Wearable acquisition foundation

Implemented October 9, 2026. The released network, weights and preprocessing code are unchanged. This layer provides deterministic raw-recording replay and a driver interface; no physical sensor driver or validated physiological signal-quality algorithm is supplied.

## Run the two recorded sources

The default still replays the 20 released, filtered PulseDB segments:

```sh
python deployment/raspberry-pi/app.py --threads 4
```

To replay a continuous raw recording:

```sh
python deployment/raspberry-pi/app.py \
  --raw-input /restricted/session/raw.csv \
  --session /restricted/session/session.json \
  --threads 1
```

Use the existing loopback/SSH dashboard procedure in [README.md](README.md). The raw source is always labeled `RAW RECORDING / replay`. It runs at the recording's original speed, without looping. It cannot be relabeled live through the CLI. At least ten seconds are needed for a complete window; a recording longer than ten seconds is useful for viewing results before EOF clears them. Startup includes model loading and collection of the first ten seconds.

Raw CSV headers must be exactly `time_s,ecg_raw,ppg_raw`. The time column starts at zero and advances by 0.008 seconds. Values must be finite, synchronized, native-unit measurements. Data are streamed in one-second chunks. Other rates, gaps and already-normalized/filtered data are not silently converted into this format. Do not pass the PulseDB example CSV or converter output to `--raw-input`.

Copy [session.example.json](session.example.json) into restricted storage and fill in device identity, firmware, placement, units and sensor-specific quality limits. The null limits deliberately fail validation until configured. `adc_min`, `adc_max` and `minimum_peak_to_peak` each contain ECG then PPG limits in native units. Thresholds are engineering configuration, not clinical acceptance criteria. Keep original recording metadata, gain, LED settings, clock validation and reference labels in the restricted session record; only a nonidentifying session token enters the dashboard. Do not commit participant recordings or completed sidecars.

## Data flow and bounded behavior

`RawCSVReader.read(timeout)` → background `BufferedSource` receiver → `WindowBuffer` → one-window mailbox → quality checks → existing `common.from_raw_125()` → canonical `SignalWindow` → released `Predictor`.

- The ring retains at most 1,250 samples per channel after each push. Windows are ten seconds long, with a 625-sample/five-second stride, including when boundaries fall inside packets.
- The mailbox keeps only the latest complete window. An overwritten unconsumed window increments `dropped_windows`; queues cannot grow indefinitely while inference is slow.
- Gaps, duplicates, session changes, sequence resets, clock discontinuities or quality flags discard accumulated samples and pending windows. A new full contiguous ten seconds is required after a fault.
- Packet flags and raw ADC saturation can invalidate displayed results before a full new window is available. Full-window low-amplitude checks run before normalization.
- The receiver has a three-second packet watchdog. Consumption uses bounded half-second waits. A blocked or disconnected reader leaves the API responsive and removes its estimate; a faulty driver is never restarted in unbounded new threads. Unexpected reader failure requires service restart. Recoverable continuity/quality faults rebuild the window automatically when valid packets resume.
- EOF is explicit and clears readings. One pending final window can be consumed before the source reports EOF. Stopping the service closes the source and handles SIGINT/SIGTERM.

## Future device-reader contract

A hardware reader must implement `read(timeout) -> RawPacket` and `close()` that unblocks reads. Idle reads raise `TimeoutError`; terminal failures raise `SourceUnavailable`. An optional `release()` closes reader-owned resources on its receiver thread. Pass the reader, a reviewed `QualityPolicy` and the unchanged raw adapter to `BufferedSource(..., mode="device_live")` only after integration and clock verification.

`RawPacket` requires:

| Field | Contract |
|---|---|
| `session_id` | Nonempty token; change on reset or acquisition-configuration change |
| `sequence` | Nonnegative integer, increments by one per packet |
| `first_sample` | Nonnegative integer native-grid sample index, strictly contiguous within a run |
| `start_monotonic` | Sample acquisition start mapped to the host monotonic clock; never substitute packet reception time |
| `waves` | Finite real numeric array `(2, N)`, `1 <= N <= 250`, ECG first, PPG second |
| `fs_hz`, `channels` | Exactly 125 and `("ECG", "PPG")` for this initial adapter |
| `flags` | Tuple drawn from `lead_off`, `poor_contact`, `motion`, `fifo_overflow`, `clock_unsynchronized`; any flag rejects the packet |

The continuity tolerance is 2 ms between a packet's declared start and the preceding packet's expected end. This is a software check, not evidence of actual channel synchronization. Hardware timing, ECG/PPG filter delay, clock mapping, units, polarity, ADC scaling and sample integrity still need bench characterization. Implement framing/integrity and reconnect behavior in the selected hardware driver; there is no generic serial/BLE transport yet. Unsupported native rates require a separately reviewed anti-aliased resampler and preprocessing version.

Raw replay rebases recording sample times onto a monotonic playback clock. The UI's age and end-to-result time describe playback, not when the original participant was recorded.

## Model, quality and freshness contracts

`bundle.json` binds the model, selection, network, predictor and shared preprocessing hashes to the input contract. The application verifies it before importing/loading the model, in addition to the existing release checksum. Changing those files requires an intentional bundle update and revalidation. Acquisition windows still enter the model as float32 `(1, 2, 1250)`, ECG then PPG, at 125 Hz.

The raw pipeline identifier is `raw125_window_zero_phase_v1`. It uses the unchanged window-local zero-phase filters and independent normalization. This is offline-compatible window processing; it does not prove equivalence to the original PulseDB filtering or a particular sensor. No causal filtering, resampling, model retraining or clinical calibration is introduced.

Raw screening rejects nonfinite data, ADC-limit contact, insufficient peak-to-peak amplitude and supplied device flags. It does **not** reliably detect every noisy waveform, low-perfusion condition or movement; a high-amplitude artifact can pass. There is no inferred confidence percentage. Threshold calibration, IMU/contact drivers, physiological signal-quality assessment and accepted-reading coverage studies remain open.

Streaming windows carry session/sample provenance and quality/preprocessing metadata. The service rejects missing provenance, repeated/out-of-order windows, future timestamps, old windows, malformed/nonfinite predictions, nonpositive DBP and SBP <= DBP. It rechecks source health and age after inference. A result is never clipped into a plausible pressure range.

- `age_seconds`: elapsed time since acquisition-window end for streamed sources; since publication for the legacy independent-window replay.
- `pipeline_ms`: window end to publication preparation; includes waiting, processing and inference, but excludes HTTP/browser delivery.
- `preprocessing_ms`: raw screening and shared preprocessing call time.
- `inference_ms`: predictor call only.
- `dropped_windows`: complete windows superseded while waiting for inference.
- `boot_id`, `sequence`: service instance and update order, used by the dashboard.
- `/api/health`: measurement readiness, 200 only for a fresh `ok` result.
- `/api/live`: HTTP process liveness; can remain 200 while acquisition is unavailable.

The 12-second maximum reading age is retained. Acquisition faults clear readings earlier. Dashboard polling is serialized, response transit time is conservatively included in displayed age, and an independent expiry timer clears values during a slow request. Browser disconnection does not stop acquisition.

## Verification and profiling

```sh
python -m unittest discover -s deployment/raspberry-pi -p 'test_*.py' -v
python -m unittest discover -s 'model training/PulseDB-based Model-V1' -p 'test_*.py' -v
node --test deployment/raspberry-pi/test_dashboard.js
```

The deployment suite opens loopback sockets. Its system test generates temporary, explicitly synthetic waveforms, runs the actual application for a 17-second recording, compares both resulting windows to offline model predictions, checks EOF/readiness and sends SIGTERM. Synthetic tests verify software behavior only. Node is needed only for dashboard tests, not on the deployed Pi.

Local verification uses macOS ARM64, Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1 and PyTorch 2.8.0 in an isolated environment. The Pi dependency files retain the historically verified Pi versions. New tests and the raw pipeline must still be rerun on the actual Pi; no new Pi latency or power evidence is claimed.

`app.py --threads 1|2|4` allows target profiling without changing the released model. `run_pi_checks.py --threads 1|2|4` records the selected thread count in its Pi-only benchmark; it remains a prepared-window microbenchmark. Compare these settings with the raw service and real sensors before choosing a battery configuration. Physical power, temperatures, resource headroom and an eight-hour hardware soak still require target equipment.

## Remaining roadmap

This branch implements the software foundation of the [optimization plan](../../docs/WEARABLE_BP_OPTIMIZATION_PLAN.md): raw packet/window contracts, bounded buffering, deterministic replay, experimental screening, acquisition freshness, invalid-output suppression, bundle checks, UI recovery and testable shutdown. Next are a selected physical sensor/transport driver, timing and electrical validation, restricted raw capture/reference collection, validated quality thresholds, actual Pi profiling, and participant-disjoint adaptation/clinical evaluation. None of those results can be inferred from the replay or synthetic tests.
