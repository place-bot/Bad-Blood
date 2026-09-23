# Raspberry Pi V1 implementation status

Checked on 2026-09-23. The implementation is in [`../../raspberry-pi/`](../../raspberry-pi/).
The current application commit on both the OMEN and Raspberry Pi is
`7a26cdd26d6e82059952486761ff9613123480ce`.

## What runs

- A Raspberry Pi 4B loads the released V1 `cnn_full` predictor once and computes
  blood-pressure estimates from ten-second, 125 Hz ECG/PPG windows.
- The current input is 20 public, recorded PulseDB segments. Reference blood
  pressure and participant identifiers are discarded before serving the page.
  The browser labels this source `SAMPLE RECORDING / PulseDB`; it does not imply
  a connected sensor or a continuous recording from one person.
- A source-independent `SignalWindow` contract accepts finite `float32`
  `(2, 1250)` ECG/PPG windows with the released filtering and normalization.
  A future device adapter must produce that same validated input. No physical
  acquisition device has been connected or validated.
- The service runs inference on the Pi independently of browser requests. It
  serves predictions, input waveforms, timing and freshness over a Pi-loopback
  JSON API. The computer displays them through an SSH tunnel; it does not run
  the model. The browser clears old readings when the service is stale or
  unreachable. HTML, CSS and JavaScript are served locally without a CDN.

## Raspberry Pi verification

On the current application commit, `test_deployment.py` passed all six tests;
the original V1 contract and release suites passed four and three tests.
`run_pi_checks.py` measured on Raspberry Pi 4 Model B Rev 1.5, aarch64,
Python 3.13.5, NumPy 2.2.6, SciPy 1.16.3 and PyTorch 2.8.0.

| Check | Result |
| --- | ---: |
| V1 `cnn_full` weight SHA-256 | `bb6742489f64128e714b365ebb2dd659b82ed2a8b8476a61416f9dff8af42ef1` |
| Public input segments compared with released predictions | 20/20 |
| Maximum absolute difference | 0.0 mmHg (limit 0.01 mmHg) |
| Predictor warmup / timed calls | 10 / 100 |
| Predictor call latency, p50 / p95 | 5.422 / 5.796 ms |
| Peak benchmark process RSS | 295,988 KiB |

Timing covers `Predictor.predict` on an already prepared window. It excludes
CSV reading, HTTP, plotting and the five-second interval between windows.
The Pi-produced detailed JSON and CSV are retained locally under
`deployment/raspberry-pi/results/` and excluded from Git because the CSV includes
source participant identifiers.

The browser was visually checked with advancing estimates, waveforms and time.
Closing it for over 15 seconds and reopening showed a later sequence; stopping
the Pi service cleared the readings, and restarting restored them. The current
page uses a source-neutral title and input-waveform section while preserving an
explicit recorded-data source label. A screenshot was viewed during that check,
but no screenshot artifact has yet been saved in this repository.

## Remaining deployment work

The service currently runs as a manually started, non-root Pi process on
`127.0.0.1:8000`. A non-root `systemd` unit template and setup instructions are
in [`../../raspberry-pi/`](../../raspberry-pi/), but the unit has not been installed
or enabled. Automatic start and recovery after a Pi reboot have therefore not
been claimed. No real ECG/PPG acquisition device or clinical validation exists.
The prompt's saved-screenshot acceptance item also remains open.

The page is reachable from the computer at `http://127.0.0.1:8000/` only while
the Pi service, local network and SSH tunnel are running. This loopback address
is the computer-side end of the tunnel; inference remains on the Pi.
