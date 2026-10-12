# Wearable software implementation status

The acquisition foundation below was completed October 9. The [V2 training and release addition](#v2-training-and-release-addition) was verified October 11 on `feature/model-v2-crc-attention`.

October 9, 2026 — branch `feature/wearable-acquisition-foundation`.

This implements the software foundation of the [optimization plan](WEARABLE_BP_OPTIMIZATION_PLAN.md). It does not complete the hardware, data-collection or clinical milestones.

| Implemented | Evidence / location |
|---|---|
| Raw 125 Hz packet contract with acquisition sample/time provenance | `deployment/raspberry-pi/acquisition.py`, `sources.py` |
| Background acquisition, bounded ring and latest-window mailbox | Exact ten-second windows/five-second stride; older pending windows counted and dropped |
| Finite, paced continuous CSV replay and sidecar configuration | `app.py --raw-input … --session …`; recorded source labeling retained |
| Fault handling | Gaps, resets, duplicates, clock discontinuities, packet flags and watchdog timeout clear accumulated data |
| Experimental quality screening before normalization | Configured ADC limits and minimum signal amplitude; not a validated physiological quality score |
| Source-age freshness and output validity | Repeated/future/stale input and invalid BP pairs suppressed; health readiness separated from liveness |
| Bundle compatibility | Hashes of released model, selection, network, predictor and shared preprocessing checked at startup |
| Dashboard recovery | Serialized requests, service boot identity, independent expiry, conservative response-age handling |
| Profiling hooks | Preprocessing, predictor and window-to-result timing; dropped-window count; selectable 1/2/4 inference threads |
| Shutdown | SIGINT/SIGTERM, cancellable consumption and source close |
| Instructions and fixtures | [Acquisition guide](../deployment/raspberry-pi/ACQUISITION.md), intentionally incomplete sensor-configuration template, synthetic tests |

Verification uses a local macOS ARM64 environment. The new software has not yet been executed on the actual Pi or connected to a physical sensor. The original V1 model weights and training/inference/preprocessing source files remain unchanged.

Verification completed: **26 deployment/acquisition/system tests, 7 original V1 tests and 5 dashboard tests passed**. All 20 released example predictions matched within the existing 0.01 mmHg tolerance at each of 1, 2 and 4 inference threads; maximum difference from the rounded reference CSV was 0.000487 mmHg. Python compilation, JavaScript syntax, documentation links and Git whitespace checks also passed.

The test suites cover raw/offline waveform parity, bounded queues, faults during processing, full-window reacquisition after watchdog expiry, source freshness, malformed predictions, API readiness, bundle tampering, browser expiry and polling, and the existing V1 release/input contracts. A real-process integration test replays 17 seconds of synthetic signals through the application, compares its two predictions with offline V1 output, checks EOF and verifies graceful termination. Synthetic input is test-only and provides no BP-accuracy evidence.

The remaining work depends on hardware and study data: select and implement the actual ECG/PPG transport driver; verify synchronization, filtering, placement and electrical design; collect restricted raw/reference recordings; validate quality thresholds and reading coverage; profile power and memory on the target Pi; and evaluate adaptation/calibration on held-out participants. Native-rate resampling, robust motion/morphology assessment, a private acquisition recorder, model retraining, quantization, a custom PCB and systemd installation are not implemented in this branch.

Run instructions and exact test commands are in [ACQUISITION.md](../deployment/raspberry-pi/ACQUISITION.md).

## V2 training and release addition

October 11, 2026 — branch `feature/model-v2-crc-attention`. Commands and precise limitations are in the [V2 implementation guide](../bp_training/README.md) and [CRC execution guide](../deployment/crc/README.md).

| Implemented | Verification |
|---|---|
| V1-compatible adapter, 419,210-parameter wider control, 425,698-parameter dual encoder/TCN | Released V1 weights produce identical adapter outputs; all candidates have finite gradients and expected shapes |
| Separate temporal-pooling, SE and combined attention variants | Pooling starts uniform; combined model has 434,099 parameters; attention shape/parameter checks pass |
| Strict configurations, hashed data manifests, exclusions and training-only transforms | Tampered files, participant leakage, invalid splits, reference input channels and holdout training are rejected |
| Explicit CUDA/CPU/MPS selection; optional CUDA mixed precision | Requested unavailable CUDA fails; actual CUDA/AMP execution remains unverified without a GPU |
| Single-device trainer, scheduler, capped sampling, seed/state logging and best/last checkpoints | CPU stop/resume and failure-mid-epoch recovery reproduce uninterrupted final weights exactly |
| Development evaluation, paired participant-bootstrap comparisons and complete-seed summaries | Participant weighting, missing/duplicate seeds and matched evaluation records checked |
| Inference-only FP32 export and explicit Pi bundle loading | Synthetic-trained candidate exports/reloads with ≤0.01 mmHg conversion difference on public fixtures at 1/2/4 threads |
| Frozen-artifact held-out evaluation | Training-manifest binding and participant exclusions required; known development participants cannot be relabeled final evidence |
| Model-aware dashboard and rollback | Selected model identity is visible; freshness/invalid-output suppression remains active; no silent fallback |
| CRC entry point and guarded scheduler templates | Shell syntax checked; account-specific resource directives deliberately remain unfilled |

Verification completed: **17 V2 tests, 26 acquisition/deployment/system tests, 7 original V1 tests and 6 dashboard tests passed (56 total)**. Tests ran locally on macOS, with PyTorch shared-memory workers and loopback HTTP tests run outside the restrictive sandbox. Python compilation, CLI/config checks, shell syntax, documentation links and Git whitespace were also checked. The V1 model implementation and released weights are unchanged.

Only small synthetic datasets were trained for software tests. Their labels and errors are not accuracy evidence. The full prepared PulseDB arrays, CRC allocation and physical Pi/sensors were unavailable in this checkout. Consequently, source-data V2 training, real CUDA/AMP behavior, model selection, wearable adaptation, prospective accuracy, Pi latency/power/soak results and clinical claims remain outstanding. ONNX/INT8, distillation, cross/self-attention and multi-source reconciliation remain evidence-gated extensions, not implemented features.
