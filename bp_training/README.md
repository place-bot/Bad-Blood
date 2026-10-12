# Model V2 training and attention experiments

Implemented October 11, 2026 on `feature/model-v2-crc-attention`. This is the executable foundation for the [implementation plan](../docs/MODEL_V2_CRC_IMPLEMENTATION_PLAN.md), including the attention experiment. Local synthetic checks establish software behavior, not BP accuracy. No CRC training, physical-sensor validation or Pi V2 performance measurement has been completed.

## What is implemented

- Strict JSON configurations; explicit CUDA, CPU or MPS selection; CUDA requests never fall back silently.
- Hashed manifests over prepared arrays, participant split/exclusion checks, canonical two-channel input validation and training-only target statistics.
- FP32 baseline training, optional CUDA BF16/FP16, capped pressure sampling, unit-aware losses, early stopping and learning-rate scheduling.
- Atomic best/last checkpoints, complete optimizer/scheduler/RNG state, exclusive writer locks and epoch-boundary resume.
- Development evaluation, participant-bootstrap intervals, paired model comparison and complete-seed summaries.
- Explicit inference-only bundles, checksummed runtime/weights, public numerical fixtures and `--bundle` integration with the existing Pi application.

| Architecture | Attention | Parameters |
|---|---|---:|
| Frozen-compatible V1 | None | 37,666 |
| Wider V1-style residual control | None | 419,210 |
| Dual ECG/PPG encoder + temporal stack | None | 425,698 |
| Dual encoder | Temporal pooling | 429,859 |
| Dual encoder | SE feature-channel gate | 429,938 |
| Dual encoder | Both | 434,099 |

The wider control uses channels 28/56/108/160 and a 96-unit head, putting it within 2% of the no-attention dual encoder's parameter count. Operation counts and activation sizes still differ and must be profiled. The V1 adapter preserves the released weight keys and numerical behavior without modifying any V1 implementation or weights. Attention pooling starts with uniform temporal weights. Model definitions live in `bp_runtime/models.py`, shared by training and deployment; the runtime does not import the training package.

The first release supports PyTorch FP32 inference. Cross/self-attention, ONNX/INT8, distillation, multi-source identity reconciliation, wearable calibration, grouped-fold generation and physical sensor drivers remain later work gated by evidence or hardware access. Do not infer that a completed training run is a validated wearable release.

## Environment

Run commands from the repository root. The local checks use Python 3.9.6, Torch 2.8.0, NumPy 2.0.2 and SciPy 1.13.1. `pyproject.toml` declares supported dependency ranges; it is not a verified CRC environment lock.

On CRC, first provision a Torch 2.8 CUDA environment compatible with the allocated GPU and driver. Run `python -m pip install --no-deps -e .` inside that environment if installing the command entry point is useful; `python -m bp_training` also works directly from the checkout. After the real GPU smoke passes, record the container digest or environment package lock. See the [CRC guide](../deployment/crc/README.md). Do not replace CRC's CUDA build with a Mac package snapshot.

## Prepare and audit data

The adapter consumes the existing V1 `waves.npy`, `y.npy`, `subjects.npy` and `valid.npy` files. It does not download or reprocess the source archive. If these arrays are absent, prepare the approved source release using the existing V1 preparation workflow and its dependencies first. They are absent in the current checkout.

Keep manifests, records, runs and reports in restricted storage outside the public checkout. In the examples, `/project/bp` is a placeholder for your actual project storage, not an assumed CRC directory.

```bash
python -m bp_training prepare \
  --data-dir /project/bp/PulseDB/train \
  --split /project/bp/PulseDB/split.json \
  --exclude-subjects /project/bp/PulseDB/test/subjects.npy /project/bp/PulseDB/aami/subjects.npy \
  --source PulseDB-official-VitalDB-subsets \
  --output /project/bp/manifests/vitaldb-development-v1.json

python -m bp_training validate-data \
  --manifest /project/bp/manifests/vitaldb-development-v1.json
```

Preparation rejects overlapping participants, missing split membership, ABP/extra input channels, bad units/shape/representation, invalid eligible labels and altered files. Masked-out records remain counted in the manifest. Identity checks operate within one canonical source identity namespace; source prefixes cannot establish cross-source independence.

Manifests contain absolute paths and full SHA-256 digests. Moving arrays requires a new manifest and run; a changed manifest is not a compatible resume. Hashing and full representation scans can be I/O-intensive. Stage one shared read-only dataset before the campaign and measure storage contention before increasing job concurrency. Concurrently mutating arrays after validation is unsupported.

## Preflight and train

```bash
python -m bp_training preflight \
  --config configs/model_v2/A_v1.json \
  --manifest /project/bp/manifests/vitaldb-development-v1.json \
  --run-root /project/bp/runs --device cuda

python -m bp_training train \
  --config configs/model_v2/A_v1.json \
  --manifest /project/bp/manifests/vitaldb-development-v1.json \
  --run-dir /project/bp/runs/A-20260923-r1 --seed 20260923
```

Preflight runs a bounded FP32 optimizer/validation smoke on the requested device and reports the actual environment, model size and available storage. It is not a full-epoch throughput estimate or a mixed-precision equivalence check. Start with FP32; verify precision-specific training before enabling BF16/FP16 in a campaign.

Configs A–F define the initial six conditions. Run each with seeds `20260923`, `20260924` and `20260925`: 18 jobs. G–I define pooling, SE and both attention variants against F, adding nine jobs when that experiment is justified. If another training recipe wins, update the baseline and all attention conditions together before comparing. The combined loss/sampling/architecture config is deliberately not chosen before the initial results. The [experiment ledger](../deployment/crc/experiments.json) records these dependencies; it does not submit jobs.

Defaults reproduce the original batch size, optimizer, target standardization, scheduler, noise augmentation and stopping rule. Config files store deltas; every run writes the complete resolved configuration. Physical-unit losses can change clipping frequency: review logged unclipped gradient norms and freeze any loss multiplier/optimizer adjustment during development before the campaign. `coverage_of_eligible_rows=1` refers only to offline scoring of eligible rows; it is not wearable reading availability. Invalid predicted BP ordering is scored and reported offline, while the live service suppresses it.

## Interrupt and resume

SIGINT/SIGTERM request a safe loop exit. `last.pt` always describes a completed epoch. If epoch `k+1` is interrupted, resume starts it again from the end of epoch `k`. The partial epoch is discarded. If no epoch completed, there is no resumable checkpoint: use a new run directory. A hard kill during the first epoch similarly cannot produce a checkpoint.

```bash
python -m bp_training train \
  --config configs/model_v2/A_v1.json \
  --manifest /project/bp/manifests/vitaldb-development-v1.json \
  --run-dir /project/bp/runs/A-20260923-r1 --seed 20260923 --resume
```

Resume requires matching resolved config, manifest, source hashes and execution environment. In particular, changing epochs, workers, device, precision or code is a new experiment. Use the same GPU type, package environment and visible-device configuration. Commit/freeze code before starting real jobs. Bitwise restart was tested on CPU; no cross-GPU determinism claim is made.

Only one writer may own a run. After a hard kill, `.writer.lock` may remain. Confirm the scheduler job and its worker processes have stopped before manually removing that one stale lock, then resume. Never delete an active run's lock. Completed runs cannot be resumed or overwritten. `best.pt` is reconstructed from the committed `last.pt` boundary on resume if a crash occurred between the two saves.

## Evaluate and compare

```bash
python -m bp_training evaluate \
  --run-dir /project/bp/runs/A-20260923-r1 \
  --output /project/bp/evaluations/A-20260923-r1

python -m bp_training compare \
  --baseline /project/bp/evaluations/A-20260923-r1 \
  --candidate /project/bp/evaluations/F-20260923-r1 \
  --output /project/bp/reports/A-vs-F-seed1.json

python -m bp_training summarize \
  --evaluations /project/bp/evaluations/A-20260923-r1 /project/bp/evaluations/A-20260924-r1 /project/bp/evaluations/A-20260925-r1 \
  --output /project/bp/reports/A-three-seeds.json
```

Evaluate each candidate before comparing its output directory. Summaries reject duplicate or incomplete seed sets and mismatched datasets. Seed SD describes training variability, separately from participant-bootstrap intervals. Reports contain participant records and belong in restricted storage. Review pressure-stratum regressions, coverage and the plan's acceptance gates; the CLI does not automatically promote the best scalar score.

## Export, deploy and benchmark

```bash
python -m bp_training export \
  --run-dir /project/bp/runs/F-20260923-r1 \
  --output /project/bp/releases/F-candidate-r1

python deployment/raspberry-pi/app.py \
  --bundle /project/bp/releases/F-candidate-r1 --threads 1

python -m bp_runtime.benchmark \
  --bundle /project/bp/releases/F-candidate-r1 --threads 1 --iterations 100
```

Choose the deployment checkpoint using a predeclared development-only rule; the example seed is not a recommendation to select a lucky seed. Export contains inference weights, contract/architecture/target metadata, exact runtime hashes and 20 public waveform-only conversion fixtures with attribution. No private training examples or participant IDs enter the bundle. The fixture outputs test numerical behavior, not BP accuracy. Ship the matching `bp_runtime` package or repository revision with the bundle.

The app fails on an incompatible bundle, modified weight or golden-output mismatch. Omitting `--bundle` intentionally selects the verified V1 release; failures never cause silent fallback. The API and dashboard display model identity. Existing acquisition, invalid-output and freshness gates apply to either model. Raw replay uses the versioned filter adapter; it does not validate equivalence to real sensors.

Benchmark 1/2/4 threads in separate processes on the actual Pi. The benchmark identifies the executing machine and measures normalization/prediction latency and process RSS. It excludes raw acquisition, filtering, HTTP, battery consumption and thermal/soak behavior. Those require hardware measurements before promotion.

## Prospective evaluation of the actual artifact

Create a separate `--role prospective` manifest for genuinely reserved participants, or `--role historical` for previously examined source tests. These roles cannot train or enter selection comparisons. Before evaluating, write a private freeze JSON containing:

```json
{
  "bundle_manifest_sha256": "SHA-256 of the selected release manifest.json",
  "dataset_hash": "hash printed by validate-data for the evaluation manifest",
  "protocol": "Versioned reference pairing, preprocessing, exclusions, coverage and calibration rules"
}
```

```bash
python -m bp_training evaluate-bundle \
  --bundle /project/bp/releases/F-candidate-r1 \
  --manifest /project/bp/manifests/prospective-v1.json \
  --training-manifest /project/bp/manifests/vitaldb-development-v1.json \
  --freeze-record /project/bp/protocols/final-freeze.json \
  --output /project/bp/evaluations/prospective-frozen-r1
```

The evaluator checks that the training manifest matches the bundle and rejects any participant overlap with training or development validation. It evaluates the actual exported predictor. Stable participant identities and cross-source reconciliation remain research responsibilities. The freeze file records a protocol commitment; it cannot prove that nobody previously inspected a cohort. Existing output directories are refused, but assigning a new directory is not permission to reuse final participants for tuning. Offline waveform evaluation does not replace real-device quality/coverage validation.

## Local verification

```bash
python -m unittest discover -s tests -p 'test_model_v2.py' -v
python -m unittest discover -s deployment/raspberry-pi -p 'test_*.py' -v
python -m unittest discover -s 'model training/PulseDB-based Model-V1' -p 'test_*.py' -v
node --test deployment/raspberry-pi/test_dashboard.js
```

The multiprocessing test needs permission to launch PyTorch's shared-memory helper. The real-process replay suite binds local loopback and runs a 17-second synthetic recording. It does not attach hardware. No test needs the private dataset or a CRC account. A real CUDA smoke, full source-data reproduction and target Pi benchmark are still required before making performance claims.
