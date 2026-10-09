"""Offline population-checkpoint prediction for filtered ECG/PPG NPY arrays.

Input contract: N x 2 x 1250, ECG then PPG, 125 Hz, already filtered using
the source-compatible acquisition/preprocessing pathway. No additional filter
is applied. Output is N x 2 float32, SBP then DBP, in mmHg.

Only trust locally controlled project checkpoints. Full training checkpoints
contain optimizer/RNG Python objects, so --trust-checkpoint explicitly permits
torch.load(weights_only=False), which can execute code from a malicious file.
Integrity checks below are provenance checks, not a sandbox for pickle loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))
from shared.models import BPNet, VARIANTS

PROTOCOL_COMMIT = "1b93cf13249931af650cdfa91e19343564032451"
CHANNEL_ORDER = ("ECG", "PPG")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_inputs(waves, *, fs_hz=125, channel_order=CHANNEL_ORDER,
                   already_normalized=False) -> np.ndarray:
    """Validate caller-declared metadata and use observed-window samples only.

    NPY stores no sampling/channel/filter metadata. These declarations are a
    caller contract; numerical values cannot prove channel order or filtering.
    """
    _require(fs_hz == 125, "The checkpoint requires 125 Hz; no resampling is performed")
    _require(tuple(channel_order) == CHANNEL_ORDER, "Channel order must be ECG followed by PPG")
    waves = np.asarray(waves)
    _require(waves.ndim == 3 and waves.shape[0] > 0 and waves.shape[1:] == (2, 1250),
             "Expected a nonempty N x 2 x 1250 ECG/PPG array")
    _require(waves.dtype.kind in "fiu", "Inputs must be real numeric waveforms")
    _require(bool(np.isfinite(waves).all()), "Input contains nonfinite samples")
    with np.errstate(over="ignore", invalid="ignore"):
        values = waves.astype(np.float32, copy=True)
        span = np.ptp(values, axis=2, keepdims=True)
    _require(bool(np.isfinite(values).all() and np.isfinite(span).all()),
             "Input amplitudes cannot be represented safely in float32")
    _require(bool((span > 1e-8).all()), "Every ECG/PPG channel must be nonconstant")
    if already_normalized:
        _require(bool((values >= 0).all() and (values <= 1).all()),
                 "--already-normalized requires all samples in [0,1]")
        return values
    return (values - values.min(axis=2, keepdims=True)) / span


def load_predictor(checkpoint_path, *, trusted_checkpoint=False):
    """Load a trusted V2 high-resolution population checkpoint on CPU.

    The self-contained .pt is sufficient. If its original manifest.json is
    present alongside it, that manifest's hash and input/target contract must
    also match. Adaptation-unit weights and historical V1 weights are not this
    full V2 population-checkpoint format.
    """
    _require(trusted_checkpoint is True,
             "Explicit trusted_checkpoint=True / --trust-checkpoint is required before pickle loading")
    path = Path(checkpoint_path).resolve()
    _require(path.is_file() and path.suffix == ".pt", "Checkpoint must be an existing local .pt file")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(saved, dict), "Expected a V2 training-checkpoint dictionary")
    config = saved.get("config")
    _require(isinstance(config, dict), "Missing canonical V2 checkpoint configuration")
    variant = saved.get("variant")
    _require(variant in VARIANTS and config.get("variant") == variant,
             "Checkpoint variant and configuration disagree")
    _require(saved.get("pool") == 1 and config.get("pool") == 1,
             "Only the exact high-resolution pool=1 architecture is supported")
    _require(config.get("protocol_commit") == PROTOCOL_COMMIT
             and config.get("toy_test_only") is False,
             "Expected the frozen, real-data V2 population source contract")
    _require(saved.get("config_hash") == _hash_json(config), "Checkpoint config hash mismatch")
    hashes = saved.get("hashes")
    _require(isinstance(hashes, dict) and config.get("hashes") == hashes,
             "Checkpoint source identity differs from its configuration")
    sources = hashes.get("source", {})
    for relative in ("shared/models.py", "shared/data.py", "PROTOCOL.md"):
        _require(sources.get(relative) == _hash_file(V2_ROOT / relative),
                 f"Checkpoint does not match the canonical source contract: {relative}")
    center = np.asarray(saved.get("center"), dtype=np.float32)
    scale = np.asarray(saved.get("scale"), dtype=np.float32)
    _require(center.shape == (2,) and scale.shape == (2,)
             and bool(np.isfinite(center).all() and np.isfinite(scale).all() and (scale > 0).all()),
             "Checkpoint must contain finite two-target center and positive scale")
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _require(_hash_json(manifest) == hashes.get("manifest"), "Adjacent data-manifest hash mismatch")
        _require(manifest.get("task") == "contemporaneous_SBP_DBP"
                 and manifest.get("fs_hz") == 125 and manifest.get("samples_per_channel") == 1250,
                 "Adjacent manifest has an incompatible input/target contract")
        _require(np.array_equal(center, np.asarray(manifest.get("target_center"), dtype=np.float32))
                 and np.array_equal(scale, np.asarray(manifest.get("target_scale"), dtype=np.float32)),
                 "Checkpoint target normalization differs from its data manifest")
    model = BPNet(pool=1, variant=variant)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, center, scale


def predict_array(model, waves, center, scale, *, batch_size=512) -> np.ndarray:
    """Predict validated/prepared arrays without augmentation or fitting."""
    _require(isinstance(batch_size, int) and batch_size > 0, "Batch size must be a positive integer")
    _require(waves.ndim == 3 and waves.shape[0] > 0 and waves.shape[1:] == (2, 1250),
             "Expected prepared N x 2 x 1250 inputs")
    _require(not model.training, "The model must be in evaluation mode")
    _require(all(parameter.device.type == "cpu" for parameter in model.parameters()),
             "This offline CLI uses CPU inference")
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(waves), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(waves[start:start + batch_size], dtype=np.float32))
            predictions.append(model(batch).numpy() * scale + center)
    output = np.concatenate(predictions).astype(np.float32, copy=False)
    _require(output.shape == (len(waves), 2) and bool(np.isfinite(output).all()),
             "Model produced invalid SBP/DBP estimates")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Filtered ECG/PPG .npy, N x 2 x 1250")
    parser.add_argument("--output", type=Path, required=True, help="New .npy file, SBP/DBP mmHg")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Explicitly trust this local checkpoint and permit Python pickle loading")
    parser.add_argument("--already-normalized", action="store_true")
    parser.add_argument("--fs-hz", type=int, default=125)
    parser.add_argument("--channel-order", nargs=2, default=list(CHANNEL_ORDER), metavar=("FIRST", "SECOND"))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    if not args.trust_checkpoint:
        parser.error("--trust-checkpoint is required; use only locally controlled, trusted checkpoints")
    if args.output.suffix != ".npy" or args.input.suffix != ".npy":
        parser.error("Input and output must use .npy extensions")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    if not args.output.parent.is_dir():
        parser.error("Output parent directory does not exist")
    waves = prepare_inputs(np.load(args.input, allow_pickle=False), fs_hz=args.fs_hz,
                           channel_order=args.channel_order, already_normalized=args.already_normalized)
    model, center, scale = load_predictor(args.checkpoint, trusted_checkpoint=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output = predict_array(model, waves, center, scale, batch_size=args.batch_size)
    with args.output.open("xb") as handle:
        np.save(handle, output, allow_pickle=False)
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(output),
                      "columns": ["SBP", "DBP"], "unit": "mmHg", "variant": model.variant,
                      "fs_hz": 125, "channel_order": list(CHANNEL_ORDER),
                      "checkpoint_sha256": _hash_file(args.checkpoint)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
