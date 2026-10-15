"""Full-cohort, epoch-resumable population experiments under PROTOCOL.md.

Only unit tests may supply ``test_settings`` or an injected data bundle. The
CLI exposes no epoch, learning-rate, cohort-size, or batch-size tuning knobs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
from typing import Any

import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))
from shared.models import VARIANTS, count_parameters, make_model

PROTOCOL_COMMIT = "1b93cf13249931af650cdfa91e19343564032451"
FIXED_SETTINGS = {
    "lr": .001, "weight_decay": .001, "batch_size": 256,
    "noise_sd": .003, "clip_norm": 5., "max_epochs": 40,
    "patience": 8, "scheduler_factor": .5, "scheduler_patience": 3,
    "inference_batch_size": 512,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _cpu_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(v) for v in value)
    return value


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _rng_state(rng: np.random.Generator, device: torch.device) -> dict:
    state = {"python": random.getstate(), "numpy_global": np.random.get_state(),
             "numpy_generator": copy.deepcopy(rng.bit_generator.state), "torch_cpu": torch.get_rng_state()}
    if device.type == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict, rng: np.random.Generator, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["torch_mps"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _seed_training(seed: int, device: torch.device) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _source_hashes() -> dict[str, str]:
    paths = [Path(__file__), V2_ROOT / "shared/models.py", V2_ROOT / "shared/data.py",
             V2_ROOT / "shared/metrics.py", V2_ROOT / "PROTOCOL.md"]
    return {str(p.relative_to(V2_ROOT)): sha256_file(p) for p in paths if p.exists()}


def _select_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this process")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this process")
    return device


def _validate_bundle(bundle: dict, test_mode: bool) -> None:
    sets = {}
    expected = {"train": (1085, 390600), "validation": (192, 69120),
                "test": (141, 56400), "aami": (116, 666)}
    for name in expected:
        part = bundle[name]
        ids = np.asarray(part["indices"])
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0 or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Invalid eligible row indices for {name}")
        if ids.min() < 0 or ids.max() >= len(part["y"]):
            raise ValueError(f"Out-of-range row indices for {name}")
        if part["waves"].shape[1:] != (2, 1250) or len(part["waves"]) != len(part["y"]) or len(part["subjects"]) != len(part["y"]):
            raise ValueError(f"Mismatched waves/labels/subjects for {name}")
        if np.asarray(part["y"]).shape[1:] != (2,):
            raise ValueError(f"Invalid label shape for {name}")
        sets[name] = set(np.asarray(part["subjects"])[ids].astype(str))
        if not test_mode and (len(sets[name]), len(ids)) != expected[name]:
            raise ValueError(f"Frozen {name} cohort mismatch: {(len(sets[name]), len(ids))} != {expected[name]}")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"), ("train", "aami"), ("validation", "aami")):
        if sets[a] & sets[b]:
            raise ValueError(f"Patient overlap: {a} and {b}")
    labels = np.asarray(bundle["train"]["y"])[bundle["train"]["indices"]]
    for key, actual in (("center", labels.mean(0, dtype=np.float64).astype(np.float32)),
                        ("scale", labels.std(0, dtype=np.float64).astype(np.float32))):
        if not np.allclose(np.asarray(bundle[key]), actual, rtol=1e-5, atol=1e-5):
            raise ValueError(f"{key} is not fitted on the eligible training rows")
    if np.any(np.asarray(bundle["scale"]) <= 0):
        raise ValueError("Training target standard deviations must be positive")


def predict(model: torch.nn.Module, part: dict, center: np.ndarray, scale: np.ndarray,
            device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(part["indices"]), batch_size):
            ids = part["indices"][start:start + batch_size]
            x = torch.from_numpy(np.array(part["waves"][ids], dtype=np.float32, copy=True)).to(device)
            predictions.append(model(x).cpu().numpy() * scale + center)
    result = np.concatenate(predictions)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite evaluation predictions")
    return result


def measure_costs(model: torch.nn.Module, sample: np.ndarray, device: torch.device,
                  test_mode: bool = False) -> dict:
    result = {"parameters": count_parameters(model), "latency": [],
              "latency_note": "Warm inference; desktop measurements, not Raspberry Pi estimates. Batch and per-example times are both reported.",
              "cpu_process_lifetime_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
              "memory_note": "CPU RSS is process-lifetime high-water mark, not a per-model peak; MPS values are current allocations, not peak."}
    devices = [torch.device("cpu")] + ([device] if device.type != "cpu" else [])
    for target in devices:
        measured = copy.deepcopy(model).to(target).eval()
        for batch in (1, min(256, len(sample))):
            x = torch.from_numpy(np.array(sample[:batch], dtype=np.float32, copy=True)).to(target)
            repetitions, warmup = (2, 1) if test_mode else (20, 5)
            times = []
            with torch.inference_mode():
                for _ in range(warmup):
                    measured(x)
                _synchronize(target)
                for _ in range(repetitions):
                    started = time.perf_counter()
                    measured(x)
                    _synchronize(target)
                    times.append((time.perf_counter() - started) * 1000)
            result["latency"].append({"device": str(target), "batch_size": batch,
                "repetitions": repetitions, "median_batch_ms": float(np.median(times)),
                "median_per_example_ms": float(np.median(times) / batch), "p95_batch_ms": float(np.quantile(times, .95))})
        del measured
    if device.type == "mps":
        result["mps_current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        result["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    return result


def run_training(variant: str, seed: int, data_root: str | Path, output_dir: str | Path,
                 device_name: str = "auto", resume: bool = False, *,
                 bundle: dict | None = None, test_settings: dict | None = None,
                 stop_after_epoch: int | None = None) -> dict:
    """Train/evaluate one run. Injected bundles are explicitly marked toy tests."""
    started = time.perf_counter()
    test_mode = bundle is not None
    if (test_settings is not None or stop_after_epoch is not None) and not test_mode:
        raise ValueError("Test-only controls require an explicitly injected toy bundle")
    if variant not in VARIANTS or seed not in (42, 43, 44):
        raise ValueError("Variant or seed is outside the frozen experiment matrix")
    if variant in ("baseline_ecg", "baseline_ppg") and seed != 42:
        raise ValueError("Single-modality ablations use seed 42 only")
    if bundle is None:
        from shared.data import load_population
        bundle = load_population(Path(data_root))
    from shared.metrics import quick_score, summarize
    _validate_bundle(bundle, test_mode)
    settings = {**FIXED_SETTINGS, **(test_settings or {})}
    device = _select_device(device_name)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output = Path(output_dir).resolve()
    hashes = {"manifest": bundle["manifest_hash"], "source": _source_hashes()}
    config = {"variant": variant, "seed": seed, "settings": settings, "pool": 1,
              "data_root": str(Path(data_root).resolve()), "device": str(device),
              "protocol_commit": PROTOCOL_COMMIT, "toy_test_only": test_mode,
              "hashes": hashes}
    config_hash = hashlib.sha256(_canonical_json(config).encode()).hexdigest()
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f"Run directory is not empty; explicit --resume required: {output}")
        if not (output / "config.json").exists() or json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Resume configuration, source, or split hash differs from the saved run")
        if (output / "completion.json").exists():
            completion = json.loads((output / "completion.json").read_text())
            if completion["best_checkpoint_sha256"] != sha256_file(output / "best.pt"):
                raise ValueError("Completed best checkpoint was modified")
            return completion
        if not (output / "last.pt").exists():
            raise ValueError("No resumable epoch checkpoint exists")
    elif resume:
        raise FileNotFoundError(f"No existing run to resume at {output}")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "config.json", config)
    _atomic_json(output / "manifest.json", bundle["manifest"])
    environment = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                   "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
                   "threads": torch.get_num_threads(), "deterministic_algorithms": True,
                   "deterministic_warn_only": True,
                   "nondeterminism_note": "Unsupported deterministic operations warn rather than fail. MPS reproducibility is best effort; CPU epoch-resume equality is unit tested."}
    _atomic_json(output / "environment.json", environment)
    center = np.asarray(bundle["center"], dtype=np.float32)
    scale = np.asarray(bundle["scale"], dtype=np.float32)
    model = make_model(variant, seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=settings["scheduler_factor"], patience=settings["scheduler_patience"])
    rng = _seed_training(seed, device)
    history, best, stale, epoch, carried_seconds = [], float("inf"), 0, 0, 0.
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["config_hash"] != config_hash or checkpoint["hashes"] != hashes:
            raise ValueError("Checkpoint configuration/source/split hashes do not match")
        if not np.array_equal(checkpoint["center"], center) or not np.array_equal(checkpoint["scale"], scale):
            raise ValueError("Checkpoint target standardization differs")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        opt.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, best, stale, epoch = checkpoint["history"], checkpoint["best"], checkpoint["stale"], checkpoint["epoch"]
        carried_seconds = checkpoint["elapsed_training_seconds"]
        _restore_rng(checkpoint["rng"], rng, device)
    train, validation = bundle["train"], bundle["validation"]
    center_t, scale_t = torch.from_numpy(center).to(device), torch.from_numpy(scale).to(device)
    mask = torch.tensor([float(i in model.available_channels) for i in (0, 1)], device=device)[None, :, None]
    training_started = time.perf_counter()
    while epoch < settings["max_epochs"] and stale < settings["patience"]:
        epoch += 1
        tick = time.perf_counter()
        model.train()
        order = rng.permutation(train["indices"])
        total_loss = torch.zeros((), device=device)
        for pos in range(0, len(order), settings["batch_size"]):
            ids = order[pos:pos + settings["batch_size"]]
            x = torch.from_numpy(np.array(train["waves"][ids], dtype=np.float32, copy=True)).to(device)
            y = torch.from_numpy(np.array(train["y"][ids], dtype=np.float32, copy=True)).to(device)
            x = (x + torch.randn_like(x) * settings["noise_sd"]) * mask
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), (y - center_t) / scale_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_norm"])
            opt.step()
            # Accumulate on-device: no per-batch .item()/.cpu() synchronization.
            total_loss.add_(loss.detach() * len(ids))
        epoch_loss = float(total_loss.cpu()) / len(order)
        if not np.isfinite(epoch_loss):
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}; last saved epoch remains resumable")
        pred = predict(model, validation, center, scale, device, settings["inference_batch_size"])
        ids = validation["indices"]
        score = quick_score(np.asarray(validation["y"])[ids], pred, np.asarray(validation["subjects"])[ids])
        value = float(np.mean(score["macro_mae"]))
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite validation selection score")
        scheduler.step(value)
        improved = value < best
        best, stale = (value, 0) if improved else (best, stale + 1)
        _synchronize(device)
        history.append({"epoch": epoch, "train_normalized_MSE": epoch_loss,
                        "validation": score, "selection_macro_mae": value,
                        "seconds": time.perf_counter() - tick, "lr": opt.param_groups[0]["lr"]})
        checkpoint = {"state_dict": _cpu_state(model.state_dict()), "center": center.tolist(), "scale": scale.tolist(),
                      "epoch": epoch, "variant": variant, "seed": seed, "pool": 1,
                      "config": config, "config_hash": config_hash, "hashes": hashes,
                      "optimizer": _cpu_state(opt.state_dict()), "scheduler": scheduler.state_dict(),
                      "rng": _rng_state(rng, device), "history": history, "best": best, "stale": stale,
                      "validation": score, "elapsed_training_seconds": carried_seconds + time.perf_counter() - training_started}
        if improved:
            _atomic_torch(output / "best.pt", checkpoint)
        _atomic_torch(output / "last.pt", checkpoint)
        _atomic_json(output / "history.json", history)
        print(json.dumps({"variant": variant, "seed": seed, **history[-1]}, default=_json_default), flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return {"status": "interrupted_for_toy_resume_test", "epoch": epoch}
    training_seconds = carried_seconds + time.perf_counter() - training_started
    evaluation_started = time.perf_counter()
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    all_metrics = {}
    artifact_hashes = {}
    for name in ("validation", "test", "aami"):
        part = bundle[name]
        ids = np.asarray(part["indices"])
        pred = predict(model, part, center, scale, device, settings["inference_batch_size"])
        y = np.asarray(part["y"])[ids]
        subjects = np.asarray(part["subjects"])[ids].astype(str)
        payload = {"pred": pred, "y": y, "subjects": subjects, "source_indices": ids,
                   "variant": variant, "seed": seed, "selected_epoch": checkpoint["epoch"],
                   "manifest_hash": bundle["manifest_hash"]}
        for field in ("record_ids", "case_ids", "timestamps", "times"):
            if field in part:
                payload[field] = np.asarray(part[field])[ids]
        path = output / f"predictions_{name}.npz"
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, path)
        artifact_hashes[path.name] = sha256_file(path)
        all_metrics[name] = summarize(y, pred, subjects)
    sample_ids = train["indices"][:256]
    costs = measure_costs(model, np.asarray(train["waves"])[sample_ids], device, test_mode)
    costs["training_seconds"] = training_seconds
    costs["evaluation_and_cost_seconds"] = time.perf_counter() - evaluation_started
    costs["epoch_seconds_sum"] = sum(row["seconds"] for row in history)
    costs["invocation_wall_seconds"] = time.perf_counter() - started
    costs["cost_note"] = "Training seconds include per-epoch validation and checkpoint writes, excluding final evaluation and cost measurement. Epoch_seconds_sum omits checkpoint writes. Resume carries prior checkpoint elapsed time."
    _atomic_json(output / "metrics.json", all_metrics)
    _atomic_json(output / "costs.json", costs)
    completion = {"status": "complete", "toy_test_only": test_mode, "variant": variant, "seed": seed,
                  "epochs_completed": epoch, "selected_epoch": checkpoint["epoch"],
                  "early_stopped": stale >= settings["patience"], "best_validation_macro_mae": checkpoint["best"],
                  "config_hash": config_hash, "manifest_hash": bundle["manifest_hash"],
                  "best_checkpoint_sha256": sha256_file(output / "best.pt"), "prediction_sha256": artifact_hashes,
                  "parameters": count_parameters(model), "parameter_count": count_parameters(model),
                  "best_epoch": checkpoint["epoch"], "actual_epochs": epoch,
                  "total_training_seconds": training_seconds, "latency": costs["latency"]}
    _atomic_json(output / "completion.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_training(args.variant, args.seed, args.data_root, args.output_dir,
                                  args.device, args.resume), indent=2), flush=True)


if __name__ == "__main__":
    main()
