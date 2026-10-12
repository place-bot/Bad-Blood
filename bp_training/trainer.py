"""Single-device training with reproducible epoch-boundary continuation."""

from contextlib import nullcontext
import os
from pathlib import Path
import platform
import random
import signal
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from bp_runtime.models import make_model
from .checkpoint import restore_rng, rng_state, save_checkpoint
from .data import WaveDataset, sampling_probabilities, training_statistics, validate_manifest
from .io import ROOT, atomic_json, code_provenance, object_hash
from .losses import objective
from .metrics import metrics


def select_device(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; request a GPU or explicitly choose --device cpu")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support BF16")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(config.device)


def environment(device):
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_build": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "threads": torch.get_num_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, config, epoch=0, order=None, validation=False):
    kwargs = {}
    if config.workers:
        kwargs.update(prefetch_factor=config.prefetch, multiprocessing_context="spawn")
    generator = torch.Generator().manual_seed((config.seed + epoch) % 2**32)
    return DataLoader(dataset, batch_size=(config.validation_batch_size if validation else config.batch_size),
                      sampler=order, num_workers=config.workers, pin_memory=config.device == "cuda",
                      worker_init_fn=seed_worker, generator=generator, **kwargs)


def predict_dataset(model, dataset, center, scale, config, device):
    model.eval()
    predictions, rows = [], []
    with torch.inference_mode():
        for waves, _labels, index in loader(dataset, config, validation=True):
            pred = model(waves.to(device, non_blocking=device.type == "cuda"))
            predictions.append(pred.float().cpu().numpy() * scale + center)
            rows.append(index.numpy())
    return np.concatenate(predictions), np.concatenate(rows)


def external_output(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Training/evaluation artifacts must be outside the public repository")
    return path


def train(config, manifest_path, run_dir, resume=False, stop_after_epochs=None):
    """stop_after_epochs is a programmatic boundary-stop hook for restart verification."""
    device = select_device(config)
    audit = validate_manifest(manifest_path)
    if audit["manifest"]["role"] != "development":
        raise ValueError("Training on a historical/prospective evaluation manifest is forbidden")
    center, scale = training_statistics(audit)
    provenance = code_provenance()
    compatibility = {"config": config.to_dict(), "dataset_hash": audit["hash"],
                     "source_hash": provenance["source_hash"], "environment": environment(device)}
    fingerprint = object_hash(compatibility)
    run_dir = external_output(run_dir)
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not run_dir.is_dir():
        raise ValueError("Resume directory does not exist")
    # O_EXCL prevents concurrent writers to a run, including concurrent resumes.
    lock = run_dir / ".writer.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    previous_handlers = {}
    stopped = threading.Event()
    started_run = False
    try:
        if resume:
            import json
            status = json.loads((run_dir / "status.json").read_text())
            if status["state"] == "completed":
                raise ValueError("A completed run cannot be resumed")
            checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
            if checkpoint["fingerprint"] != fingerprint:
                raise ValueError("Resume config, data, code or execution environment differs")
        else:
            checkpoint = None
            atomic_json(run_dir / "config.json", config.to_dict())
            atomic_json(run_dir / "dataset.json", audit["manifest"])
            atomic_json(run_dir / "provenance.json", {**compatibility, "code": provenance})
        started_run = True
        atomic_json(run_dir / "status.json", {"state": "running", "fingerprint": fingerprint})
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, lambda *_: stopped.set())
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.cuda.reset_peak_memory_stats(device)
        sampler = np.random.default_rng(config.seed)
        model = make_model(config.architecture, config.attention).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience)
        scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
        history, best, stale, global_step, start_epoch = [], float("inf"), 0, 0, 1
        if checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            history, best, stale = checkpoint["history"], checkpoint["best"], checkpoint["stale"]
            global_step, start_epoch = checkpoint["global_step"], checkpoint["epoch"] + 1
            restore_rng(checkpoint["rng"], sampler, device)
            # Repair a crash between saving best.pt and last.pt using the committed boundary.
            best_copy = dict(checkpoint)
            best_copy["model"] = checkpoint["best_model"]
            best_copy["selected_epoch"] = checkpoint["best_epoch"]
            save_checkpoint(run_dir / "best.pt", best_copy)
        best_model = checkpoint["best_model"] if checkpoint else None
        best_epoch = checkpoint["best_epoch"] if checkpoint else None
        train_ids, validation_ids = audit["indices"]["train"], audit["indices"]["validation"]
        training = WaveDataset(audit["manifest"], train_ids)
        validation = WaveDataset(audit["manifest"], validation_ids)
        probability = sampling_probabilities(audit["labels"][train_ids], audit["subjects"][train_ids], config)
        center_t = torch.tensor(center, device=device)
        scale_t = torch.tensor(scale, device=device)
        completed_epochs = 0
        terminal_epoch = start_epoch - 1
        for epoch in range(start_epoch, config.epochs + 1):
            if stopped.is_set() or stale >= config.early_stopping:
                break
            began = time.perf_counter()
            order = (sampler.permutation(len(training)) if probability is None else
                     sampler.choice(len(training), len(training), replace=True, p=probability))
            model.train()
            totals = torch.zeros(4, device=device)
            seen, steps, data_wait = 0, 0, 0.0
            last_step = time.perf_counter()
            for waves, labels, _indices in loader(training, config, epoch, order.tolist()):
                data_wait += time.perf_counter() - last_step
                if stopped.is_set():
                    break
                waves = waves.to(device, non_blocking=device.type == "cuda")
                labels = labels.to(device, non_blocking=device.type == "cuda")
                waves = waves + torch.randn_like(waves) * config.noise_std
                optimizer.zero_grad(set_to_none=True)
                ctx = (nullcontext() if config.precision == "fp32" else torch.autocast(
                    "cuda", dtype=torch.bfloat16 if config.precision == "bf16" else torch.float16))
                with ctx:
                    pred = model(waves)
                    loss, raw = objective(pred, labels, center_t, scale_t, config)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm,
                                                      error_if_nonfinite=config.precision != "fp16")
                scaler.step(optimizer)
                scaler.update()
                finite_norm = torch.isfinite(norm)
                totals += torch.stack((raw * len(waves),
                                       torch.where(finite_norm, norm.detach(), torch.zeros_like(norm)),
                                       ((norm > config.clip_norm) & finite_norm).float(),
                                       (~finite_norm).float()))
                seen += len(waves)
                steps += 1
                global_step += 1
                last_step = time.perf_counter()
            if stopped.is_set():
                break  # Preserve last completed epoch; partial gradients/metrics never commit.
            predictions, rows = predict_dataset(model, validation, center, scale, config, device)
            score = metrics(audit["labels"][rows], predictions, audit["subjects"][rows])
            scheduler.step(score["primary"])
            improved = score["primary"] < best
            if improved:
                best, stale, best_epoch = score["primary"], 0, epoch
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - began
            totals_cpu = totals.detach().cpu().tolist()
            successful_steps = steps - int(totals_cpu[3])
            if not successful_steps or not np.isfinite(totals_cpu[0]):
                raise FloatingPointError("No finite optimization steps or nonfinite training loss")
            row = {"epoch": epoch, "train_loss_unscaled": totals_cpu[0] / seen,
                   "gradient_norm_mean": totals_cpu[1] / successful_steps,
                   "clipped_fraction": totals_cpu[2] / successful_steps,
                   "amp_skipped_steps": int(totals_cpu[3]),
                   "validation": score, "seconds": seconds, "data_wait_seconds": data_wait,
                   "training_samples_per_total_epoch_second": seen / seconds,
                   "lr": optimizer.param_groups[0]["lr"], "effective_batch_size": config.batch_size,
                   "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
            history.append(row)
            payload = {"schema": "bp_checkpoint_v1", "fingerprint": fingerprint,
                       "config": config.to_dict(), "dataset_hash": audit["hash"],
                       "code": provenance, "environment": environment(device),
                       "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "scaler": scaler.state_dict(), "rng": rng_state(sampler, device),
                       "center": center.tolist(), "scale": scale.tolist(),
                       "epoch": epoch, "global_step": global_step, "best": best, "stale": stale,
                       "best_epoch": best_epoch, "best_model": best_model, "history": history}
            if improved:
                selected = dict(payload, selected_epoch=epoch)
                save_checkpoint(run_dir / "best.pt", selected)
            save_checkpoint(run_dir / "last.pt", payload)
            atomic_json(run_dir / "history.json", history)
            print(f"epoch={epoch} macro_MAE={score['macro_mae']} seconds={seconds:.2f}", flush=True)
            terminal_epoch = epoch
            completed_epochs += 1
            if stop_after_epochs and completed_epochs >= stop_after_epochs:
                stopped.set()
        finished = terminal_epoch >= config.epochs or stale >= config.early_stopping
        state = "completed" if finished else "interrupted"
        atomic_json(run_dir / "status.json", {"state": state, "fingerprint": fingerprint,
                                               "epoch": terminal_epoch, "best_epoch": best_epoch})
        return {"state": state, "run_dir": str(run_dir), "best_epoch": best_epoch,
                "best_primary": None if best_model is None else best}
    except BaseException as error:
        if started_run:
            atomic_json(run_dir / "status.json", {"state": "failed", "fingerprint": fingerprint,
                                                   "error_type": type(error).__name__})
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        lock.unlink()
