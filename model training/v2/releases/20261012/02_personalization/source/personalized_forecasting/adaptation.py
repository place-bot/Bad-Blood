"""Support-only calibration; later query data are intentionally absent from this API."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


NEURAL_METHODS = ("head", "last_block", "full")


def _matrix(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not len(value):
        raise ValueError(f"{name} must have nonempty shape (N, 2)")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    return value


def fit_calibration(method: str, support_predictions: Any, support_labels: Any,
                    center: Any, scale: Any, penalty: float | None = None) -> dict:
    """Fit scalar calibration only on supplied support, in standardized units."""
    predictions = _matrix(support_predictions, "support_predictions")
    labels = _matrix(support_labels, "support_labels")
    if labels.shape != predictions.shape:
        raise ValueError("Support labels and predictions must align")
    center, scale = np.asarray(center, float), np.asarray(scale, float)
    if center.shape != (2,) or scale.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Population center/scale must be finite two-vectors; scale > 0")
    x, y = (predictions - center) / scale, (labels - center) / scale
    slope, intercept = np.ones(2), np.zeros(2)
    if method == "bias":
        intercept = (y - x).mean(axis=0)
    elif method == "affine":
        if len(y) < 3:
            raise ValueError("Affine calibration requires K >= 3")
        if penalty is None or penalty <= 0:
            raise ValueError("Affine penalty must be positive")
        dx, dy = x - x.mean(axis=0), y - y.mean(axis=0)
        slope = ((dx * dy).sum(axis=0) + penalty) / ((dx * dx).sum(axis=0) + penalty)
        intercept = y.mean(axis=0) - slope * x.mean(axis=0)
    elif method == "last_reference":
        slope, intercept = np.zeros(2), y[-1]
    elif method == "calibration_mean":
        slope, intercept = np.zeros(2), y.mean(axis=0)
    elif method != "population":
        raise ValueError(f"Unknown scalar method: {method}")
    return {"method": method, "slope": slope.tolist(), "intercept": intercept.tolist(),
            "center": center.tolist(), "scale": scale.tolist(), "penalty": penalty,
            "n_support": len(y)}


def apply_calibration(calibration: dict, predictions: Any) -> np.ndarray:
    predictions = _matrix(predictions, "predictions")
    center, scale = np.array(calibration["center"]), np.array(calibration["scale"])
    return ((predictions - center) / scale * np.array(calibration["slope"])
            + np.array(calibration["intercept"])) * scale + center


def configure_trainable(model: nn.Module, strategy: str) -> list[str]:
    """Freeze all BN running state and dropout, enabling only the chosen depth."""
    if strategy not in NEURAL_METHODS:
        raise ValueError(f"Unknown neural strategy: {strategy}")
    if not hasattr(model, "head") or not hasattr(model, "body"):
        raise ValueError("Expected the shared BPNet body/head interface")
    # BPNet's unchanged V1 final residual is body[6]; fail rather than train
    # an unintended module after an architecture change.
    last_block = model.body[6]
    if strategy == "last_block" and not (hasattr(last_block, "net") and hasattr(last_block, "skip")):
        raise ValueError("Expected V1 residual block at body[6]")
    enabled = {id(p) for p in model.head.parameters()}
    if strategy == "last_block":
        enabled.update(id(p) for p in last_block.parameters())
    if strategy == "full":
        enabled = {id(p) for p in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in enabled)
    model.eval()  # eval does not prevent gradients; it fixes BN state/dropout.
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def synchronize(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class AdaptationResult:
    model: nn.Module
    audit: dict


def adapt_model(population_model: nn.Module, support_x: Any, support_y: Any,
                center: Any, scale: Any, *, strategy: str, lr: float,
                steps: int, device: str = "cpu", seed: int = 42) -> AdaptationResult:
    """Fresh fixed-step AdamW adaptation. No query arguments or early stopping."""
    total_started = time.perf_counter()
    if lr <= 0 or steps <= 0:
        raise ValueError("lr and steps must be positive")
    labels = _matrix(support_y, "support_y")
    inputs = np.asarray(support_x, dtype=np.float32)
    if inputs.ndim != 3 or inputs.shape[0] != len(labels) or inputs.shape[1:] != (2, 1250):
        raise ValueError("Aligned support_x must have shape (K, 2, 1250)")
    if not np.isfinite(inputs).all():
        raise ValueError("Nonfinite support inputs")
    center, scale = np.asarray(center, np.float32), np.asarray(scale, np.float32)
    if center.shape != (2,) or scale.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid population target normalization")
    torch.manual_seed(seed)
    model = copy.deepcopy(population_model).to(device)
    trainable = configure_trainable(model, strategy)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor((labels - center) / scale, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=lr, weight_decay=0.001)
    losses = []
    synchronize(device)
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite support loss")
        loss.backward()
        nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    synchronize(device)
    elapsed = time.perf_counter() - started
    named_parameters = dict(model.named_parameters())
    changed, forbidden = {}, []
    for name, value in model.state_dict().items():
        new_value = value.detach().cpu()
        if not torch.equal(before[name], new_value):
            if name not in trainable:
                forbidden.append(name)
            changed[name] = float((new_value.double() - before[name].double()).norm())
    if forbidden:
        raise AssertionError(f"Frozen parameters/buffers changed: {forbidden}")
    audit = {"strategy": strategy, "lr": lr, "steps": steps, "seed": seed,
             "support_count": len(labels), "weight_decay": 0.001, "gradient_clip": 5.0,
             "dropout_active": False, "batchnorm_running_statistics_frozen": True,
             "trainable_parameter_names": trainable,
             "trainable_parameters": sum(named_parameters[name].numel() for name in trainable),
             "total_parameters": sum(p.numel() for p in model.parameters()),
             "changed_parameter_l2": changed, "changed_parameter_count": len(changed),
             "support_loss_history": losses, "adaptation_seconds": elapsed,
             "adaptation_timing_scope": "optimizer update loop; excludes model cloning, transfer and optimizer setup",
             "total_adaptation_seconds": time.perf_counter() - total_started,
             "device": str(device), "query_used_for_adaptation": False}
    return AdaptationResult(model=model, audit=audit)


def predict(model: nn.Module, x: Any, center: Any, scale: Any,
            device: str = "cpu", batch_size: int = 256) -> tuple[np.ndarray, dict]:
    """Return mmHg predictions and measured host/device-synchronized latency."""
    model.eval().to(device)
    x = np.asarray(x, np.float32)
    if x.ndim != 3 or x.shape[1:] != (2, 1250) or not len(x):
        raise ValueError("Prediction inputs require nonempty (N, 2, 1250)")
    outputs = []
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(x), batch_size):
            outputs.append(model(torch.as_tensor(x[offset:offset + batch_size], device=device)).cpu().numpy())
    synchronize(device)
    elapsed = time.perf_counter() - started
    predictions = np.concatenate(outputs) * np.asarray(scale) + np.asarray(center)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite predictions")
    return predictions, {"device": str(device), "windows": len(x), "batch_size": batch_size,
                         "prediction_seconds": elapsed, "seconds_per_window": elapsed / len(x),
                         "latency_scope": "batched inference including input/output transfer, excluding model transfer; no warmup exclusion",
                         "hardware_claim": "local workstation, not Raspberry Pi"}
