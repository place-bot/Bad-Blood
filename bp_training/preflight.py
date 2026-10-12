"""Bounded device/model/data smoke check before a scheduled training run."""

import shutil
import time

import numpy as np
import torch

from bp_runtime.models import make_model
from .data import training_statistics, validate_manifest, WaveDataset
from .losses import objective
from .trainer import environment, external_output, select_device


def preflight(config, manifest, run_root):
    device = select_device(config)
    audit = validate_manifest(manifest)
    center, scale = training_statistics(audit)
    run_root = external_output(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryFile(dir=run_root) as stream:
        stream.write(b"preflight")
        stream.flush()
    ds = WaveDataset(audit["manifest"], audit["indices"]["train"])
    count = min(config.batch_size, len(ds))
    batch = [ds[i] for i in range(count)]
    x = torch.from_numpy(np.stack([r[0] for r in batch])).to(device)
    y = torch.from_numpy(np.stack([r[1] for r in batch])).to(device)
    model = make_model(config.architecture, config.attention).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start = time.perf_counter()
    pred = model(x)
    loss, _ = objective(pred, y, torch.tensor(center, device=device), torch.tensor(scale, device=device), config)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    val = WaveDataset(audit["manifest"], audit["indices"]["validation"])
    with torch.inference_mode():
        checked = model(torch.from_numpy(val[0][0][None]).to(device))
    if not torch.isfinite(checked).all():
        raise ValueError("Preflight produced nonfinite predictions")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"environment": environment(device), "dataset_hash": audit["hash"],
            "eligible_rows": audit["eligible_rows"], "participants": audit["participants"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "smoke_batch_size": count, "optimization_and_validation_seconds": time.perf_counter() - start,
            "gradient_norm": float(norm), "free_storage_bytes": shutil.disk_usage(run_root).free,
            "note": "Bounded FP32 functional smoke; not accuracy evidence or a full-epoch throughput estimate"}
