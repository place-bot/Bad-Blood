"""Export a completed run to a portable FP32 research candidate bundle."""

import csv
from pathlib import Path

import numpy as np
import torch

from bp_runtime.models import ARCHITECTURE_VERSION, make_model
from bp_runtime.preprocessing import CONTRACT, normalize
from bp_runtime.release import BundlePredictor, runtime_hashes, sha256
from .evaluate import completed_checkpoint
from .io import ROOT, atomic_json, object_hash
from .trainer import external_output


def public_fixtures():
    path = ROOT / "data/PulseDB/examples/real_20_segments.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    waves = np.asarray([[[float(r[f"{c}_{j:04d}"]) for j in range(1250)]
                         for c in ("ecg", "ppg")] for r in rows], dtype=np.float32)
    return np.stack([normalize(w) for w in waves])


def export_run(run_dir, output):
    cp = completed_checkpoint(run_dir)
    config = cp["config"]
    output = external_output(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = make_model(config["architecture"], config["attention"]).eval()
    model.load_state_dict(cp["model"])
    waves = public_fixtures()
    with torch.inference_mode():
        predictions = (model(torch.from_numpy(waves)).numpy() *
                       np.asarray(cp["scale"], dtype=np.float32) + np.asarray(cp["center"], dtype=np.float32))
    torch.save(model.state_dict(), output / "weights.pt")
    np.savez(output / "golden.npz", waves=waves, predictions=predictions)
    manifest = {"schema": "bp_release_v2", "contract": CONTRACT,
                "architecture": config["architecture"], "attention": config["attention"],
                "architecture_version": ARCHITECTURE_VERSION, "backend": "pytorch_fp32",
                "center": cp["center"], "scale": cp["scale"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "runtime_sha256": runtime_hashes(),
                "sha256": {n: sha256(output / n) for n in ("weights.pt", "golden.npz")},
                "training": {"fingerprint": cp["fingerprint"], "dataset_hash": cp["dataset_hash"],
                             "selected_epoch": cp["selected_epoch"], "code": cp["code"]},
                "status": "research_candidate_requires_device_and_prospective_validation",
                "golden_provenance": "Public PulseDB-derived examples; waveform-only conversion fixtures, not evaluation",
                "data_terms": "PulseDB-derived fixtures: CC BY-NC-SA 4.0; see data/PulseDB/LICENSE.txt"}
    manifest["bundle_id"] = "v2-" + object_hash(manifest)[:16]
    atomic_json(output / "manifest.json", manifest)
    (output / "DATA_LICENSE.txt").write_text((ROOT / "data/PulseDB/LICENSE.txt").read_text())
    loaded = BundlePredictor(output)
    difference = float(np.max(np.abs(loaded.predict(waves) - predictions)))
    if difference > .01:
        raise ValueError("Export exceeded the FP32 parity tolerance")
    atomic_json(output / "export_check.json", {"max_abs_difference_mmHg": difference,
                                              "fixtures": len(waves), "scope": "conversion_only"})
    return {"bundle_id": manifest["bundle_id"], "output": str(output), "max_abs_difference_mmHg": difference}
