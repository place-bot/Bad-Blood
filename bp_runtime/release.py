"""Explicit, checksummed FP32 release loading without training dependencies."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .models import ARCHITECTURE_VERSION, make_model
from .preprocessing import CONTRACT, from_raw_125, normalize

RUNTIME_FILES = ("models.py", "preprocessing.py", "release.py")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_hashes():
    return {name: sha256(Path(__file__).with_name(name)) for name in RUNTIME_FILES}


class BundlePredictor:
    def __init__(self, root, threads=1):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if (m.get("schema") != "bp_release_v2" or m.get("contract") != CONTRACT or
                m.get("architecture_version") != ARCHITECTURE_VERSION or
                m.get("backend") != "pytorch_fp32" or
                m.get("runtime_sha256") != runtime_hashes() or
                set(m.get("sha256", {})) != {"weights.pt", "golden.npz"}):
            raise ValueError("Incompatible release/runtime contract")
        if type(threads) is not int or threads not in (1, 2, 4):
            raise ValueError("Inference threads must be 1, 2 or 4")
        for name, expected in m["sha256"].items():
            if sha256(self.root / name) != expected:
                raise ValueError(f"Release hash mismatch: {name}")
        self.center = np.asarray(m["center"], dtype=np.float32)
        self.scale = np.asarray(m["scale"], dtype=np.float32)
        if (self.center.shape != (2,) or self.scale.shape != (2,) or
                not np.isfinite(self.center).all() or not np.isfinite(self.scale).all() or
                np.any(self.scale <= 0)):
            raise ValueError("Invalid target transformation")
        torch.set_num_threads(threads)
        self.model = make_model(m["architecture"], m["attention"]).eval()
        if sum(p.numel() for p in self.model.parameters()) != m["parameters"]:
            raise ValueError("Release parameter count mismatch")
        self.model.load_state_dict(torch.load(self.root / "weights.pt", map_location="cpu", weights_only=True))
        self.bundle_id = m["bundle_id"]
        self.model_label = f"V2 candidate / {m['architecture']} / attention={m['attention']}"
        with np.load(self.root / "golden.npz", allow_pickle=False) as fixtures:
            actual = self.predict(fixtures["waves"])
            expected = fixtures["predictions"]
            if (actual.shape != expected.shape or not np.isfinite(expected).all() or
                    not np.allclose(actual, expected, atol=.01, rtol=0)):
                raise ValueError("Release golden prediction parity failed")

    def predict(self, waves, input_format="filtered"):
        if input_format not in ("filtered", "raw125"):
            raise ValueError("input_format must be filtered or raw125")
        waves = np.asarray(waves)
        if waves.ndim != 3 or waves.shape[1:] != (2, 1250) or not len(waves):
            raise ValueError("Expected nonempty (N,2,1250) waveforms")
        transform = from_raw_125 if input_format == "raw125" else normalize
        x = np.stack([transform(w) for w in waves])
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(x)).numpy() * self.scale + self.center
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite prediction")
        return prediction
