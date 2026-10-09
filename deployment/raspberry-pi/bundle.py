"""Fail closed when the released predictor and preprocessing no longer match."""

import hashlib
import json
from pathlib import Path

from acquisition import RAW_VERSION
from sources import CHANNELS, FS_HZ, PREPROCESSING, SAMPLES

MANIFEST = Path(__file__).with_name("bundle.json")
REQUIRED_FILES = {"common.py", "network.py", "inference.py", "models/selection.json",
                  "models/cnn_full.pt"}


def verify_bundle(model_root, manifest_path=MANIFEST):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = {"fs_hz": FS_HZ, "channels": list(CHANNELS), "samples": SAMPLES,
                "preprocessing": PREPROCESSING, "raw_adapter": RAW_VERSION}
    if (manifest.get("schema") != "bad_blood_bundle_v1"
            or manifest.get("contract") != expected
            or set(manifest.get("sha256", {})) != REQUIRED_FILES):
        raise ValueError("Deployment bundle contract does not match this runtime")
    for name, digest in manifest["sha256"].items():
        actual = hashlib.sha256((Path(model_root) / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Deployment bundle hash mismatch: {name}")
    return manifest["bundle_id"]
