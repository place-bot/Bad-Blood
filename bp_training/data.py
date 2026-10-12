"""Hashed array manifests, participant isolation and lazy waveform access."""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from bp_runtime.preprocessing import CONTRACT
from .io import atomic_json, digest, object_hash


def build_manifest(data_dir, output, split=None, exclusions=(), source="PulseDB/VitalDB",
                   role="development"):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Manifest exists; use a new versioned path")
    paths = {name: Path(data_dir) / filename for name, filename in
             {"waves": "waves.npy", "labels": "y.npy", "subjects": "subjects.npy",
              "valid": "valid.npy"}.items()}
    if split is not None:
        paths["split"] = Path(split)
    paths.update({f"excluded_{i}": Path(p) for i, p in enumerate(exclusions)})
    value = {"schema": "bp_dataset_v1", "contract": CONTRACT, "source": source,
             "role": role, "identity_scope": "single_source_canonical_participants",
             "files": {k: {"path": str(p.resolve()), "sha256": digest(p)} for k, p in paths.items()}}
    validate_manifest(value)
    atomic_json(output, value)
    return value


def validate_manifest(value):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if value.get("schema") != "bp_dataset_v1" or value.get("contract") != CONTRACT:
        raise ValueError("Unsupported dataset contract")
    if value.get("role") not in ("development", "historical", "prospective"):
        raise ValueError("Unsupported dataset role")
    if not isinstance(value.get("source"), str) or not value["source"].strip():
        raise ValueError("Source release/provenance is required")
    if value.get("identity_scope") != "single_source_canonical_participants":
        raise ValueError("Cross-source identity reconciliation is not implemented")
    files = value["files"]
    if not {"waves", "labels", "subjects", "valid"} <= set(files):
        raise ValueError("Incomplete dataset files")
    for entry in files.values():
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("Dataset hash mismatch")
    arrays = {name: np.load(files[name]["path"], mmap_mode="r", allow_pickle=False)
              for name in ("waves", "labels", "subjects", "valid")}
    waves, y, subjects, valid = [arrays[k] for k in ("waves", "labels", "subjects", "valid")]
    n = len(waves)
    if waves.shape != (n, 2, 1250) or waves.dtype != np.float32:
        raise ValueError("Waveforms must be float32 (N,2,1250)")
    if y.shape != (n, 2) or y.dtype != np.float32:
        raise ValueError("Labels must be float32 (N,2) in SBP/DBP mmHg order")
    if subjects.shape != (n,) or subjects.dtype.kind not in "US":
        raise ValueError("Expected one non-object participant ID per window")
    if valid.shape != (n,) or valid.dtype != np.bool_ or not valid.any():
        raise ValueError("Expected a nonempty boolean eligibility mask")
    ids = subjects.astype(str)
    if np.any(np.char.str_len(ids) == 0):
        raise ValueError("Empty participant ID")
    for start in range(0, n, 512):
        mask = valid[start:start + 512]
        x = waves[start:start + 512][mask]
        labels = y[start:start + 512][mask]
        if not np.isfinite(x).all() or not np.isfinite(labels).all():
            raise ValueError("Eligible rows contain nonfinite values")
        if len(x) and (np.any(np.abs(x.min(2)) > 1e-5) or
                       np.any(np.abs(x.max(2) - 1) > 1e-5)):
            raise ValueError("Eligible waves must use the frozen per-window min-max representation")
        if np.any(labels[:, 0] <= labels[:, 1]) or np.any(labels[:, 1] <= 0):
            raise ValueError("Eligible rows have invalid BP labels")
    exclusions = set()
    for key, entry in files.items():
        if key.startswith("excluded_"):
            excluded = np.load(entry["path"], allow_pickle=False)
            if excluded.ndim != 1 or excluded.dtype.kind not in "US":
                raise ValueError("Excluded participants must be a one-dimensional string array")
            exclusions.update(excluded.astype(str).tolist())
    if exclusions & set(ids):
        raise ValueError("Participant overlap with excluded cohort")
    indices = {"all": np.flatnonzero(valid)}
    if value["role"] == "development":
        if "split" not in files or not any(k.startswith("excluded_") for k in files):
            raise ValueError("Development manifest requires split and explicit holdout identity files")
        split = json.loads(Path(files["split"]["path"]).read_text())
        train, validation = split["train"], split["validation"]
        if (any(not isinstance(s, str) for s in train + validation) or
                len(set(train)) != len(train) or len(set(validation)) != len(validation) or
                set(train) & set(validation) or set(train + validation) != set(ids)):
            raise ValueError("Invalid or overlapping participant split")
        for name, members in (("train", train), ("validation", validation)):
            indices[name] = np.flatnonzero(valid & np.isin(ids, members))
            if not len(indices[name]) or set(ids[indices[name]]) != set(members):
                raise ValueError("Every split participant needs eligible records")
    return {"manifest": value, "hash": object_hash(value), "indices": indices,
            "labels": y, "subjects": ids, "rows": n,
            "eligible_rows": int(valid.sum()), "participants": len(set(ids))}


class WaveDataset(Dataset):
    def __init__(self, manifest, indices):
        self.wave_path = manifest["files"]["waves"]["path"]
        self.label_path = manifest["files"]["labels"]["path"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self._waves = self._labels = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_waves"] = state["_labels"] = None
        return state

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self._waves is None:
            self._waves = np.load(self.wave_path, mmap_mode="r", allow_pickle=False)
            self._labels = np.load(self.label_path, mmap_mode="r", allow_pickle=False)
        row = self.indices[index]
        return np.array(self._waves[row], copy=True), np.array(self._labels[row], copy=True), int(row)


def training_statistics(audit):
    if audit["manifest"]["role"] != "development":
        raise ValueError("Target statistics may only be fitted on development training data")
    y = audit["labels"][audit["indices"]["train"]]
    center, scale = y.mean(0), y.std(0)
    if not np.isfinite(scale).all() or np.any(scale < 1e-6):
        raise ValueError("Training targets need nonzero variance")
    return center, scale


def sampling_probabilities(y, subjects, config):
    if not config.balancing:
        return None
    bins = np.digitize(y[:, 0], config.sbp_edges) * (len(config.dbp_edges) + 1)
    bins += np.digitize(y[:, 1], config.dbp_edges)
    weights = np.ones(len(y), dtype=np.float64)
    counts = {b: int((bins == b).sum()) for b in np.unique(bins)}
    largest = max(counts.values())
    for b, count in counts.items():
        mask = bins == b
        if len(np.unique(subjects[mask])) >= config.balance_min_subjects:
            weights[mask] = min(config.balance_cap, largest / count)
    return weights / weights.sum()
