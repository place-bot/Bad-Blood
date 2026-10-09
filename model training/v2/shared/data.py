"""Frozen VitalDB data contracts for the two contemporaneous BP experiments.

Population rows have no recoverable timestamps. Personal rows retain the
original MATLAB column indices, with support/query membership read from the
versioned audit rather than selected using outcomes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


V2_ROOT = Path(__file__).resolve().parents[1]
EXCLUSION_PATH = (
    V2_ROOT.parent / "PulseDB-based Model-V1/reports/prior_exploratory_subjects.json"
)
DEFAULT_AUDIT_PATH = V2_ROOT / "audit_snapshot.json"
SOURCE_COUNTS = {"train": (465480, 1293), "test": (57600, 144), "aami": (666, 116)}
POPULATION_COUNTS = {
    "train": (390600, 1085), "validation": (69120, 192),
    "test": (56400, 141), "aami": (666, 116),
}
EXCLUDED_PATIENT_COUNT = 31
FS = 125
SAMPLES = 1250
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _file_hash(path: Path, *, cache: bool = False) -> str:
    """Hash all bytes; only large waveform arrays use the process-local cache.

    A cached identity is reusable only when resolved path, byte count, and
    modification time in nanoseconds all match. Nothing is written to source
    data or Git. Small identity/label/metadata files are always rehashed.
    """
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if cache and key in _HASH_CACHE:
        return _HASH_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    _require((stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             f"File changed while hashing: {path}")
    result = digest.hexdigest()
    if cache:
        _HASH_CACHE[key] = result
    return result


def _identity(path: Path, *, cache: bool = False) -> dict:
    return {"bytes": path.stat().st_size, "sha256": _file_hash(path, cache=cache)}


def _canonical_subjects(values) -> np.ndarray:
    values = np.asarray(values)
    _require(values.ndim == 1, "Subject identifiers must be a vector")
    unique, inverse = np.unique(values.astype(str), return_inverse=True)
    canonical = []
    for subject in unique:
        _require(len(subject) == 9 and subject.startswith("p")
                 and subject[1:7].isdigit() and subject.endswith("_1"),
                 f"Expected a VitalDB source-qualified pXXXXXX_1 ID, got {subject!r}")
        canonical.append("vitaldb:" + subject[:7])
    return np.asarray(canonical, dtype="U15")[inverse]


def _validate_waveforms(waves: np.ndarray, label: str) -> None:
    _require(waves.ndim == 3 and waves.shape[1:] == (2, SAMPLES),
             f"{label}: expected N x 2 x 1250 waveforms")
    _require(waves.dtype == np.float32, f"{label}: expected float32 waveforms")
    for start in range(0, len(waves), 1024):
        chunk = waves[start:start + 1024]
        _require(bool(np.isfinite(chunk).all()), f"{label}: nonfinite input waveforms")
        span = np.ptp(chunk, axis=-1)
        _require(bool((span > 1e-8).all()), f"{label}: flat input waveforms")
        _require(bool((chunk.min() >= -1e-6) and (chunk.max() <= 1 + 1e-6)),
                 f"{label}: input is not the frozen min-max representation")


def _validate_labels(y: np.ndarray, rows: int, label: str) -> None:
    _require(y.shape == (rows, 2), f"{label}: expected N x 2 SBP/DBP labels")
    _require(bool(np.isfinite(y).all()), f"{label}: nonfinite pressure reference")
    _require(bool((y[:, 0] > y[:, 1]).all()), f"{label}: SBP must exceed DBP")


def load_population(root) -> dict:
    """Load complete frozen population partitions without copying large waves.

    Each partition exposes full source arrays and its eligible row ``indices``.
    Train/validation share exactly the same source-array objects. The manifest
    includes full content hashes; absolute locations are excluded from its hash
    so an identical copy remains reproducible at another storage location.
    """
    root = Path(root).resolve()
    split_path = root / "split.json"
    split = json.loads(split_path.read_text())
    exclusion_path = Path(EXCLUSION_PATH)
    excluded_raw = json.loads(exclusion_path.read_text())
    excluded = set(_canonical_subjects(excluded_raw).tolist())
    _require(len(excluded_raw) == len(excluded) == EXCLUDED_PATIENT_COUNT,
             "The historical population-exclusion identity list changed")
    split_sets = {}
    for name in ("train", "validation"):
        values = _canonical_subjects(split[name])
        _require(len(values) == len(set(values)), f"Duplicate IDs in {name} split")
        split_sets[name] = set(values.tolist())
    _require(not split_sets["train"] & split_sets["validation"],
             "Patient overlap between original training and validation")

    manifest = {
        "schema_version": 1, "task": "contemporaneous_SBP_DBP",
        "fs_hz": FS, "samples_per_channel": SAMPLES,
        "source_identity": {"split.json": _identity(split_path),
                            "prior_exploratory_subjects.json": _identity(exclusion_path)},
        "excluded_patients": sorted(excluded), "partitions": {},
        "waveform_hash_cache": "process-local; validated by path, size, mtime_ns",
    }
    source = {}
    for name, (expected_rows, expected_patients) in SOURCE_COUNTS.items():
        folder = root / name
        metadata_path = folder / "complete.json"
        metadata = json.loads(metadata_path.read_text())
        arrays = {key: np.load(folder / f"{key}.npy", mmap_mode="r" if key == "waves" else None,
                               allow_pickle=False)
                  for key in ("waves", "y", "subjects", "valid")}
        subjects = _canonical_subjects(arrays["subjects"])
        rows = len(subjects)
        _require((rows, len(set(subjects))) == (expected_rows, expected_patients),
                 f"{name}: source row or patient count changed")
        _require(arrays["waves"].shape == (rows, 2, SAMPLES),
                 f"{name}: waveform axes or row alignment changed")
        _require(arrays["valid"].shape == (rows,) and arrays["valid"].dtype == bool,
                 f"{name}: invalid validity-mask contract")
        _require(metadata["rows"] == rows and metadata["subjects"] == expected_patients
                 and metadata["invalid"] == int((~arrays["valid"]).sum()),
                 f"{name}: completion metadata disagrees with arrays")
        _validate_waveforms(arrays["waves"], name)
        _validate_labels(arrays["y"], rows, name)
        arrays["subjects"] = subjects
        source[name] = arrays
        manifest["source_identity"][f"{name}/complete.json"] = _identity(metadata_path)
        for key in ("waves", "y", "subjects", "valid"):
            manifest["source_identity"][f"{name}/{key}.npy"] = _identity(
                folder / f"{key}.npy", cache=key == "waves")

    _require(split_sets["train"] | split_sets["validation"] == set(source["train"]["subjects"]),
             "Original split manifest does not cover exactly the training source patients")
    result = {}
    patient_sets = {}
    for name in POPULATION_COUNTS:
        arrays = source["train" if name == "validation" else name]
        subjects = arrays["subjects"]
        keep = arrays["valid"] & ~np.isin(subjects, sorted(excluded))
        if name in split_sets:
            keep &= np.isin(subjects, sorted(split_sets[name]))
        indices = np.flatnonzero(keep)
        patient_sets[name] = set(subjects[indices])
        _require((len(indices), len(patient_sets[name])) == POPULATION_COUNTS[name],
                 f"{name}: corrected partition counts do not match the frozen protocol")
        result[name] = {key: arrays[key] for key in ("waves", "y", "subjects")}
        result[name]["indices"] = indices
        manifest["partitions"][name] = {
            "rows": len(indices), "patients": sorted(patient_sets[name]),
            "source_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        }
    names = list(patient_sets)
    for position, name in enumerate(names):
        for other in names[position + 1:]:
            _require(not patient_sets[name] & patient_sets[other],
                     f"Patient leakage between {name} and {other}")
    selected_y = result["train"]["y"][result["train"]["indices"]]
    center = selected_y.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = selected_y.std(axis=0, dtype=np.float64).astype(np.float32)
    _require(bool(np.isfinite(scale).all() and (scale > 0).all()),
             "Training target scale is zero or nonfinite")
    manifest["target_center"] = center.tolist()
    manifest["target_scale"] = scale.tolist()
    result.update(manifest=manifest, manifest_hash=_json_hash(manifest), center=center, scale=scale)
    return result


def _read_ref(handle, group, field: str, index: int) -> np.ndarray:
    return np.asarray(handle[group[field][0, index]]).reshape(-1)


def _read_string(handle, group, field: str, index: int) -> str:
    return "".join(chr(int(value)) for value in _read_ref(handle, group, field, index))


def _manifest_indices(entries, start, end, field: str) -> np.ndarray:
    values = np.asarray(entries, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 3 and len(values) > 0,
             f"{field}: invalid frozen source-index/time table")
    indices = values[:, 0].astype(np.int64)
    _require(bool(np.equal(values[:, 0], indices).all())
             and len(set(indices)) == len(indices)
             and bool(((indices >= 0) & (indices < len(start))).all()),
             f"{field}: duplicate, fractional, or out-of-range source indices")
    _require(bool(np.allclose(start[indices], values[:, 1], rtol=0, atol=1e-6))
             and bool(np.allclose(end[indices], values[:, 2], rtol=0, atol=1e-6)),
             f"{field}: frozen timestamps disagree with original source rows")
    return indices


def load_personal_cohort(individual_root, audit_path=DEFAULT_AUDIT_PATH) -> dict:
    """Read frozen support/query records and normalize only ECG_F/PPG_F.

    No pressure samples, reference-derived landmarks, subject identifiers, or
    timestamps enter ``x``. ABP-derived scalar labels remain separate in ``y``.
    The full original row numbering is retained; failures abort, never drop rows.
    """
    root = Path(individual_root).resolve()
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text())
    protocol = audit["final_personalization_protocol"]
    _require(protocol["calibration_budgets"] == [1, 2, 3, 5, 10], "Calibration budgets changed")
    file_metadata = {Path(row["file"]).name: row for row in audit["individual_files"]["files"]}
    manifest = {"audit_sha256": _file_hash(audit_path), "source_identity": {},
                "source_time_diagnostics": {}}
    result = {"development": [], "target": []}
    seen = set()
    for record in protocol["patients"]:
        patient = record["patient_id"]
        subject = record["subject_id"]
        case = record["case_id"]
        role = record["role"]
        _require(patient == "vitaldb:" + subject and patient not in seen,
                 "Repeated or inconsistent personal patient identity")
        _require(role in result, f"Unknown personal partition: {role}")
        seen.add(patient)
        path = root / (subject + ".mat")
        metadata = file_metadata[path.name]
        _require(metadata["status"] == "complete"
                 and path.stat().st_size == metadata["expected_bytes"] == metadata["actual_bytes"],
                 f"Incomplete or changed personal source: {path.name}")
        manifest["source_identity"][path.name] = _identity(path)
        with h5py.File(path, "r") as handle:
            group = handle["Subj_Wins"]
            n = group["T"].size
            _require(n == record["total_retained_windows"], f"{subject}: row count changed")
            x = np.empty((n, 2, SAMPLES), dtype=np.float32)
            y = np.empty((n, 2), dtype=np.float32)
            start = np.empty(n, dtype=np.float64)
            end = np.empty(n, dtype=np.float64)
            for i in range(n):
                _require(_read_string(handle, group, "SubjectID", i) == subject
                         and _read_string(handle, group, "CaseID", i) == case,
                         f"{subject}: patient/case mismatch at row {i}")
                timestamps = _read_ref(handle, group, "T", i)
                _require(timestamps.shape == (SAMPLES,) and bool(np.isfinite(timestamps).all())
                         and bool(np.allclose(np.diff(timestamps), 1 / FS, rtol=0, atol=1e-6)),
                         f"{subject}: invalid sample times at row {i}")
                start[i], end[i] = timestamps[0], timestamps[-1] + 1 / FS
                channels = [_read_ref(handle, group, key, i) for key in ("ECG_F", "PPG_F")]
                _require(all(channel.shape == (SAMPLES,) for channel in channels),
                         f"{subject}: invalid input shape at row {i}")
                wave = np.asarray(channels, dtype=np.float32)
                span = np.ptp(wave, axis=1, keepdims=True)
                _require(bool(np.isfinite(wave).all() and (span > 1e-8).all()),
                         f"{subject}: nonfinite or flat input at row {i}")
                x[i] = (wave - wave.min(axis=1, keepdims=True)) / span
                for column, key in enumerate(("SegSBP", "SegDBP")):
                    value = _read_ref(handle, group, key, i)
                    _require(value.shape == (1,), f"{subject}: non-scalar reference at row {i}")
                    y[i, column] = value[0]
        _validate_labels(y, n, subject)
        _require(bool((np.diff(start) > 0).all()), f"{subject}: unordered source windows")
        # Two audited query pairs overlap by one source sample. Preserve the
        # frozen rows and report this upstream timestamp anomaly; the separate
        # support/query boundary checks below still forbid calibration leakage.
        overlap_rows = np.flatnonzero(start[1:] < end[:-1] - 1e-6)
        manifest["source_time_diagnostics"][patient] = {
            "overlapping_adjacent_source_windows": [
                {"source_row": int(i), "next_source_row": int(i + 1),
                 "overlap_seconds": float(end[i] - start[i + 1])}
                for i in overlap_rows],
        }
        boundary = float(record["calibration_boundary_s"])
        expected_boundary = start[0] + .2 * (end[-1] - start[0])
        _require(abs(boundary - expected_boundary) < 1e-6,
                 f"{subject}: elapsed-time boundary changed")
        support = _manifest_indices(record["support"], start, end, "support")
        query = _manifest_indices(record["queries"], start, end, "query")
        _require(len(support) == 10 and len(query) == record["query_count"],
                 f"{subject}: support/query counts changed")
        _require(bool((np.diff(start[support]) >= 60 - 1e-6).all())
                 and bool((end[support] <= boundary + 1e-6).all()),
                 f"{subject}: support budget crosses spacing or calibration boundary")
        _require(bool((start[query] >= boundary + 10 - 1e-6).all())
                 and not set(support) & set(query), f"{subject}: support/query temporal leakage")
        expected_query = np.flatnonzero(start >= boundary + 10 - 1e-6)
        _require(np.array_equal(query, expected_query), f"{subject}: query manifest omits or reorders rows")
        pool = []
        for index in np.flatnonzero(end <= boundary + 1e-6):
            if not pool or start[index] - start[pool[-1]] >= 60 - 1e-6:
                pool.append(int(index))
        _require(len(pool) >= 10, f"{subject}: insufficient independent-time support windows")
        expected_support = np.asarray(pool)[np.rint(np.linspace(0, len(pool) - 1, 10)).astype(int)]
        _require(np.array_equal(support, expected_support), f"{subject}: support selection changed")
        result[role].append({
            "patient_id": patient, "case_id": case, "x": x, "y": y,
            "start": start, "end_exclusive": end, "source_indices": np.arange(n, dtype=np.int64),
            "support_indices": support, "query_indices": query, "calibration_boundary": boundary,
        })
    for role in ("development", "target"):
        _require({row["patient_id"] for row in result[role]}
                 == {"vitaldb:" + sid for sid in protocol[role + "_subjects"]},
                 f"{role}: cohort manifest membership changed")
    result["manifest"] = manifest
    result["manifest_hash"] = _json_hash(manifest)
    return result
