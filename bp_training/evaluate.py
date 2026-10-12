"""Completed-run evaluation and paired comparisons with explicit dataset roles."""

import json
from pathlib import Path

import numpy as np
import torch

from bp_runtime.models import make_model
from bp_runtime.release import BundlePredictor, sha256
from .config import Config
from .data import validate_manifest, WaveDataset
from .io import ROOT, atomic_json, digest, object_hash
from .metrics import metrics, paired_difference
from .trainer import external_output, predict_dataset, select_device


def completed_checkpoint(run_dir):
    run_dir = Path(run_dir)
    status = json.loads((run_dir / "status.json").read_text())
    if status["state"] != "completed":
        raise ValueError("Only completed runs are eligible")
    cp = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if cp.get("schema") != "bp_checkpoint_v1" or cp["fingerprint"] != status["fingerprint"]:
        raise ValueError("Checkpoint does not match run status")
    for name in ("bp_runtime/models.py", "bp_runtime/preprocessing.py"):
        if cp["code"]["files"].get(name) != digest(ROOT / name):
            raise ValueError("Inference implementation differs from the training revision")
    return cp


def evaluate_run(run_dir, output):
    """Development comparison only; final cohorts use evaluate_bundle."""
    if external_output(output).exists():
        raise FileExistsError(output)
    cp = completed_checkpoint(run_dir)
    audit = validate_manifest(Path(run_dir) / "dataset.json")
    if audit["hash"] != cp["dataset_hash"]:
        raise ValueError("Evaluation data differs from the training manifest")
    config = Config(**{**cp["config"], "device": "cpu", "precision": "fp32", "workers": 0})
    device = select_device(config)
    model = make_model(config.architecture, config.attention)
    model.load_state_dict(cp["model"])
    ds = WaveDataset(audit["manifest"], audit["indices"]["validation"])
    predictions, rows = predict_dataset(model, ds, np.asarray(cp["center"], dtype=np.float32),
                                       np.asarray(cp["scale"], dtype=np.float32), config, device)
    return write_evaluation(output, audit, rows, predictions, {
        "kind": "development", "dataset_hash": audit["hash"], "config": cp["config"],
        "fingerprint": cp["fingerprint"], "selected_epoch": cp["selected_epoch"],
        "checkpoint_sha256": sha256(Path(run_dir) / "best.pt")})


def evaluate_bundle(bundle_dir, manifest, output, freeze_record=None, training_manifest=None):
    if external_output(output).exists():
        raise FileExistsError(output)
    audit = validate_manifest(manifest)
    predictor = BundlePredictor(bundle_dir)
    role = audit["manifest"]["role"]
    if role == "development":
        rows = audit["indices"]["validation"]
    else:
        rows = audit["indices"]["all"]
        if freeze_record is None:
            raise ValueError("Historical/prospective evaluation requires a written artifact/protocol freeze record")
        freeze = json.loads(Path(freeze_record).read_text())
        if (freeze.get("bundle_manifest_sha256") != sha256(Path(bundle_dir) / "manifest.json") or
                freeze.get("dataset_hash") != audit["hash"] or
                not isinstance(freeze.get("protocol"), str) or not freeze["protocol"].strip()):
            raise ValueError("Freeze record must match bundle, dataset and a nonempty evaluation protocol")
        if training_manifest is None:
            raise ValueError("Held-out evaluation requires the private training manifest for participant exclusion")
        training = validate_manifest(training_manifest)
        if (training["manifest"]["role"] != "development" or
                training["hash"] != predictor.manifest["training"]["dataset_hash"]):
            raise ValueError("Training manifest does not match the selected bundle")
        if set(training["subjects"]) & set(audit["subjects"]):
            raise ValueError("Held-out participants overlap training or development validation participants")
    ds = WaveDataset(audit["manifest"], rows)
    predictions = []
    for start in range(0, len(ds), 64):
        waves = np.stack([ds[i][0] for i in range(start, min(start + 64, len(ds)))])
        predictions.append(predictor.predict(waves))
    return write_evaluation(output, audit, rows, np.concatenate(predictions), {
        "kind": role, "dataset_hash": audit["hash"],
        "bundle_manifest_sha256": sha256(Path(bundle_dir) / "manifest.json"),
        "freeze_record_sha256": sha256(freeze_record) if freeze_record else None,
        "training_manifest_hash": training["hash"] if role != "development" else None,
        "scope": "Offline waveform accuracy; acquisition/quality/coverage require hardware evaluation"})


def write_evaluation(output, audit, rows, predictions, provenance):
    output = external_output(output)
    output.mkdir(parents=True, exist_ok=False)
    labels, subjects = audit["labels"][rows], audit["subjects"][rows]
    report = {"provenance": provenance, "metrics": metrics(labels, predictions, subjects, bootstrap=2000),
              "eligible_fraction_of_manifest": audit["eligible_rows"] / audit["rows"]}
    # Private outputs: participant identifiers are never copied into release bundles.
    np.savez(output / "predictions.npz", rows=rows, labels=labels, subjects=subjects, predictions=predictions)
    atomic_json(output / "report.json", report)
    return report


def compare_evaluations(baseline, candidate, output):
    a = json.loads((Path(baseline) / "report.json").read_text())
    b = json.loads((Path(candidate) / "report.json").read_text())
    if any(r["provenance"]["kind"] != "development" for r in (a, b)):
        raise ValueError("Model-selection comparisons accept development evaluations only")
    if a["provenance"]["dataset_hash"] != b["provenance"]["dataset_hash"]:
        raise ValueError("Paired comparisons require the same dataset/split")
    with np.load(Path(baseline) / "predictions.npz", allow_pickle=False) as aa, \
            np.load(Path(candidate) / "predictions.npz", allow_pickle=False) as bb:
        for key in ("rows", "labels", "subjects"):
            if not np.array_equal(aa[key], bb[key]):
                raise ValueError("Comparison records are not aligned")
        result = paired_difference(aa["labels"], aa["predictions"], bb["predictions"], aa["subjects"])
    result["baseline"] = str(Path(baseline).resolve())
    result["candidate"] = str(Path(candidate).resolve())
    output = external_output(output)
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, result)
    return result


def summarize_evaluations(evaluations, output, seeds=(20260923, 20260924, 20260925)):
    """Keep training-seed variation separate from participant confidence intervals."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected seeds must be distinct and nonempty")
    output = external_output(output)
    if output.exists():
        raise FileExistsError(output)
    groups, dataset_hash = {}, None
    for directory in evaluations:
        report = json.loads((Path(directory) / "report.json").read_text())
        provenance = report["provenance"]
        if provenance["kind"] != "development" or "config" not in provenance:
            raise ValueError("Seed summaries require completed-run development evaluations")
        if dataset_hash is None:
            dataset_hash = provenance["dataset_hash"]
        if provenance["dataset_hash"] != dataset_hash:
            raise ValueError("Summaries must use the same dataset and split")
        config = dict(provenance["config"])
        seed = config.pop("seed")
        key = object_hash(config)
        group = groups.setdefault(key, {"config_without_seed": config, "runs": {}})
        if seed in group["runs"]:
            raise ValueError("Duplicate seed for an experiment condition")
        group["runs"][seed] = {"directory": str(Path(directory).resolve()),
                               "macro_mae": report["metrics"]["macro_mae"],
                               "primary": report["metrics"]["primary"]}
    if not groups:
        raise ValueError("No evaluations supplied")
    result = {"dataset_hash": dataset_hash, "expected_seeds": list(seeds), "conditions": []}
    for group in groups.values():
        if set(group["runs"]) != set(seeds):
            raise ValueError("Incomplete or unexpected seed set; do not compare partial campaigns")
        values = np.asarray([group["runs"][s]["macro_mae"] for s in seeds], dtype=np.float64)
        if values.shape != (len(seeds), 2) or not np.isfinite(values).all():
            raise ValueError("Invalid seed metrics")
        group["mean_macro_mae"] = values.mean(0).tolist()
        group["training_seed_sd"] = values.std(0, ddof=1).tolist() if len(seeds) > 1 else None
        group["primary_mean"] = float(values.mean())
        result["conditions"].append(group)
    result["note"] = "Seed SD is training variability, not a participant confidence interval or a promotion decision"
    atomic_json(output, result)
    return result
