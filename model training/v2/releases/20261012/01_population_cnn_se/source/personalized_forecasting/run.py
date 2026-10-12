"""Frozen Track B: early personal calibration, later contemporaneous SBP/DBP.

The historical directory name does not imply open-loop forecasting. Development
and final evaluation are separate CLI phases; final evaluation cannot select
hyperparameters. Resume restarts an incomplete independent unit from baseline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from personalized_forecasting.adaptation import (  # noqa: E402
    NEURAL_METHODS, adapt_model, apply_calibration, fit_calibration, predict,
)
from shared.metrics import summarize  # noqa: E402
from shared.models import BPNet  # noqa: E402

SEEDS = (42, 43, 44)
BUDGETS = (1, 2, 3, 5, 10)
DEV_IDS = tuple("vitaldb:" + value for value in ("p005548", "p005647", "p005968", "p006090"))
TARGET_IDS = tuple("vitaldb:" + value for value in ("p005879", "p005885", "p005977", "p005984", "p006009", "p006052"))
CONTROLS = ("population", "bias", "last_reference", "calibration_mean")
DRIFT_BINS = ((0, 15), (15, 30), (30, 60), (60, None))


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def array_hash(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_population_checkpoint(path: Path, seed: int) -> dict:
    """Reject partial fits, V1/reused weights, toy data and personal leakage."""
    directory = path.parent
    required = (path, directory / "completion.json", directory / "config.json", directory / "manifest.json")
    if any(not item.exists() for item in required):
        raise FileNotFoundError(f"Completed population fit (including completion/config/manifest) required: {directory}")
    completion = json.loads((directory / "completion.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    saved = torch.load(path, map_location="cpu", weights_only=False)
    digest = file_hash(path)
    if completion.get("status") != "complete" or completion.get("toy_test_only") is not False or config.get("toy_test_only") is not False:
        raise ValueError("Only completed real population experiments may supply Track B")
    if completion["best_checkpoint_sha256"] != digest:
        raise ValueError("Population best checkpoint does not match completed fit")
    if saved["variant"] != "baseline" or saved["seed"] != seed or saved["pool"] != 1:
        raise ValueError("Track B requires the matched-seed unchanged high-resolution baseline")
    if config.get("variant") != "baseline" or config.get("seed") != seed or config.get("pool") != 1:
        raise ValueError("Population training configuration does not match requested baseline seed")
    if saved["config"] != config or saved["config_hash"] != canonical_hash(config) or completion["config_hash"] != saved["config_hash"]:
        raise ValueError("Population checkpoint configuration mismatch")
    if not config.get("protocol_commit", "").startswith("1b93cf1"):
        raise ValueError("Population fit is not tied to the frozen v2 protocol")
    manifest_hash = canonical_hash(manifest)
    if saved["hashes"]["manifest"] != manifest_hash or completion["manifest_hash"] != manifest_hash:
        raise ValueError("Population split manifest mismatch")
    forbidden = set(DEV_IDS) | set(TARGET_IDS)
    excluded = set(manifest["excluded_patients"])
    if len(excluded) != 31 or not forbidden <= excluded:
        raise ValueError("Population fit did not exclude all personal patients and historical exclusions")
    for name, partition in manifest["partitions"].items():
        if forbidden & set(partition["patients"]):
            raise ValueError(f"Personal cohort leakage in population {name}")
    if int(saved["epoch"]) != int(completion["selected_epoch"]):
        raise ValueError("Population checkpoint is not the selected best epoch")
    for key, manifest_key in (("center", "target_center"), ("scale", "target_scale")):
        if not np.array_equal(saved[key], manifest[manifest_key]):
            raise ValueError("Population-only target normalization mismatch")
    return saved


def validate_patient(patient: dict) -> None:
    """Defense in depth against alignment, boundary or support-selection drift."""
    x, y = np.asarray(patient["x"]), np.asarray(patient["y"])
    n = len(x)
    start, end = np.asarray(patient["start"]), np.asarray(patient["end_exclusive"])
    source = np.asarray(patient["source_indices"])
    support, query = np.asarray(patient["support_indices"]), np.asarray(patient["query_indices"])
    if x.shape != (n, 2, 1250) or y.shape != (n, 2):
        raise ValueError("Patient input/label shape mismatch")
    if any(a.shape != (n,) for a in (start, end, source)) or len(np.unique(source)) != n:
        raise ValueError("Patient timestamps and unique source indices must align")
    if not all(np.isfinite(a).all() for a in (x, y, start, end)):
        raise ValueError("Patient contains nonfinite data")
    if np.any(np.diff(start) < 0) or np.any(end <= start):
        raise ValueError("Patient data must be chronological, positive-duration windows")
    if support.shape != (10,) or len(np.unique(support)) != 10 or not len(query):
        raise ValueError("Frozen cohort requires ten unique supports and nonempty query")
    for indices in (support, query):
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("Invalid patient row indices")
    boundary = float(patient["calibration_boundary"])
    expected_boundary = start[0] + 0.2 * (end[-1] - start[0])
    if not np.isclose(boundary, expected_boundary, rtol=0, atol=1e-5):
        raise ValueError("Calibration boundary is not first 20% of elapsed span")
    expected_query = np.flatnonzero(start >= boundary + 10 - 1e-6)
    if not np.array_equal(query, expected_query):
        raise ValueError("Query must include all whole windows after the guard")
    pool = []
    for index in np.flatnonzero(end <= boundary + 1e-6):
        if not pool or start[index] - start[pool[-1]] >= 60 - 1e-6:
            pool.append(index)
    if len(pool) < 10:
        raise ValueError("Insufficient early >=60s-spaced references; do not impute budgets")
    expected_support = np.asarray(pool)[np.rint(np.linspace(0, len(pool) - 1, 10)).astype(int)]
    if not np.array_equal(support, expected_support):
        raise ValueError("Support selection differs from frozen metadata-only rule")
    if np.intersect1d(support, query).size or np.max(end[support]) > boundary + 1e-6 or np.min(start[query]) < boundary + 10 - 1e-6:
        raise ValueError("Support/query leakage or guard violation")


def settings_for(method: str, budget: int) -> list[dict]:
    if method in NEURAL_METHODS:
        return [{"lr": lr, "steps": steps} for steps in (20, 100) for lr in (1e-4, 1e-3)]
    if method == "affine":
        return [{"penalty": penalty} for penalty in (1, 10, 100)] if budget >= 3 else []
    return [{}]


def setting_name(setting: dict) -> str:
    if "lr" in setting:
        return f"lr{setting['lr']:g}_steps{setting['steps']}"
    if "penalty" in setting:
        return f"lambda{setting['penalty']:g}"
    return "fixed"


def choose_settings(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Require every dev patient/seed; equal weights, deterministic tie-breaks."""
    selected, grid = {}, []
    expected_pairs = {(patient, seed) for patient in DEV_IDS for seed in SEEDS}
    for method in (*NEURAL_METHODS, "affine"):
        for budget in BUDGETS:
            candidates = []
            for setting in settings_for(method, budget):
                matched = [row for row in rows if row["strategy"] == method and
                           row["budget"] == budget and row["setting"] == setting]
                pairs = {(row["patient"], row["seed"]) for row in matched}
                if pairs != expected_pairs or len(matched) != len(expected_pairs):
                    raise ValueError(f"Incomplete or duplicated development grid: {method}/{budget}/{setting}")
                score = float(np.mean([row["selection_score"] for row in matched]))
                if not np.isfinite(score):
                    raise ValueError("Nonfinite development score")
                candidate = {"strategy": method, "budget": budget, "setting": setting,
                             "score": score, "n_patients": 4, "n_seeds": 3,
                             "patient_seed_scores": [{"patient": row["patient"], "seed": row["seed"],
                                                      "score": row["selection_score"]} for row in matched]}
                grid.append(candidate)
                candidates.append(candidate)
            if candidates:
                if method == "affine":
                    best = min(candidates, key=lambda row: (row["score"], -row["setting"]["penalty"]))
                else:
                    best = min(candidates, key=lambda row: (row["score"], row["setting"]["steps"], row["setting"]["lr"]))
                selected[f"{method}/K{budget}"] = best["setting"]
    return selected, grid


def drift_summary(y, prediction, subjects, elapsed) -> list[dict]:
    result = []
    for lower, upper in DRIFT_BINS:
        mask = elapsed >= lower
        if upper is not None:
            mask &= elapsed < upper
        result.append({"lower_minutes_inclusive": lower, "upper_minutes_exclusive": upper,
                       "n_windows": int(mask.sum()), "n_patients": len(np.unique(subjects[mask])),
                       "summary": summarize(y[mask], prediction[mask], subjects[mask]) if mask.any() else None})
    return result


def load_complete(unit_dir: Path, config: dict, resume: bool) -> dict | None:
    record_path = unit_dir / "unit.json"
    if not record_path.exists():
        return None
    if not resume:
        raise FileExistsError(f"Completed unit already exists; use --resume: {unit_dir}")
    record = json.loads(record_path.read_text())
    if record["config_hash"] != canonical_hash(config):
        raise ValueError(f"Refusing to resume different configuration: {unit_dir}")
    for name, digest in record["artifact_hashes"].items():
        if not (unit_dir / name).exists() or file_hash(unit_dir / name) != digest:
            raise ValueError(f"Completed artifact missing/corrupt: {unit_dir / name}")
    return record


def run_unit(patient, baseline, checkpoint, budget, method, setting, phase, seed,
             output_root, provenance_hash, base_support_pred, base_query_pred,
             baseline_latency, device, resume, selection_hash=None):
    unit_dir = output_root / phase / patient["patient_id"] / f"seed{seed}" / f"K{budget}" / method / setting_name(setting)
    support, query = patient["support_indices"][:budget], patient["query_indices"]
    config = {"patient": patient["patient_id"], "case": str(patient["case_id"]), "seed": seed,
              "budget": budget, "strategy": method, "setting": setting, "phase": phase,
              "provenance_hash": provenance_hash, "checkpoint_sha256": checkpoint["sha256"],
              "selection_hash": selection_hash, "device": device,
              "query_indices_hash": array_hash(patient["source_indices"][query]),
              "support_indices_hash": array_hash(patient["source_indices"][support])}
    existing = load_complete(unit_dir, config, resume)
    if existing is not None:
        return existing
    unit_dir.mkdir(parents=True, exist_ok=True)
    center, scale = checkpoint["center"], checkpoint["scale"]
    artifacts = {}
    if method in NEURAL_METHODS:
        result = adapt_model(baseline, patient["x"][support], patient["y"][support], center, scale,
                             strategy=method, device=device, seed=seed, **setting)
        prediction, latency = predict(result.model, patient["x"][query], center, scale, device)
        audit = result.audit
        state_path = unit_dir / "adapted.pt"
        torch.save({"state_dict": {name: value.detach().cpu() for name, value in result.model.state_dict().items()},
                    "center": center, "scale": scale, "variant": "baseline", "config": config},
                   state_path.with_suffix(".pt.tmp"))
        state_path.with_suffix(".pt.tmp").replace(state_path)
        artifacts[state_path.name] = file_hash(state_path)
        del result
    else:
        started = time.perf_counter()
        calibration = fit_calibration(method, base_support_pred[:budget], patient["y"][support],
                                      center, scale, **setting)
        adaptation_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = apply_calibration(calibration, base_query_pred)
        transform_seconds = time.perf_counter() - started
        audit = {"strategy": method, "support_count": budget, "calibration": calibration,
                 "trainable_parameters": 0, "total_parameters": sum(p.numel() for p in baseline.parameters()),
                 "trainable_parameter_names": [], "changed_parameter_l2": {}, "changed_parameter_count": 0,
                 "fitted_scalar_coefficients": 0 if method == "population" else (4 if method == "affine" else 2),
                 "adaptation_seconds": adaptation_seconds, "query_used_for_adaptation": False}
        # Reference-only controls do not need ECG/PPG inference. Other controls
        # reuse the one measured shared population prediction, not a new timing.
        latency = {"scalar_transform_seconds": transform_seconds,
                   "shared_population_inference": baseline_latency if method in ("population", "bias", "affine") else None,
                   "hardware_claim": "local workstation, not Raspberry Pi"}
        state_path = unit_dir / "calibration.json"
        atomic_json(state_path, calibration)
        artifacts[state_path.name] = file_hash(state_path)
    y = patient["y"][query]
    subjects = np.repeat(patient["patient_id"], len(query))
    last_end = float(patient["end_exclusive"][support[-1]])
    since_reference = (patient["end_exclusive"][query] - last_end) / 60.0
    since_boundary = (patient["end_exclusive"][query] - patient["calibration_boundary"]) / 60.0
    prediction_path = unit_dir / "predictions.npz"
    with prediction_path.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, y=y, pred=prediction, subjects=subjects,
                            source_indices=patient["source_indices"][query],
                            start=patient["start"][query], end_exclusive=patient["end_exclusive"][query],
                            query_indices=query, support_indices=support,
                            support_source_indices=patient["source_indices"][support],
                            support_start=patient["start"][support],
                            support_end_exclusive=patient["end_exclusive"][support],
                            support_y=patient["y"][support], support_population_pred=base_support_pred[:budget],
                            elapsed_since_reference_min=since_reference, elapsed_since_boundary_min=since_boundary,
                            calibration_boundary=patient["calibration_boundary"], case_id=str(patient["case_id"]),
                            config_hash=canonical_hash(config), checkpoint_sha256=checkpoint["sha256"])
    prediction_path.with_suffix(".npz.tmp").replace(prediction_path)
    artifacts[prediction_path.name] = file_hash(prediction_path)
    summary = summarize(y, prediction, subjects)
    record = {**config, "role": "development" if phase == "development" else "target", "config": config,
              "config_hash": canonical_hash(config), "artifact_hashes": artifacts,
              "last_reference_end": last_end, "calibration_boundary": float(patient["calibration_boundary"]),
              "population_epoch": checkpoint["epoch"], "summary": summary,
              "selection_score": float(np.mean(summary["macro"]["mae"])) if phase == "development" else None,
              "drift_since_reference": drift_summary(y, prediction, subjects, since_reference),
              "drift_since_boundary_secondary": drift_summary(y, prediction, subjects, since_boundary),
              "adaptation": audit, "prediction_latency": latency,
              "process_peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "peak_rss_units": "bytes on macOS; KiB on Linux; process cumulative, not model-only",
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "unit_path": str(unit_dir.relative_to(output_root))}
    atomic_json(unit_dir / "unit.json", record)  # Written last: unit completion marker.
    return record


def build_provenance(cohort, checkpoint_paths, individual_root, device) -> dict:
    source_paths = [Path(__file__), Path(__file__).with_name("adaptation.py"),
                    V2_ROOT / "shared/models.py", V2_ROOT / "shared/data.py", V2_ROOT / "shared/metrics.py",
                    V2_ROOT / "PROTOCOL.md"]
    all_patients = cohort["development"] + cohort["target"]
    return {"protocol": "v2-personal-calibration-v1", "manifest_hash": cohort["manifest_hash"],
            "data_manifest": cohort.get("manifest", {}),
            "individual_root": str(individual_root.resolve()), "development_ids": list(DEV_IDS),
            "target_ids": list(TARGET_IDS), "seeds": list(SEEDS), "budgets": list(BUDGETS),
            "checkpoint_hashes": {str(seed): file_hash(path) for seed, path in checkpoint_paths.items()},
            "population_provenance_file_hashes": {
                str(seed): {name: file_hash(path.parent / name) for name in ("config.json", "manifest.json", "completion.json")}
                for seed, path in checkpoint_paths.items()},
            "source_hashes": {str(path.relative_to(V2_ROOT)): file_hash(path) for path in source_paths},
            "patient_data_hashes": {p["patient_id"]: array_hash(p["x"], p["y"], p["start"], p["end_exclusive"],
                                                               p["source_indices"], p["support_indices"], p["query_indices"])
                                    for p in all_patients},
            "device": device, "torch_threads": torch.get_num_threads(),
            "environment": {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
                            "platform": platform.platform(), "machine": platform.machine(),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
            "limitations": ["ABP-derived sparse-reference simulation, not real cuffs",
                            "Same-case later observations, not cross-visit validation",
                            "Provided upstream offline-filtered inputs; native streaming causality unverified"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("development", "test"))
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--individual-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    from shared.data import load_personal_cohort
    checkpoint_paths = {seed: args.population_root / f"baseline_seed{seed}" / "best.pt" for seed in SEEDS}
    missing = [str(path) for path in checkpoint_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("All three freshly trained baseline checkpoints are required: " + ", ".join(missing))
    saved_checkpoints = {seed: load_population_checkpoint(path, seed) for seed, path in checkpoint_paths.items()}
    cohort = load_personal_cohort(args.individual_root)
    for role, expected in (("development", DEV_IDS), ("target", TARGET_IDS)):
        if sorted(p["patient_id"] for p in cohort[role]) != sorted(expected):
            raise ValueError(f"{role} cohort differs from frozen allocation")
        for patient in cohort[role]:
            validate_patient(patient)
    provenance = build_provenance(cohort, checkpoint_paths, args.individual_root, args.device)
    provenance_hash = canonical_hash(provenance)
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != provenance:
            raise ValueError("Output directory belongs to a different data/source/checkpoint/environment configuration")
        if not args.resume and args.phase == "development":
            raise FileExistsError("Run already exists; use --resume")
    else:
        if args.phase == "test":
            raise FileNotFoundError("Development run_manifest.json and selection.json must precede test")
        atomic_json(manifest_path, provenance)
    selection_path = args.output_dir / "selection.json"
    selection = None
    if args.phase == "test":
        if not selection_path.exists():
            raise FileNotFoundError("Complete development and freeze selection.json before test")
        selection = json.loads(selection_path.read_text())
        supplied_hash = selection.pop("selection_hash")
        if supplied_hash != canonical_hash(selection) or selection["provenance_hash"] != provenance_hash:
            raise ValueError("Frozen development selection hash/provenance mismatch")
        selection["selection_hash"] = supplied_hash
    patients = cohort["development" if args.phase == "development" else "target"]
    records = []
    for seed, checkpoint_path in checkpoint_paths.items():
        saved = saved_checkpoints[seed]
        baseline = BPNet(pool=1, variant="baseline")
        baseline.load_state_dict(saved["state_dict"], strict=True)
        baseline.eval()
        checkpoint = {"center": np.asarray(saved["center"]).tolist(), "scale": np.asarray(saved["scale"]).tolist(),
                      "epoch": int(saved["epoch"]), "sha256": file_hash(checkpoint_path)}
        for patient in patients:
            support_pred, support_latency = predict(baseline, patient["x"][patient["support_indices"]], checkpoint["center"], checkpoint["scale"], args.device)
            query_pred, query_latency = predict(baseline, patient["x"][patient["query_indices"]], checkpoint["center"], checkpoint["scale"], args.device)
            latency = {"query": query_latency, "all_ten_support_references": support_latency,
                       "reuse_note": "One actual population inference per patient/seed reused across budgets/methods; support timing is K=10, not a per-K estimate"}
            for budget in BUDGETS:
                for method in (*CONTROLS, "affine", *NEURAL_METHODS):
                    settings = settings_for(method, budget)
                    if args.phase == "test" and method not in CONTROLS and settings:
                        settings = [selection["selected"][f"{method}/K{budget}"]]
                    for setting in settings:
                        record = run_unit(patient, baseline, checkpoint, budget, method, setting, args.phase, seed,
                                          args.output_dir, provenance_hash, support_pred, query_pred, latency,
                                          args.device, args.resume, selection["selection_hash"] if selection else None)
                        records.append(record)
                print(json.dumps({"phase": args.phase, "seed": seed, "patient": patient["patient_id"],
                                  "budget": budget, "completed_units": len(records)}), flush=True)
    if args.phase == "development":
        chosen, grid = choose_settings(records)
        frozen = {"provenance_hash": provenance_hash, "selected": chosen, "grid": grid,
                  "selection_rule": "mean patient-macro SBP/DBP MAE across 4 development patients and 3 seeds",
                  "neural_tie_break": "fewer steps then smaller lr", "affine_tie_break": "larger penalty",
                  "query_labels_used": "development only; never final target"}
        frozen["selection_hash"] = canonical_hash(frozen)
        if selection_path.exists() and json.loads(selection_path.read_text()) != frozen:
            raise ValueError("Refusing to alter already frozen selection")
        atomic_json(selection_path, frozen)
    index = [{key: row[key] for key in ("patient", "seed", "budget", "strategy", "setting", "unit_path", "config_hash")}
             for row in records]
    atomic_json(args.output_dir / f"{args.phase}_index.json", {"complete": True, "n_units": len(index),
                                                             "provenance_hash": provenance_hash, "units": index,
                                                             "affine_unavailable_budgets": [1, 2]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
