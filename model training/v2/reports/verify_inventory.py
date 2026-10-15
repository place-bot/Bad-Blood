"""Read-only, fail-closed verification of real experiment report inputs.

Directory names and declared unit counts are not evidence of experiment identity.
These checks bind source, data, configuration, checkpoints and predictions, then
enumerate the complete frozen matrix before any report can claim completion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from attention_cnn.train import FIXED_SETTINGS, PROTOCOL_COMMIT
from personalized_forecasting.run import (
    BUDGETS, CONTROLS, DEV_IDS, NEURAL_METHODS, SEEDS, TARGET_IDS,
    array_hash, choose_settings, setting_name, settings_for, validate_patient,
)
from shared.metrics import summarize
from shared.models import EXPECTED_PARAMETERS, make_model

A_VARIANTS = ("baseline", "se_last", "temporal_last", "cbam_last", "additive_last", "se_all",
              "baseline_ecg", "baseline_ppg")
_POPULATION_CACHE = {}
_PERSONAL_CACHE = {}
_HASH_CACHE = {}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def file_hash(path: Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
        after = path.stat()
        require((stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"File changed while hashing: {path}")
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Artifact path escapes its root: {relative}")
    return path


def _verify_source_hashes(source_hashes: dict) -> None:
    require(isinstance(source_hashes, dict) and bool(source_hashes), "Missing source hashes")
    for name, expected in source_hashes.items():
        path = _inside(V2, name)
        require(file_hash(path) == expected, f"Current source differs from recorded source: {name}")


def _signature(paths):
    return tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


def _verify_population_inputs(config: dict, manifest: dict, *, population_data_root: Path | None = None) -> dict:
    """Revalidate identical source bytes, optionally at a relocated data root.

    The override changes only file discovery. The recorded configuration and
    canonical manifest remain untouched and every identity check still applies.
    """
    from shared.data import EXCLUSION_PATH, load_population
    root = Path(config["data_root"] if population_data_root is None else population_data_root)
    files = [root / "split.json", Path(EXCLUSION_PATH)]
    files.extend(root / name / filename for name in ("train", "test", "aami")
                 for filename in ("waves.npy", "y.npy", "subjects.npy", "valid.npy", "complete.json"))
    signature = _signature(files)
    if signature not in _POPULATION_CACHE:
        _POPULATION_CACHE[signature] = load_population(root)
    bundle = _POPULATION_CACHE[signature]
    require(bundle["manifest"] == manifest, "Population manifest differs from the live, audited data/split")
    return bundle


def _verify_personal_inputs(manifest: dict, *, individual_root: Path | None = None) -> dict:
    """Locate personal files without rewriting their original provenance."""
    from shared.data import DEFAULT_AUDIT_PATH, load_personal_cohort
    root = Path(manifest["individual_root"] if individual_root is None else individual_root)
    patient_ids = DEV_IDS + TARGET_IDS
    paths = [Path(DEFAULT_AUDIT_PATH)] + [root / (p.split(":")[1] + ".mat") for p in patient_ids]
    signature = _signature(paths)
    if signature not in _PERSONAL_CACHE:
        _PERSONAL_CACHE[signature] = load_personal_cohort(root)
    cohort = _PERSONAL_CACHE[signature]
    require(cohort["manifest_hash"] == manifest["manifest_hash"], "Personal data manifest hash mismatch")
    require(cohort["manifest"] == manifest["data_manifest"], "Personal data source manifest mismatch")
    patients = cohort["development"] + cohort["target"]
    actual = {p["patient_id"]: array_hash(p["x"], p["y"], p["start"], p["end_exclusive"],
              p["source_indices"], p["support_indices"], p["query_indices"]) for p in patients}
    require(actual == manifest["patient_data_hashes"], "Personal input/reference/timestamp hashes differ")
    return cohort


def _same(actual, expected, message: str) -> None:
    require(np.array_equal(np.asarray(actual), np.asarray(expected)), message)


def _close_structure(actual, expected, where: str = "value") -> None:
    """Compare recomputed numeric metrics without depending on JSON float spelling."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Metric keys differ: {where}")
        for key, value in expected.items():
            _close_structure(actual[key], value, f"{where}/{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Metric vector differs: {where}")
        for i, value in enumerate(expected):
            _close_structure(actual[i], value, f"{where}/{i}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and np.isclose(actual, expected, rtol=1e-9, atol=1e-9), f"Metric differs: {where}")
    else:
        require(actual == expected, f"Metadata differs: {where}")


def verify_a_folder(folder: Path, variant: str, seed: int, *, population_data_root: Path | None = None) -> dict:
    """Verify one expected model/seed, including relocated identical source data."""
    folder = Path(folder).resolve()
    require(variant in EXPECTED_PARAMETERS and seed in SEEDS, "Unknown expected A identity")
    require(not variant.startswith("baseline_") or seed == 42, "Modality ablations have one frozen seed")
    completion = read_json(folder / "completion.json")
    config = read_json(folder / "config.json")
    manifest = read_json(folder / "manifest.json")
    history = read_json(folder / "history.json")
    costs = read_json(folder / "costs.json")
    environment = read_json(folder / "environment.json")
    require(completion.get("status") == "complete" and completion.get("toy_test_only") is False
            and config.get("toy_test_only") is False, "Only completed real A runs may enter the report")
    for name, record in (("completion", completion), ("config", config)):
        require(record.get("variant") == variant and record.get("seed") == seed, f"Expected {variant}/seed{seed}, found different {name} identity")
    require(config["settings"] == FIXED_SETTINGS and config["pool"] == 1
            and config["protocol_commit"] == PROTOCOL_COMMIT, "Population settings differ from frozen protocol")
    require(canonical_hash(config) == completion["config_hash"], "Population config hash mismatch")
    manifest_hash = canonical_hash(manifest)
    require(manifest_hash == completion["manifest_hash"] == config["hashes"]["manifest"], "Population manifest hash mismatch")
    source_hashes = config["hashes"]["source"]
    require(set(source_hashes) == {"attention_cnn/train.py", "shared/models.py", "shared/data.py", "shared/metrics.py", "PROTOCOL.md"}, "Population source inventory is incomplete")
    _verify_source_hashes(source_hashes)
    data_location = {} if population_data_root is None else {"population_data_root": population_data_root}
    bundle = _verify_population_inputs(config, manifest, **data_location)
    checkpoint_path = folder / "best.pt"
    require(file_hash(checkpoint_path) == completion["best_checkpoint_sha256"], "Population checkpoint bytes changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["variant"] == variant and checkpoint["seed"] == seed and checkpoint["pool"] == 1, "Checkpoint model/seed mismatch")
    require(checkpoint["config"] == config and checkpoint["config_hash"] == completion["config_hash"]
            and checkpoint["hashes"] == config["hashes"], "Checkpoint configuration/provenance mismatch")
    _same(checkpoint["center"], bundle["center"], "Checkpoint center is not training-only")
    _same(checkpoint["scale"], bundle["scale"], "Checkpoint scale is not training-only")
    require(checkpoint["epoch"] == completion["selected_epoch"] == completion["best_epoch"], "Selected checkpoint epoch mismatch")
    require(len(history) == completion["actual_epochs"] == completion["epochs_completed"], "Actual epoch count mismatch")
    require([r["epoch"] for r in history] == list(range(1, len(history) + 1)), "Training history has missing/duplicate epochs")
    require(1 <= len(history) <= 40 and all(np.isfinite(r["selection_macro_mae"]) for r in history), "Invalid training history")
    best_row = min(history, key=lambda row: row["selection_macro_mae"])
    require(best_row["epoch"] == checkpoint["epoch"], "Report checkpoint was not validation-selected")
    require(np.isclose(best_row["selection_macro_mae"], completion["best_validation_macro_mae"], rtol=0, atol=1e-10), "Best selection metric mismatch")
    expected_count = EXPECTED_PARAMETERS[variant]
    require(completion["parameters"] == completion["parameter_count"] == costs["parameters"] == expected_count, "Parameter count mismatch")
    model = make_model(variant, seed)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite checkpoint parameters")
    require(environment["device"] == config["device"], "Population device metadata differs")
    predictions = {}
    require(set(completion["prediction_sha256"]) == {f"predictions_{s}.npz" for s in ("validation", "test", "aami")}, "Prediction inventory incomplete")
    for split in ("validation", "test", "aami"):
        path = folder / f"predictions_{split}.npz"
        require(file_hash(path) == completion["prediction_sha256"][path.name], f"Prediction bytes changed: {split}")
        with np.load(path, allow_pickle=False) as archive:
            p = dict(archive)
        require(p["variant"].item() == variant and p["seed"].item() == seed, f"Prediction seed/model metadata mismatch: {split}")
        require(p["manifest_hash"].item() == manifest_hash and p["selected_epoch"].item() == checkpoint["epoch"], f"Prediction split/epoch metadata mismatch: {split}")
        part = bundle[split]
        ids = part["indices"]
        _same(p["source_indices"], ids, f"Prediction source rows differ: {split}")
        _same(p["y"], np.asarray(part["y"])[ids], f"Prediction labels differ from source: {split}")
        _same(p["subjects"], np.asarray(part["subjects"])[ids], f"Prediction patients differ from source: {split}")
        require(p["pred"].shape == p["y"].shape and np.isfinite(p["pred"]).all(), f"Invalid predictions: {split}")
        predictions[split] = p
    return {"completion": completion, "config": config, "manifest": manifest, "history": history,
            "costs": costs, "environment": environment, "predictions": predictions,
            "checkpoint_metadata": {k: checkpoint[k] for k in ("variant", "seed", "epoch", "pool", "center", "scale", "config_hash", "hashes")}}


def expected_b_keys(phase: str, selected: dict | None = None) -> set[tuple]:
    require(phase in ("development", "test"), "Unknown personal phase")
    require(phase != "test" or selected is not None, "Test inventory requires frozen selections")
    expected = set()
    for patient in DEV_IDS if phase == "development" else TARGET_IDS:
        for seed in SEEDS:
            for budget in BUDGETS:
                for method in (*CONTROLS, "affine", *NEURAL_METHODS):
                    settings = settings_for(method, budget)
                    if phase == "test" and method not in CONTROLS and settings:
                        choice = selected[f"{method}/K{budget}"]
                        require(choice in settings, f"Selected personal setting is outside the grid: {method}/K{budget}")
                        settings = [choice]
                    for setting in settings:
                        expected.add((patient, seed, budget, method, setting_name(setting)))
    return expected


def _unit_key(item):
    return (item["patient"], item["seed"], item["budget"], item["strategy"], setting_name(item["setting"]))


def _verify_unit(folder: Path, item: dict, phase: str, patient: dict, provenance: dict,
                 provenance_hash: str, selection: dict) -> dict:
    unit_dir = _inside(folder, item["unit_path"])
    record = read_json(unit_dir / "unit.json")
    config = record["config"]
    require(record["config_hash"] == item["config_hash"] == canonical_hash(config), "Personal unit configuration hash mismatch")
    require(record["unit_path"] == item["unit_path"], "Personal unit path mismatch")
    for key, value in config.items():
        require(record.get(key) == value, f"Personal outer/config metadata differ: {key}")
    for key in ("patient", "seed", "budget", "strategy", "setting"):
        require(config[key] == item[key], f"Personal index/record metadata differ: {key}")
    require(config["phase"] == phase and record["role"] == ("development" if phase == "development" else "target"), "Personal phase/role mismatch")
    require(config["patient"] == patient["patient_id"] and config["case"] == str(patient["case_id"]), "Personal patient/case mismatch")
    require(config["provenance_hash"] == provenance_hash and config["device"] == provenance["device"], "Personal provenance/device mismatch")
    seed = config["seed"]
    checkpoint_hash = provenance["checkpoint_hashes"][str(seed)]
    require(config["checkpoint_sha256"] == checkpoint_hash, "Personal checkpoint seed mismatch")
    expected_selection_hash = selection["selection_hash"] if phase == "test" else None
    require(config["selection_hash"] == expected_selection_hash, "Personal unit used different frozen selection")
    neural = config["strategy"] in NEURAL_METHODS
    expected_artifacts = {"predictions.npz", "adapted.pt" if neural else "calibration.json"}
    require(set(record["artifact_hashes"]) == expected_artifacts, "Personal artifact inventory incomplete")
    for name, expected in record["artifact_hashes"].items():
        require(file_hash(_inside(unit_dir, name)) == expected, f"Personal artifact bytes changed: {name}")
    if neural:
        saved = torch.load(unit_dir / "adapted.pt", map_location="cpu", weights_only=False)
        require(saved["config"] == config and saved["variant"] == "baseline", "Adapted checkpoint config mismatch")
    else:
        calibration = read_json(unit_dir / "calibration.json")
        require(calibration == record["adaptation"]["calibration"], "Scalar calibration metadata mismatch")
    require(record["adaptation"]["query_used_for_adaptation"] is False, "Unit claims query-dependent adaptation")
    support = patient["support_indices"][:config["budget"]]
    query = patient["query_indices"]
    require(config["query_indices_hash"] == array_hash(patient["source_indices"][query]) and
            config["support_indices_hash"] == array_hash(patient["source_indices"][support]), "Personal row hash mismatch")
    with np.load(unit_dir / "predictions.npz", allow_pickle=False) as archive:
        p = dict(archive)
    require(p["config_hash"].item() == record["config_hash"] and p["checkpoint_sha256"].item() == checkpoint_hash, "Prediction configuration/checkpoint metadata mismatch")
    require(str(p["case_id"].item()) == str(patient["case_id"]), "Prediction case mismatch")
    expected_arrays = {
        "y": patient["y"][query], "subjects": np.repeat(patient["patient_id"], len(query)),
        "source_indices": patient["source_indices"][query], "start": patient["start"][query],
        "end_exclusive": patient["end_exclusive"][query], "query_indices": query,
        "support_indices": support, "support_source_indices": patient["source_indices"][support],
        "support_start": patient["start"][support], "support_end_exclusive": patient["end_exclusive"][support],
        "support_y": patient["y"][support], "calibration_boundary": patient["calibration_boundary"],
        "elapsed_since_reference_min": (patient["end_exclusive"][query] - patient["end_exclusive"][support[-1]]) / 60.,
        "elapsed_since_boundary_min": (patient["end_exclusive"][query] - patient["calibration_boundary"]) / 60.,
    }
    for name, value in expected_arrays.items():
        _same(p[name], value, f"Personal prediction/source alignment differs: {name}")
    recomputed = summarize(p["y"], p["pred"], p["subjects"])
    _close_structure(record["summary"], recomputed, "personal_summary")
    if phase == "test":
        require(record["selection_score"] is None, "Final target must not expose a selection score")
    else:
        require(np.isclose(record["selection_score"], np.mean(recomputed["macro"]["mae"]), rtol=0, atol=1e-10), "Development selection score differs from predictions")
    return record


def verify_b_inventory(folder: Path, *, population_data_root: Path | None = None,
                       individual_root: Path | None = None) -> dict:
    """Verify every personal unit, optionally locating identical data elsewhere.

    Data-root overrides do not alter original run configuration/provenance
    hashes. Population checkpoint folders remain relative to the artifact root.
    """
    folder = Path(folder).resolve()
    manifest = read_json(folder / "run_manifest.json")
    provenance_hash = canonical_hash(manifest)
    require(manifest["development_ids"] == list(DEV_IDS) and manifest["target_ids"] == list(TARGET_IDS)
            and manifest["seeds"] == list(SEEDS) and manifest["budgets"] == list(BUDGETS), "Personal cohort/seed/budget inventory changed")
    source_hashes = manifest["source_hashes"]
    require(set(source_hashes) == {"personalized_forecasting/run.py", "personalized_forecasting/adaptation.py",
            "shared/models.py", "shared/data.py", "shared/metrics.py", "PROTOCOL.md"}, "Personal source inventory incomplete")
    _verify_source_hashes(source_hashes)
    personal_location = {} if individual_root is None else {"individual_root": individual_root}
    population_location = {} if population_data_root is None else {"population_data_root": population_data_root}
    cohort = _verify_personal_inputs(manifest, **personal_location)
    patient_map = {p["patient_id"]: p for p in cohort["development"] + cohort["target"]}
    require(set(patient_map) == set(DEV_IDS + TARGET_IDS), "Personal live cohort is incomplete")
    for patient in patient_map.values():
        validate_patient(patient)
    for seed in SEEDS:
        population_folder = folder.parent / "attention" / f"baseline_seed{seed}"
        verified = verify_a_folder(population_folder, "baseline", seed, **population_location)
        require(verified["completion"]["best_checkpoint_sha256"] == manifest["checkpoint_hashes"][str(seed)], "Personal baseline differs from population report baseline")
        expected = manifest["population_provenance_file_hashes"][str(seed)]
        require(set(expected) == {"config.json", "manifest.json", "completion.json"}, "Personal population-provenance inventory incomplete")
        for name, value in expected.items():
            require(file_hash(population_folder / name) == value, "Personal baseline provenance file changed")
    selection = read_json(folder / "selection.json")
    selection_payload = {k: v for k, v in selection.items() if k != "selection_hash"}
    require(canonical_hash(selection_payload) == selection["selection_hash"] and selection["provenance_hash"] == provenance_hash, "Frozen personal selection hash/provenance mismatch")
    output = {"manifest": manifest, "selection": selection}
    for phase, expected_count in (("development", 1068), ("test", 684)):
        index = read_json(folder / f"{phase}_index.json")
        require(index.get("complete") is True and index["n_units"] == len(index["units"]) == expected_count, f"Actual {phase} unit count is incomplete")
        require(index["provenance_hash"] == provenance_hash and index["affine_unavailable_budgets"] == [1, 2], f"{phase} index provenance mismatch")
        keys = [_unit_key(item) for item in index["units"]]
        require(len(set(keys)) == len(keys), f"Duplicate {phase} unit identities")
        require(set(keys) == expected_b_keys(phase, selection["selected"]), f"{phase} experiment matrix differs from frozen expected identities")
        require(len({item["unit_path"] for item in index["units"]}) == len(keys), f"Duplicate {phase} artifact paths")
        records = [_verify_unit(folder, item, phase, patient_map[item["patient"]], manifest,
                                provenance_hash, selection) for item in index["units"]]
        output[f"{phase}_index"] = index
        output[f"{phase}_records"] = records
    chosen, grid = choose_settings(output["development_records"])
    require(chosen == selection["selected"], "Frozen selections differ from verified development scores/tie-breaks")
    _close_structure(selection["grid"], grid, "development_grid")
    return output


def verify_a_inventory(root: Path, include_eca: bool = False, *, population_data_root: Path | None = None) -> dict:
    """Require every population run, optionally reading identical relocated data."""
    variants = A_VARIANTS + (("eca_last",) if include_eca else ())
    result = {}
    data_location = {} if population_data_root is None else {"population_data_root": population_data_root}
    for variant in variants:
        for seed in (42,) if variant.startswith("baseline_") else SEEDS:
            result[f"{variant}_seed{seed}"] = verify_a_folder(
                Path(root) / f"{variant}_seed{seed}", variant, seed, **data_location)
    require(len({r["completion"]["manifest_hash"] for r in result.values()}) == 1, "Population runs use different split manifests")
    for split in ("validation", "test", "aami"):
        reference = result["baseline_seed42"]["predictions"][split]
        for run in result.values():
            for key in ("y", "subjects", "source_indices"):
                _same(run["predictions"][split][key], reference[key], f"Across-model/seed {split} pairing differs: {key}")
    return result
