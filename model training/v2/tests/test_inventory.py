"""Fail-closed inventory tests using isolated synthetic metadata and arrays."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
from attention_cnn.train import FIXED_SETTINGS, PROTOCOL_COMMIT
from personalized_forecasting.run import (BUDGETS, CONTROLS, DEV_IDS, NEURAL_METHODS, SEEDS, TARGET_IDS,
                                          atomic_json, choose_settings, run_unit, setting_name, settings_for)
from reports import verify_inventory as vi
from shared.models import BPNet, EXPECTED_PARAMETERS, make_model


def a_fixture(folder):
    """A complete one-epoch metadata fixture, not a trained study model."""
    folder.mkdir(parents=True, exist_ok=True)
    variant, seed = "baseline", 42
    manifest = {"unit_test_fixture": True}
    config = {"variant": variant, "seed": seed, "pool": 1, "toy_test_only": False,
              "settings": FIXED_SETTINGS, "protocol_commit": PROTOCOL_COMMIT, "device": "cpu", "data_root": "unit-test-only",
              "hashes": {"manifest": vi.canonical_hash(manifest), "source": {name: "unit-test-source-hash" for name in
                 ("attention_cnn/train.py", "shared/models.py", "shared/data.py", "shared/metrics.py", "PROTOCOL.md")}}}
    config_hash = vi.canonical_hash(config)
    checkpoint = {"state_dict": make_model(variant, seed).state_dict(), "center": [120., 80.], "scale": [20., 10.],
                  "variant": variant, "seed": seed, "pool": 1, "config": config, "config_hash": config_hash,
                  "hashes": config["hashes"], "epoch": 1}
    torch.save(checkpoint, folder / "best.pt")
    bundle = {"center": np.array([120., 80.]), "scale": np.array([20., 10.])}
    prediction_hashes = {}
    for split in ("validation", "test", "aami"):
        y = np.array([[100., 60.], [110., 70.], [120., 80.], [130., 90.], [140., 100.]])
        subjects = np.array([f"{split}-unit-patient-{i}" for i in range(5)])
        indices = np.array([1, 4])
        bundle[split] = {"y": y, "subjects": subjects, "indices": indices}
        path = folder / f"predictions_{split}.npz"
        np.savez_compressed(path, y=y[indices], pred=y[indices] + 2, subjects=subjects[indices], source_indices=indices,
                            variant=variant, seed=seed, manifest_hash=vi.canonical_hash(manifest), selected_epoch=1)
        prediction_hashes[path.name] = vi.file_hash(path)
    completion = {"status": "complete", "toy_test_only": False, "variant": variant, "seed": seed,
                  "config_hash": config_hash, "manifest_hash": vi.canonical_hash(manifest),
                  "best_checkpoint_sha256": vi.file_hash(folder / "best.pt"), "selected_epoch": 1, "best_epoch": 1,
                  "actual_epochs": 1, "epochs_completed": 1, "best_validation_macro_mae": 2.0,
                  "parameters": EXPECTED_PARAMETERS[variant], "parameter_count": EXPECTED_PARAMETERS[variant],
                  "prediction_sha256": prediction_hashes}
    for name, value in (("completion", completion), ("config", config), ("manifest", manifest),
                        ("history", [{"epoch": 1, "selection_macro_mae": 2.0}]),
                        ("costs", {"parameters": EXPECTED_PARAMETERS[variant]}), ("environment", {"device": "cpu"})):
        atomic_json(folder / f"{name}.json", value)
    return bundle


def patient_fixture(patient_id):
    n = 600
    start = np.arange(n, dtype=float) * 10
    boundary = (start[-1] + 10) * .2
    pool = np.flatnonzero((start + 10 <= boundary) & (np.arange(n) % 6 == 0))
    support = pool[np.rint(np.linspace(0, len(pool) - 1, 10)).astype(int)]
    return {"patient_id": patient_id, "case_id": "unit-test-case", "x": np.zeros((n, 2, 1250), np.float32),
            "y": np.column_stack((100 + np.arange(n) / 100, 60 + np.arange(n) / 200)).astype(np.float32),
            "start": start, "end_exclusive": start + 10, "source_indices": np.arange(n, dtype=np.int64) * 3,
            "support_indices": support, "query_indices": np.flatnonzero(start >= boundary + 10),
            "calibration_boundary": boundary}


def b_fixture(folder):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"development_ids": list(DEV_IDS), "target_ids": list(TARGET_IDS), "seeds": list(SEEDS),
                "budgets": list(BUDGETS), "device": "cpu", "source_hashes": {name: "unit-test-hash" for name in
                   ("personalized_forecasting/run.py", "personalized_forecasting/adaptation.py", "shared/models.py",
                    "shared/data.py", "shared/metrics.py", "PROTOCOL.md")},
                "checkpoint_hashes": {str(seed): f"checkpoint-{seed}" for seed in SEEDS},
                "population_provenance_file_hashes": {str(seed): {name: "fixture-file-hash" for name in
                    ("config.json", "manifest.json", "completion.json")} for seed in SEEDS}}
    provenance_hash = vi.canonical_hash(manifest)
    records = {}
    indexes = {}
    for phase in ("development", "test"):
        rows, items = [], []
        for patient in DEV_IDS if phase == "development" else TARGET_IDS:
            for seed in SEEDS:
                for budget in BUDGETS:
                    for method in (*CONTROLS, "affine", *NEURAL_METHODS):
                        settings = settings_for(method, budget)
                        if phase == "test" and method not in CONTROLS and settings:
                            settings = [selected[f"{method}/K{budget}"]]
                        for setting in settings:
                            path = f"{phase}/{patient}/{seed}/{budget}/{method}/{setting_name(setting)}"
                            row = {"patient": patient, "seed": seed, "budget": budget, "strategy": method,
                                   "setting": setting, "unit_path": path, "config_hash": "fixture-config",
                                   "selection_score": 2.0 if phase == "development" else None}
                            rows.append(row)
                            records[path] = row
                            items.append({key: value for key, value in row.items() if key != "selection_score"})
        if phase == "development":
            selected, grid = choose_settings(rows)
        index = {"complete": True, "n_units": len(items), "units": items,
                 "provenance_hash": provenance_hash, "affine_unavailable_budgets": [1, 2]}
        atomic_json(folder / f"{phase}_index.json", index)
        indexes[phase] = index
    selection = {"provenance_hash": provenance_hash, "selected": selected, "grid": grid}
    selection["selection_hash"] = vi.canonical_hash(selection)
    atomic_json(folder / "selection.json", selection)
    atomic_json(folder / "run_manifest.json", manifest)
    base = patient_fixture("unused-unit-fixture")
    cohort = {"development": [{**base, "patient_id": p} for p in DEV_IDS],
              "target": [{**base, "patient_id": p} for p in TARGET_IDS]}
    return records, indexes, selection, cohort


class InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_population_identity_config_and_source_row_checks(self):
        with tempfile.TemporaryDirectory(prefix="bad-blood-inventory-unit-fixture-") as temp:
            folder = Path(temp)
            bundle = a_fixture(folder)
            with patch.object(vi, "_verify_population_inputs", return_value=bundle), patch.object(vi, "_verify_source_hashes"):
                valid = vi.verify_a_folder(folder, "baseline", 42)
                self.assertEqual(valid["completion"]["parameters"], 37666)
                done = vi.read_json(folder / "completion.json")
                for field, value in (("seed", 43), ("variant", "se_last"), ("toy_test_only", True)):
                    atomic_json(folder / "completion.json", {**done, field: value})
                    with self.assertRaises(ValueError):
                        vi.verify_a_folder(folder, "baseline", 42)
                atomic_json(folder / "completion.json", done)
                config = vi.read_json(folder / "config.json")
                atomic_json(folder / "config.json", {**config, "pool": 5})
                with self.assertRaisesRegex(ValueError, "settings"):
                    vi.verify_a_folder(folder, "baseline", 42)
                atomic_json(folder / "config.json", config)
                manifest = vi.read_json(folder / "manifest.json")
                atomic_json(folder / "manifest.json", {**manifest, "corrupt": True})
                with self.assertRaisesRegex(ValueError, "manifest hash"):
                    vi.verify_a_folder(folder, "baseline", 42)
                atomic_json(folder / "manifest.json", manifest)
                path = folder / "predictions_test.npz"
                with np.load(path) as archive:
                    original = dict(archive)
                for key, value in (("seed", 43), ("variant", "se_last"),
                                   ("source_indices", original["source_indices"][::-1]),
                                   ("y", original["y"] + 10)):
                    np.savez_compressed(path, **{**original, key: value})
                    done["prediction_sha256"][path.name] = vi.file_hash(path)
                    atomic_json(folder / "completion.json", done)
                    with self.assertRaises(ValueError):
                        vi.verify_a_folder(folder, "baseline", 42)

    def test_exact_personal_matrix_duplicate_missing_and_selection_corruption(self):
        with tempfile.TemporaryDirectory(prefix="bad-blood-inventory-unit-fixture-") as temp:
            folder = Path(temp) / "personalization"
            records, indexes, selection, cohort = b_fixture(folder)
            self.assertEqual(len(vi.expected_b_keys("development")), 1068)
            self.assertEqual(len(vi.expected_b_keys("test", selection["selected"])), 684)
            with ExitStack() as stack:
                stack.enter_context(patch.object(vi, "_verify_personal_inputs", return_value=cohort))
                stack.enter_context(patch.object(vi, "_verify_source_hashes"))
                stack.enter_context(patch.object(vi, "verify_a_folder", side_effect=lambda folder, variant, seed:
                                                {"completion": {"best_checkpoint_sha256": f"checkpoint-{seed}"}}))
                stack.enter_context(patch.object(vi, "file_hash", return_value="fixture-file-hash"))
                stack.enter_context(patch.object(vi, "_verify_unit", side_effect=lambda folder, item, *args: records[item["unit_path"]]))
                verified = vi.verify_b_inventory(folder)
                self.assertEqual(len(verified["development_records"]), 1068)
                self.assertEqual(len(verified["test_records"]), 684)
                for phase in ("development", "test"):
                    missing = copy.deepcopy(indexes[phase])
                    missing["units"].pop()  # Keep stale declared count: must still fail.
                    atomic_json(folder / f"{phase}_index.json", missing)
                    with self.assertRaisesRegex(ValueError, "count"):
                        vi.verify_b_inventory(folder)
                    duplicate = copy.deepcopy(indexes[phase])
                    duplicate["units"][-1] = duplicate["units"][0]
                    atomic_json(folder / f"{phase}_index.json", duplicate)
                    with self.assertRaisesRegex(ValueError, "Duplicate"):
                        vi.verify_b_inventory(folder)
                    wrong_identity = copy.deepcopy(indexes[phase])
                    wrong_identity["units"][0]["seed"] = 99
                    atomic_json(folder / f"{phase}_index.json", wrong_identity)
                    with self.assertRaisesRegex(ValueError, "matrix"):
                        vi.verify_b_inventory(folder)
                    atomic_json(folder / f"{phase}_index.json", indexes[phase])
                atomic_json(folder / "selection.json", {**selection, "selection_hash": "tampered"})
                with self.assertRaisesRegex(ValueError, "selection hash"):
                    vi.verify_b_inventory(folder)

    def test_personal_unit_revalidates_labels_timestamps_and_metadata(self):
        patient = patient_fixture(DEV_IDS[0])
        baseline = BPNet()
        checkpoint = {"center": [120, 80], "scale": [20, 10], "epoch": 1, "sha256": "fixture-checkpoint"}
        provenance = {"device": "cpu", "checkpoint_hashes": {"42": "fixture-checkpoint"}}
        provenance_hash = vi.canonical_hash(provenance)
        with tempfile.TemporaryDirectory(prefix="bad-blood-inventory-unit-fixture-") as temp:
            folder = Path(temp)
            record = run_unit(patient, baseline, checkpoint, 3, "bias", {}, "development", 42,
                              folder, provenance_hash, patient["y"][patient["support_indices"]] - 2,
                              np.tile([120., 80.], (len(patient["query_indices"]), 1)), {}, "cpu", False)
            item = {key: record[key] for key in ("patient", "seed", "budget", "strategy", "setting", "unit_path", "config_hash")}
            args = (folder, item, "development", patient, provenance, provenance_hash, {"selection_hash": "unused"})
            self.assertEqual(vi._verify_unit(*args)["budget"], 3)
            unit = folder / record["unit_path"]
            path = unit / "predictions.npz"
            with np.load(path) as archive:
                original = dict(archive)
            for key, value in (("y", original["y"] + 10), ("support_y", original["support_y"] + 1),
                               ("start", original["start"] + 1),
                               ("elapsed_since_reference_min", original["elapsed_since_reference_min"] + 1),
                               ("config_hash", "changed-config")):
                np.savez_compressed(path, **{**original, key: value})
                record["artifact_hashes"][path.name] = vi.file_hash(path)
                atomic_json(unit / "unit.json", record)
                with self.assertRaises(ValueError):
                    vi._verify_unit(*args)

    def test_artifact_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "escapes"):
                vi._inside(root, "../outside-artifact")


if __name__ == "__main__":
    unittest.main()
