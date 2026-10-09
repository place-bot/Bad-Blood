"""Synthetic unit fixtures only, never replacement experiment data."""
import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
from personalized_forecasting.adaptation import (
    adapt_model, apply_calibration, configure_trainable, fit_calibration,
)
from personalized_forecasting.run import (
    BUDGETS, DEV_IDS, TARGET_IDS, NEURAL_METHODS, SEEDS, atomic_json, canonical_hash,
    choose_settings, file_hash, load_complete, load_population_checkpoint,
    main, run_unit, settings_for, validate_patient,
)
from shared.models import BPNet


def toy_patient():
    n = 600
    start = np.arange(n, dtype=np.float64) * 10
    boundary = 0.2 * (start[-1] + 10)
    pool = np.flatnonzero((start + 10 <= boundary) & (np.arange(n) % 6 == 0))
    support = pool[np.rint(np.linspace(0, len(pool) - 1, 10)).astype(int)]
    y = np.column_stack((100 + np.arange(n) / 100, 60 + np.arange(n) / 200)).astype(np.float32)
    return {"patient_id": DEV_IDS[0], "case_id": "synthetic-unit-case", "x": np.zeros((n, 2, 1250), np.float32),
            "y": y, "start": start, "end_exclusive": start + 10,
            "source_indices": np.arange(n, dtype=np.int64) * 3 + 7,
            "support_indices": support, "query_indices": np.flatnonzero(start >= boundary + 10),
            "calibration_boundary": boundary}


class PersonalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_analytic_calibration_and_regularization(self):
        center, scale = np.array([120, 80]), np.array([20, 10])
        prediction = np.array([[100, 60], [120, 80], [140, 100]], float)
        y = prediction + [5, -3]
        for method in ("bias", "affine"):
            state = fit_calibration(method, prediction, y, center, scale,
                                    penalty=1 if method == "affine" else None)
            np.testing.assert_allclose(apply_calibration(state, prediction), y)
        state = fit_calibration("last_reference", prediction, y, center, scale)
        np.testing.assert_allclose(apply_calibration(state, prediction), np.tile(y[-1], (3, 1)))
        state = fit_calibration("calibration_mean", prediction, y, center, scale)
        np.testing.assert_allclose(apply_calibration(state, prediction), np.tile(y.mean(0), (3, 1)))
        target = center + ((prediction - center) / scale * 2) * scale
        weak = fit_calibration("affine", prediction, target, center, scale, penalty=1)
        strong = fit_calibration("affine", prediction, target, center, scale, penalty=100)
        self.assertTrue(np.all(np.abs(np.array(strong["slope"]) - 1) < np.abs(np.array(weak["slope"]) - 1)))
        with self.assertRaises(ValueError):
            fit_calibration("affine", prediction[:2], y[:2], center, scale, penalty=1)

    def test_freeze_masks_bn_state_dropout_and_fresh_baseline(self):
        rng = np.random.default_rng(12)
        x = rng.uniform(0, 1, (3, 2, 1250)).astype(np.float32)
        y = np.array([[110, 70], [120, 80], [130, 90]], np.float32)
        torch.manual_seed(42)
        baseline = BPNet(pool=1, variant="baseline")
        original = copy.deepcopy(baseline.state_dict())
        for strategy in NEURAL_METHODS:
            result = adapt_model(baseline, x, y, [120, 80], [20, 10], strategy=strategy,
                                 lr=1e-3, steps=2, device="cpu", seed=42)
            mask = result.audit["trainable_parameter_names"]
            expected = [name for name, _ in baseline.named_parameters()
                        if strategy == "full" or name.startswith("head.")
                        or (strategy == "last_block" and name.startswith("body.6."))]
            self.assertEqual(mask, expected)
            self.assertGreater(result.audit["changed_parameter_count"], 0)
            self.assertTrue(set(result.audit["changed_parameter_l2"]) <= set(mask))
            self.assertFalse(result.audit["query_used_for_adaptation"])
            for module in result.model.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.Dropout)):
                    self.assertFalse(module.training)
            for name, value in baseline.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
                if name not in mask:
                    torch.testing.assert_close(result.model.state_dict()[name], original[name], rtol=0, atol=0)
        self.assertNotIn("query", " ".join(inspect.signature(adapt_model).parameters))

    def test_support_alignment_guard_and_reference_selection(self):
        patient = toy_patient()
        validate_patient(patient)
        for budget in BUDGETS:
            self.assertTrue(np.all(patient["end_exclusive"][patient["support_indices"][:budget]] <= patient["calibration_boundary"]))
        invalid = copy.deepcopy(patient)
        invalid["query_indices"][0] = patient["support_indices"][0]
        with self.assertRaises(ValueError):
            validate_patient(invalid)
        invalid = copy.deepcopy(patient)
        invalid["support_indices"] = invalid["support_indices"][::-1]
        with self.assertRaises(ValueError):
            validate_patient(invalid)
        invalid = copy.deepcopy(patient)
        invalid["source_indices"] = invalid["source_indices"][:-1]
        with self.assertRaises(ValueError):
            validate_patient(invalid)

    def test_development_grid_complete_equal_weight_and_ties(self):
        rows = []
        for method in (*NEURAL_METHODS, "affine"):
            for budget in BUDGETS:
                for setting in settings_for(method, budget):
                    for patient in DEV_IDS:
                        for seed in SEEDS:
                            rows.append({"strategy": method, "budget": budget, "setting": setting,
                                         "patient": patient, "seed": seed, "selection_score": 1.0})
        chosen, grid = choose_settings(rows)
        self.assertEqual(len(grid), 69)
        self.assertEqual(len(chosen), 18)
        self.assertEqual(chosen["head/K1"], {"lr": 1e-4, "steps": 20})
        self.assertEqual(chosen["affine/K3"], {"penalty": 100})
        with self.assertRaises(ValueError):
            choose_settings(rows[:-1])
        with self.assertRaises(ValueError):
            choose_settings(rows + [rows[0]])
        for row in rows:
            if row["strategy"] == "head" and row["budget"] == 1 and row["setting"] == {"lr": 1e-3, "steps": 100}:
                row["selection_score"] = 0.5
        chosen, _ = choose_settings(rows)
        self.assertEqual(chosen["head/K1"], {"lr": 1e-3, "steps": 100})

    def test_query_labels_cannot_affect_calibration_and_artifacts_resume(self):
        patient = toy_patient()
        model = BPNet(pool=1, variant="baseline")
        checkpoint = {"center": [120, 80], "scale": [20, 10], "epoch": 5, "sha256": "unit-test-checkpoint"}
        sp = patient["y"][patient["support_indices"]] - [5, -2]
        qp = np.tile([120., 80.], (len(patient["query_indices"]), 1))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kwargs = dict(baseline=model, checkpoint=checkpoint, budget=3, method="bias", setting={},
                          phase="development", seed=42, provenance_hash="test-only", base_support_pred=sp,
                          base_query_pred=qp, baseline_latency={}, device="cpu", resume=False)
            first = run_unit(patient, output_root=root / "first", **kwargs)
            modified = copy.deepcopy(patient)
            modified["y"][modified["query_indices"]] += 100
            second = run_unit(modified, output_root=root / "second", **kwargs)
            self.assertEqual(first["adaptation"]["calibration"], second["adaptation"]["calibration"])
            self.assertNotEqual(first["selection_score"], second["selection_score"])
            unit_dir = root / "first" / first["unit_path"]
            with np.load(unit_dir / "predictions.npz") as artifact:
                np.testing.assert_array_equal(artifact["y"], patient["y"][patient["query_indices"]])
                np.testing.assert_array_equal(artifact["source_indices"], patient["source_indices"][patient["query_indices"]])
                np.testing.assert_array_equal(artifact["support_y"], patient["y"][patient["support_indices"][:3]])
                expected_age = (patient["end_exclusive"][patient["query_indices"]] - patient["end_exclusive"][patient["support_indices"][2]]) / 60
                np.testing.assert_allclose(artifact["elapsed_since_reference_min"], expected_age)
            resumed = load_complete(unit_dir, first["config"], True)
            self.assertEqual(first, resumed)
            with self.assertRaises(FileExistsError):
                load_complete(unit_dir, first["config"], False)
            with self.assertRaises(ValueError):
                load_complete(unit_dir, {**first["config"], "budget": 5}, True)
            (unit_dir / "calibration.json").write_text("corrupt test fixture")
            with self.assertRaises(ValueError):
                load_complete(unit_dir, first["config"], True)

    def test_neural_query_reference_no_peek(self):
        patient = toy_patient()
        # Real BPNet, two update steps only: an isolated unit test, not a fit.
        patient["x"][:] = np.linspace(0, 1, 1250)[None, None, :]
        model = BPNet(pool=1, variant="baseline")
        checkpoint = {"center": [120, 80], "scale": [20, 10], "epoch": 5, "sha256": "unit-test-checkpoint"}
        sp = patient["y"][patient["support_indices"]]
        qp = np.tile([120., 80.], (len(patient["query_indices"]), 1))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kwargs = dict(baseline=model, checkpoint=checkpoint, budget=3, method="head", setting={"lr": 1e-3, "steps": 2},
                          phase="test", seed=42, provenance_hash="test-only", base_support_pred=sp, base_query_pred=qp,
                          baseline_latency={}, device="cpu", resume=False, selection_hash="test-only-frozen")
            a = run_unit(patient, output_root=root / "a", **kwargs)
            modified = copy.deepcopy(patient)
            modified["y"][modified["query_indices"]] += 100
            b = run_unit(modified, output_root=root / "b", **kwargs)
            self.assertIsNone(a["selection_score"])
            self.assertIsNone(b["selection_score"])
            state_a = torch.load(root / "a" / a["unit_path"] / "adapted.pt", weights_only=False)["state_dict"]
            state_b = torch.load(root / "b" / b["unit_path"] / "adapted.pt", weights_only=False)["state_dict"]
            for name in state_a:
                torch.testing.assert_close(state_a[name], state_b[name], rtol=0, atol=0)
            with np.load(root / "a" / a["unit_path"] / "predictions.npz") as pred_a, np.load(root / "b" / b["unit_path"] / "predictions.npz") as pred_b:
                np.testing.assert_array_equal(pred_a["pred"], pred_b["pred"])

    def test_population_checkpoint_completion_and_exclusion_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "best.pt"
            with self.assertRaises(FileNotFoundError):
                load_population_checkpoint(path, 42)
            excluded = list(DEV_IDS + TARGET_IDS) + [f"vitaldb:historical_unit_fixture_{i}" for i in range(21)]
            manifest = {"excluded_patients": excluded, "partitions": {"train": {"patients": ["unrelated_unit_fixture"]}},
                        "target_center": [120, 80], "target_scale": [20, 10]}
            config = {"variant": "baseline", "seed": 42, "pool": 1, "protocol_commit": "1b93cf1-test-fixture",
                      "toy_test_only": False}
            saved = {"variant": "baseline", "seed": 42, "pool": 1, "config": config, "config_hash": canonical_hash(config),
                     "hashes": {"manifest": canonical_hash(manifest)}, "epoch": 4, "center": [120, 80], "scale": [20, 10]}
            torch.save(saved, path)
            completion = {"status": "complete", "toy_test_only": False, "best_checkpoint_sha256": file_hash(path),
                          "config_hash": canonical_hash(config), "manifest_hash": canonical_hash(manifest), "selected_epoch": 4}
            for name, data in (("completion", completion), ("config", config), ("manifest", manifest)):
                atomic_json(directory / f"{name}.json", data)
            self.assertEqual(load_population_checkpoint(path, 42)["epoch"], 4)
            completion["toy_test_only"] = True
            atomic_json(directory / "completion.json", completion)
            with self.assertRaises(ValueError):
                load_population_checkpoint(path, 42)
            completion["toy_test_only"] = False
            manifest["partitions"]["train"]["patients"].append(TARGET_IDS[0])
            saved["hashes"]["manifest"] = canonical_hash(manifest)
            torch.save(saved, path)
            completion.update(best_checkpoint_sha256=file_hash(path), manifest_hash=canonical_hash(manifest))
            atomic_json(directory / "manifest.json", manifest)
            atomic_json(directory / "completion.json", completion)
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_population_checkpoint(path, 42)

    def test_cli_rejects_missing_baselines_before_writing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(FileNotFoundError):
                main(["--phase", "test", "--population-root", str(root / "absent"),
                      "--individual-root", str(root / "individual"), "--output-dir", str(root / "output")])
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
