"""Synthetic inference fixtures only; no training or research-score selection."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personalized_forecasting.adaptation import apply_calibration, configure_trainable, predict
from personalized_forecasting.predict import (NEURAL_METHODS, SCALAR_METHODS, SOURCE_FILES,
                                               load_personal_predictor, main, predict_personal)
from shared.models import BPNet
from shared.predict import PROTOCOL_COMMIT, V2_ROOT, _hash_file, _hash_json, prepare_inputs


class PersonalPredictionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_root = self.root / "saved_run"
        self.run_root.mkdir()
        self.checkpoint_root = self.root / "population"
        self.checkpoint_root.mkdir()
        self.checkpoint = self.checkpoint_root / "best.pt"
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.baseline = BPNet(pool=1, variant="baseline").eval()
        self.center = np.array([115.41764831542969, 62.9281005859375], dtype=np.float64)
        self.scale = np.array([18.941879272460938, 12.083216667175293], dtype=np.float64)
        hashes = {"manifest": "unused-in-detached-fixture", "source": {
            name: _hash_file(V2_ROOT / name) for name in ("shared/models.py", "shared/data.py", "PROTOCOL.md")}}
        config = {"variant": "baseline", "pool": 1, "protocol_commit": PROTOCOL_COMMIT,
                  "toy_test_only": False, "hashes": hashes}
        torch.save({"variant": "baseline", "pool": 1, "seed": 42, "config": config,
                    "config_hash": _hash_json(config), "hashes": hashes,
                    "state_dict": self.baseline.state_dict(), "center": self.center.tolist(),
                    "scale": self.scale.tolist()}, self.checkpoint)
        self.manifest = {"protocol": "v2-personal-calibration-v1", "target_ids": ["fixture:person"],
                         "development_ids": [], "checkpoint_hashes": {"42": _hash_file(self.checkpoint)},
                         "source_hashes": {name: _hash_file(V2_ROOT / name) for name in SOURCE_FILES}}
        self.manifest_path = self.run_root / "run_manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.waves = np.random.default_rng(17).normal(size=(3, 2, 1250)).astype(np.float32)

    def make_unit(self, method):
        directory = self.run_root / method
        directory.mkdir(exist_ok=True)
        setting = ({"lr": 0.0001, "steps": 20} if method in NEURAL_METHODS else
                   {"penalty": 1} if method == "affine" else {})
        config = {"patient": "fixture:person", "case": "fixture:case", "seed": 42,
                  "budget": 10, "strategy": method, "setting": setting, "phase": "test",
                  "checkpoint_sha256": _hash_file(self.checkpoint), "provenance_hash": _hash_json(self.manifest),
                  "selection_hash": "fixture-only", "device": "cpu",
                  "support_indices_hash": "fixture-support", "query_indices_hash": "fixture-query"}
        audit = {"strategy": method, "support_count": 10, "query_used_for_adaptation": False}
        model = copy.deepcopy(self.baseline)
        if method in NEURAL_METHODS:
            enabled = configure_trainable(model, method)
            # A known deterministic parameter edit, not fitting/training.
            with torch.no_grad():
                dict(model.named_parameters())[enabled[0]].add_(0.0125)
            saved = {"state_dict": model.state_dict(), "center": self.center.tolist(),
                     "scale": self.scale.tolist(), "variant": "baseline", "config": config}
            artifact = directory / "adapted.pt"
            torch.save(saved, artifact)
            audit.update({"trainable_parameter_names": enabled,
                          "batchnorm_running_statistics_frozen": True, "dropout_active": False})
        else:
            slope, intercept = [1., 1.], [0., 0.]
            if method == "bias":
                intercept = [-0.7, 0.2]
            elif method == "affine":
                slope, intercept = [0.6, 1.2], [-0.8, 0.4]
            elif method == "last_reference":
                slope, intercept = [0., 0.], [-0.5, 0.7]
            elif method == "calibration_mean":
                slope, intercept = [0., 0.], [-0.2, 0.3]
            saved = {"method": method, "slope": slope, "intercept": intercept,
                     "center": self.center.tolist(), "scale": self.scale.tolist(),
                     "penalty": 1 if method == "affine" else None, "n_support": 10}
            artifact = directory / "calibration.json"
            artifact.write_text(json.dumps(saved))
            audit["calibration"] = saved
        record = {**config, "config": config, "config_hash": _hash_json(config), "role": "target",
                  "completed_at_utc": "fixture-only-not-a-research-run", "adaptation": audit,
                  "artifact_hashes": {artifact.name: _hash_file(artifact)}}
        (directory / "unit.json").write_text(json.dumps(record))
        return artifact, saved, record, model

    def load(self, artifact, **kwargs):
        return load_personal_predictor(artifact, self.checkpoint,
                                       expected_patient_id="fixture:person", trusted_checkpoint=True, **kwargs)

    def rewrite(self, artifact, saved, record):
        if artifact.suffix == ".pt":
            torch.save(saved, artifact)
        else:
            artifact.write_text(json.dumps(saved))
        record["artifact_hashes"][artifact.name] = _hash_file(artifact)
        (artifact.parent / "unit.json").write_text(json.dumps(record))

    def test_all_eight_strategies_match_original_prediction_arithmetic(self):
        prepared = prepare_inputs(self.waves)
        for method in (*SCALAR_METHODS, *NEURAL_METHODS):
            with self.subTest(method=method):
                artifact, saved, _, model = self.make_unit(method)
                loaded = self.load(artifact)
                base, _ = predict(model, prepared, self.center, self.scale)
                expected = apply_calibration(saved, base) if method in SCALAR_METHODS else base
                actual = predict_personal(loaded, self.waves)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual, predict_personal(loaded, prepared, already_normalized=True))
                self.assertEqual(actual.shape, (3, 2))
                self.assertEqual(actual.dtype, np.float64)
                self.assertFalse(loaded.model.training)
                self.assertFalse(any(parameter.requires_grad for parameter in loaded.model.parameters()))
                self.assertFalse((artifact.parent / "predictions.npz").exists())

    def test_reference_controls_do_not_run_waveform_model(self):
        for method in ("last_reference", "calibration_mean"):
            artifact, _, _, _ = self.make_unit(method)
            loaded = self.load(artifact)
            with patch.object(loaded.model, "forward", side_effect=AssertionError("must not infer")):
                result = predict_personal(loaded, self.waves)
            np.testing.assert_array_equal(result, np.repeat(result[:1], len(result), axis=0))

    def test_trust_patient_hash_and_manifest_checks_precede_pickle(self):
        artifact, _, _, _ = self.make_unit("head")
        with patch("shared.predict.torch.load") as loading:
            with self.assertRaisesRegex(ValueError, "Explicit trusted"):
                load_personal_predictor(artifact, self.checkpoint, expected_patient_id="fixture:person")
            with self.assertRaisesRegex(ValueError, "Patient identity"):
                load_personal_predictor(artifact, self.checkpoint, expected_patient_id="wrong", trusted_checkpoint=True)
            bad_checkpoint = self.root / "wrong.pt"
            bad_checkpoint.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(ValueError, "Population checkpoint SHA256"):
                load_personal_predictor(artifact, bad_checkpoint, expected_patient_id="fixture:person", trusted_checkpoint=True)
            self.manifest_path.write_text(json.dumps({**self.manifest, "protocol": "changed"}))
            with self.assertRaisesRegex(ValueError, "provenance hash"):
                self.load(artifact)
            loading.assert_not_called()

    def test_rejects_modified_artifact_missing_completion_and_config(self):
        artifact, saved, record, _ = self.make_unit("bias")
        artifact.write_text(artifact.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "artifact SHA256"):
            self.load(artifact)
        self.rewrite(artifact, saved, record)
        record["config_hash"] = "wrong"
        self.rewrite(artifact, saved, record)
        with self.assertRaisesRegex(ValueError, "configuration hash"):
            self.load(artifact)
        record["config_hash"] = _hash_json(record["config"])
        record.pop("completed_at_utc")
        self.rewrite(artifact, saved, record)
        with self.assertRaisesRegex(ValueError, "completed development or test"):
            self.load(artifact)

    def test_detached_bundle_requires_explicit_original_manifest(self):
        artifact, _, _, _ = self.make_unit("head")
        detached = self.root / "detached"
        detached.mkdir()
        for path in (artifact, artifact.parent / "unit.json"):
            (detached / path.name).write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "run_manifest.json is required"):
            self.load(detached / artifact.name)
        loaded = self.load(detached / artifact.name, run_manifest_path=self.manifest_path)
        self.assertEqual(loaded.config["strategy"], "head")

    def test_rejects_source_drift_before_pickle(self):
        self.manifest["source_hashes"]["personalized_forecasting/adaptation.py"] = "wrong"
        self.manifest_path.write_text(json.dumps(self.manifest))
        artifact, _, _, _ = self.make_unit("head")
        with patch("shared.predict.torch.load") as loading:
            with self.assertRaisesRegex(ValueError, "source contract"):
                self.load(artifact)
            loading.assert_not_called()

    def test_rejects_frozen_buffer_or_parameter_changes_and_nonfinite_state(self):
        for kind in ("buffer", "parameter", "nonfinite"):
            with self.subTest(kind=kind):
                artifact, saved, record, _ = self.make_unit("head")
                state = saved["state_dict"]
                key = (next(name for name in state if name.endswith("running_mean")) if kind == "buffer"
                       else next(name for name in state if not name.startswith("head.") and name.endswith("weight")))
                state[key] = state[key] + 1
                if kind == "nonfinite":
                    state[key].flatten()[0] = float("nan")
                self.rewrite(artifact, saved, record)
                with self.assertRaisesRegex(ValueError, "Frozen parameter or BN buffer|Invalid adapted tensor"):
                    self.load(artifact)

    def test_rejects_normalization_and_calibration_contract_drift(self):
        for method in ("head", "affine", "population"):
            artifact, saved, record, _ = self.make_unit(method)
            saved["center"][0] += 0.125
            self.rewrite(artifact, saved, record)
            with self.assertRaisesRegex(ValueError, "normalization differs"):
                self.load(artifact)
        for invalid in ([], [1, 2, 3], [float("nan"), 1]):
            artifact, saved, record, _ = self.make_unit("bias")
            saved["slope"] = invalid
            self.rewrite(artifact, saved, record)
            with self.assertRaises(ValueError):
                self.load(artifact)
        artifact, saved, record, _ = self.make_unit("population")
        saved["intercept"] = [1, 0]
        self.rewrite(artifact, saved, record)
        with self.assertRaisesRegex(ValueError, "nonzero intercept"):
            self.load(artifact)

    def test_rejects_input_metadata_nonfinite_constant_shape_and_bad_batch(self):
        artifact, _, _, _ = self.make_unit("head")
        loaded = self.load(artifact)
        for options in ({"fs_hz": 250}, {"channel_order": ("PPG", "ECG")},
                        {"already_normalized": True}, {"batch_size": 0}, {"batch_size": True}):
            with self.assertRaises(ValueError):
                predict_personal(loaded, self.waves, **options)
        nonfinite = self.waves.copy()
        nonfinite[0, 0, 0] = np.nan
        for waves in (nonfinite, np.zeros_like(self.waves), self.waves.transpose(0, 2, 1), self.waves[:0]):
            with self.assertRaises(ValueError):
                predict_personal(loaded, waves)

    def test_cli_writes_new_prediction_file_only_and_protects_experiments(self):
        artifact, _, _, _ = self.make_unit("bias")
        inputs = self.root / "waves.npy"
        np.save(inputs, self.waves, allow_pickle=False)
        output = self.root / "result.npy"
        arguments = ["--artifact", str(artifact), "--population-checkpoint", str(self.checkpoint),
                     "--patient-id", "fixture:person", "--input", str(inputs), "--trust-checkpoint"]
        before = {path: _hash_file(path) for path in self.run_root.rglob("*") if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(arguments + ["--output", str(output)]), 0)
        np.testing.assert_array_equal(np.load(output, allow_pickle=False),
                                      predict_personal(self.load(artifact), self.waves))
        for blocked in (output, self.run_root / "new.npy", artifact.parent / "new.npy", self.checkpoint_root / "new.npy"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(arguments + ["--output", str(blocked)])
        after = {path: _hash_file(path) for path in self.run_root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
