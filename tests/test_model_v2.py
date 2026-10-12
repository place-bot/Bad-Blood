"""Contract and failure-mode checks with synthetic training data, never BP evidence."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from bp_runtime.models import make_model, SEGate, TemporalPool
from bp_runtime.preprocessing import from_raw_125, normalize
from bp_runtime.release import BundlePredictor, sha256
from bp_training.config import Config, load_config
from bp_training.data import build_manifest, sampling_probabilities, training_statistics, validate_manifest
from bp_training.evaluate import compare_evaluations, evaluate_bundle, evaluate_run, summarize_evaluations
from bp_training.io import ROOT, atomic_json, digest, object_hash
from bp_training.losses import objective
from bp_training.metrics import metrics, paired_difference
from bp_training.preflight import preflight
from bp_training.release import export_run, public_fixtures
from bp_training.trainer import select_device, train

torch.set_num_threads(1)
V1 = ROOT / "model training/PulseDB-based Model-V1"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root):
    data = root / "arrays"
    data.mkdir()
    rng = np.random.default_rng(42)
    waves = np.stack([normalize(w) for w in rng.normal(size=(16, 2, 1250))])
    labels = np.column_stack((100 + np.arange(16) * 3, 60 + np.arange(16) % 5 * 2)).astype("float32")
    subjects = np.repeat(["synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d"], 4)
    np.save(data / "waves.npy", waves)
    np.save(data / "y.npy", labels)
    np.save(data / "subjects.npy", subjects)
    np.save(data / "valid.npy", np.ones(16, dtype=bool))
    np.save(root / "excluded.npy", np.asarray(["synthetic-heldout"]))
    atomic_json(root / "split.json", {"train": subjects[::4][:3].tolist(),
                                      "validation": ["synthetic-d"]})
    manifest = root / "manifest.json"
    build_manifest(data, manifest, root / "split.json", [root / "excluded.npy"], source="synthetic-test-only")
    return manifest


class ConfigAndModelTests(unittest.TestCase):
    def test_strict_config_and_explicit_cuda_failure(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.json"
            path.write_text('{"batch_szie": 12}')
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_config(path)
        for kwargs in ({"precision": "fp16", "device": "cpu"}, {"seed": -1},
                       {"batch_size": True}, {"architecture": "v1", "attention": "pool"},
                       {"loss_multiplier": float("nan")}, {"sbp_edges": [130, 90]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                select_device(Config())

    def test_all_candidates_have_finite_gradients_and_contract_shapes(self):
        for architecture, attention in (("v1", "none"), ("wider_resnet", "none"),
                                         ("dual_tcn", "none"), ("dual_tcn", "pool"),
                                         ("dual_tcn", "se"), ("dual_tcn", "both")):
            with self.subTest(architecture=architecture, attention=attention):
                model = make_model(architecture, attention)
                result = model(torch.randn(2, 2, 1250))
                self.assertEqual(tuple(result.shape), (2, 2))
                result.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
        model = make_model("dual_tcn")
        self.assertEqual(tuple(model.ecg(torch.randn(2, 1, 1250)).shape), (2, 96, 313))

    def test_attention_starts_at_uniform_pooling_and_preserves_time(self):
        features = torch.randn(3, 128, 313)
        pool = TemporalPool()
        torch.testing.assert_close(pool(features), features.mean(-1))
        self.assertEqual(sum(p.numel() for p in pool.parameters()), 4161)
        gate = SEGate()
        self.assertEqual(sum(p.numel() for p in gate.parameters()), 4240)
        # Per-channel multiplication cannot shift or reindex temporal positions.
        actual = gate(features)
        expected_gate = gate.gate(features.mean(-1)).unsqueeze(-1)
        torch.testing.assert_close(actual, features * expected_gate)

    def test_frozen_v1_weights_and_preprocessing_have_parity(self):
        original_network = load_module("frozen_network", V1 / "network.py")
        original_common = load_module("frozen_common", V1 / "common.py")
        checkpoint = torch.load(V1 / "models/cnn_full.pt", map_location="cpu", weights_only=True)
        reference = original_network.BPNet(pool=1).eval()
        reference.load_state_dict(checkpoint["state_dict"])
        model = make_model("v1").eval()
        model.load_state_dict(checkpoint["state_dict"])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 37666)
        x = public_fixtures()
        with torch.inference_mode():
            torch.testing.assert_close(model(torch.from_numpy(x)), reference(torch.from_numpy(x)), rtol=0, atol=0)
        t = np.arange(1250) / 125
        raw = np.stack([np.sin(t * 8) + .2 * np.sin(t * 31), np.sin(t * 8 - .4)])
        np.testing.assert_array_equal(from_raw_125(raw), original_common.from_raw_125(raw))
        np.testing.assert_array_equal(normalize(raw), original_common.normalize(raw))

    def test_losses_use_declared_units(self):
        center, scale = torch.tensor([120., 70.]), torch.tensor([20., 10.])
        pred, labels = torch.zeros(1, 2), torch.tensor([[130., 80.]])
        expected = {"standardized_mse": .625, "mmhg_mse": 100., "mmhg_huber": 37.5}
        for loss, value in expected.items():
            scaled, raw = objective(pred, labels, center, scale,
                                    Config(loss=loss, loss_multiplier=.25))
            self.assertAlmostEqual(raw.item(), value)
            self.assertAlmostEqual(scaled.item(), value * .25)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_training_statistics_exclude_validation(self):
        audit = validate_manifest(self.manifest)
        center, scale = training_statistics(audit)
        expected = audit["labels"][:12]
        np.testing.assert_array_equal(center, expected.mean(0))
        np.testing.assert_array_equal(scale, expected.std(0))
        self.assertEqual(len(audit["indices"]["validation"]), 4)

    def test_overlap_and_changed_file_fail_closed(self):
        np.save(self.root / "excluded.npy", np.asarray(["synthetic-a"]))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_manifest(self.manifest)
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_manifest(self.root / "arrays", self.root / "overlap.json", self.root / "split.json",
                           [self.root / "excluded.npy"])
        self.assertFalse((self.root / "overlap.json").exists())

    def test_split_overlap_and_reference_channel_fail(self):
        atomic_json(self.root / "split.json", {"train": ["synthetic-a", "synthetic-b", "synthetic-c"],
                                               "validation": ["synthetic-a", "synthetic-d"]})
        with self.assertRaisesRegex(ValueError, "split"):
            build_manifest(self.root / "arrays", self.root / "overlap.json", self.root / "split.json",
                           [self.root / "excluded.npy"])
        x = np.load(self.root / "arrays/waves.npy")
        np.save(self.root / "arrays/waves.npy", np.concatenate([x, x[:, :1]], axis=1))
        with self.assertRaisesRegex(ValueError, "Waveforms"):
            build_manifest(self.root / "arrays", self.root / "abp.json", self.root / "split.json",
                           [self.root / "excluded.npy"])

    def test_evaluation_roles_cannot_train_or_fit_statistics(self):
        path = self.root / "holdout.json"
        build_manifest(self.root / "arrays", path, role="prospective")
        with self.assertRaisesRegex(ValueError, "only"):
            training_statistics(validate_manifest(path))
        with self.assertRaisesRegex(ValueError, "forbidden"):
            train(Config(device="cpu", threads=1), path, self.root / "forbidden")
        self.assertFalse((self.root / "forbidden").exists())

    def test_balancing_is_capped_and_sparse_bins_are_not_amplified(self):
        labels = np.asarray([[120, 75]] * 20 + [[180, 110]] * 2, dtype=np.float32)
        subjects = np.asarray([f"s{i}" for i in range(22)])
        config = Config(balancing=True, balance_min_subjects=2)
        p = sampling_probabilities(labels, subjects, config)
        self.assertAlmostEqual(p[-1] / p[0], 4)
        subjects[-2:] = "one"
        p = sampling_probabilities(labels, subjects, config)
        np.testing.assert_allclose(p, np.full(22, 1 / 22))

    def test_metrics_weight_people_not_number_of_overlapping_windows(self):
        labels = np.asarray([[120, 70]] * 4)
        pred = labels + np.asarray([[0, 0], [0, 0], [0, 0], [20, 10]])
        subjects = np.asarray(["a", "a", "a", "b"])
        result = metrics(labels, pred, subjects, bootstrap=10)
        self.assertEqual(result["macro_mae"], [10, 5])
        self.assertEqual(result["mae"], [5, 2.5])
        paired = paired_difference(labels, pred, pred, subjects, bootstrap=10)
        self.assertEqual(paired["paired_participant_ci95"], [[0, 0], [0, 0]])

    def test_seed_summary_requires_complete_distinct_matched_conditions(self):
        directories = []
        for i, seed in enumerate((20260923, 20260924, 20260925)):
            directory = self.root / str(seed)
            directories.append(directory)
            config = Config(seed=seed).to_dict()
            atomic_json(directory / "report.json", {
                "provenance": {"kind": "development", "dataset_hash": "test-only", "config": config},
                "metrics": {"macro_mae": [10 + i, 5 + i], "primary": 7.5 + i}})
        result = summarize_evaluations(directories, self.root / "summary.json")
        self.assertEqual(result["conditions"][0]["mean_macro_mae"], [11, 6])
        self.assertEqual(result["conditions"][0]["training_seed_sd"], [1, 1])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize_evaluations(directories[:2], self.root / "incomplete.json")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_evaluations(directories + directories[:1], self.root / "duplicate.json")


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)
        self.config = Config(device="cpu", threads=1, epochs=2, batch_size=4, validation_batch_size=4)

    def tearDown(self):
        self.temp.cleanup()

    def test_epoch_boundary_and_mid_epoch_failure_resume_exactly(self):
        full = self.root / "full"
        train(self.config, self.manifest, full)
        expected = torch.load(full / "last.pt", weights_only=True)
        for mode in ("boundary", "partial"):
            run = self.root / mode
            if mode == "boundary":
                result = train(self.config, self.manifest, run, stop_after_epochs=1)
                self.assertEqual(result["state"], "interrupted")
            else:
                calls = 0

                def fail_after_committed_epoch(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 5:
                        raise RuntimeError("Simulated interruption after partial optimizer work")
                    return objective(*args)

                with mock.patch("bp_training.trainer.objective", side_effect=fail_after_committed_epoch):
                    with self.assertRaisesRegex(RuntimeError, "Simulated"):
                        train(self.config, self.manifest, run)
                self.assertEqual(torch.load(run / "last.pt", weights_only=True)["epoch"], 1)
            result = train(self.config, self.manifest, run, resume=True)
            self.assertEqual(result["state"], "completed")
            actual = torch.load(run / "last.pt", weights_only=True)
            for key, value in expected["model"].items():
                torch.testing.assert_close(value, actual["model"][key], rtol=0, atol=0)
            self.assertEqual(expected["global_step"], actual["global_step"])
            self.assertEqual(expected["history"][-1]["validation"], actual["history"][-1]["validation"])
            self.assertEqual(expected["rng"]["sampler"], actual["rng"]["sampler"])

    def test_incompatible_resume_and_run_collision_do_not_overwrite(self):
        run = self.root / "run"
        train(self.config, self.manifest, run, stop_after_epochs=1)
        before = digest(run / "last.pt")
        with self.assertRaises(FileExistsError):
            train(self.config, self.manifest, run)
        changed = Config(**{**self.config.to_dict(), "seed": 42})
        with self.assertRaisesRegex(ValueError, "differs"):
            train(changed, self.manifest, run, resume=True)
        self.assertEqual(before, digest(run / "last.pt"))
        with self.assertRaisesRegex(ValueError, "completed"):
            evaluate_run(run, self.root / "not-evaluated")

    def test_train_evaluate_export_and_explicit_deployment_loader(self):
        config = Config(**{**self.config.to_dict(), "architecture": "dual_tcn", "attention": "both", "epochs": 1})
        run, release = self.root / "run", self.root / "release"
        train(config, self.manifest, run)
        report = evaluate_run(run, self.root / "evaluation")
        self.assertEqual(report["metrics"]["records"], 4)
        exported = export_run(run, release)
        self.assertLessEqual(exported["max_abs_difference_mmHg"], .01)
        predictor = BundlePredictor(release)
        batch = public_fixtures()[:2]
        reference = predictor.predict(batch)
        for threads in (1, 2, 4):
            np.testing.assert_allclose(BundlePredictor(release, threads).predict(batch), reference, atol=.01, rtol=0)
        import sys
        sys.path.insert(0, str(ROOT / "deployment/raspberry-pi"))
        from app import load_predictor
        loaded, _, bundle_id, label = load_predictor(release, threads=1)
        self.assertEqual(bundle_id, predictor.bundle_id)
        self.assertIn("attention=both", label)
        np.testing.assert_allclose(loaded.predict(batch), reference, atol=.01, rtol=0)
        from types import SimpleNamespace
        import time
        from acquisition import RAW_VERSION
        from service import InferenceState
        from sources import Acquisition, SignalWindow, SourceUnavailable
        state = InferenceState(SimpleNamespace(mode="device_live"), loaded.predict, model_label=label)
        window = SignalWindow("synthetic-contract-test", batch[0],
                              acquisition=Acquisition("test", 0, 1250, time.monotonic()),
                              quality={"status": "screened", "preprocessing": RAW_VERSION})
        state.process_window(window)
        self.assertEqual(state.snapshot()["model"], label)
        self.assertEqual(state.snapshot()["status"], "ok")
        with self.assertRaises(SourceUnavailable):
            state.process_window(window)
        state._acquisition_end -= 13
        self.assertEqual(state.snapshot()["status"], "stale")
        self.assertIsNone(state.snapshot()["prediction"])
        bundle_report = evaluate_bundle(release, self.manifest, self.root / "bundle-eval")
        comparison = compare_evaluations(self.root / "evaluation", self.root / "bundle-eval",
                                         self.root / "comparison.json")
        np.testing.assert_allclose(comparison["candidate_minus_baseline_macro_mae"], 0, atol=.01)
        self.assertEqual(bundle_report["provenance"]["kind"], "development")
        with self.assertRaises(FileExistsError):
            evaluate_run(run, self.root / "evaluation")
        # Runtime failure must not silently fall back to V1.
        with (release / "weights.pt").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_predictor(release, threads=1)

    def test_heldout_evaluation_requires_matching_frozen_artifact(self):
        config = Config(**{**self.config.to_dict(), "epochs": 1})
        run, release = self.root / "run", self.root / "release"
        train(config, self.manifest, run)
        export_run(run, release)
        # A role label cannot turn development participants into a held-out cohort.
        heldout = self.root / "historical.json"
        build_manifest(self.root / "arrays", heldout, role="historical")
        with self.assertRaisesRegex(ValueError, "freeze record"):
            evaluate_bundle(release, heldout, self.root / "blocked")
        freeze = self.root / "freeze.json"
        atomic_json(freeze, {"bundle_manifest_sha256": "wrong", "dataset_hash": "wrong", "protocol": "test"})
        with self.assertRaisesRegex(ValueError, "Freeze record"):
            evaluate_bundle(release, heldout, self.root / "blocked", freeze)
        atomic_json(freeze, {"bundle_manifest_sha256": sha256(release / "manifest.json"),
                            "dataset_hash": validate_manifest(heldout)["hash"], "protocol": "synthetic test"})
        with self.assertRaisesRegex(ValueError, "training manifest"):
            evaluate_bundle(release, heldout, self.root / "blocked", freeze)
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_bundle(release, heldout, self.root / "blocked", freeze, self.manifest)
        new = self.root / "new-people"
        new.mkdir()
        for name in ("waves", "y", "valid"):
            np.save(new / f"{name}.npy", np.load(self.root / f"arrays/{name}.npy"))
        np.save(new / "subjects.npy", np.repeat(["new-a", "new-b", "new-c", "new-d"], 4))
        prospective = self.root / "prospective.json"
        build_manifest(new, prospective, role="prospective", source="synthetic-test-only")
        atomic_json(freeze, {"bundle_manifest_sha256": sha256(release / "manifest.json"),
                            "dataset_hash": validate_manifest(prospective)["hash"], "protocol": "synthetic test"})
        result = evaluate_bundle(release, prospective, self.root / "final-evaluation", freeze, self.manifest)
        self.assertEqual(result["provenance"]["kind"], "prospective")

    def test_preflight_and_spawn_workers(self):
        smoke = preflight(self.config, self.manifest, self.root / "outputs")
        self.assertEqual(smoke["parameters"], 37666)
        self.assertEqual(smoke["environment"]["device"], "cpu")
        config = Config(**{**self.config.to_dict(), "workers": 1, "epochs": 1})
        result = train(config, self.manifest, self.root / "workers")
        self.assertEqual(result["state"], "completed")


if __name__ == "__main__":
    unittest.main()
