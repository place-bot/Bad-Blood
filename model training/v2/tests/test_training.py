"""Toy-only epoch resume and label-selection safeguards."""
import copy
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
from attention_cnn.train import _validate_bundle, run_training


def toy_bundle():
    rng = np.random.default_rng(1234)
    data = {}
    for name in ("train", "validation", "test", "aami"):
        data[name] = {
            "waves": rng.uniform(0, 1, (8, 2, 1250)).astype(np.float32),
            "y": np.column_stack((rng.uniform(95, 150, 8), rng.uniform(55, 95, 8))).astype(np.float32),
            "subjects": np.array([f"{name}_{i // 4}" for i in range(8)], dtype="U32"),
            # Non-contiguous eligible indices detect implicit slicing mistakes.
            "indices": np.array([0, 2, 3, 4, 6, 7]),
        }
    y = data["train"]["y"][data["train"]["indices"]]
    data.update(center=y.mean(0), scale=y.std(0), manifest={"toy": True}, manifest_hash="toy-verified-manifest")
    return data


SETTINGS = {"max_epochs": 3, "batch_size": 3, "inference_batch_size": 3}


class TrainingTests(unittest.TestCase):
    def test_cpu_epoch_resume_is_exact_and_saves_aligned_best_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = toy_bundle()
            full = run_training("se_last", 42, root, root / "full", "cpu", bundle=data, test_settings=SETTINGS)
            interrupted = run_training("se_last", 42, root, root / "resumed", "cpu", bundle=data,
                                       test_settings=SETTINGS, stop_after_epoch=1)
            self.assertEqual(interrupted["epoch"], 1)
            resumed = run_training("se_last", 42, root, root / "resumed", "cpu", True,
                                   bundle=data, test_settings=SETTINGS)
            self.assertTrue(full["toy_test_only"])
            self.assertEqual(full["selected_epoch"], resumed["selected_epoch"])
            for checkpoint_name in ("last.pt", "best.pt"):
                a = torch.load(root / "full" / checkpoint_name, weights_only=False)
                b = torch.load(root / "resumed" / checkpoint_name, weights_only=False)
                for key in a["state_dict"]:
                    torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)
                self.assertEqual(a["scheduler"], b["scheduler"])
                self.assertEqual(a["stale"], b["stale"])
            for split in ("validation", "test", "aami"):
                a = np.load(root / "full" / f"predictions_{split}.npz")
                b = np.load(root / "resumed" / f"predictions_{split}.npz")
                np.testing.assert_array_equal(a["pred"], b["pred"])
                np.testing.assert_array_equal(a["source_indices"], data[split]["indices"])
                np.testing.assert_array_equal(a["y"], data[split]["y"][data[split]["indices"]])
                np.testing.assert_array_equal(a["subjects"], data[split]["subjects"][data[split]["indices"]])
            for filename in ("config.json", "manifest.json", "environment.json", "history.json", "metrics.json", "costs.json", "completion.json"):
                self.assertTrue((root / "full" / filename).is_file())
            completed = run_training("se_last", 42, root, root / "resumed", "cpu", True,
                                     bundle=data, test_settings=SETTINGS)
            self.assertEqual(completed, resumed)
            with self.assertRaises(FileExistsError):
                run_training("se_last", 42, root, root / "full", "cpu", bundle=data, test_settings=SETTINGS)
            with self.assertRaises(ValueError):
                run_training("se_last", 43, root, root / "resumed", "cpu", True, bundle=data, test_settings=SETTINGS)
            changed = copy.deepcopy(data)
            changed["manifest_hash"] = "changed-split"
            with self.assertRaises(ValueError):
                run_training("se_last", 42, root, root / "resumed", "cpu", True, bundle=changed, test_settings=SETTINGS)

    def test_held_out_labels_never_change_training_or_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = toy_bundle()
            changed = copy.deepcopy(data)
            changed["test"]["y"] += 50
            changed["aami"]["y"] -= 20
            settings = {**SETTINGS, "max_epochs": 1}
            run_training("baseline", 42, root, root / "original", "cpu", bundle=data, test_settings=settings)
            run_training("baseline", 42, root, root / "changed", "cpu", bundle=changed, test_settings=settings)
            a = torch.load(root / "original/best.pt", weights_only=False)
            b = torch.load(root / "changed/best.pt", weights_only=False)
            self.assertEqual(a["epoch"], b["epoch"])
            self.assertEqual(a["validation"], b["validation"])
            for key in a["state_dict"]:
                torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)

    def test_bundle_validation_detects_leakage_and_train_only_normalization(self):
        data = toy_bundle()
        _validate_bundle(data, True)
        overlap = copy.deepcopy(data)
        overlap["test"]["subjects"][0] = data["train"]["subjects"][0]
        with self.assertRaises(ValueError):
            _validate_bundle(overlap, True)
        wrong_center = copy.deepcopy(data)
        wrong_center["center"] += 1
        with self.assertRaises(ValueError):
            _validate_bundle(wrong_center, True)
        with self.assertRaises(ValueError):
            _validate_bundle(data, False)


if __name__ == "__main__":
    unittest.main()
