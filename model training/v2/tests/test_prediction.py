"""Offline inference contracts; fixtures do not count as research runs."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.models import BPNet
from shared.predict import (PROTOCOL_COMMIT, V2_ROOT, _hash_file, _hash_json,
                            load_predictor, main, predict_array, prepare_inputs)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "best.pt"
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = BPNet(pool=1, variant="baseline").eval()
        self.center = np.array([120, 75], dtype=np.float32)
        self.scale = np.array([20, 12], dtype=np.float32)
        manifest = {"task": "contemporaneous_SBP_DBP", "fs_hz": 125, "samples_per_channel": 1250,
                    "target_center": self.center.tolist(), "target_scale": self.scale.tolist()}
        hashes = {"manifest": _hash_json(manifest), "source": {
            path: _hash_file(V2_ROOT / path) for path in ("shared/models.py", "shared/data.py", "PROTOCOL.md")}}
        config = {"variant": "baseline", "pool": 1, "protocol_commit": PROTOCOL_COMMIT,
                  "toy_test_only": False, "hashes": hashes}
        self.saved = {"variant": "baseline", "pool": 1, "config": config,
                      "config_hash": _hash_json(config), "hashes": hashes,
                      "state_dict": self.model.state_dict(), "center": self.center.tolist(),
                      "scale": self.scale.tolist()}
        torch.save(self.saved, self.checkpoint)
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        self.waves = np.random.default_rng(2).normal(size=(3, 2, 1250)).astype(np.float32)

    def test_predictions_equal_direct_model_and_are_repeatable(self):
        prepared = prepare_inputs(self.waves)
        model, center, scale = load_predictor(self.checkpoint, trusted_checkpoint=True)
        actual = predict_array(model, prepared, center, scale)
        with torch.inference_mode():
            expected = self.model(torch.from_numpy(prepared)).numpy() * self.scale + self.center
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, predict_array(model, prepared, center, scale))
        self.assertEqual(actual.shape, (3, 2))
        self.assertEqual(actual.dtype, np.float32)
        self.assertFalse(model.training)

    def test_input_normalization_and_source_metadata(self):
        prepared = prepare_inputs(self.waves)
        np.testing.assert_array_equal(prepared.min(axis=-1), np.zeros((3, 2)))
        np.testing.assert_array_equal(prepared.max(axis=-1), np.ones((3, 2)))
        np.testing.assert_array_equal(prepare_inputs(prepared, already_normalized=True), prepared)
        for options in ({"fs_hz": 250}, {"channel_order": ("PPG", "ECG")}):
            with self.assertRaisesRegex(ValueError, "125 Hz|Channel order"):
                prepare_inputs(self.waves, **options)
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            prepare_inputs(self.waves, already_normalized=True)

    def test_invalid_shape_nonfinite_and_constant_channels(self):
        for invalid in (np.zeros((2, 1250)), np.zeros((1, 1250, 2)), np.zeros((0, 2, 1250))):
            with self.assertRaisesRegex(ValueError, "N x 2 x 1250"):
                prepare_inputs(invalid)
        for value in (np.nan, np.inf):
            invalid = self.waves.copy()
            invalid[0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                prepare_inputs(invalid)
        invalid = self.waves.copy()
        invalid[1, 1] = 7
        with self.assertRaisesRegex(ValueError, "nonconstant"):
            prepare_inputs(invalid)

    def test_explicit_trust_precedes_deserialization(self):
        with patch("shared.predict.torch.load") as loading:
            with self.assertRaisesRegex(ValueError, "Explicit trusted"):
                load_predictor(self.checkpoint)
            loading.assert_not_called()

    def test_rejects_noncanonical_pool_or_manifest(self):
        self.saved["pool"] = 5
        torch.save(self.saved, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "pool=1"):
            load_predictor(self.checkpoint, trusted_checkpoint=True)
        self.saved["pool"] = 1
        torch.save(self.saved, self.checkpoint)
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["fs_hz"] = 250
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            load_predictor(self.checkpoint, trusted_checkpoint=True)

    def test_cli_writes_only_predictions_and_refuses_overwrite(self):
        input_path, output_path = self.root / "waves.npy", self.root / "predictions.npy"
        np.save(input_path, self.waves)
        arguments = ["--checkpoint", str(self.checkpoint), "--input", str(input_path),
                     "--output", str(output_path), "--trust-checkpoint"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(arguments), 0)
        actual = np.load(output_path, allow_pickle=False)
        self.assertEqual(actual.shape, (3, 2))
        before = output_path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(arguments)
        self.assertEqual(output_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
