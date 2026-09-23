"""Source-boundary and local API checks; no training or fabricated Pi results."""

import json
import csv
import sys
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from service import InferenceState, make_handler
from sources import PulseDBReplay, SignalWindow, validate_window

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "PulseDB" / "examples" / "real_20_segments.csv"


class DeviceStub:
    """A future device adapter can satisfy precisely the replay contract."""

    mode = "device_live"

    def __init__(self, waves):
        self.waves = waves

    def next_window(self):
        return SignalWindow("device-window-1", self.waves.copy())


class DeploymentTests(unittest.TestCase):
    def test_unrecognized_mode_is_rejected(self):
        source = DeviceStub(PulseDBReplay(CSV).next_window().waves)
        source.mode = "unknown_source"
        with self.assertRaises(ValueError):
            InferenceState(source, lambda *_args, **_kwargs: None)

    def test_all_recorded_rows_are_canonical_and_loop(self):
        source = PulseDBReplay(CSV)
        windows = [source.next_window() for _ in range(21)]
        self.assertEqual(len({w.segment_id for w in windows[:20]}), 20)
        self.assertEqual(windows[0].segment_id, windows[20].segment_id)
        self.assertEqual(windows[20].source_details["cycle"], 2)
        for window in windows:
            self.assertEqual(validate_window(window).shape, (2, 1250))
            self.assertEqual(window.waves.dtype, np.float32)
            self.assertFalse(any("sbp" in key.lower() or "dbp" in key.lower()
                                 for key in window.source_details))

    def test_device_stub_uses_same_predictor_entry(self):
        waves = PulseDBReplay(CSV).next_window().waves
        calls = []

        def fake_predict(batch, input_format):
            calls.append((batch.copy(), input_format))
            return np.array([[120.0, 75.0]], dtype=np.float32)

        for source in (PulseDBReplay(CSV), DeviceStub(waves)):
            state = InferenceState(source, fake_predict, interval_seconds=60)
            stop = threading.Event()
            worker = threading.Thread(target=state.run, args=(stop,), daemon=True)
            worker.start()
            deadline = time.monotonic() + 3
            while state.snapshot()["sequence"] == 0 and time.monotonic() < deadline:
                time.sleep(.01)
            stop.set()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(state.snapshot()["prediction"],
                             {"SBP_mmHg": 120.0, "DBP_mmHg": 75.0})
        self.assertEqual(len(calls), 2)
        for batch, input_format in calls:
            self.assertEqual(batch.shape, (1, 2, 1250))
            self.assertEqual(batch.dtype, np.float32)
            self.assertEqual(input_format, "filtered")

    def test_released_predictor_accepts_canonical_replay_window(self):
        v1 = ROOT / "model training" / "PulseDB-based Model-V1"
        sys.path.insert(0, str(v1))
        from inference import Predictor

        window = PulseDBReplay(CSV).next_window()
        actual = Predictor(v1 / "models").predict(
            validate_window(window)[None], input_format="filtered")[0]
        with (v1 / "reports" / "example_predictions.csv").open(newline="") as stream:
            reference = next(csv.DictReader(stream))
        self.assertEqual(window.segment_id, reference["segment_id"])
        self.assertLessEqual(abs(float(actual[0]) - float(reference["pred_SBP_mmHg"])), .01)
        self.assertLessEqual(abs(float(actual[1]) - float(reference["pred_DBP_mmHg"])), .01)

    def test_rejects_wrong_order_shape_dtype_and_nonfinite(self):
        waves = PulseDBReplay(CSV).next_window().waves
        bad_windows = [
            SignalWindow("x", waves, channels=("PPG", "ECG")),
            SignalWindow("x", waves[:, :-1]),
            SignalWindow("x", waves.astype(np.float64)),
            SignalWindow("x", np.full((2, 1250), np.nan, dtype=np.float32)),
            SignalWindow("x", waves, preprocessing="raw"),
        ]
        for window in bad_windows:
            with self.subTest(window=window):
                with self.assertRaises(ValueError):
                    validate_window(window)

    def test_local_api_and_stale_clears_prediction(self):
        state = InferenceState(PulseDBReplay(CSV), lambda *_args, **_kwargs: None)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            state.publish(status="ok", sequence=1,
                          prediction={"SBP_mmHg": 120.0, "DBP_mmHg": 75.0},
                          waveforms={"ecg": [0.1], "ppg": [0.2]})
            with urlopen(url + "/api/state") as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.load(response)["prediction"]["SBP_mmHg"], 120.0)
            with urlopen(url + "/") as response:
                self.assertIn(b"RECORDED DATA - PulseDB replay", response.read())
            with self.assertRaises(HTTPError) as raised:
                urlopen(url + "/../../models/cnn_full.pt")
            self.assertEqual(raised.exception.code, 404)
            state._updated_monotonic -= 13
            snapshot = state.snapshot()
            self.assertEqual(snapshot["status"], "stale")
            self.assertIsNone(snapshot["prediction"])
            self.assertIsNone(snapshot["waveforms"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(3)


if __name__ == "__main__":
    unittest.main()
