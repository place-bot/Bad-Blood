"""Real-process, real-time raw replay smoke test. Synthetic signals are test-only."""

import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np

import test_acquisition as fixtures
from test_acquisition import session_config, waves, from_raw_125, V1


class SystemTests(unittest.TestCase):
    def test_raw_replay_predictions_eof_health_and_sigterm(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw.csv"
            fixtures.CSVTests().write_csv(raw, count=2125)  # 17 seconds, two complete windows
            config = root / "session.json"
            config.write_text(json.dumps(session_config()))
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("app.py")),
                 "--raw-input", str(raw), "--session", str(config), "--port", str(port),
                 "--threads", "1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            url = f"http://127.0.0.1:{port}"
            records = {}
            ended = False
            try:
                deadline = time.monotonic() + 35
                while time.monotonic() < deadline:
                    self.assertIsNone(process.poll(), "App exited before smoke test completed")
                    try:
                        with urlopen(url + "/api/state", timeout=1) as response:
                            state = json.load(response)
                    except (URLError, TimeoutError):
                        time.sleep(.1)
                        continue
                    self.assertEqual(state["mode"], "recorded_raw_replay")
                    if state["status"] == "ok":
                        records[state["acquisition"]["first_sample"]] = state
                        # A displayed window naturally ages during the five-second stride.
                        self.assertLess(state["age_seconds"], 6)
                        self.assertGreaterEqual(state["preprocessing_ms"], 0)
                    if state["status"] == "ended":
                        self.assertIsNone(state["prediction"])
                        with self.assertRaises(HTTPError) as error:
                            urlopen(url + "/api/health")
                        self.assertEqual(error.exception.code, 503)
                        with urlopen(url + "/api/live") as response:
                            self.assertEqual(response.status, 200)
                        ended = True
                        break
                    time.sleep(.1)
                self.assertTrue(ended, "Replay did not report EOF")
                self.assertEqual(set(records), {0, 625})
                from inference import Predictor
                predictor = Predictor(V1 / "models")
                for start, record in records.items():
                    expected = predictor.predict(from_raw_125(waves(start, 1250))[None])[0]
                    np.testing.assert_allclose(
                        [record["prediction"]["SBP_mmHg"], record["prediction"]["DBP_mmHg"]],
                        expected, atol=.01, rtol=0)
            finally:
                process.terminate()
                try:
                    _out, err = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    _out, err = process.communicate()
                    self.fail("Service did not stop after SIGTERM")
            self.assertEqual(process.returncode, 0, err)


if __name__ == "__main__":
    unittest.main()
