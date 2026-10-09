"""Fault injection and parity checks for the wearable software foundation."""

import csv
import json
import queue
import shutil
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "model training" / "PulseDB-based Model-V1"
sys.path.insert(0, str(V1))
from common import from_raw_125
from acquisition import (RAW_VERSION, BufferedSource, QualityPolicy, RawCSVReader,
                         RawPacket, WindowBuffer)
from bundle import MANIFEST, REQUIRED_FILES, verify_bundle
from service import InferenceState
from sources import Acquisition, PulseDBReplay, SignalWindow, SourceUnavailable


def waves(start=0, count=125):
    t = np.arange(start, start + count) / 125
    return np.stack([np.sin(2 * np.pi * 1.2 * t), .7 * np.sin(2 * np.pi * 1.2 * t - .5)])


def packet(index, anchor=0, session="test", count=125):
    return RawPacket(session, index, index * count, anchor + index * count / 125,
                     waves(index * count, count))


def policy_config():
    return {"version": "synthetic-test-only", "minimum_peak_to_peak": [.1, .1],
            "adc_min": [-2, -2], "adc_max": [2, 2]}


def session_config():
    return {"schema": "bad_blood_raw125_v1", "session_id": "synthetic",
            "device": "test-fixture", "firmware": "test", "units": ["test", "test"],
            "placement": "simulator", "fs_hz": 125, "channels": ["ECG", "PPG"],
            "quality": policy_config()}


class QueueReader:
    def __init__(self):
        self.packets = queue.Queue()
        self.closed = threading.Event()

    def read(self, timeout):
        if self.closed.is_set():
            raise SourceUnavailable("disconnected", "Closed")
        try:
            result = self.packets.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError() from None
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed.set()
        self.packets.put(SourceUnavailable("disconnected", "Closed"))


class AcquisitionTests(unittest.TestCase):
    def test_exact_windows_stride_and_bounded_storage(self):
        buffer = WindowBuffer()
        outputs = []
        for i in range(100):
            outputs.extend(buffer.push(packet(i)))
            self.assertLessEqual(buffer._count, 1250)
        self.assertEqual(len(outputs), 19)
        for i, window in enumerate(outputs):
            self.assertEqual(window.acquisition.first_sample, i * 625)
            self.assertEqual(window.acquisition.end_sample, i * 625 + 1250)
            np.testing.assert_array_equal(window.waves, waves(i * 625, 1250))

    def test_windows_can_end_inside_packets(self):
        buffer = WindowBuffer()
        outputs = []
        for i in range(30):
            outputs.extend(buffer.push(packet(i, count=100)))
        self.assertEqual([w.acquisition.end_sample for w in outputs], [1250, 1875, 2500])
        np.testing.assert_array_equal(outputs[1].waves, waves(625, 1250))

    def test_discontinuities_never_join_old_and_new_samples(self):
        mutations = [lambda p: replace(p, sequence=1),
                     lambda p: replace(p, first_sample=p.first_sample + 1),
                     lambda p: replace(p, start_monotonic=p.start_monotonic + .01),
                     lambda p: replace(p, session_id="reset"),
                     lambda p: replace(p, flags=("motion",)),
                     lambda p: replace(p, fs_hz=250),
                     lambda p: replace(p, channels=("PPG", "ECG"))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                buffer = WindowBuffer()
                for i in range(9):
                    self.assertEqual(buffer.push(packet(i)), [])
                with self.assertRaises(SourceUnavailable):
                    buffer.push(mutate(packet(9)))
                for i in range(10, 19):
                    self.assertEqual(buffer.push(packet(i)), [])
                result = buffer.push(packet(19))
                self.assertEqual(result[0].acquisition.first_sample, 1250)

    def test_quality_is_checked_before_normalizing_noise_or_clipping(self):
        policy = QualityPolicy(policy_config())
        for raw in (waves(count=1250) * 1e-5, np.zeros((2, 1250)),
                    np.clip(waves(count=1250) * 5, -2, 2)):
            with self.assertRaises(SourceUnavailable):
                policy.check(raw)
        self.assertEqual(policy.check(waves(count=1250))["status"], "screened")
        with self.assertRaises(ValueError):
            QualityPolicy({**policy_config(), "minimum_peak_to_peak": [0, 0]})

    def test_latest_window_mailbox_and_preprocessing_parity(self):
        reader = QueueReader()
        source = BufferedSource(reader, from_raw_125, QualityPolicy(policy_config()))
        anchor = time.monotonic() - 30
        source.start()
        try:
            for i in range(30):
                reader.packets.put(packet(i, anchor))
            deadline = time.monotonic() + 2
            while source.dropped_windows < 4 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(source.dropped_windows, 4)
            window = source.next_window()
            self.assertEqual(window.acquisition.first_sample, 2500)
            np.testing.assert_array_equal(window.waves, from_raw_125(waves(2500, 1250)))
            self.assertEqual(window.quality["preprocessing"], RAW_VERSION)
        finally:
            source.close()
        self.assertFalse(source._thread.is_alive())

    def test_fault_invalidates_pending_window(self):
        reader = QueueReader()
        source = BufferedSource(reader, from_raw_125, QualityPolicy(policy_config()))
        source.start()
        try:
            for i in range(10):
                reader.packets.put(packet(i))
            reader.packets.put(replace(packet(10), flags=("lead_off",)))
            deadline = time.monotonic() + 2
            while source._generation == 0 and time.monotonic() < deadline:
                time.sleep(.005)
            with self.assertRaises(SourceUnavailable):
                source.next_window()
            self.assertIsNone(source._pending)
        finally:
            source.close()

    def test_fault_during_preprocessing_does_not_escape(self):
        reader = QueueReader()
        source = None

        def preprocess(raw):
            source._set_fault(SourceUnavailable("poor_signal", "lead off"))
            return from_raw_125(raw)

        source = BufferedSource(reader, preprocess, QualityPolicy(policy_config()))
        try:
            for i in range(10):
                reader.packets.put(packet(i))
            with self.assertRaises(SourceUnavailable):
                source.next_window(timeout=1)
        finally:
            source.close()

    def test_idle_reader_cannot_block_consumption_or_shutdown(self):
        source = BufferedSource(QueueReader(), from_raw_125, QualityPolicy(policy_config()))
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            source.next_window(timeout=.02)
        self.assertLess(time.monotonic() - start, .3)
        source._last_packet = time.monotonic() - 4
        with self.assertRaises(SourceUnavailable):
            source.health()
        source.close()
        self.assertFalse(source._thread.is_alive())

    def test_eof_preserves_pending_window_for_consumer(self):
        reader = QueueReader()
        source = BufferedSource(reader, from_raw_125, QualityPolicy(policy_config()))
        source.start()
        try:
            for i in range(10):
                reader.packets.put(packet(i))
            reader.packets.put(SourceUnavailable("ended", "End"))
            source._thread.join(timeout=1)
            self.assertEqual(source.next_window().waves.shape, (2, 1250))
            with self.assertRaises(SourceUnavailable) as error:
                source.next_window()
            self.assertEqual(error.exception.status, "ended")
        finally:
            source.close()

    def test_watchdog_discards_partial_window_before_reconnection(self):
        reader = QueueReader()
        source = BufferedSource(reader, from_raw_125, QualityPolicy(policy_config()))
        source.start()
        try:
            for i in range(9):
                reader.packets.put(packet(i))
            deadline = time.monotonic() + 2
            while source.received_packets < 9 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(source.received_packets, 9)
            with source._condition:
                source._last_packet = time.monotonic() - 4
            reader.packets.put(TimeoutError())
            with self.assertRaises(SourceUnavailable) as error:
                source.next_window(timeout=1)
            self.assertEqual(error.exception.status, "disconnected")
            for i in range(9, 18):
                reader.packets.put(packet(i))
            with self.assertRaises(TimeoutError):
                source.next_window(timeout=.05)
            reader.packets.put(packet(18))
            self.assertEqual(source.next_window(timeout=1).acquisition.first_sample, 1125)
        finally:
            source.close()


class CSVTests(unittest.TestCase):
    def write_csv(self, path, count=250, bad_time=False):
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "ecg_raw", "ppg_raw"])
            for i, row in enumerate(waves(count=count).T):
                writer.writerow([i / 125 + (.01 if bad_time and i == 3 else 0), *row])

    def test_reader_paces_packets_and_preserves_raw_values(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "raw.csv"
            self.write_csv(path)
            clock = [100.0]
            reader = RawCSVReader(path, session_config(), clock=lambda: clock[0])
            try:
                with self.assertRaises(TimeoutError):
                    reader.read(timeout=0)
                clock[0] = 101
                first = reader.read(timeout=0)
                np.testing.assert_allclose(first.waves, waves())
                self.assertEqual(first.start_monotonic, 100)
                clock[0] = 102
                self.assertEqual(reader.read(timeout=0).first_sample, 125)
                with self.assertRaises(SourceUnavailable) as error:
                    reader.read(timeout=0)
                self.assertEqual(error.exception.status, "ended")
            finally:
                reader.close()
                reader.release()

    def test_invalid_timeline_and_wrong_rate_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "raw.csv"
            self.write_csv(path, bad_time=True)
            reader = RawCSVReader(path, session_config())
            try:
                with self.assertRaises(SourceUnavailable):
                    reader.read(timeout=0)
            finally:
                reader.close()
                reader.release()
            with self.assertRaises(ValueError):
                RawCSVReader(path, {**session_config(), "fs_hz": 250})


class StateTests(unittest.TestCase):
    def setUp(self):
        self.source = type("Source", (), {"mode": "device_live"})()
        self.state = InferenceState(self.source, lambda *_a, **_k: np.array([[120., 75.]]))
        self.window = SignalWindow("test", from_raw_125(waves(count=1250)),
                                   acquisition=Acquisition("session", 0, 1250, time.monotonic()),
                                   quality={"status": "screened", "preprocessing": RAW_VERSION})

    def test_repeat_and_retimestamped_old_samples_rejected(self):
        self.state.process_window(self.window)
        for window in (self.window, replace(self.window, acquisition=replace(
                self.window.acquisition, end_monotonic=time.monotonic()))):
            with self.assertRaises(SourceUnavailable):
                self.state.process_window(window)

    def test_stale_future_and_unprovenanced_input_never_runs_model(self):
        calls = []
        self.state.predict = lambda *_a, **_k: calls.append(True)
        for acquisition in (None, replace(self.window.acquisition, end_monotonic=time.monotonic() - 13),
                            replace(self.window.acquisition, end_monotonic=time.monotonic() + 1)):
            with self.assertRaises(SourceUnavailable):
                self.state.process_window(replace(self.window, acquisition=acquisition))
        self.assertEqual(calls, [])

    def test_publication_does_not_refresh_acquisition(self):
        self.state.process_window(self.window)
        self.state._acquisition_end -= 13
        self.state.publish(message="A new publication")
        self.assertEqual(self.state.snapshot()["status"], "stale")
        self.assertIsNone(self.state.snapshot()["prediction"])

    def test_invalid_predictions_are_suppressed(self):
        for result in ([[70, 90]], [[120, 0]], [[float("nan"), 70]], [120, 75], [[120, 75], [121, 76]]):
            self.state = InferenceState(self.source, lambda *_a, **_k: np.array(result))
            with self.assertRaises(SourceUnavailable):
                self.state.process_window(self.window)
            self.assertIsNone(self.state.snapshot()["prediction"])

    def test_input_that_ages_during_inference_is_suppressed(self):
        self.window = replace(self.window, acquisition=replace(self.window.acquisition, end_monotonic=100))
        with patch("service.time.monotonic", side_effect=[101, 113]):
            with self.assertRaises(SourceUnavailable):
                self.state.process_window(self.window)

    def test_service_stop_clears_values_and_closes_source(self):
        reader = QueueReader()
        source = BufferedSource(reader, from_raw_125, QualityPolicy(policy_config()), mode="device_live")
        state = InferenceState(source, lambda *_a, **_k: np.array([[120, 75]]))
        stop = threading.Event()
        worker = threading.Thread(target=state.run, args=(stop,))
        worker.start()
        stop.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(reader.closed.is_set())
        self.assertIsNone(state.snapshot()["prediction"])

    def test_model_bundle_detects_code_and_contract_tampering(self):
        self.assertEqual(verify_bundle(V1), "v1-cnn-full-raw125-1")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in REQUIRED_FILES:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(V1 / name, root / name)
            (root / "common.py").write_text("changed")
            with self.assertRaises(ValueError):
                verify_bundle(root)
            manifest = json.loads(MANIFEST.read_text())
            manifest["contract"]["fs_hz"] = 250
            altered = root / "bundle.json"
            altered.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify_bundle(V1, altered)


if __name__ == "__main__":
    unittest.main()
