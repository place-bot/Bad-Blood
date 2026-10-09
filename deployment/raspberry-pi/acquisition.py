"""Bounded raw acquisition and deterministic replay; no hardware driver is implied.

The initial raw contract is synchronized 125 Hz ECG/PPG. Other native rates
must be resampled and timing-validated in a future, versioned device adapter.
"""

import csv
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sources import Acquisition, FS_HZ, SAMPLES, SignalWindow, SourceUnavailable

RAW_VERSION = "raw125_window_zero_phase_v1"
STRIDE = 625
MAX_PACKET_SAMPLES = 250
PACKET_TIMEOUT_SECONDS = 3
QUALITY_FLAGS = {"lead_off", "poor_contact", "motion", "fifo_overflow", "clock_unsynchronized"}


@dataclass(frozen=True)
class RawPacket:
    session_id: str
    sequence: int
    first_sample: int
    start_monotonic: float
    waves: np.ndarray
    fs_hz: int = FS_HZ
    channels: tuple = ("ECG", "PPG")
    flags: tuple = ()

    def validate(self):
        if (not isinstance(self.session_id, str) or not self.session_id.strip()
                or type(self.sequence) is not int or self.sequence < 0
                or type(self.first_sample) is not int or self.first_sample < 0
                or not np.isfinite(self.start_monotonic)
                or self.fs_hz != FS_HZ or self.channels != ("ECG", "PPG")
                or not isinstance(self.waves, np.ndarray) or self.waves.ndim != 2
                or self.waves.shape[0] != 2
                or not 1 <= self.waves.shape[1] <= MAX_PACKET_SAMPLES
                or not isinstance(self.flags, tuple)
                or any(flag not in QUALITY_FLAGS for flag in self.flags)
                or not np.issubdtype(self.waves.dtype, np.number)
                or not np.isrealobj(self.waves) or not np.isfinite(self.waves).all()):
            raise SourceUnavailable("invalid_input", "Invalid raw packet or unsupported sample contract.")


@dataclass(frozen=True)
class RawWindow:
    waves: np.ndarray
    acquisition: Acquisition


class WindowBuffer:
    """At most 1,250 raw samples/channel; windows follow sample counts.

    Any discontinuity clears accumulated data. A later packet can begin a
    new ten-second run, but no window is ever assembled across a fault.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._chunks = deque()
        self._count = 0
        self._previous = None
        self._until_window = SAMPLES

    def push(self, packet):
        try:
            packet.validate()
            if packet.flags:
                reasons = {"lead_off": "ECG electrode disconnected", "poor_contact": "poor sensor contact",
                           "motion": "excessive movement", "fifo_overflow": "sensor FIFO overflow",
                           "clock_unsynchronized": "sensor clock not synchronized"}
                raise SourceUnavailable("poor_signal", "; ".join(reasons[flag] for flag in packet.flags) + ".")
            previous = self._previous
            if previous is not None:
                size = previous.waves.shape[1]
                expected_time = previous.start_monotonic + size / FS_HZ
                if (packet.session_id != previous.session_id
                        or packet.sequence != previous.sequence + 1
                        or packet.first_sample != previous.first_sample + size
                        or abs(packet.start_monotonic - expected_time) > .002):
                    raise SourceUnavailable("discontinuity", "Sample gap, duplicate, reset or clock discontinuity; collecting a new window.")
        except (SourceUnavailable, ValueError, TypeError):
            self.reset()
            raise
        self._previous = packet
        result = []
        offset = 0
        while offset < packet.waves.shape[1]:
            take = min(self._until_window, packet.waves.shape[1] - offset)
            self._chunks.append(packet.waves[:, offset:offset + take].astype(np.float64, copy=True))
            self._count += take
            self._until_window -= take
            offset += take
            while self._count > SAMPLES:
                drop = min(self._count - SAMPLES, self._chunks[0].shape[1])
                first = self._chunks.popleft()
                if drop < first.shape[1]:
                    self._chunks.appendleft(first[:, drop:])
                self._count -= drop
            if self._until_window == 0:
                end_sample = packet.first_sample + offset
                end_time = packet.start_monotonic + offset / FS_HZ
                result.append(RawWindow(np.concatenate(self._chunks, axis=1),
                                       Acquisition(packet.session_id, end_sample - SAMPLES,
                                                   end_sample, end_time)))
                self._until_window = STRIDE
        return result


class QualityPolicy:
    """Explicit sensor-specific raw amplitude limits, not clinical SQI validation."""

    def __init__(self, config):
        self.version = config["version"]
        self.minimum = np.asarray(config["minimum_peak_to_peak"], dtype=float)
        self.low = np.asarray(config["adc_min"], dtype=float)
        self.high = np.asarray(config["adc_max"], dtype=float)
        if (not isinstance(self.version, str) or not self.version
                or any(a.shape != (2,) or not np.isfinite(a).all()
                       for a in (self.minimum, self.low, self.high))
                or np.any(self.minimum <= 0) or np.any(self.high <= self.low)
                or np.any(self.minimum >= self.high - self.low)):
            raise ValueError("Quality policy needs version and two valid sensor-specific channel limits")

    def check(self, raw):
        if raw.shape != (2, SAMPLES) or not np.isfinite(raw).all():
            raise SourceUnavailable("invalid_input", "Quality screening needs a finite complete raw window.")
        if np.any(raw.min(axis=1) <= self.low) or np.any(raw.max(axis=1) >= self.high):
            raise SourceUnavailable("poor_signal", "Raw signal reaches the configured ADC limits.")
        if np.any(np.ptp(raw, axis=1) < self.minimum):
            raise SourceUnavailable("poor_signal", "Raw signal amplitude is below the configured contact threshold.")
        return {"status": "screened", "policy": self.version,
                "preprocessing": RAW_VERSION,
                "scope": "Raw amplitude and packet flags only; physiological quality unvalidated."}


class RawCSVReader:
    """Paced, finite replay of a continuous ND-format CSV and session sidecar.

    Rebased monotonic timestamps simulate acquisition age, not original capture
    time. No looping or speedup is allowed; this source is always labeled replay.
    CSV rows are streamed in bounded chunks rather than loading the recording.
    """

    def __init__(self, path, config, clock=time.monotonic):
        required = ("session_id", "device", "firmware", "units", "placement")
        if (config.get("schema") != "bad_blood_raw125_v1" or config.get("fs_hz") != FS_HZ
                or any(not config.get(k) for k in required)
                or config.get("channels") != ["ECG", "PPG"]
                or not isinstance(config["session_id"], str)
                or not isinstance(config["units"], list) or len(config["units"]) != 2
                or any(not isinstance(unit, str) or not unit for unit in config["units"])):
            raise ValueError("Invalid raw125 session sidecar")
        self.config = config
        self.policy = QualityPolicy(config["quality"])
        self._file = Path(path).open(newline="", encoding="utf-8")
        self._rows = csv.DictReader(self._file)
        if self._rows.fieldnames != ["time_s", "ecg_raw", "ppg_raw"]:
            self._file.close()
            raise ValueError("Raw CSV header must be time_s,ecg_raw,ppg_raw")
        self._clock = clock
        self._anchor = clock()
        self._sample = 0
        self._sequence = 0
        self._pending = None
        self._closed = threading.Event()

    def read(self, timeout=.5):
        if self._closed.is_set():
            raise SourceUnavailable("disconnected", "Source closed.")
        if self._pending is None:
            values = []
            for _ in range(125):
                row = next(self._rows, None)
                if row is None:
                    break
                try:
                    timestamp = float(row["time_s"])
                    pair = [float(row["ecg_raw"]), float(row["ppg_raw"])]
                    if (not np.isfinite(timestamp)
                            or abs(timestamp - (self._sample + len(values)) / FS_HZ) > 1e-6
                            or not np.isfinite(pair).all()):
                        raise ValueError()
                except (ValueError, TypeError, KeyError):
                    raise SourceUnavailable("invalid_input", "Raw CSV has invalid values or a nonuniform 125 Hz timeline.") from None
                values.append(pair)
            if not values:
                raise SourceUnavailable("ended", "Raw recording ended; no live device is connected.")
            self._pending = RawPacket(self.config["session_id"], self._sequence, self._sample,
                                      self._anchor + self._sample / FS_HZ,
                                      np.asarray(values, dtype=np.float64).T)
        packet = self._pending
        due = packet.start_monotonic + packet.waves.shape[1] / FS_HZ
        delay = max(0, due - self._clock())
        if self._closed.wait(min(delay, timeout)):
            raise SourceUnavailable("disconnected", "Source closed.")
        if self._clock() + 1e-6 < due:
            raise TimeoutError()
        self._sample += packet.waves.shape[1]
        self._sequence += 1
        self._pending = None
        return packet

    def close(self):
        self._closed.set()

    def release(self):
        self._file.close()


class BufferedSource:
    """Background receiver with a one-window mailbox and cancellable consumption.

    Readers must implement read(timeout), raising TimeoutError on an idle poll,
    and close() that unblocks reads. A broken reader cannot block inference/API;
    it is not restarted in new threads, avoiding leaks and duplicate hardware access.
    """

    def __init__(self, reader, preprocess, policy, mode="recorded_raw_replay"):
        if mode not in ("recorded_raw_replay", "device_live"):
            raise ValueError("Buffered raw source needs an explicit raw input mode")
        self.mode = mode
        self.reader = reader
        self.preprocess = preprocess
        self.policy = policy
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._pending = None
        self._fault = None
        self._terminal = False
        self._ended = None
        self._last_packet = time.monotonic()
        self._generation = 0
        self.dropped_windows = 0
        self.received_packets = 0
        self._thread = None

    def start(self):
        with self._condition:
            if self._thread is None:
                self._thread = threading.Thread(target=self._receive, name="raw-acquisition", daemon=True)
                self._thread.start()

    def _receive(self):
        buffer = WindowBuffer()
        watchdog_latched = False

        def check_watchdog():
            nonlocal watchdog_latched
            if not watchdog_latched and time.monotonic() - self._last_packet > PACKET_TIMEOUT_SECONDS:
                buffer.reset()
                watchdog_latched = True
                self._set_fault(SourceUnavailable("disconnected", "Packet watchdog expired; collecting a new window."))

        try:
            while not self._stop.is_set():
                try:
                    packet = self.reader.read(timeout=.5)
                except TimeoutError:
                    check_watchdog()
                    continue
                try:
                    # Also catch a driver that returns after ignoring its read timeout.
                    check_watchdog()
                    # Catch raw quality faults immediately, before a complete window.
                    packet.validate()
                    # Packet-level saturation is unambiguous; amplitude needs a full window.
                    if (np.any(packet.waves.min(axis=1) <= self.policy.low)
                            or np.any(packet.waves.max(axis=1) >= self.policy.high)):
                        raise SourceUnavailable("poor_signal", "Raw signal reaches the configured ADC limits.")
                    windows = buffer.push(packet)
                    with self._condition:
                        self._last_packet = time.monotonic()
                        self.received_packets += 1
                        watchdog_latched = False
                        for window in windows:
                            if self._pending is not None:
                                self.dropped_windows += 1
                            self._pending = window
                        self._condition.notify_all()
                except (SourceUnavailable, ValueError, TypeError) as error:
                    buffer.reset()
                    self._set_fault(error if isinstance(error, SourceUnavailable) else
                                    SourceUnavailable("invalid_input", "Invalid acquisition packet."))
        except SourceUnavailable as error:
            if error.status == "ended":
                with self._condition:
                    self._ended = SourceUnavailable(error.status, str(error))
                    self._condition.notify_all()
            else:
                self._set_fault(error, terminal=True)
        except Exception:
            self._set_fault(SourceUnavailable("unavailable", "Acquisition failed; restart the source after inspection."), terminal=True)
        finally:
            if hasattr(self.reader, "release"):
                self.reader.release()

    def _set_fault(self, error, terminal=False):
        with self._condition:
            self._pending = None
            # Do not retain reader frames or grow a traceback by re-raising one instance.
            self._fault = SourceUnavailable(error.status, str(error))
            self._generation += 1
            self._terminal = terminal
            self._condition.notify_all()

    def health(self):
        with self._condition:
            if self._fault is not None:
                raise SourceUnavailable(self._fault.status, str(self._fault))
            if self._stop.is_set():
                raise SourceUnavailable("disconnected", "Source closed.")
            if time.monotonic() - self._last_packet > PACKET_TIMEOUT_SECONDS:
                raise SourceUnavailable("disconnected", "No valid acquisition packets for three seconds.")
            return self._generation

    def next_window(self, timeout=.5):
        self.start()
        with self._condition:
            self._condition.wait_for(lambda: self._pending is not None or self._fault is not None
                                     or self._ended is not None or self._stop.is_set(), timeout=timeout)
            if self._fault is not None:
                error = SourceUnavailable(self._fault.status, str(self._fault))
                if not self._terminal:
                    self._fault = None
                raise error
            if self._stop.is_set():
                raise SourceUnavailable("disconnected", "Source closed.")
            if self._pending is None:
                if self._ended is not None:
                    raise SourceUnavailable(self._ended.status, str(self._ended))
                raise TimeoutError()
            window = self._pending
            self._pending = None
            generation = self._generation
        started = time.perf_counter_ns()
        quality = self.policy.check(window.waves)
        waves = self.preprocess(window.waves)
        preprocessing_ms = (time.perf_counter_ns() - started) / 1e6
        self.health()
        with self._condition:
            if generation != self._generation:
                raise SourceUnavailable("discontinuity", "Acquisition changed during preprocessing.")
        acquisition = window.acquisition
        return SignalWindow(f"{acquisition.session_id}:{acquisition.first_sample}", waves,
                            acquisition=acquisition, quality=quality,
                            source_details={"dropped_windows": self.dropped_windows,
                                            "received_packets": self.received_packets,
                                            "preprocessing_ms": round(preprocessing_ms, 3),
                                            "generation": generation})

    def close(self):
        self._stop.set()
        self.reader.close()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1)
        elif hasattr(self.reader, "release"):
            self.reader.release()


def raw_replay(path, session_path, preprocess):
    config = json.loads(Path(session_path).read_text(encoding="utf-8"))
    reader = RawCSVReader(path, config)
    return BufferedSource(reader, preprocess, reader.policy)
