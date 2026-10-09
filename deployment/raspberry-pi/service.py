"""Source-independent inference worker and local dashboard server."""

import json
import logging
import math
import socket
import threading
import time
import signal
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import numpy as np

from sources import FS_HZ, SAMPLES, SOURCE_MODES, SourceUnavailable, WaveformSource, validate_window
from acquisition import RAW_VERSION

ASSETS = Path(__file__).with_name("web")
STALE_AFTER_SECONDS = 12


class InferenceState:
    def __init__(self, source: WaveformSource, predict: Callable, interval_seconds=5.0):
        if source.mode not in SOURCE_MODES:
            raise ValueError(f"Unsupported source mode: {source.mode}")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Update interval must be positive and finite")
        self.source = source
        self.predict = predict
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._updated_monotonic = None
        self._acquisition_end = None
        self._last_acquisition = None
        self._generation = None
        self._state = {
            "mode": source.mode, "model": "V1 / cnn_full", "sequence": 0,
            "boot_id": uuid.uuid4().hex,
            "status": "initializing", "hostname": socket.gethostname(),
            "updated_at": None, "segment_id": None, "position": None,
            "total": None, "cycle": None, "loop_replay": False,
            "prediction": None, "inference_ms": None, "waveforms": None,
            "acquisition": None, "quality": None, "pipeline_ms": None,
            "preprocessing_ms": None, "dropped_windows": 0,
            "message": "Waiting for a complete input window.",
        }

    def publish(self, **changes):
        with self._lock:
            self._state.update(changes)
            self._updated_monotonic = time.monotonic()

    def snapshot(self):
        with self._lock:
            result = dict(self._state)
            origin = (self._updated_monotonic if self.source.mode == "recorded_pulsedb_replay"
                      else self._acquisition_end)
            age = None if origin is None else time.monotonic() - origin
            result["age_seconds"] = None if age is None else round(max(0, age), 2)
            generation = self._generation
        if result["prediction"] is not None:
            try:
                if age is None or age > STALE_AFTER_SECONDS:
                    raise SourceUnavailable("stale", "Input window is stale; waiting for fresh data.")
                if hasattr(self.source, "health"):
                    current = self.source.health()
                    if generation != current:
                        raise SourceUnavailable("discontinuity", "Acquisition changed; waiting for a new valid window.")
            except SourceUnavailable as error:
                result.update(status=error.status, prediction=None, waveforms=None,
                              inference_ms=None, pipeline_ms=None, preprocessing_ms=None,
                              message=str(error))
        return result

    def unavailable(self, status, message, segment_id=None):
        self.publish(sequence=self.snapshot()["sequence"] + 1, status=status,
                     prediction=None, waveforms=None, inference_ms=None, pipeline_ms=None,
                     preprocessing_ms=None,
                     quality=None, acquisition=None, segment_id=segment_id,
                     position=None, total=None, cycle=None, loop_replay=False,
                     updated_at=datetime.now(timezone.utc).isoformat(), message=message)

    def process_window(self, window):
        """Validate, infer, then recheck acquisition health before publication."""
        waves = validate_window(window)
        acquisition = window.acquisition
        if self.source.mode != "recorded_pulsedb_replay":
            if acquisition is None:
                raise SourceUnavailable("invalid_input", "Input lacks acquisition timestamps.")
            if (window.quality.get("status") != "screened"
                    or window.quality.get("preprocessing") != RAW_VERSION):
                raise SourceUnavailable("poor_signal", "Input lacks raw quality screening and versioned preprocessing.")
            age = time.monotonic() - acquisition.end_monotonic
            if age < -.002 or age > STALE_AFTER_SECONDS:
                raise SourceUnavailable("stale", "Acquisition time is stale or ahead of the host clock.")
            previous = self._last_acquisition
            if previous is not None and (
                    acquisition.end_monotonic <= previous.end_monotonic
                    or (acquisition.session_id == previous.session_id
                        and acquisition.end_sample <= previous.end_sample)):
                raise SourceUnavailable("discontinuity", "Repeated or out-of-order acquisition window.")
            # Consume provenance even if inference fails; repeated buffers stay rejected.
            self._last_acquisition = acquisition
        started = time.perf_counter_ns()
        batch = np.asarray(self.predict(waves[None], input_format="filtered"))
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if batch.shape != (1, 2) or not np.isfinite(batch).all():
            raise SourceUnavailable("invalid_prediction", "Model returned an invalid BP pair.")
        prediction = batch[0]
        if prediction[0] <= prediction[1] or prediction[1] <= 0:
            raise SourceUnavailable("invalid_prediction", "Model returned an inconsistent BP pair; estimate suppressed.")
        generation = window.source_details.get("generation")
        if hasattr(self.source, "health") and self.source.health() != generation:
            raise SourceUnavailable("discontinuity", "Acquisition changed during inference.")
        pipeline_ms = None
        acquisition_info = None
        if acquisition is not None:
            age = time.monotonic() - acquisition.end_monotonic
            if age < -.002 or age > STALE_AFTER_SECONDS:
                raise SourceUnavailable("stale", "Input became stale during inference.")
            pipeline_ms = round(max(0, age) * 1000, 2)
            acquisition_info = {"session_id": acquisition.session_id,
                                "first_sample": acquisition.first_sample,
                                "end_sample": acquisition.end_sample,
                                "duration_seconds": SAMPLES / FS_HZ}
        messages = {
            "recorded_pulsedb_replay": "Recorded PulseDB segment. Not a live sensor reading.",
            "recorded_raw_replay": "Recorded raw ECG/PPG replay. Quality screening is experimental; no device is connected.",
            "device_live": "Connected-device window. Experimental quality screening; prototype estimate only.",
        }
        with self._lock:
            self._acquisition_end = acquisition.end_monotonic if acquisition else None
            self._generation = generation
            self._updated_monotonic = time.monotonic()
            self._state.update(
                sequence=self._state["sequence"] + 1, status="ok",
                updated_at=datetime.now(timezone.utc).isoformat(), segment_id=window.segment_id,
                position=window.source_details.get("position"), total=window.source_details.get("total"),
                cycle=window.source_details.get("cycle"),
                loop_replay=window.source_details.get("cycle", 0) > 1,
                prediction={"SBP_mmHg": float(prediction[0]), "DBP_mmHg": float(prediction[1])},
                inference_ms=round(elapsed_ms, 3), pipeline_ms=pipeline_ms,
                preprocessing_ms=window.source_details.get("preprocessing_ms"),
                dropped_windows=window.source_details.get("dropped_windows", 0),
                waveforms={"ecg": waves[0].astype(float).tolist(), "ppg": waves[1].astype(float).tolist()},
                acquisition=acquisition_info, quality=window.quality or None,
                message=messages[self.source.mode],
            )

    def run(self, stop: threading.Event):
        streaming = self.source.mode != "recorded_pulsedb_replay"
        next_tick = time.monotonic()
        try:
            while not stop.is_set():
                try:
                    window = (self.source.next_window(timeout=.5) if streaming
                              else self.source.next_window())
                    if stop.is_set():
                        break
                    self.process_window(window)
                except TimeoutError:
                    if hasattr(self.source, "health"):
                        try:
                            self.source.health()
                        except SourceUnavailable as error:
                            self.unavailable(error.status, str(error))
                except SourceUnavailable as error:
                    self.unavailable(error.status, str(error))
                    if error.status == "ended":
                        stop.wait()  # Finite replay remains ended until service shutdown.
                    else:
                        stop.wait(.1)  # Bounded retries; failures must not busy-spin.
                except Exception:
                    logging.exception("Input update failed")
                    self.unavailable("unavailable", "Input processing failed; inspect local service logs.")
                    stop.wait(.1)
                if not streaming:
                    next_tick = max(next_tick + self.interval_seconds, time.monotonic())
                    stop.wait(max(0, next_tick - time.monotonic()))
        finally:
            if hasattr(self.source, "close"):
                self.source.close()
            self.unavailable("disconnected", "Acquisition stopped.")


def make_handler(state: InferenceState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/state":
                self._respond(200, "application/json; charset=utf-8", state.snapshot())
            elif self.path == "/api/health":
                snapshot = state.snapshot()
                self._respond(200 if snapshot["status"] == "ok" else 503,
                              "application/json; charset=utf-8",
                              {"status": snapshot["status"], "sequence": snapshot["sequence"]})
            elif self.path == "/api/live":
                self._respond(200, "application/json; charset=utf-8", {"status": "alive"})
            elif self.path in ("/", "/app.js", "/style.css"):
                path = ASSETS / {"/": "index.html", "/app.js": "app.js",
                                 "/style.css": "style.css"}[self.path]
                content_type = {".html": "text/html; charset=utf-8",
                                ".js": "text/javascript; charset=utf-8",
                                ".css": "text/css; charset=utf-8"}[path.suffix]
                self._respond(200, content_type, path.read_bytes())
            else:
                self._respond(404, "text/plain; charset=utf-8", b"Not found")

        def _respond(self, code, content_type, content):
            if not isinstance(content, bytes):
                content = json.dumps(content, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "connect-src 'self'; img-src 'self'; object-src 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, fmt, *args):
            logging.info("HTTP %s", fmt % args)

    return Handler


def serve(state: InferenceState, host="127.0.0.1", port=8000):
    if host != "127.0.0.1":
        raise ValueError("This version only binds Pi loopback; use an SSH tunnel")
    stop = threading.Event()
    worker = threading.Thread(target=state.run, args=(stop,), daemon=True,
                              name="waveform-inference")
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.timeout = .5
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
    worker.start()
    try:
        logging.info("Bad-Blood inference service listening on http://%s:%d", host, port)
        while not stop.is_set():
            server.handle_request()
    finally:
        stop.set()
        server.server_close()
        if hasattr(state.source, "close"):
            state.source.close()
        worker.join(timeout=3)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
