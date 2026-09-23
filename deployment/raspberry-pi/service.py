"""Local replay worker and JSON-only dashboard server."""

import json
import logging
import math
import socket
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import numpy as np

from sources import WaveformSource, validate_window

ASSETS = Path(__file__).with_name("web")
STALE_AFTER_SECONDS = 12


class InferenceState:
    def __init__(self, source: WaveformSource, predict: Callable, interval_seconds=5.0):
        if interval_seconds <= 0:
            raise ValueError("Replay interval must be positive")
        self.source = source
        self.predict = predict
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._updated_monotonic = None
        self._state = {
            "mode": source.mode, "model": "V1 / cnn_full", "sequence": 0,
            "status": "initializing", "hostname": socket.gethostname(),
            "updated_at": None, "segment_id": None, "position": None,
            "total": None, "cycle": None, "loop_replay": False,
            "prediction": None, "inference_ms": None, "waveforms": None,
            "message": "Waiting for the first recorded segment.",
        }

    def publish(self, **changes):
        with self._lock:
            self._state.update(changes)
            self._updated_monotonic = time.monotonic()

    def snapshot(self):
        with self._lock:
            result = dict(self._state)
            if self._updated_monotonic is not None:
                age = time.monotonic() - self._updated_monotonic
                result["age_seconds"] = round(age, 2)
                if age > STALE_AFTER_SECONDS:
                    result.update(status="stale", prediction=None, waveforms=None,
                                  inference_ms=None, message="Replay update is stale.")
            else:
                result["age_seconds"] = None
            return result

    def run(self, stop: threading.Event):
        next_tick = time.monotonic()
        while not stop.is_set():
            window = None
            try:
                window = self.source.next_window()
                waves = validate_window(window)
                # CSV parsing and array construction happened before this timer.
                # Predictor.predict includes the original V1 normalization,
                # network call, and target inverse scaling; no replay wait here.
                started = time.perf_counter_ns()
                prediction = np.asarray(
                    self.predict(waves[None], input_format="filtered")[0],
                    dtype=np.float64,
                )
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                if prediction.shape != (2,) or not np.isfinite(prediction).all():
                    raise ValueError("V1 returned an invalid SBP/DBP pair")
                if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
                    raise ValueError("Inference timer returned an invalid value")
                self.publish(
                    sequence=self.snapshot()["sequence"] + 1,
                    status="ok" if prediction[0] > prediction[1] else "review_prediction",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    segment_id=window.segment_id,
                    position=window.source_details.get("position"),
                    total=window.source_details.get("total"),
                    cycle=window.source_details.get("cycle"),
                    loop_replay=window.source_details.get("cycle", 0) > 1,
                    prediction={"SBP_mmHg": float(prediction[0]),
                                "DBP_mmHg": float(prediction[1])},
                    inference_ms=round(elapsed_ms, 3),
                    waveforms={"ecg": waves[0].astype(float).tolist(),
                               "ppg": waves[1].astype(float).tolist()},
                    message=("Recorded PulseDB segment. Not a live sensor reading."
                             if self.source.mode == "recorded_pulsedb_replay"
                             else "Source window processed by the V1 model."),
                )
            except Exception as error:
                logging.exception("Replay update failed")
                self.publish(
                    sequence=self.snapshot()["sequence"] + 1,
                    status="unavailable", prediction=None, waveforms=None,
                    inference_ms=None, updated_at=datetime.now(timezone.utc).isoformat(),
                    segment_id=window.segment_id if window else None,
                    message=f"Replay unavailable: {type(error).__name__}",
                )
            next_tick += self.interval_seconds
            next_tick = max(next_tick, time.monotonic())
            stop.wait(max(0, next_tick - time.monotonic()))


def make_handler(state: InferenceState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/state":
                self._respond(200, "application/json; charset=utf-8", state.snapshot())
            elif self.path == "/api/health":
                snapshot = state.snapshot()
                self._respond(200 if snapshot["status"] in ("ok", "review_prediction") else 503,
                              "application/json; charset=utf-8",
                              {"status": snapshot["status"], "sequence": snapshot["sequence"]})
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
                              name="recorded-replay")
    server = ThreadingHTTPServer((host, port), make_handler(state))
    worker.start()
    try:
        logging.info("Bad-Blood replay listening on http://%s:%d", host, port)
        server.serve_forever(poll_interval=0.2)
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=10)
