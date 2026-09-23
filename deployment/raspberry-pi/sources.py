"""Waveform-source boundary. Only the recorded PulseDB source is implemented."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


FS_HZ = 125
SAMPLES = 1250
CHANNELS = ("ECG", "PPG")
PREPROCESSING = "pulsedb_filtered_minmax_v2"


@dataclass(frozen=True)
class SignalWindow:
    """Canonical V1 input from either replay or a future verified device adapter.

    Only ``waves`` (with an added batch axis) reaches Predictor. Source metadata
    and reference blood-pressure labels never enter the model.
    """

    segment_id: str
    waves: np.ndarray  # float32, (ECG, PPG) x 1250, filtered and min-max scaled
    fs_hz: int = FS_HZ
    channels: tuple[str, str] = CHANNELS
    preprocessing: str = PREPROCESSING
    source_details: dict = field(default_factory=dict)


def validate_window(window: SignalWindow) -> np.ndarray:
    """Reject a mismatched source before the unchanged V1 Predictor is called."""
    if (window.fs_hz != FS_HZ or window.channels != CHANNELS
            or window.preprocessing != PREPROCESSING):
        raise ValueError("Source does not meet the V1 125 Hz ECG/PPG preprocessing contract")
    waves = window.waves
    if (not isinstance(waves, np.ndarray) or waves.dtype != np.float32
            or waves.shape != (2, SAMPLES) or not np.isfinite(waves).all()):
        raise ValueError("Source needs finite float32 ECG/PPG shaped (2, 1250)")
    if waves.min() < -1e-5 or waves.max() > 1 + 1e-5:
        raise ValueError("Source waveform must be min-max scaled into [0, 1]")
    if np.any(np.ptp(waves, axis=1) < 1e-8):
        raise ValueError("Source waveform must not be flat")
    if not window.segment_id:
        raise ValueError("Source window needs an ID")
    return waves


class WaveformSource(Protocol):
    """Replay and future device adapters both emit the same canonical window."""

    mode: str

    def next_window(self) -> SignalWindow: ...


class PulseDBReplay:
    mode = "recorded_pulsedb_replay"

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        with self.csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 20:
            raise ValueError(f"Expected the 20 public PulseDB segments, found {len(rows)}")
        self._segments = []
        for row in rows:
            if float(row["fs_hz"]) != FS_HZ:
                raise ValueError("PulseDB replay needs 125 Hz segments")
            waves = np.asarray(
                [[float(row[f"{channel}_{index:04d}"]) for index in range(SAMPLES)]
                 for channel in ("ecg", "ppg")], dtype=np.float32,
            )
            window = SignalWindow(row.get("segment_id", ""), waves)
            validate_window(window)
            # SBP/DBP reference labels, participant IDs and other columns are not
            # stored or passed to the model by this adapter.
            self._segments.append(window)
        self._cursor = 0

    def next_window(self) -> SignalWindow:
        index = self._cursor % len(self._segments)
        cycle = self._cursor // len(self._segments) + 1
        recorded = self._segments[index]
        self._cursor += 1
        return SignalWindow(recorded.segment_id, recorded.waves,
                            source_details={"cycle": cycle, "position": index + 1,
                                            "total": len(self._segments)})
