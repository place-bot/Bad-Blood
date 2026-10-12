"""The V1 numerical input contract, independently usable on the Pi."""

import numpy as np

CONTRACT = {"fs_hz": 125, "channels": ["ECG", "PPG"], "samples": 1250,
            "dtype": "float32", "preprocessing": "pulsedb_filtered_minmax_v2",
            "raw_adapter": "raw125_window_zero_phase_v1",
            "targets": ["SBP", "DBP"], "units": "mmHg"}


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (2, 1250) or not np.isfinite(x).all():
        raise ValueError("Expected finite ECG/PPG array shaped (2,1250)")
    spread = np.ptp(x, axis=1, keepdims=True)
    if np.any(spread < 1e-8):
        raise ValueError("Flat waveform")
    return (x - x.min(axis=1, keepdims=True)) / spread


def from_raw_125(x):
    from scipy.signal import butter, cheby2, sosfiltfilt
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (2, 1250) or not np.isfinite(x).all():
        raise ValueError("Expected (2,1250) raw samples at 125 Hz")
    ecg = sosfiltfilt(butter(4, [.5, 40], btype="bandpass", fs=125, output="sos"), x[0])
    ppg = sosfiltfilt(cheby2(4, 20, [.5, 8], btype="bandpass", fs=125, output="sos"), x[1])
    return normalize(np.stack([ecg, ppg]))
