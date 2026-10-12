"""Measure a bundle on the executing machine; never label host results as Pi data."""

import argparse
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np

from .release import BundlePredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    predictor = BundlePredictor(args.bundle, threads=args.threads)
    with np.load(args.bundle / "golden.npz", allow_pickle=False) as f:
        waves = f["waves"]
    for i in range(10):
        predictor.predict(waves[i % len(waves)][None])
    times = []
    for i in range(args.iterations):
        start = time.perf_counter_ns()
        predictor.predict(waves[i % len(waves)][None])
        times.append((time.perf_counter_ns() - start) / 1e6)
    device_tree = Path("/proc/device-tree/model")
    hardware = device_tree.read_bytes().replace(b"\0", b"").decode() if device_tree.exists() else platform.machine()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    print(json.dumps({"bundle_id": predictor.bundle_id, "hardware": hardware,
                      "platform": platform.platform(), "threads": args.threads,
                      "iterations": args.iterations, "p50_ms": float(np.quantile(times, .5)),
                      "p95_ms": float(np.quantile(times, .95)), "peak_process_rss_bytes": rss_bytes,
                      "scope": "Prepared-window normalization + prediction only; excludes acquisition, raw filtering, UI and power"}, indent=2))


if __name__ == "__main__":
    main()
