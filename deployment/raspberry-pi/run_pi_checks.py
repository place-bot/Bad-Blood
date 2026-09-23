"""Run release checks and a reproducible V1 predictor benchmark on a Pi 4."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sources import PulseDBReplay, validate_window

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "model training" / "PulseDB-based Model-V1"
EXAMPLES = ROOT / "data" / "PulseDB" / "examples" / "real_20_segments.csv"
REFERENCE = V1 / "reports" / "example_predictions.csv"


def run(command):
    completed = subprocess.run(command, cwd=V1, capture_output=True, text=True)
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(f"Failed with exit {completed.returncode}: {command}")
    return completed.stdout


def compare_predictions(actual_path):
    with actual_path.open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    with REFERENCE.open(newline="") as stream:
        expected = list(csv.DictReader(stream))
    if len(actual) != len(expected) or len(actual) != 20:
        raise AssertionError("Expected exactly 20 predictions and 20 references")
    differences = []
    for index, (new, old) in enumerate(zip(actual, expected), start=1):
        if new["segment_id"] != old["segment_id"] or new["status"] != old["status"]:
            raise AssertionError(f"Reference ID/status mismatch at row {index}")
        for target in ("SBP", "DBP"):
            gap = abs(float(new[f"pred_{target}_mmHg"])
                      - float(old[f"pred_{target}_mmHg"]))
            differences.append(gap)
    maximum = max(differences)
    if maximum > 0.01:
        raise AssertionError(f"Pi reference deviation {maximum:.6f} mmHg > 0.01")
    return {"rows": 20, "max_absolute_difference_mmHg": maximum,
            "tolerance_mmHg": 0.01}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    model_text = Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode()
    if platform.machine() != "aarch64" or "Raspberry Pi 4" not in model_text:
        parser.error("This evidence script must run on the actual Raspberry Pi 4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prediction_csv = args.output_dir / f"pi_predictions_{stamp}.csv"
    evidence_json = args.output_dir / f"pi_evidence_{stamp}.json"
    if prediction_csv.exists() or evidence_json.exists():
        raise FileExistsError("Evidence filenames already exist; rerun in a new second")

    test_results = {}
    for name in ("test_contract.py", "test_release.py"):
        test_results[name] = run([sys.executable, name])
    run([sys.executable, "predict.py", str(EXAMPLES), "--format", "filtered",
         "--output", str(prediction_csv), "--model-dir", str(V1 / "models")])
    comparison = compare_predictions(prediction_csv)

    weight = V1 / "models" / "cnn_full.pt"
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    manifest = json.loads((V1 / "model_checksums.json").read_text())
    if digest != manifest["models/cnn_full.pt"]:
        raise AssertionError("Released V1 model weight checksum changed")
    selection = json.loads((V1 / "models" / "selection.json").read_text())
    if selection["cnn_name"] != "cnn_full" or selection["tree_weight"] != 0:
        raise AssertionError("Unexpected V1 model selection")

    sys.path.insert(0, str(V1))
    from inference import Predictor
    predictor = Predictor(V1 / "models")
    count = sum(parameter.numel() for parameter in predictor.cnn.parameters())
    if count != 37666:
        raise AssertionError(f"Unexpected V1 parameter count: {count}")
    waves = validate_window(PulseDBReplay(EXAMPLES).next_window()).copy()
    batch = waves[None]
    for _ in range(10):
        predictor.predict(batch, input_format="filtered")
    timings_ms = []
    for _ in range(100):
        start = time.perf_counter_ns()
        output = predictor.predict(batch, input_format="filtered")
        timings_ms.append((time.perf_counter_ns() - start) / 1e6)
        if output.shape != (1, 2) or not np.isfinite(output).all():
            raise AssertionError("Benchmark prediction is invalid")
    report = {
        "note": "Measured on Raspberry Pi 4; Predictor.predict includes V1 normalization, CNN and inverse target scaling; excludes CSV reading, HTTP and replay wait.",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": model_text, "architecture": platform.machine(),
        "os": platform.platform(), "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "torch")},
        "git_commit": run(["git", "rev-parse", "HEAD"]).strip(),
        "model": "V1 cnn_full", "parameters": count,
        "weight_sha256": digest, "source": "public PulseDB 20-segment filtered CSV",
        "tests": {name: "passed" for name in test_results},
        "reference_comparison": comparison,
        "benchmark": {"warmup_calls": 10, "timed_calls": len(timings_ms),
                      "p50_ms": float(np.percentile(timings_ms, 50)),
                      "p95_ms": float(np.percentile(timings_ms, 95)),
                      "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "timings_ms": timings_ms},
        "predictions_csv": str(prediction_csv),
    }
    with evidence_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"evidence_json": str(evidence_json),
                      "predictions_csv": str(prediction_csv),
                      "reference_comparison": comparison,
                      "p50_ms": report["benchmark"]["p50_ms"],
                      "p95_ms": report["benchmark"]["p95_ms"]}, indent=2))


if __name__ == "__main__":
    main()
