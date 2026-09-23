"""Run the existing V1 Predictor on a Raspberry Pi with a loopback dashboard."""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from service import InferenceState, serve
from sources import PulseDBReplay

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "model training" / "PulseDB-based Model-V1"
DEFAULT_CSV = ROOT / "data" / "PulseDB" / "examples" / "real_20_segments.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV,
                        help="The 20 recorded, already filtered PulseDB windows")
    parser.add_argument("--port", type=int, default=8000,
                        help="Pi loopback port for the SSH-tunneled dashboard")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    manifest = json.loads((V1 / "model_checksums.json").read_text())
    weight = V1 / "models" / "cnn_full.pt"
    actual = hashlib.sha256(weight.read_bytes()).hexdigest()
    if actual != manifest["models/cnn_full.pt"]:
        raise ValueError("Selected V1 weight hash differs from the release manifest")

    sys.path.insert(0, str(V1))
    from inference import Predictor  # pylint: disable=import-outside-toplevel

    predictor = Predictor(V1 / "models")
    if (predictor.config["cnn_name"] != "cnn_full"
            or predictor.config["tree_weight"] != 0):
        raise ValueError("This deployment requires the released cnn_full only")
    parameters = sum(value.numel() for value in predictor.cnn.parameters())
    if parameters != 37666:
        raise ValueError(f"Unexpected V1 parameter count: {parameters}")
    logging.info("Selected V1 cnn_full; parameters=%d; weight_sha256=%s",
                 parameters, actual)

    source = PulseDBReplay(args.input)
    state = InferenceState(source, predictor.predict, interval_seconds=5.0)
    serve(state, port=args.port)


if __name__ == "__main__":
    main()
