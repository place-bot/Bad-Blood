"""Run the existing V1 Predictor on a Raspberry Pi with a loopback dashboard."""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from service import InferenceState, serve
from sources import PulseDBReplay
from acquisition import raw_replay
from bundle import verify_bundle

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "model training" / "PulseDB-based Model-V1"
DEFAULT_CSV = ROOT / "data" / "PulseDB" / "examples" / "real_20_segments.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, default=DEFAULT_CSV,
                        help="The 20 recorded, already filtered PulseDB windows")
    inputs.add_argument("--raw-input", type=Path,
                        help="Continuous raw125 CSV; paced recorded replay, never labeled live")
    parser.add_argument("--session", type=Path,
                        help="Required raw replay session/quality-policy JSON sidecar")
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), default=4,
                        help="Inference CPU threads; benchmark on the target Pi (default: 4)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Pi loopback port for the SSH-tunneled dashboard")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    if bool(args.raw_input) != bool(args.session):
        parser.error("--raw-input and --session must be supplied together")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    bundle_id = verify_bundle(V1)
    manifest = json.loads((V1 / "model_checksums.json").read_text())
    weight = V1 / "models" / "cnn_full.pt"
    actual = hashlib.sha256(weight.read_bytes()).hexdigest()
    if actual != manifest["models/cnn_full.pt"]:
        raise ValueError("Selected V1 weight hash differs from the release manifest")

    sys.path.insert(0, str(V1))
    from inference import Predictor  # pylint: disable=import-outside-toplevel

    predictor = Predictor(V1 / "models")
    import torch
    torch.set_num_threads(args.threads)
    if (predictor.config["cnn_name"] != "cnn_full"
            or predictor.config["tree_weight"] != 0):
        raise ValueError("This deployment requires the released cnn_full only")
    parameters = sum(value.numel() for value in predictor.cnn.parameters())
    if parameters != 37666:
        raise ValueError(f"Unexpected V1 parameter count: {parameters}")
    logging.info("Selected V1 cnn_full; parameters=%d; weight_sha256=%s",
                 parameters, actual)

    from common import from_raw_125
    source = (raw_replay(args.raw_input, args.session, from_raw_125) if args.raw_input
              else PulseDBReplay(args.input))
    state = InferenceState(source, predictor.predict, interval_seconds=5.0)
    state.publish(bundle_id=bundle_id, inference_threads=args.threads)
    serve(state, port=args.port)


if __name__ == "__main__":
    main()
