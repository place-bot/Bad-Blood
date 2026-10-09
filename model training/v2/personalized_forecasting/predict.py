"""Label-free offline inference from an existing, completed Track B unit.

Inputs are already-observed, source-compatible filtered ECG/PPG at 125 Hz,
N x 2 x 1250, ECG first. Outputs are N x 2 float64 SBP/DBP in mmHg, preserving
the original Track B arithmetic. This command does not fit or select a model.
The historical package name does not mean open-loop future forecasting.

The population checkpoint uses Python pickle: explicitly trust its origin.
Hashes establish consistency, not authenticity or safety of an untrusted file.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from personalized_forecasting.adaptation import (  # noqa: E402
    NEURAL_METHODS, apply_calibration, configure_trainable, predict,
)
from shared.predict import (  # noqa: E402
    CHANNEL_ORDER, _hash_file, _hash_json, _require, load_predictor, prepare_inputs,
)

SCALAR_METHODS = ("population", "bias", "affine", "last_reference", "calibration_mean")
SOURCE_FILES = ("personalized_forecasting/run.py", "personalized_forecasting/adaptation.py",
                "shared/models.py", "shared/data.py", "shared/metrics.py", "PROTOCOL.md")


def _read_json(path: Path) -> dict:
    _require(path.is_file(), f"Required local JSON file is missing: {path}")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    # Reject nonfinite JSON numbers, including unused metadata.
    _hash_json(value)
    return value


def _normalization(saved: dict, center, scale):
    saved_center = np.asarray(saved.get("center"), dtype=np.float64)
    saved_scale = np.asarray(saved.get("scale"), dtype=np.float64)
    _require(saved_center.shape == (2,) and saved_scale.shape == (2,)
             and np.array_equal(saved_center, np.asarray(center, dtype=np.float64))
             and np.array_equal(saved_scale, np.asarray(scale, dtype=np.float64)),
             "Personal artifact normalization differs from the population checkpoint")
    return saved_center, saved_scale


@dataclass(frozen=True)
class PersonalPredictor:
    model: torch.nn.Module
    center: np.ndarray
    scale: np.ndarray
    calibration: dict | None
    config: dict
    artifact_path: Path
    population_checkpoint_path: Path
    run_manifest_path: Path
    artifact_sha256: str


def load_personal_predictor(artifact_path, population_checkpoint_path, *,
                            expected_patient_id: str, trusted_checkpoint=False,
                            run_manifest_path=None) -> PersonalPredictor:
    """Load a completed unit, without loading predictions or later labels.

    The adjacent unit.json and original run_manifest.json are mandatory. The
    latter is discovered in artifact ancestors unless explicitly supplied;
    copied bundles need not preserve original absolute data paths. No source
    corpus is opened. This verifies deployment-package identity, not the full
    experiment inventory (use reports/verify_inventory.py for that audit).
    """
    _require(trusted_checkpoint is True,
             "Explicit trusted_checkpoint=True / --trust-checkpoint is required before pickle loading")
    artifact = Path(artifact_path).resolve()
    checkpoint = Path(population_checkpoint_path).resolve()
    _require(artifact.is_file() and artifact.name in ("adapted.pt", "calibration.json"),
             "Artifact must be an existing adapted.pt or calibration.json")
    _require(checkpoint.is_file() and checkpoint.suffix == ".pt", "Population checkpoint must be a local .pt file")
    unit = _read_json(artifact.parent / "unit.json")
    config = unit.get("config")
    _require(isinstance(config, dict) and unit.get("config_hash") == _hash_json(config),
             "Completed unit configuration hash mismatch")
    _require(all(unit.get(key) == value for key, value in config.items()),
             "Unit metadata differs from its canonical configuration")
    _require(isinstance(expected_patient_id, str) and bool(expected_patient_id)
             and config.get("patient") == expected_patient_id,
             "Patient identity mismatch; a personal artifact must not be applied to another person")
    method = config.get("strategy")
    budget, phase = config.get("budget"), config.get("phase")
    _require(method in (*SCALAR_METHODS, *NEURAL_METHODS), "Unknown personal strategy")
    _require(type(budget) is int and budget in (1, 2, 3, 5, 10), "Unknown frozen support budget")
    _require(phase in ("development", "test") and bool(unit.get("completed_at_utc")),
             "A completed development or test unit is required")
    _require(unit.get("role") == ("development" if phase == "development" else "target"),
             "Unit role and phase disagree")
    _require(artifact.name == ("adapted.pt" if method in NEURAL_METHODS else "calibration.json"),
             "Artifact type does not match its strategy")
    artifact_hash = _hash_file(artifact)
    _require(unit.get("artifact_hashes", {}).get(artifact.name) == artifact_hash,
             "Personal artifact SHA256 mismatch")
    # These checks precede the population checkpoint's unsafe pickle load.
    checkpoint_hash = _hash_file(checkpoint)
    _require(config.get("checkpoint_sha256") == checkpoint_hash,
             "Population checkpoint SHA256 does not match this personal unit")
    if run_manifest_path is None:
        run_manifest_path = next((parent / "run_manifest.json" for parent in artifact.parents
                                  if (parent / "run_manifest.json").is_file()), None)
    _require(run_manifest_path is not None, "Original run_manifest.json is required; use --run-manifest for a copied bundle")
    manifest_path = Path(run_manifest_path).resolve()
    manifest = _read_json(manifest_path)
    _require(_hash_json(manifest) == config.get("provenance_hash"), "Run-manifest provenance hash mismatch")
    _require(manifest.get("protocol") == "v2-personal-calibration-v1"
             and manifest.get("checkpoint_hashes", {}).get(str(config.get("seed"))) == checkpoint_hash,
             "Run manifest has an incompatible protocol or population checkpoint")
    role_ids = manifest.get("development_ids" if phase == "development" else "target_ids", [])
    _require(expected_patient_id in role_ids, "Patient is not in the recorded run cohort")
    for relative in SOURCE_FILES:
        _require(manifest.get("source_hashes", {}).get(relative) == _hash_file(V2_ROOT / relative),
                 f"Personal run does not match the frozen source contract: {relative}")
    audit = unit.get("adaptation", {})
    _require(audit.get("strategy") == method and audit.get("support_count") == budget
             and audit.get("query_used_for_adaptation") is False,
             "Unit is missing its support-only adaptation audit")
    model, center, scale = load_predictor(checkpoint, trusted_checkpoint=True)
    _require(model.variant == "baseline", "Track B requires the unchanged population baseline")
    _require(all(torch.isfinite(value).all().item() for value in model.state_dict().values()),
             "Population checkpoint has nonfinite model state")
    calibration = None
    if method in NEURAL_METHODS:
        saved = torch.load(artifact, map_location="cpu", weights_only=True)
        _require(isinstance(saved, dict) and saved.get("config") == config
                 and saved.get("variant") == "baseline", "Adapted checkpoint configuration mismatch")
        center, scale = _normalization(saved, center, scale)
        state = saved.get("state_dict")
        original = model.state_dict()
        _require(isinstance(state, dict) and set(state) == set(original), "Adapted state keys differ from BPNet")
        enabled = configure_trainable(model, method)
        _require(audit.get("trainable_parameter_names") == enabled
                 and audit.get("batchnorm_running_statistics_frozen") is True
                 and audit.get("dropout_active") is False,
                 "Adaptation depth, BN or dropout audit mismatch")
        for name, value in state.items():
            _require(isinstance(value, torch.Tensor) and value.shape == original[name].shape
                     and value.dtype == original[name].dtype and torch.isfinite(value).all().item(),
                     f"Invalid adapted tensor: {name}")
            if name not in enabled:
                _require(torch.equal(value, original[name]), f"Frozen parameter or BN buffer changed: {name}")
        model.load_state_dict(state, strict=True)
    else:
        calibration = _read_json(artifact)
        _require(calibration.get("method") == method and calibration.get("n_support") == budget
                 and audit.get("calibration") == calibration, "Saved calibration and unit audit disagree")
        center, scale = _normalization(calibration, center, scale)
        slope = np.asarray(calibration.get("slope"), dtype=np.float64)
        intercept = np.asarray(calibration.get("intercept"), dtype=np.float64)
        _require(slope.shape == (2,) and intercept.shape == (2,)
                 and np.isfinite(slope).all() and np.isfinite(intercept).all(),
                 "Calibration coefficients must be finite two-vectors")
        if method == "affine":
            _require(budget >= 3 and calibration.get("penalty") in (1, 10, 100)
                     and config.get("setting") == {"penalty": calibration["penalty"]},
                     "Affine calibration does not match the frozen K/penalty contract")
        else:
            _require(calibration.get("penalty") is None and config.get("setting") == {},
                     "Fixed control must not contain a fitted hyperparameter setting")
            expected_slope = np.zeros(2) if method in ("last_reference", "calibration_mean") else np.ones(2)
            _require(np.array_equal(slope, expected_slope), "Control has an invalid slope")
            if method == "population":
                _require(np.array_equal(intercept, np.zeros(2)), "Population control has a nonzero intercept")
    model.eval()
    model.requires_grad_(False)
    return PersonalPredictor(model, center, scale, calibration, config, artifact,
                             checkpoint, manifest_path, artifact_hash)


def predict_personal(predictor: PersonalPredictor, waves, *, batch_size=256,
                     fs_hz=125, channel_order=CHANNEL_ORDER,
                     already_normalized=False) -> np.ndarray:
    """Infer only: no support/query labels, optimization, or model selection."""
    _require(type(batch_size) is int and batch_size > 0, "Batch size must be a positive integer")
    values = prepare_inputs(waves, fs_hz=fs_hz, channel_order=channel_order,
                            already_normalized=already_normalized)
    if predictor.config["strategy"] in ("last_reference", "calibration_mean"):
        # These controls deliberately ignore observed signals after validation.
        base = np.zeros((len(values), 2), dtype=np.float64)
    else:
        base, _ = predict(predictor.model, values, predictor.center, predictor.scale,
                          device="cpu", batch_size=batch_size)
    result = apply_calibration(predictor.calibration, base) if predictor.calibration is not None else base
    _require(result.shape == (len(values), 2) and np.isfinite(result).all(), "Invalid SBP/DBP estimates")
    return np.asarray(result, dtype=np.float64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="adapted.pt or calibration.json, with adjacent unit.json")
    parser.add_argument("--population-checkpoint", type=Path, required=True)
    parser.add_argument("--patient-id", required=True, help="Identity check only; must exactly match the saved unit")
    parser.add_argument("--run-manifest", type=Path, help="Original run_manifest.json; otherwise discovered in artifact ancestors")
    parser.add_argument("--input", type=Path, required=True, help="Filtered ECG/PPG .npy, N x 2 x 1250")
    parser.add_argument("--output", type=Path, required=True, help="New .npy outside saved experiment directories")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Permit pickle loading of the trusted local population checkpoint")
    parser.add_argument("--already-normalized", action="store_true")
    parser.add_argument("--fs-hz", type=int, default=125)
    parser.add_argument("--channel-order", nargs=2, default=list(CHANNEL_ORDER), metavar=("FIRST", "SECOND"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if not args.trust_checkpoint:
        parser.error("--trust-checkpoint is required; hashes do not make an untrusted pickle safe")
    if args.input.suffix != ".npy" or args.output.suffix != ".npy":
        parser.error("Input and output must use .npy extensions")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Output must be a new file in an existing directory")
    predictor = load_personal_predictor(args.artifact, args.population_checkpoint,
                                       expected_patient_id=args.patient_id, trusted_checkpoint=True,
                                       run_manifest_path=args.run_manifest)
    protected = (predictor.artifact_path.parent, predictor.population_checkpoint_path.parent,
                 predictor.run_manifest_path.parent)
    if any(args.output.resolve().is_relative_to(directory) for directory in protected):
        parser.error("Output must be outside saved unit, population-checkpoint and run-manifest directories")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    result = predict_personal(predictor, np.load(args.input, allow_pickle=False),
                              batch_size=args.batch_size, fs_hz=args.fs_hz,
                              channel_order=args.channel_order, already_normalized=args.already_normalized)
    with args.output.open("xb") as stream:
        np.save(stream, result, allow_pickle=False)
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(result),
                      "columns": ["SBP", "DBP"], "unit": "mmHg", "dtype": str(result.dtype),
                      "patient": args.patient_id, "calibration_case": predictor.config["case"],
                      "strategy": predictor.config["strategy"], "budget": predictor.config["budget"],
                      "seed": predictor.config["seed"], "fs_hz": 125, "channel_order": list(CHANNEL_ORDER),
                      "artifact_sha256": predictor.artifact_sha256,
                      "population_checkpoint_sha256": predictor.config["checkpoint_sha256"],
                      "config_hash": _hash_json(predictor.config),
                      "provenance_hash": predictor.config["provenance_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
