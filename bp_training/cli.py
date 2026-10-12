"""Command-line entry points for the CRC research workflow."""

import argparse
import json
from pathlib import Path

from .config import load_config


def training_args(parser):
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Hash and validate already prepared arrays; no archive download")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--split", type=Path)
    prepare.add_argument("--exclude-subjects", nargs="*", default=[], type=Path)
    prepare.add_argument("--source", default="PulseDB/VitalDB")
    prepare.add_argument("--role", choices=("development", "historical", "prospective"), default="development")
    prepare.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-data")
    validate.add_argument("--manifest", type=Path, required=True)
    fit = commands.add_parser("train")
    training_args(fit)
    fit.add_argument("--run-dir", type=Path, required=True)
    fit.add_argument("--resume", action="store_true")
    smoke = commands.add_parser("preflight")
    training_args(smoke)
    smoke.add_argument("--run-root", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate a completed run on its development validation split")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="Aggregate complete seed sets without selecting the best seed")
    summarize.add_argument("--evaluations", nargs="+", type=Path, required=True)
    summarize.add_argument("--seeds", nargs="+", type=int, default=[20260923, 20260924, 20260925])
    summarize.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    bundle = commands.add_parser("evaluate-bundle", help="Evaluate the actual Pi artifact, with a freeze record for held-out cohorts")
    bundle.add_argument("--bundle", type=Path, required=True)
    bundle.add_argument("--manifest", type=Path, required=True)
    bundle.add_argument("--freeze-record", type=Path)
    bundle.add_argument("--training-manifest", type=Path,
                        help="Required for held-out evaluation; verifies exclusion of development participants")
    bundle.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("train", "preflight"):
        config = load_config(args.config, device=args.device, seed=args.seed)
        if args.command == "train":
            from .trainer import train
            result = train(config, args.manifest, args.run_dir, resume=args.resume)
        else:
            from .preflight import preflight
            result = preflight(config, args.manifest, args.run_root)
    elif args.command == "prepare":
        from .data import build_manifest
        from .trainer import external_output
        from .io import object_hash
        result = build_manifest(args.data_dir, external_output(args.output), args.split,
                                args.exclude_subjects, args.source, args.role)
        result = {"manifest": str(args.output), "dataset_hash": object_hash(result)}
    elif args.command == "validate-data":
        from .data import validate_manifest
        audit = validate_manifest(args.manifest)
        result = {k: audit[k] for k in ("hash", "rows", "eligible_rows", "participants")}
    elif args.command == "evaluate":
        from .evaluate import evaluate_run
        result = evaluate_run(args.run_dir, args.output)
    elif args.command == "compare":
        from .evaluate import compare_evaluations
        result = compare_evaluations(args.baseline, args.candidate, args.output)
    elif args.command == "export":
        from .release import export_run
        result = export_run(args.run_dir, args.output)
    elif args.command == "summarize":
        from .evaluate import summarize_evaluations
        result = summarize_evaluations(args.evaluations, args.output, args.seeds)
    else:
        from .evaluate import evaluate_bundle
        result = evaluate_bundle(args.bundle, args.manifest, args.output, args.freeze_record,
                                 args.training_manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
