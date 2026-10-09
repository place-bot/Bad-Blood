"""Execute the frozen inventory without changing training budgets or cohorts.

Population fits run sequentially. Track B development must finish and freeze
its settings before target evaluation. The artifact root stays outside Git.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("population", "personalization", "all"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--individual-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--personal-device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-eca", action="store_true", help="Also run the frozen supplementary ECA comparison")
    args = parser.parse_args()
    artifact = args.artifact_root.resolve()
    if artifact.is_relative_to(ROOT.parents[1]):
        parser.error("Put runtime artifacts outside the Git repository")
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "logs").mkdir(exist_ok=True)
    matrix = json.loads((ROOT / "experiment_matrix.json").read_text())
    status = {"stage": args.stage, "started_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "completed_this_invocation": [], "failed": None}
    status_path = artifact / f"suite_{args.stage}.json"

    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        os.replace(temporary, status_path)

    def run(name, command):
        status["current"] = name
        status["command"] = command
        save()
        print(f"START {name}", flush=True)
        log = artifact / "logs" / f"{name}.log"
        with log.open("a" if args.resume else "x") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            status["status"], status["failed"] = "failed", name
            status["returncode"] = result.returncode
            save()
            raise SystemExit(f"{name} failed; inspect {log}. Resume after resolving the cause.")
        status["completed_this_invocation"].append(name)
        save()
        print(f"DONE {name}", flush=True)

    save()
    if args.stage in ("population", "all"):
        p = matrix["population"]
        variants = p["required_three_seed_variants"] + (
            p["supplementary_three_seed_variants"] if args.include_eca else [])
        runs = [(v, s) for v in variants for s in p["seeds"]]
        runs += [(v, p["single_seed"]) for v in p["required_single_seed_variants"]]
        for variant, seed in runs:
            name = f"{variant}_seed{seed}"
            output = artifact / "attention" / name
            command = [sys.executable, str(ROOT / "attention_cnn/train.py"),
                       "--variant", variant, "--seed", str(seed),
                       "--data-root", str(args.data_root.resolve()),
                       "--output-dir", str(output), "--device", args.device]
            if output.exists() and any(output.iterdir()):
                if not args.resume:
                    raise SystemExit(f"Existing run {output}; --resume is required, never overwrite")
                command.append("--resume")
            run(name, command)
    if args.stage in ("personalization", "all"):
        for seed in matrix["personalization"]["seeds"]:
            completion = artifact / "attention" / f"baseline_seed{seed}" / "completion.json"
            if not completion.exists():
                raise SystemExit(f"Required population initialization is not complete: {completion}")
            if json.loads(completion.read_text()).get("toy_test_only", True):
                raise SystemExit("Toy tests cannot initialize the real personalization experiment")
        for phase in ("development", "test"):
            command = [sys.executable, str(ROOT / "personalized_forecasting/run.py"),
                       "--phase", phase, "--population-root", str(artifact / "attention"),
                       "--individual-root", str(args.individual_root.resolve()),
                       "--output-dir", str(artifact / "personalization"),
                       "--device", args.personal_device]
            if args.resume:
                command.append("--resume")
            run(f"personalization_{phase}", command)
    status.update(status="complete", current=None, finished_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == "__main__":
    main()
