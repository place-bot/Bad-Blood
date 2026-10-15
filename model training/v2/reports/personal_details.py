"""Personalization detail tables and compact provenance from actual run records.

This module never chooses a strategy or modifies an experiment. Its caller must
first verify the complete inventory; no synthetic/default accuracy is supplied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("population", "bias", "affine", "head", "last_block", "full", "last_reference", "calibration_mean")
BUDGETS = (1, 2, 3, 5, 10)
SEEDS = (42, 43, 44)
DISPLAY = {"population": "Population", "bias": "Bias", "affine": "Affine", "head": "Head",
           "last_block": "Last block", "full": "Full", "last_reference": "Last reference",
           "calibration_mean": "Reference mean"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seed_statistics(summaries, method, budget, metric, target):
    """Equal-seed average of already patient-macro metrics; not pooled windows."""
    if method == "affine" and budget < 3:
        return None
    values = np.array([summaries[(method, budget, seed)]["macro"][metric][target] for seed in SEEDS], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Missing/nonfinite mandatory metric: {method}/{budget}/{metric}/{target}")
    return {"mean": float(values.mean()), "seed_sd": float(values.std(ddof=1)),
            "seed_values": values.tolist(), "n_seeds": 3, "seed_sd_ddof": 1}


def prediction_seconds(record):
    """Measured elapsed inference, with shared population inference explicit."""
    latency = record["prediction_latency"]
    if "prediction_seconds" in latency:
        return float(latency["prediction_seconds"])
    transform = float(latency["scalar_transform_seconds"])
    shared = latency["shared_population_inference"]
    if shared is not None:
        return transform + float(shared["query"]["prediction_seconds"])
    return transform


def compact_unit(record):
    adaptation = record["adaptation"]
    prediction_time = prediction_seconds(record)
    total_fit = adaptation.get("total_adaptation_seconds", adaptation["adaptation_seconds"])
    return {"patient": record["patient"], "case": record["case"], "role": record["role"],
            "strategy": record["strategy"], "budget": record["budget"], "seed": record["seed"],
            "setting": record["setting"], "unit_path": record["unit_path"],
            "n_queries": record["summary"]["n_windows"], "observed_reference_signal_seconds": 10 * record["budget"],
            "last_reference_end": record["last_reference_end"], "calibration_boundary": record["calibration_boundary"],
            "population_epoch": record["population_epoch"], "checkpoint_sha256": record["checkpoint_sha256"],
            "config_hash": record["config_hash"], "provenance_hash": record["provenance_hash"],
            "selection_hash": record["selection_hash"], "artifact_hashes": record["artifact_hashes"],
            "trainable_parameters": adaptation["trainable_parameters"],
            "fitted_scalar_coefficients": adaptation.get("fitted_scalar_coefficients", 0),
            "changed_parameter_l2": adaptation["changed_parameter_l2"],
            "changed_parameter_count": adaptation["changed_parameter_count"],
            "optimizer_or_scalar_fit_seconds": adaptation["adaptation_seconds"],
            "total_adaptation_seconds": total_fit, "prediction_seconds": prediction_time,
            "milliseconds_per_query": 1000 * prediction_time / record["summary"]["n_windows"],
            "prediction_timing_detail": record["prediction_latency"],
            "process_peak_rss_native_units": record["process_peak_rss_native_units"],
            "peak_rss_units": record["peak_rss_units"]}


def build_personal_details(out, bsummaries, units, selection, *, provenance):
    """Save full-budget metrics/costs and return supplemental ACL LaTeX tables."""
    out = Path(out)
    provenance_hash = _hash(provenance)
    if selection["provenance_hash"] != provenance_hash:
        raise ValueError("Personal detail provenance differs from frozen selection")
    selected_without_hash = {key: value for key, value in selection.items() if key != "selection_hash"}
    if _hash(selected_without_hash) != selection["selection_hash"]:
        raise ValueError("Frozen selection hash is invalid")
    compact, masks = [], {}
    for record in units:
        if record["provenance_hash"] != provenance_hash or record["selection_hash"] != selection["selection_hash"]:
            raise ValueError("Unit has different data/source provenance or selection")
        names = record["adaptation"]["trainable_parameter_names"]
        mask = {"parameter_names": names, "trainable_parameters": record["adaptation"]["trainable_parameters"],
                "total_parameters": record["adaptation"]["total_parameters"],
                "fitted_scalar_coefficients": record["adaptation"].get("fitted_scalar_coefficients", 0),
                "batchnorm_running_statistics_frozen": True, "dropout_active": False}
        method = record["strategy"]
        if method in masks and mask != masks[method]:
            raise ValueError(f"Inconsistent parameter mask within method: {method}")
        masks[method] = mask
        compact.append(compact_unit(record))
    tables, grid = [], {}
    for target, label in enumerate(("SBP", "DBP")):
        for metric in ("mae", "rmse"):
            lines = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{lrrrrr}\toprule",
                     f"{label} {metric.upper()} & K=1 & K=2 & K=3 & K=5 & K=10\\\\\\midrule"]
            for method in METHODS:
                cells = []
                for budget in BUDGETS:
                    row = seed_statistics(bsummaries, method, budget, metric, target)
                    grid[f"{method}/K{budget}/{label}/{metric}"] = row
                    cells.append("--" if row is None else f"${row['mean']:.2f}\\pm{row['seed_sd']:.2f}$")
                lines.append(DISPLAY[method] + " & " + " & ".join(cells) + r"\\")
            lines.extend([r"\bottomrule\end{tabular}",
                f"\\caption{{All-budget patient-macro {label} {metric.upper()} in mmHg: mean $\\pm$ sample SD across three matched population seeds, after equal weighting of the six final patients within each seed. Seed SD is not a patient confidence interval. Affine calibration is prespecified as unavailable for K below three.}}\\end{{table*}}"])
            tables.append("\n".join(lines))
    lines = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{lrrl}\toprule",
             r"Method & Neural weights & Scalar coefficients & Enabled neural modules\\\midrule"]
    modules = {"head": "Two head linear layers", "last_block": "Final residual block and head",
               "full": "Entire CNN; BN state fixed"}
    for method in METHODS:
        mask = masks[method]
        lines.append(f"{DISPLAY[method]} & {mask['trainable_parameters']:,} & {mask['fitted_scalar_coefficients']} & {modules.get(method, 'None')}" + r"\\")
    lines.extend([r"\bottomrule\end{tabular}",
                  r"\caption{Personal adaptation capacity. Scalar coefficients are offset/reference values (two), or per-target slope/intercept (four); they are not extra neural weights. BN affine parameters train only inside enabled blocks. BN running statistics and dropout remain fixed; exact named masks and actual tensor changes are saved in personal\_details.json.}\end{table*}"])
    tables.append("\n".join(lines))
    lines = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{lrrr}\toprule",
             r"Method, K=10 & Update/fit seconds & Total adaptation seconds & Inference ms/query\\\midrule"]
    costs = {}
    for method in METHODS:
        rows = [unit for unit in compact if unit["strategy"] == method and unit["budget"] == 10]
        if len(rows) != 18:
            raise ValueError(f"K=10 cost table needs 6 patients x 3 seeds: {method}")
        costs[method] = {"n_patient_seed_runs": 18}
        cells = []
        for field in ("optimizer_or_scalar_fit_seconds", "total_adaptation_seconds", "milliseconds_per_query"):
            values = np.array([row[field] for row in rows], float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid measured personal cost")
            costs[method][field] = {"median": float(np.median(values)), "minimum": float(values.min()), "maximum": float(values.max())}
            cells.append(f"{np.median(values):.4f}")
        lines.append(DISPLAY[method] + " & " + " & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule\end{tabular}",
        r"\caption{Actual local-workstation costs at K=10: median across 18 patient/seed runs. Neural update time excludes setup; total adaptation includes cloning, setup and audit. Scalar rows time calibration arithmetic, not reference acquisition. Population performs only an identity transform, not a fit. Inference is batched elapsed time divided by query count, including data transfers, not single-window latency or Raspberry Pi timing. Population inference is measured once per patient/seed and reused by bias/affine, so these rows must not be summed as independent total-run costs. Reference-only controls exclude neural inference. Complete unit measurements and ranges are saved in personal\_details.json.}\end{table*}"])
    tables.append("\n".join(lines))
    caveat = ("Nested budgets change both reference count and last-reference age: these curves are not "
              "a pure sample-complexity experiment. Primary reference-relative bins and secondary "
              "boundary-relative bins expose this distinction; no clinical minimum cuff count is inferred.")
    data = {"provenance_hash": provenance_hash, "provenance": provenance, "selection": selection,
            "full_budget_patient_macro_mean_seed_sd": grid, "parameter_masks": masks,
            "k10_measured_costs": costs, "units": compact, "n_units": len(compact),
            "budget_interpretation": caveat,
            "cost_scope": "No physiological reference-acquisition time; shared population inference is not summed across reused units",
            "data_scope": "Real completed experiment records only; no raw waveforms/prediction arrays embedded"}
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "personal_details.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return caveat + "\n\n" + "\n\n".join(tables)
