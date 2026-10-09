"""Build the final report only from a complete, hash-verified real inventory."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from shared.metrics import summarize, paired_patient_change
from reports.analysis_details import scientific_details, insert_gap_breaks
from reports.personal_details import build_personal_details
from reports.task_figure import build_task_figure
from reports.extended_tables import build_extended_tables
from reports.personal_interpretation import build_personal_interpretation
from reports.attention_interpretation import build_attention_interpretation, REQUIRED_THREE_SEED
from reports.personal_failure_context import build_personal_failure_context
from reports.result_storage import write_compressed_json, build_metric_overview
from reports.verify_inventory import verify_a_inventory, verify_b_inventory

A_VARIANTS = ["baseline", "se_last", "temporal_last", "cbam_last", "additive_last", "se_all",
              "baseline_ecg", "baseline_ppg"]
B_METHODS = ["population", "bias", "affine", "head", "last_block", "full",
             "last_reference", "calibration_mean"]
DISPLAY = {"baseline": "CNN", "se_last": "SE-last", "temporal_last": "Temporal",
           "cbam_last": "CBAM1D", "additive_last": "Additive", "se_all": "SE-all",
           "baseline_ecg": "ECG only", "baseline_ppg": "PPG only", "eca_last": "ECA",
           "population": "Population", "bias": "Bias", "affine": "Affine", "head": "Head",
           "last_block": "Last block", "full": "Full", "last_reference": "Last reference",
           "calibration_mean": "Reference mean"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def tex(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def number(value, places=2):
    return "--" if value is None or not np.isfinite(value) else f"{value:.{places}f}"


def assert_pair(left, right):
    for key in ("y", "subjects", "source_indices"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"Paired comparison differs in {key}")


def figure(out, name):
    plt.savefig(out / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def aggregate_a(root, population_data_root=None):
    runs, predictions = {}, {}
    variants = list(A_VARIANTS)
    if all((root / "attention" / f"eca_last_seed{s}" / "completion.json").exists() for s in (42, 43, 44)):
        variants.append("eca_last")
    verified = verify_a_inventory(root / "attention", include_eca="eca_last" in variants,
                                  population_data_root=population_data_root)
    for variant in variants:
        for seed in ([42] if variant.startswith("baseline_") else [42, 43, 44]):
            key = f"{variant}_seed{seed}"
            folder = root / "attention" / key
            done = verified[key]["completion"]
            if done["status"] != "complete" or done["toy_test_only"]:
                raise ValueError(f"Not a completed real experiment: {key}")
            if digest(folder / "best.pt") != done["best_checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed: {key}")
            run = {"completion": done, "costs": verified[key]["costs"],
                   "history": verified[key]["history"], "config": verified[key]["config"],
                   "environment": verified[key]["environment"], "metrics": {}}
            for split in ("validation", "test", "aami"):
                file = folder / f"predictions_{split}.npz"
                if digest(file) != done["prediction_sha256"][file.name]:
                    raise ValueError(f"Prediction artifact changed: {file}")
                p = verified[key]["predictions"][split]
                predictions[(variant, seed, split)] = p
                run["metrics"][split] = summarize(p["y"], p["pred"], p["subjects"])
            runs[key] = run
    paired = {}
    for variant in variants:
        if variant == "baseline":
            continue
        seeds = [42] if variant.startswith("baseline_") else [42, 43, 44]
        for seed in seeds:
            assert_pair(predictions[(variant, seed, "test")], predictions[("baseline", seed, "test")])
        paired[variant] = {m: paired_patient_change(
            [runs[f"{variant}_seed{s}"]["metrics"]["test"] for s in seeds],
            [runs[f"baseline_seed{s}"]["metrics"]["test"] for s in seeds], m) for m in ("mae", "rmse")}
    return variants, runs, predictions, paired


def aggregate_b(root, population_data_root=None, individual_root=None):
    folder = root / "personalization"
    verified = verify_b_inventory(folder, population_data_root=population_data_root,
                                  individual_root=individual_root)
    development, test = verified["development_index"], verified["test_index"]
    if not development["complete"] or development["n_units"] != 1068:
        raise ValueError("The full development matrix is incomplete")
    if not test["complete"] or test["n_units"] != 684:
        raise ValueError("The full personal test matrix is incomplete")
    selection = verified["selection"]
    records = {r["unit_path"]: r for r in verified["test_records"]}
    groups = defaultdict(list)
    units = []
    for item in test["units"]:
        unit = folder / item["unit_path"]
        record = records[item["unit_path"]]
        for filename, expected in record["artifact_hashes"].items():
            if digest(unit / filename) != expected:
                raise ValueError(f"Personal artifact changed: {unit / filename}")
        if record["selection_hash"] != selection["selection_hash"]:
            raise ValueError("Final unit used a different development selection")
        p = dict(np.load(unit / "predictions.npz", allow_pickle=False))
        groups[(item["strategy"], item["budget"], item["seed"])].append((item["patient"], p, record))
        units.append(record)
    summaries, predictions, drifts = {}, {}, {}
    for key, items in groups.items():
        items.sort(key=lambda row: row[0])
        if len(items) != 6 or len({x[0] for x in items}) != 6:
            raise ValueError(f"Personal group missing patients: {key}")
        p = {k: np.concatenate([item[1][k] for item in items]) for k in
             ("y", "pred", "subjects", "source_indices", "start", "end_exclusive",
              "elapsed_since_reference_min", "elapsed_since_boundary_min")}
        predictions[key] = p
        summaries[key] = summarize(p["y"], p["pred"], p["subjects"])
        drifts[key] = []
        for axis in ("elapsed_since_reference_min", "elapsed_since_boundary_min"):
            for lo, hi in ((0, 15), (15, 30), (30, 60), (60, np.inf)):
                mask = (p[axis] >= lo) & (p[axis] < hi)
                drifts[key].append({"axis": axis, "lower": lo, "upper": None if np.isinf(hi) else hi,
                    "n_windows": int(mask.sum()), "n_patients": int(len(np.unique(p["subjects"][mask]))),
                    "metrics": summarize(p["y"][mask], p["pred"][mask], p["subjects"][mask]) if mask.any() else None})
    paired = {}
    for method in B_METHODS:
        if method == "population":
            continue
        for budget in (1, 2, 3, 5, 10):
            if (method, budget, 42) not in summaries:
                continue
            for seed in (42, 43, 44):
                assert_pair(predictions[(method, budget, seed)], predictions[("population", budget, seed)])
            paired[(method, budget)] = {m: paired_patient_change(
                [summaries[(method, budget, s)] for s in (42, 43, 44)],
                [summaries[("population", budget, s)] for s in (42, 43, 44)], m) for m in ("mae", "rmse")}
    return summaries, predictions, drifts, paired, selection, units


def mean_metric(rows, metric, aggregation="macro"):
    values = np.asarray([row[aggregation][metric] for row in rows], dtype=float)
    counts = np.isfinite(values).sum(0)
    return np.divide(np.nansum(values, axis=0), counts,
                     out=np.full(2, np.nan), where=counts > 0)


def tables_and_figures(out, variants, aruns, apred, apaired, bsummaries, bpred, bdrifts, bpaired):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140, "pdf.fonttype": 42})
    (out / "figures").mkdir(exist_ok=True)
    tables = []
    def arows(v, split="test"):
        return [r["metrics"][split] for name, r in aruns.items() if name.startswith(v + "_seed")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for col, metric in enumerate(("mae", "rmse")):
        for q, target in enumerate(("SBP", "DBP")):
            vals = [np.array([r["macro"][metric][q] for r in arows(v)]) for v in variants]
            axes[q, col].bar(range(len(variants)), [x.mean() for x in vals])
            repeated = [i for i, x in enumerate(vals) if len(x) > 1]
            axes[q, col].errorbar(repeated, [vals[i].mean() for i in repeated],
                                 yerr=[vals[i].std(ddof=1) for i in repeated],
                                 fmt="none", color="black", capsize=3)
            axes[q, col].set_xticks(range(len(variants)), [DISPLAY[v] for v in variants], rotation=45, ha="right")
            axes[q, col].set_ylabel(f"{target} patient-macro {metric.upper()} (mmHg)")
    figure(out, "attention_results")
    for split in ("test", "aami"):
        for q, target in enumerate(("SBP", "DBP")):
            table = [r"\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrrrrrrrr}\toprule",
                     f"{target}, {tex(split)} & MAE & RMSE & Bias & SD & MedAE & Q90 & Q95 & $\\leq5$ & $\\leq10$ & $\\leq15$ & $r$ & $R^2$\\\\\\midrule"]
            for variant in variants:
                rows = arows(variant, split)
                metrics = [mean_metric(rows, m, "pooled")[q] for m in
                           ("mae", "rmse", "bias", "error_sd", "median_ae", "q90_ae", "q95_ae", "within_5", "within_10", "within_15", "pearson_r", "r2")]
                metrics[7:10] = [100 * x for x in metrics[7:10]]
                table.append(DISPLAY[variant] + " & " + " & ".join(number(x) for x in metrics) + r"\\")
            table += [r"\bottomrule\end{tabular}",
                      f"\\caption{{Pooled-window {target} metrics on {tex(split)}, averaged across matched seeds (single seed for modality ablations). SD is sample error SD. Threshold columns are percentages. Complete per-patient and macro metrics are in the machine-readable results.}}\\end{{table*}}"]
            tables.append("\n".join(table))
    # Budget curves preserve the same final queries and show all methods.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for method in B_METHODS:
        budgets = [k for k in (1, 2, 3, 5, 10) if (method, k, 42) in bsummaries]
        for col, metric in enumerate(("mae", "rmse")):
            for q, target in enumerate(("SBP", "DBP")):
                values = [mean_metric([bsummaries[(method, k, s)] for s in (42, 43, 44)], metric)[q] for k in budgets]
                axes[q, col].plot(budgets, values, "o-", label=DISPLAY[method], linewidth=1.3, markersize=3)
                axes[q, col].set_ylabel(f"{target} patient-macro {metric.upper()} (mmHg)")
                axes[q, col].set_xlabel("ABP-derived calibration references")
                axes[q, col].set_xticks([1, 2, 3, 5, 10])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure(out, "personal_budget")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for method in ("population", "bias", "head", "last_block", "full", "calibration_mean"):
        for q, target in enumerate(("SBP", "DBP")):
            values = []
            for bin_index in range(4):
                entries = [bdrifts[(method, 10, s)][bin_index]["metrics"] for s in (42, 43, 44)]
                values.append(np.mean([x["macro"]["mae"][q] for x in entries if x is not None]) if any(x is not None for x in entries) else np.nan)
            axes[q].plot(range(4), values, "o-", label=DISPLAY[method])
            axes[q].set_xticks(range(4), ["0-15", "15-30", "30-60", "60+"])
            axes[q].set_xlabel("Minutes since last reference (K=10)")
            axes[q].set_ylabel(f"{target} patient-macro MAE (mmHg)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    figure(out, "personal_drift")
    for q, target in enumerate(("SBP", "DBP")):
        table = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{lrrrrrr}\toprule",
                 f"{target}, K=10 & Macro MAE & Macro RMSE & Bias & Error SD & $\\Delta$ MAE (95\\% CI) & $\\leq 10$\\\\\\midrule"]
        for method in B_METHODS:
            rows = [bsummaries[(method, 10, s)] for s in (42, 43, 44)]
            values = [mean_metric(rows, m, ag)[q] for m, ag in (("mae", "macro"), ("rmse", "macro"), ("bias", "pooled"), ("error_sd", "pooled"))]
            if method != "population":
                ci = bpaired[(method, 10)]["mae"]
                delta = f"{number(ci['estimate'][q])} [{number(ci['ci_lower'][q])}, {number(ci['ci_upper'][q])}]"
            else:
                delta = "Reference"
            table.append(DISPLAY[method] + " & " + " & ".join(number(x) for x in values) + " & " + delta + " & " + number(100 * mean_metric(rows, "within_10", "pooled")[q]) + r"\\")
        table += [r"\bottomrule\end{tabular}", f"\\caption{{Personal {target} results at ten simulated references. Paired changes are model minus population, averaged within patient across seeds before a 2,000-resample patient bootstrap. Intervals are descriptive for six patients.}}\\end{{table*}}"]
        tables.append("\n".join(table))
    for q, target in enumerate(("SBP", "DBP")):
        table = [r"\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
                 r"\begin{tabular}{lrrrrrrrrrrrr}\toprule",
                 f"{target}, K=10 & MAE & RMSE & Bias & SD & MedAE & Q90 & Q95 & $\\leq5$ & $\\leq10$ & $\\leq15$ & $r$ & $R^2$\\\\\\midrule"]
        for method in B_METHODS:
            rows = [bsummaries[(method, 10, s)] for s in (42, 43, 44)]
            values = [mean_metric(rows, m, "pooled")[q] for m in
                      ("mae", "rmse", "bias", "error_sd", "median_ae", "q90_ae", "q95_ae", "within_5", "within_10", "within_15", "pearson_r", "r2")]
            values[7:10] = [100 * v for v in values[7:10]]
            table.append(DISPLAY[method] + " & " + " & ".join(number(x) for x in values) + r"\\")
        table += [r"\bottomrule\end{tabular}",
                  f"\\caption{{Complete pooled-window personal {target} metrics at K=10, averaged across three seeds. Thresholds are percentages; undefined correlations are marked unavailable. Reference-only controls may correlate when pooling different patient means, despite being constant within each patient.}}\\end{{table*}}"]
        tables.append("\n".join(table))
    methods = [m for m in B_METHODS if m != "population"]
    patients = sorted(bpaired[("bias", 10)]["mae"]["patient_changes"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for q, target in enumerate(("SBP", "DBP")):
        values = np.array([[bpaired[(m, 10)]["mae"]["patient_changes"][p][q] for m in methods] for p in patients])
        lim = max(1, np.max(np.abs(values)))
        im = axes[q].imshow(values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        axes[q].set_xticks(range(len(methods)), [DISPLAY[m] for m in methods], rotation=45, ha="right")
        axes[q].set_yticks(range(len(patients)), [p.split(":")[-1] for p in patients])
        axes[q].set_title(f"{target}: paired MAE change, K=10")
        for i in range(len(patients)):
            for j in range(len(methods)):
                axes[q].text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=axes[q], label="mmHg (negative is better)", shrink=.8)
    figure(out, "patient_changes")
    # Examples selected by baseline error, not by adaptation improvement.
    reference = bsummaries[("population", 10, 42)]["per_patient"]
    ranked = sorted(reference, key=lambda p: np.mean(reference[p]["mae"]))
    examples = [ranked[len(ranked) // 2], ranked[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for r, patient in enumerate(examples):
        for q, target in enumerate(("SBP", "DBP")):
            p = bpred[("population", 10, 42)]
            mask = p["subjects"] == patient
            time = p["elapsed_since_reference_min"][mask]
            starts, ends = p["start"][mask], p["end_exclusive"][mask]
            axes[r, q].plot(*insert_gap_breaks(time, p["y"][mask, q], starts, ends),
                            color="black", linewidth=.7, label="Reference")
            for method in ("population", "bias", "full"):
                axes[r, q].plot(*insert_gap_breaks(time, bpred[(method, 10, 42)]["pred"][mask, q], starts, ends),
                                linewidth=.7, alpha=.8, label=DISPLAY[method])
            axes[r, q].set_title(f"{patient.split(':')[-1]}: {target}")
            axes[r, q].set_xlabel("Minutes since final reference")
            axes[r, q].set_ylabel("mmHg")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    figure(out, "temporal_examples")
    return "\n\n".join(tables), examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        help="Optional relocated population arrays; original byte hashes still must match")
    parser.add_argument("--individual-root", type=Path, required=True,
                        help="Timestamped files for byte-verified personal inventory and observed-waveform figures")
    args = parser.parse_args()
    root, out = args.artifact_root.resolve(), Path(__file__).resolve().parent
    variants, aruns, apred, apaired = aggregate_a(root, args.data_root)
    bs, bp, bd, bc, selection, units = aggregate_b(root, args.data_root, args.individual_root)
    results = out / "results"
    results.mkdir(exist_ok=True)
    seeds = (42, 43, 44)
    principal = [v for v in REQUIRED_THREE_SEED if v in variants]
    winner = min(principal, key=lambda v: np.mean([aruns[f"{v}_seed{s}"]["metrics"]["validation"]["macro"]["mae"] for s in seeds]))
    table_tex, examples = tables_and_figures(out, variants, aruns, apred, apaired, bs, bp, bd, bc)
    extra_tables, extra_notes = scientific_details(out, variants, aruns, apred, apaired, bs, bp, bd, bc,
        args.individual_root, examples, winner,
        expected_manifest_hash=read(root / "personalization/run_manifest.json")["manifest_hash"])
    personal_tables = build_personal_details(out, bs, units, selection,
                                             provenance=read(root / "personalization/run_manifest.json"))
    extended_tables = build_extended_tables(aruns, bs, bc, selection)
    personal_interpretation, personal_interpretation_data = build_personal_interpretation(bs, bc, bd)
    attention_interpretation, attention_interpretation_data = build_attention_interpretation(aruns, winner)
    context_units = []
    for record in units:
        if (record["strategy"], record["budget"], record["seed"]) == ("population", 10, 42):
            with np.load(root / "personalization" / record["unit_path"] / "predictions.npz",
                         allow_pickle=False) as saved:
                context_units.append({"record": record, "predictions": dict(saved)})
    failure_context, failure_context_data = build_personal_failure_context(context_units)
    build_task_figure(out)
    write_compressed_json(results / "attention.json.gz", {"runs": aruns, "paired": apaired})
    write_compressed_json(results / "personalization.json.gz", {
        "summaries": {"/".join(map(str, k)): v for k, v in bs.items()},
        "paired": {"/".join(map(str, k)): v for k, v in bc.items()},
        "drift": {"/".join(map(str, k)): v for k, v in bd.items()}, "selection": selection})
    write_json(results / "personal_interpretation.json", personal_interpretation_data)
    write_json(results / "attention_interpretation.json", attention_interpretation_data)
    write_json(results / "personal_failure_context.json", failure_context_data)
    write_json(results / "metric_overview.json", build_metric_overview(
        aruns, bs, attention_interpretation_data["selection"]))
    write_json(results / "inventory.json", {"population_completed": len(aruns), "population_required": 20,
        "personal_development_completed": 1068, "personal_test_completed": len(units),
        "supplementary_ECA_completed": "eca_last" in variants,
        "artifact_root": str(root), "protocol_commit": "1b93cf13249931af650cdfa91e19343564032451",
        "representative_patients": examples, "raw_predictions_in_git": False,
        "full_metric_archives": ["attention.json.gz", "personalization.json.gz"],
        "browsable_metric_summary": "metric_overview.json"})
    win_mae = mean_metric([aruns[f"{winner}_seed{s}"]["metrics"]["test"] for s in seeds], "mae")
    base_mae = mean_metric([aruns[f"baseline_seed{s}"]["metrics"]["test"] for s in seeds], "mae")
    attention_choice = attention_interpretation_data["selection"]["attention_winner"]
    attention_choice_mae = mean_metric(
        [aruns[f"{attention_choice}_seed{s}"]["metrics"]["test"] for s in seeds], "mae")
    personal_abstract = {
        method: mean_metric([bs[(method, 10, s)] for s in seeds], "mae")
        for method in ("population", "full", "bias", "calibration_mean")}
    abstract_result = (
        f"Within the prospective four-model attention comparison, validation selects {DISPLAY[attention_choice]}, "
        f"with primary-test patient-macro SBP/DBP MAE {attention_choice_mae[0]:.2f}/{attention_choice_mae[1]:.2f} mmHg "
        f"versus {base_mae[0]:.2f}/{base_mae[1]:.2f} for the unchanged CNN. "
        f"With ten simulated personal references, population and full-adaptation MAE are "
        f"{personal_abstract['population'][0]:.2f}/{personal_abstract['population'][1]:.2f} and "
        f"{personal_abstract['full'][0]:.2f}/{personal_abstract['full'][1]:.2f} mmHg; "
        f"bias correction gives {personal_abstract['bias'][0]:.2f}/{personal_abstract['bias'][1]:.2f}, "
        f"and reference mean gives {personal_abstract['calibration_mean'][0]:.2f}/"
        f"{personal_abstract['calibration_mean'][1]:.2f}. "
        "Paired patient intervals and all individual cases qualify these descriptive comparisons.\n")
    observations = []
    for method in ("bias", "head", "last_block", "full"):
        c = bc[(method, 10)]["mae"]
        gains = np.array(list(c["patient_changes"].values()))
        observations.append(f"At K=10, {DISPLAY[method]} changes SBP/DBP MAE by "
            f"{number(c['estimate'][0])}/{number(c['estimate'][1])} mmHg; "
            f"{int((gains[:,0]<0).sum())}/6 patients improve in SBP and {int((gains[:,1]<0).sum())}/6 in DBP.")
    costs = []
    for variant in variants:
        rows = [r for k, r in aruns.items() if k.startswith(variant + "_seed")]
        epochs = ",".join(str(r["completion"]["actual_epochs"]) for r in rows)
        best = ",".join(str(r["completion"]["best_epoch"]) for r in rows)
        costs.append(f"{DISPLAY[variant]} & {rows[0]['completion']['parameters']:,} & {epochs} & {best} & "
                     f"{sum(r['costs']['training_seconds'] for r in rows)/60:.1f}\\\\")
    final = r"""\documentclass[11pt]{article}
\usepackage[preprint]{acl}
\usepackage{times}\renewcommand{\ttdefault}{lmtt}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage{microtype,amsmath,amssymb,booktabs,graphicx,url}
\extrafloats{100}
\title{Attention and Personal Calibration for VitalDB Blood-Pressure Estimation: Bad-Blood v2}
\author{Bad-Blood Project\\Research and Engineering Report}
\begin{document}\maketitle
\begin{abstract}
We evaluate lightweight attention interventions on the actual high-resolution
Bad-Blood CNN and early personal calibration for later contemporaneous blood-
pressure estimation. Waveform models receive ten-second ECG/PPG windows and output
scalar SBP/DBP. The controlled population comparison excludes previously
explored personal patients and retrains the unchanged baseline. A separate
six-patient chronological pilot evaluates sparse ABP-reference calibration,
with adaptation settings chosen on four different patients. Results below
are computed from completed, hash-verified experiments, not simulations of
model accuracy. Sparse references are simulated from real invasive labels;
they are not actual cuff events.
""" + abstract_result + r"""
\end{abstract}
\section{Research questions and motivation}
The application is a population-trained estimator that can be calibrated to
one person and then use newly observed ECG/PPG to estimate their current
pressure. Track A asks whether attention improves the existing architecture
at matched training and data budgets. Track B asks how adaptation depth and
reference budget affect later error and how performance changes over time.
These hypotheses do not assume that larger models or more adaptation improve
every patient. Later ECG/PPG is a legitimate input; later BP is evaluator-only.
The task is not open-loop future-waveform prediction.

\section{Related work}
SE, ECA and CBAM motivate feature-channel and temporal gates
\citep{hu2018se,wang2020eca,woo2018cbam}. Learned feature channels are not
individual sensors after convolutional mixing. Physiological hybrids offer
relevant precedents but differ in calibration, scale and splitting; their
reported numbers are not interchangeable benchmarks. One-reference conditioning
\citep{schlesinger2020} and restricted-layer transfer \citep{leitner2022} motivate
the personal comparison. Shuffled within-patient evaluation does not establish
durability. Real longer-term evidence is sensor- and model-specific
\citep{yavarimanesh2022}, while strong reference-only comparators are essential
\citep{mieloszyk2022}. The two accompanying ACL literature reviews document
paper-level definitions, reported metrics, provenance and limitations.

\section{Data and provenance audit}
Official VitalDB-specific PulseDB subsets provide the population data
\citep{wang2023pulsedb}. Source suffix 1 identifies VitalDB, not a visit. All 31
case-to-subject links in complete local individual files were checked against
public VitalDB metadata. The population subsets contain no timestamps or case
IDs; temporal evaluation instead uses individual files with original T and
CaseID. Raw ABP, derived labels and QC fields never enter the inference inputs.

Excluding all 31 historically explored patients leaves 1,085 training patients
(390,600 windows), 192 validation patients (69,120), 141 primary-test patients
(56,400), and 116 stress-test patients (666). Original V1 scores are historical
and are not treated as the new baseline. Previous project work inspected the
test cohort, so this is a controlled rerun rather than untouched external
validation. All new population weights exclude the personal patients.

Thirty complete individual files contain 30 patients, 31 cases and 15,486
windows. One truncated and one absent file are excluded; two patients have
only two isolated windows. A multi-case patient is not used for the main
within-case pilot. Availability-only allocation gives four development and
six final patients. One initially selected patient lacked ten spaced early
references and was replaced before training, with the complete audit trail
retained. Final patients provide 4,284 later queries; development provides 1,855.
Two query-to-query timestamp pairs overlap by 8 ms upstream; no support/query
boundary is crossed. No local actual cuff/NIBP events or dated repeat visits
support clinical cuff or cross-visit claims.

\section{Methods and controlled design}
\subsection{Architecture and population fitting}
The V1 source last changed at commit \texttt{1f33cdc}; the research base is
\texttt{7553248}. The exact 37,666-parameter BPNet(pool=1) has a 2-to-8 stem
(kernel 7, stride 2), residual blocks 8-to-16-to-32-to-48 (kernels 7 and 5),
BatchNorm/SiLU, global mean pooling and a 48-to-32-to-2 head with dropout 0.15.
Residual feature lengths are 313, 157 and 79.
SE-last uses a 48-to-6-to-48 MLP; temporal attention uses mean/max channel
pooling and a seven-tap convolution; CBAM1D applies channel then temporal
gates. A 630-parameter additive pooled-context control matches SE capacity.
SE-all tests placement. ECG-only/PPG-only ablations zero the absent channel.
ECA is explicitly supplementary. All parameters and state keys are tested
against the original implementation, rather than approximating the backbone.

Inputs use provided filtered fields with per-window/channel min--max scaling.
For observed window $X$, the physical-unit prediction is
\begin{equation}
g_\theta(X)=\mu_{\rm tr}+\sigma_{\rm tr}\odot f_\theta(N(X)),
\end{equation}
where $N$ scales each input channel within that window. Labels use
training-only centering/scaling: $\mu_{\rm tr}=(115.418,62.928)$ and
$\sigma_{\rm tr}=(18.942,12.083)$ mmHg, in SBP/DBP order. All fits use normalized MSE,
AdamW (learning rate 0.001, weight decay 0.001), batch size 256, noise SD 0.003,
gradient clipping at norm 5, maximum 40 epochs, and patience 8. A factor-0.5,
patience-3 plateau scheduler and best-checkpoint selection use mean validation
patient-macro SBP/DBP MAE. Seeds 42/43/44 are matched for principal comparisons;
modality ablations use 42 only. Source,
split, config and checkpoint hashes accompany every run. Epoch-boundary
resume restores optimizer, scheduler and RNG states.

\subsection{Personal calibration}
Each case's first 20\% of elapsed span defines the support period; query windows
start after a 10-second boundary guard. Gaps are retained. From a greedy pool
with starts at least 60 seconds apart, ten approximately evenly spaced references
are selected; budgets 1, 2, 3, 5 and 10 use nested prefixes. Queries stay identical
across budgets. Labels are invasive-ABP-derived sparse references, not cuff
measurements or independent evidence merely because they are spaced.

New unchanged baseline checkpoints initialize all strategies at matched seeds.
Comparisons are no adaptation, additive residual bias, shrinkage affine
($K\geq3$), head-only, last-block-plus-head, full-model, last-reference and
calibration-mean controls. The last two do not need ECG/PPG at prediction.
Head, last-block and full train 1,634, 25,730 and 37,666 parameters respectively.
BN running state and dropout remain fixed. Neural learning rates 0.0001/0.001
and steps 20/100, and affine penalties 1/10/100, are selected only on the four
development patients averaged across seeds. No final patient's later labels
control adaptation stopping, parameter selection or normalization.
Bias calibration adds the support mean residual,
\begin{equation}
\delta_i=\frac{1}{K}\sum_{j=1}^{K}\bigl(y_{ij}-g_\theta(X_{ij})\bigr),
\end{equation}
to each later prediction. For each pressure separately, affine calibration
fits $a z_j+b$ to standardized support labels $u_j$ by minimizing
$\sum_j(u_j-a z_j-b)^2+\lambda(a-1)^2$; the intercept is unpenalized.
Last-reference and reference-mean controls predict the corresponding scalar
support pressure without using later waveforms. Neural adaptation minimizes
the same standardized two-output MSE on all $K$ support windows for the
development-selected number of updates; it never monitors later BP loss.
The nested chronological prefixes vary both reference count and reference
recency. Budget curves therefore describe this acquisition policy, not pure
sample complexity with reference age held constant.

\subsection{Evaluation and uncertainty}
MAE/RMSE, bias, sample error SD, median and 90th/95th error quantiles,
within-5/10/15-mmHg fractions, correlation and $R^2$ are saved separately
for SBP and DBP. Patient-macro metrics weight people equally; pooled metrics
weight windows equally. Paired changes average matched seeds within patient,
then resample patients 2,000 times for descriptive 95\% intervals. Seed SD and
patient uncertainty are separate. Reference-only predictors have identical
predictions across seeds by construction; zero seed SD does not imply zero
patient uncertainty or three independent replications. Counts of patients
with lower error describe its sign, not a minimum practically important change.
Pressure strata and all patient failures
remain visible. Repeated-measurement Bland--Altman summaries decompose
between/within-person error; serial correlation remains a limitation.
Drift uses time from the actual last support reference, not silently the 20\%
boundary. Empty bins are unavailable, not zero error. The saved secondary
boundary-relative analysis uses a distinct label.
For one pressure and errors $e_{ij}=\hat y_{ij}-y_{ij}$, the primary
patient-macro definitions are
\begin{align}
{\rm MAE}_{\rm macro}&=\frac{1}{P}\sum_i\frac{1}{n_i}\sum_j|e_{ij}|,\\
{\rm RMSE}_{\rm macro}&=\frac{1}{P}\sum_i\sqrt{\frac{1}{n_i}\sum_j e_{ij}^2}.
\end{align}
Pooled RMSE instead takes one square root after averaging all window errors.
Error SD uses denominator $N-1$; threshold fractions include equality.
Correlations and $R^2$ with zero required variance are undefined, not zero;
macro summaries retain the number of patients for whom each metric exists.

\section{Results}
"""
    final += f"Across the six required three-seed architectures, including capacity and placement controls, validation selects {DISPLAY[winner]}. Its primary-test patient-macro SBP/DBP MAE is {win_mae[0]:.2f}/{win_mae[1]:.2f} mmHg, versus {base_mae[0]:.2f}/{base_mae[1]:.2f} for the unchanged CNN. The four-model attention ranking is reported separately below; neither ranking uses test accuracy.\n\n"
    final += "\n\n".join(tex(x) for x in observations)
    final += "\n\n" + extra_notes
    final += "\n\n" + attention_interpretation
    final += "\n\n" + personal_interpretation
    for name, caption in (("attention_results", "Population comparison: patient-macro errors; bars show across-seed SD, absent for single-seed modality ablations."),
        ("personal_budget", "Calibration-budget curves on common later queries. Every curve averages six patients equally and three seeds; reference age changes with nested budget."),
        ("personal_drift", "Within-case error versus minutes since the actual final reference at K=10. Patient counts vary by interval; complete counts are saved with the results."),
        ("patient_changes", "Every final patient's paired change at K=10. Negative values improve on the matched population model; heterogeneous failures must not be hidden."),
        ("temporal_examples", "Representative upper-median and worst patients under baseline seed 42 query MAE, shown at K=10. Lines break at recording gaps. Bias and full-model alternatives were prespecified; no post-hoc time alignment is applied."),
        ("architecture", "The unchanged high-resolution backbone and final-block intervention. Shapes omit batch size. Inputs are observed, contemporaneous ECG/PPG; outputs are scalar pressures."),
        ("personal_task", "Personal calibration and later contemporaneous inference. Only early invasive-ABP-derived scalar references fit the personal parameters. Later observed ECG/PPG enters the estimator; later blood-pressure labels enter only the evaluator. The 20\\% split uses elapsed case span and preserves recording gaps."),
        ("drift_gain", "Paired patient-macro MAE change versus the population model in each time interval, at K=10. Negative values favor calibration. Changing patient coverage is shown in the appendix."),
        ("repeated_bland_altman", "Descriptive repeated-measurement Bland--Altman plots for seed 42, K=10. Colors identify the six patients. Every fourth window is displayed; all windows determine the bias and random-intercept limits of agreement. Serial dependence remains, so these are not confidence bounds or clinical validation limits."),
        ("observed_waveforms", "Actual normalized ten-second ECG/PPG input windows from the prespecified example patients. These are observed signals, not predicted future waveforms. Corresponding scalar reference and prediction values are tabulated in the appendix.")):
        final += f"\n\\begin{{figure*}}[t]\\centering\\includegraphics[width=\\textwidth]{{figures/{name}.pdf}}\\caption{{{caption}}}\\end{{figure*}}\n"
    final += table_tex
    final += r"""
\section{Computational cost and engineering verification}
\begin{table*}[t]\centering\small
\begin{tabular}{lrrrr}\toprule
Variant & Parameters & Actual epochs & Selected epochs & Total training minutes\\\midrule
""" + "\n".join(costs) + r"""
\bottomrule\end{tabular}
\caption{Real fitting costs across included seeds. Training time includes per-epoch validation and checkpointing. Complete CPU/accelerator inference timings and memory definitions are in the result JSON.}\end{table*}
The recorded environment is an Apple M5 with 24 GiB memory and PyTorch 2.14.0.
Population fitting uses MPS; personal adaptation uses CPU. Measured desktop
costs are not Raspberry Pi timings. Personal CPU execution overlapped later
MPS population fits on the same workstation; the saved timing ranges therefore
include shared-host workload variation, rather than isolated-hardware benchmarks.
Full source waveform hashes and patient
selection hashes are verified. Tests cover V1 equality, attention shapes and
counts, split separation, label/time alignment, support-only fitting, exact
hand-computed metrics, frozen BN and parameters, and epoch-resume equality.
No toy-test outputs enter the real inventory. The development and final
personal phases require 1,068 and 684 completed units respectively.

\section{Interpretation and limitations}
The controlled attention intervention tests mechanisms, placement, modality
availability and matched parameter count; equal capacity does not guarantee
equal optimization. A validation-selected architecture can still have mixed
test outcomes, and intervals spanning zero do not support a stable gain.
The per-patient table and calibration curves, not a single best number,
determine whether deeper adaptation consistently helps. A neural strategy
that fails to outperform reference-only controls adds complexity without
demonstrated incremental predictive value on this pilot. High-pressure and
low-pressure strata can reveal failures concealed by overall averages.

Six personal targets cannot establish a universal calibration budget or
durability time. The references are real invasive labels sampled to simulate
sparse calibration, whereas intended ND measurements will be cuffs. Sensor,
population and reference differences remain separate transfer questions.
Provided offline filtering, upstream reference-based quality selection and
an incompletely documented resampling kernel constrain online interpretation.
The study evaluates later known ECG/PPG, not future waveforms. It does not
test across visits or establish clinical certification. Exact negative,
inconsistent and positive effects are retained in saved per-patient outputs.

\section{Reproduction and artifact traceability}
The review/protocol gate is commit \texttt{1b93cf1}. The repository preserves V1 and uses
a dedicated research branch. Follow the README commands with Python 3.12 and
the pinned requirements. Run unit tests, the population stage, then personal
development and test; build this report only after the full real inventory
passes hash checks. Raw recordings, detailed predictions and model weights
remain outside Git under the documented artifact root. Committed JSON
summaries retain run/config/checkpoint hashes and per-patient metrics.
"""
    final += r"\clearpage\appendix\section{Complete paired, pressure-range and budget results}" + "\n"
    final += extra_tables + "\n" + personal_tables + "\n" + extended_tables + "\n" + failure_context
    final += r"\clearpage\bibliography{references}\end{document}" + "\n"
    (out / "final_report.tex").write_text(final)
    # Bibliographies have distinct keys; preserve verified source entries.
    (out / "references.bib").write_text((V2 / "attention_cnn/references.bib").read_text() + "\n" +
                                       (V2 / "personalized_forecasting/references.bib").read_text())
    print(json.dumps({"status": "generated_from_complete_inventory", "validation_selected": winner,
                      "attention_runs": len(aruns), "personal_test_units": len(units),
                      "report": str(out / "final_report.tex")}, indent=2))


if __name__ == "__main__":
    main()
