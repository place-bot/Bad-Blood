"""Descriptive plots of complete, already-verified population fit histories.

Call after ``build_report.aggregate_a`` verifies the full real inventory. This
module reads no artifacts itself, never accesses test metrics, and neither
selects new checkpoints nor diagnoses a cause for the recorded curve shapes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

PRIMARY = ("baseline", "se_last", "temporal_last", "cbam_last")
REQUIRED_THREE_SEED = (*PRIMARY, "additive_last", "se_all")
SINGLE_SEED = ("baseline_ecg", "baseline_ppg")
SEEDS = (42, 43, 44)
DISPLAY = {"baseline": "CNN", "se_last": "SE-last", "temporal_last": "Temporal",
           "cbam_last": "CBAM1D", "additive_last": "Additive", "se_all": "SE-all",
           "baseline_ecg": "ECG only", "baseline_ppg": "PPG only", "eca_last": "ECA"}
COLORS = {42: "#0072B2", 43: "#D55E00", 44: "#009E73"}
STYLES = {42: "-", 43: "--", 44: ":"}
FIGURES = {"training": "attention_training_mse.pdf", "validation": "attention_validation_mae.pdf"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_nonnegative(value, name):
    _require(not isinstance(value, (bool, str)) and np.isscalar(value)
             and np.isfinite(value) and value >= 0, f"Invalid {name}")


def _canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def summarize_training_histories(aruns):
    """Validate completeness before plotting and return copied, JSON-safe evidence.

    This is a defensive summary contract, not a replacement for upstream raw
    artifact/hash verification. Optional ECA is accepted only as three complete
    runs and retained in JSON, never substituted into the four primary panels.
    """
    variants = (*REQUIRED_THREE_SEED, *SINGLE_SEED)
    if any(key.startswith("eca_last_seed") for key in aruns):
        variants += ("eca_last",)
    expected = {f"{variant}_seed{seed}" for variant in variants
                for seed in ((42,) if variant in SINGLE_SEED else SEEDS)}
    _require(set(aruns) == expected, "Training diagnostics require all 20 complete required fits and no partial optional inventory")
    records = {}
    for key in sorted(expected):
        run = aruns[key]
        config, completion, history = run["config"], run["completion"], run["history"]
        variant, seed = config["variant"], config["seed"]
        _require(key == f"{variant}_seed{seed}", "Run identity differs from its inventory key")
        _require(completion["variant"] == variant and completion["seed"] == seed,
                 "Completion identity differs from its configured run")
        _require(completion.get("status") == "complete" and completion.get("toy_test_only") is False
                 and config.get("toy_test_only") is False, "Only completed real fits can enter training diagnostics")
        actual, selected = completion["actual_epochs"], completion["selected_epoch"]
        _require(isinstance(actual, int) and not isinstance(actual, bool) and 1 <= actual <= 40
                 and actual == completion["epochs_completed"] == len(history), "Actual epoch count differs from complete history")
        _require(isinstance(selected, int) and not isinstance(selected, bool)
                 and 1 <= selected <= actual and selected == completion["best_epoch"], "Invalid selected epoch")
        _require(all(isinstance(row["epoch"], int) and not isinstance(row["epoch"], bool) for row in history)
                 and [row["epoch"] for row in history] == list(range(1, actual + 1)),
                 "History has missing, reordered or duplicated epochs")
        for row in history:
            _finite_nonnegative(row["train_normalized_MSE"], "training normalized MSE")
            _finite_nonnegative(row["selection_macro_mae"], "validation selection MAE")
            _finite_nonnegative(row["seconds"], "epoch seconds")
            _finite_nonnegative(row["lr"], "learning rate")
            _require(row["lr"] > 0, "Learning rate must remain positive")
            target_mae = np.asarray(row["validation"]["macro_mae"], float)
            _require(target_mae.shape == (2,) and np.isfinite(target_mae).all() and np.all(target_mae >= 0)
                     and np.isclose(target_mae.mean(), row["selection_macro_mae"], rtol=0, atol=1e-10),
                     "Validation scalar is not mean patient-macro SBP/DBP MAE")
        # The trainer keeps the earliest strict minimum, including exact ties.
        best_row = min(history, key=lambda row: row["selection_macro_mae"])
        _require(best_row["epoch"] == selected, "Selected marker is not the saved first validation minimum")
        _require(np.isclose(best_row["selection_macro_mae"], completion["best_validation_macro_mae"],
                            rtol=0, atol=1e-10), "Completion and selected-history validation score disagree")
        copied = copy.deepcopy(history)
        records[key] = {
            "variant": variant, "seed": seed, "shown_in_primary_figures": variant in PRIMARY,
            "actual_epochs": actual, "selected_epoch": selected,
            "early_stopped": completion["early_stopped"],
            "selected_validation_mean_macro_mae_mmhg": float(best_row["selection_macro_mae"]),
            "training_normalized_mse_at_selected_epoch": float(best_row["train_normalized_MSE"]),
            "source_history_relative_path": f"attention/{key}/history.json",
            "canonical_history_sha256": _canonical_digest(copied),
            "history": copied,
            "config_hash": completion["config_hash"], "manifest_hash": completion["manifest_hash"],
            "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
        }
    return {
        "required_completed_runs": 20, "total_completed_runs": len(records),
        "supplementary_eca_completed": "eca_last" in variants,
        "figure_variant_order": list(PRIMARY), "figure_seed_order": list(SEEDS),
        "figures": {kind: f"figures/{name}" for kind, name in FIGURES.items()},
        "source_scope": "Already-verified saved per-epoch training and validation histories only; no test metrics are accessed",
        "training_objective": "Dimensionless MSE of train-standardized SBP/DBP, averaged over minibatches during parameter updates with augmentation and training-mode layers",
        "validation_objective": "Mean of SBP and DBP equal-patient macro MAE in mmHg, evaluated after each epoch",
        "selected_marker": "Saved earliest strict minimum of validation selection_macro_mae; not a new selection",
        "history_digest_definition": "SHA-256 of canonical JSON history values, not the source history.json byte hash",
        "plot_policy": "One curve per saved seed; straight connections between recorded epochs only, no smoothing, extrapolation, seed averaging or causal diagnosis",
        "interpretation_limit": "Different objectives, scales and evaluation conditions; not a directly comparable train-validation generalization gap or proof of causal overfitting",
        "runs": records,
    }


def _draw_history_figure(details, kind):
    field = "train_normalized_MSE" if kind == "training" else "selection_macro_mae"
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.2), sharex=True, sharey=True,
                             constrained_layout=True)
    all_values = []
    max_epoch = max(details["runs"][f"{variant}_seed{seed}"]["actual_epochs"]
                    for variant in PRIMARY for seed in SEEDS)
    for panel, (axis, variant) in enumerate(zip(axes.flat, PRIMARY)):
        for seed in SEEDS:
            record = details["runs"][f"{variant}_seed{seed}"]
            epochs = [row["epoch"] for row in record["history"]]
            values = [row[field] for row in record["history"]]
            all_values.extend(values)
            axis.plot(epochs, values, color=COLORS[seed], linestyle=STYLES[seed],
                      linewidth=1.4, label=f"Seed {seed}")
            if kind == "validation":
                selected = record["selected_epoch"]
                axis.plot(selected, values[selected - 1], marker="o", markersize=5,
                          markerfacecolor=COLORS[seed], markeredgecolor="black",
                          markeredgewidth=.65, linestyle="none", zorder=5)
        axis.set_title(DISPLAY[variant], fontsize=10, pad=5)
        axis.set_xlabel("Recorded epoch", fontsize=9)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.tick_params(labelsize=8)
        axis.tick_params(axis="x", labelbottom=True)
        axis.grid(axis="y", alpha=.2, linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
        if panel % 2 == 0:
            axis.set_ylabel("Training MSE\n(standardized targets)" if kind == "training" else
                            "Validation mean macro MAE\n(mmHg; SBP/DBP mean)", fontsize=9)
    # Common scales permit comparisons within a figure, not between objectives.
    xmin, xmax = (.5, 1.5) if max_epoch == 1 else (.75, max_epoch + .25)
    axes.flat[0].set_xlim(xmin, xmax)
    low, high = min(all_values), max(all_values)
    pad = max((high - low) * .08, high * .015, .01)
    axes.flat[0].set_ylim(max(0, low - pad), high + pad)
    handles = [Line2D([], [], color=COLORS[seed], linestyle=STYLES[seed],
                      linewidth=1.4, label=f"Seed {seed}") for seed in SEEDS]
    if kind == "validation":
        handles.append(Line2D([], [], color="black", marker="o", markerfacecolor="white",
                              linestyle="none", markersize=5, label="Selected epoch"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False, fontsize=9, handlelength=2.2, columnspacing=1.4)
    return fig


def build_training_diagnostics(out, aruns):
    """Write two primary-history PDFs; return short LaTeX and JSON-ready records.

    All validation happens before making a directory or writing a figure. Unit
    tests replace ``Figure.savefig`` so fixture data never produce real-looking
    research artifacts. Actual figures must be visually checked by the caller.
    """
    details = summarize_training_histories(aruns)
    destination = Path(out) / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "figure.dpi": 160}):
        for kind, name in FIGURES.items():
            fig = _draw_history_figure(details, kind)
            try:
                fig.savefig(destination / name, bbox_inches="tight")
            finally:
                plt.close(fig)
    counts = []
    for variant in PRIMARY:
        rows = [details["runs"][f"{variant}_seed{seed}"] for seed in SEEDS]
        actual = "/".join(str(row["actual_epochs"]) for row in rows)
        selected = "/".join(str(row["selected_epoch"]) for row in rows)
        counts.append(f"{DISPLAY[variant]} {actual} completed, {selected} selected")
    latex = (r"\paragraph{Recorded optimization histories.} "
        r"Figures~\ref{fig:attention_training_mse} and~\ref{fig:attention_validation_mae} show the four-model attention comparison's complete saved histories, one curve per seed, without smoothing or extrapolation. "
        "Training MSE is dimensionless, using train-standardized targets and minibatch losses recorded during parameter updates with augmentation and training-mode layers. "
        "Validation instead shows mean patient-macro SBP/DBP MAE in mmHg after each epoch; circles mark the already-saved validation-selected epochs. "
        "These different objectives and evaluation conditions are not a directly comparable training-validation gap. "
        "In seed order 42/43/44, the actual/selected epochs are: " + "; ".join(counts) + ". "
        "The plots describe the fixed runs; they do not establish a causal overfitting mechanism or motivate retuning. "
        "All twenty required fits are complete before these diagnostics are produced; optional ECA histories are retained only in the companion JSON.")
    return latex, details
