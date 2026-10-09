"""Additional result-derived scientific tables and diagnostics.

These functions describe fixed experiments; they never select training settings.
The only architecture choice is supplied by the validation-based report caller.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from shared.data import load_personal_cohort

SEEDS = (42, 43, 44)
DISPLAY = {"baseline": "CNN", "se_last": "SE-last", "temporal_last": "Temporal",
           "cbam_last": "CBAM1D", "additive_last": "Additive", "se_all": "SE-all",
           "baseline_ecg": "ECG only", "baseline_ppg": "PPG only", "eca_last": "ECA",
           "population": "Population", "bias": "Bias", "affine": "Affine", "head": "Head",
           "last_block": "Last block", "full": "Full", "last_reference": "Last reference",
           "calibration_mean": "Reference mean"}


def fmt(value):
    return "--" if value is None or not np.isfinite(value) else f"{value:.2f}"


def spread(values):
    values = np.asarray(values, float)
    return f"{values.mean():.2f} $\\pm$ {values.std(ddof=1):.2f}" if len(values) > 1 else fmt(values[0])


def insert_gap_breaks(x, y, starts, ends, tolerance=1e-6):
    """Break plotted lines across missing recording intervals; never interpolate."""
    x, y, starts, ends = [np.asarray(a, dtype=float) for a in (x, y, starts, ends)]
    if x.ndim != 1 or any(a.shape != x.shape for a in (y, starts, ends)):
        raise ValueError("Plot coordinates and observation intervals must align")
    if not all(np.isfinite(a).all() for a in (x, y, starts, ends)):
        raise ValueError("Only finite observations can define gap breaks")
    if np.any(ends <= starts) or np.any(np.diff(starts) < 0):
        raise ValueError("Observation intervals must be positive and chronological")
    breaks = np.flatnonzero(starts[1:] > ends[:-1] + tolerance) + 1
    return np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan)


def table(headers, rows, caption, size="small"):
    return "\n".join([f"\\begin{{table*}}[t]\\centering\\{size}",
        "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}\\toprule",
        " & ".join(headers) + r"\\\midrule", *[" & ".join(row) + r"\\" for row in rows],
        r"\bottomrule\end{tabular}", f"\\caption{{{caption}}}\\end{{table*}}"])


def architecture(out):
    fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    ax.set_axis_off()
    boxes = [(.08, .73, "ECG + PPG\n2 x 1250"), (.29, .73, "Conv7 / BN / SiLU\n8 x 625"),
             (.50, .73, "Residual block\n16 x 313"), (.71, .73, "Residual block\n32 x 157"),
             (.92, .73, "Residual block\n48 x 79"), (.92, .20, "Gate / control\n48 x 79"),
             (.66, .20, "Global mean\n48 features"), (.40, .20, "Dropout + MLP\n48 > 32 > 2"),
             (.14, .20, "Denormalize\nSBP / DBP (mmHg)")]
    for i, (x, y, label) in enumerate(boxes):
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=.5", "facecolor": "#edf3f7", "edgecolor": "#38536a"})
        if i:
            px, py, _ = boxes[i - 1]
            dx, dy = x - px, y - py
            a = (px + .095 * np.sign(dx), py + .12 * np.sign(dy))
            b = (x - .095 * np.sign(dx), y - .12 * np.sign(dy))
            ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#38536a"})
    ax.text(.08, -.02, "SE-all adds gates after each residual block; the baseline has no gate.",
            fontsize=9, transform=ax.transAxes)
    ax.set_xlim(-.035, 1.035)
    ax.set_ylim(-.02, 1)
    fig.savefig(out / "figures/architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def scientific_details(out, variants, aruns, apred, apaired, bs, bp, bd, bc,
                       individual_root, examples, validation_winner, *, expected_manifest_hash):
    tables, notes = [], []
    # Seed variability is different from the patient bootstrap below.
    rows = []
    for v in variants:
        seeds = (42,) if v.startswith("baseline_") else SEEDS
        vals = [aruns[f"{v}_seed{s}"]["metrics"]["test"] for s in seeds]
        rows.append([DISPLAY[v]] + [spread([x["macro"][m][q] for x in vals])
                    for m in ("mae", "rmse") for q in range(2)])
    tables.append(table(["Model", "SBP MAE", "DBP MAE", "SBP RMSE", "DBP RMSE"], rows,
        "Primary-test patient-macro errors (mmHg), mean and sample SD across training seeds. "
        "The modality ablations have one seed. These SDs describe training variability, not patient uncertainty."))
    for metric in ("mae", "rmse"):
        rows = []
        for v, pair in apaired.items():
            result = pair[metric]
            rows.append([DISPLAY[v]] + [f"{fmt(result['estimate'][q])} [{fmt(result['ci_lower'][q])}, "
                        f"{fmt(result['ci_upper'][q])}]" for q in range(2)])
        tables.append(table(["Model minus CNN", "SBP change [95\\% CI]", "DBP change [95\\% CI]"], rows,
            f"Patient-paired {metric.upper()} changes (mmHg) on the primary test. Negative values favor the intervention. "
            "Seeds are averaged within patient before resampling 141 patients; no window-level bootstrap is used."))
    # Capacity and placement differences are descriptive matched-seed contrasts.
    for left, right, concept in (("se_last", "additive_last", "multiplicative attention versus matched additive capacity"),
                                 ("se_all", "se_last", "attention placement")):
        delta = np.mean([np.asarray(aruns[f"{left}_seed{s}"]["metrics"]["test"]["macro"]["mae"]) -
                         np.asarray(aruns[f"{right}_seed{s}"]["metrics"]["test"]["macro"]["mae"]) for s in SEEDS], 0)
        notes.append(f"For {concept}, {DISPLAY[left]} minus {DISPLAY[right]} gives "
                     f"{delta[0]:.2f}/{delta[1]:.2f} mmHg SBP/DBP patient-macro MAE. "
                     "This is an architectural contrast under the fixed optimizer, not evidence that all possible tuning would rank them similarly.")
    # Pressure-range sample counts are fixed by labels and never selected by errors.
    for target, q in (("SBP", 0), ("DBP", 1)):
        rows = []
        for v in dict.fromkeys(("baseline", validation_winner)):
            metrics = [aruns[f"{v}_seed{s}"]["metrics"]["test"] for s in SEEDS]
            for index in range(3):
                r = [m["pressure_ranges"][target][index] for m in metrics]
                rows.append([DISPLAY[v] + ": " + r[0]["range"].replace("_", " "),
                             str(r[0]["n_windows"]), str(r[0]["n_patients"])] +
                            [fmt(np.mean([x[m] for x in r])) if r[0]["n_windows"] else "--" for m in ("mae", "rmse", "bias")])
        tables.append(table([target + " stratum", "Windows", "Patients", "MAE", "RMSE", "Bias"], rows,
            f"Primary-test {target} pressure-range errors (mmHg), averaged across seeds. "
            "Stratum errors are pooled within each label-defined range; counts are not multiplied by seeds. "
            "Only the unchanged baseline and validation-selected architecture are shown; all models are in the JSON."))
    for target in ("SBP", "DBP"):
        rows = []
        for method in ("population", "bias", "head", "last_block", "full", "calibration_mean"):
            for index in range(3):
                r = [bs[(method, 10, s)]["pressure_ranges"][target][index] for s in SEEDS]
                rows.append([DISPLAY[method] + ": " + r[0]["range"].replace("_", " "),
                             str(r[0]["n_windows"]), str(r[0]["n_patients"])] +
                            [fmt(np.mean([x[m] for x in r])) if r[0]["n_windows"] else "--" for m in ("mae", "rmse")])
        tables.append(table([target + " stratum, K=10", "Windows", "Patients", "MAE", "RMSE"], rows,
            f"Personal-query {target} pressure-range errors (mmHg), averaged across seeds. "
            "Empty reference-pressure ranges are unavailable, not perfect performance. More than one range may contain the same patient.", "scriptsize"))
    # Count and gain versus time, without confusing the common boundary with K-specific reference age.
    rows = []
    for budget in (1, 5, 10):
        for j, label in enumerate(("0--15", "15--30", "30--60", "60+")):
            base = [bd[("population", budget, s)][j] for s in SEEDS]
            row = [f"K={budget}, {label} min", str(base[0]["n_windows"]), str(base[0]["n_patients"])]
            for method in ("population", "bias", "full"):
                vals = [bd[(method, budget, s)][j]["metrics"] for s in SEEDS]
                row += [fmt(np.mean([x["macro"]["mae"][q] for x in vals])) if vals[0] else "--" for q in range(2)]
            rows.append(row)
    tables.append(table(["Age from last reference", "Windows", "Patients", "Pop. S", "Pop. D", "Bias S", "Bias D", "Full S", "Full D"], rows,
        "Drift-bin coverage and patient-macro MAE (mmHg) at three budgets. S/D mean SBP/DBP. "
        "Counts are unique windows/patients, not multiplied by seeds. Differences across intervals also reflect which patients remain observed.", "scriptsize"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for method in ("bias", "head", "last_block", "full", "calibration_mean"):
        for q, target in enumerate(("SBP", "DBP")):
            changes = []
            for j in range(4):
                a = [bd[(method, 10, s)][j]["metrics"] for s in SEEDS]
                b = [bd[("population", 10, s)][j]["metrics"] for s in SEEDS]
                changes.append(np.mean([x["macro"]["mae"][q] - y["macro"]["mae"][q] for x, y in zip(a, b)]) if a[0] else np.nan)
            axes[q].plot(range(4), changes, "o-", label=DISPLAY[method])
            axes[q].axhline(0, c="gray", lw=.7)
            axes[q].set_xticks(range(4), ["0-15", "15-30", "30-60", "60+"])
            axes[q].set_xlabel("Minutes since final reference (K=10)")
            axes[q].set_ylabel(f"{target} MAE change (mmHg)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(out / "figures/drift_gain.pdf", bbox_inches="tight")
    plt.close(fig)
    # Repeated-measurement limits accompany errors, rather than claiming each window is independent.
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True,
                             sharex="row", sharey="row")
    for col, method in enumerate(("population", "bias", "full")):
        p = bp[(method, 10, 42)]
        patients = sorted(np.unique(p["subjects"]))
        ba = bs[(method, 10, 42)]["repeated_measurement_bland_altman"]
        for q, target in enumerate(("SBP", "DBP")):
            for patient in patients:
                mask = np.flatnonzero(p["subjects"] == patient)[::4]
                axes[q, col].scatter((p["pred"][mask, q] + p["y"][mask, q]) / 2,
                                    p["pred"][mask, q] - p["y"][mask, q], s=3, alpha=.35,
                                    label=patient.split(":")[-1])
            for key, style in (("pooled_bias", "-"), ("loa_lower", "--"), ("loa_upper", "--")):
                axes[q, col].axhline(ba[key][q], color="black", ls=style, lw=.8)
            axes[q, col].set_title(f"{DISPLAY[method]}: {target}")
            axes[q, col].set_xlabel("Mean of estimate/reference (mmHg)")
            axes[q, col].set_ylabel("Estimate - reference (mmHg)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=6,
               frameon=False, markerscale=3)
    fig.savefig(out / "figures/repeated_bland_altman.pdf", bbox_inches="tight")
    plt.close(fig)
    # Actual observed input waveforms: first common query of the prespecified examples.
    cohort = load_personal_cohort(individual_root)
    if cohort["manifest_hash"] != expected_manifest_hash:
        raise ValueError("Observed-waveform source differs from the verified personal experiment")
    cohort_rows = []
    for role in ("development", "target"):
        for p in cohort[role]:
            cohort_rows.append([p["patient_id"].split(":")[-1] + " (" + ("Dev" if role == "development" else "Test") + ")",
                                str(p["case_id"]), str(len(p["x"])),
                                fmt(len(p["x"]) / 6), fmt((p["end_exclusive"][-1] - p["start"][0]) / 60),
                                str(len(p["query_indices"])),
                                fmt((p["calibration_boundary"] - p["start"][0]) / 60)])
    tables.append(table(["Patient (role)", "Case", "Windows", "Observed min", "Elapsed min", "Queries", "Early span min"],
        cohort_rows, "Chronological personal cohort. Observed duration counts retained ten-second windows; "
        "elapsed duration includes gaps. Every patient has ten available spaced reference windows; K selects a nested prefix. "
        "Queries are identical across all methods and budgets, and all ten patients were excluded from population fitting.", "scriptsize"))
    patients = {p["patient_id"]: p for p in cohort["target"]}
    example_metadata = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), constrained_layout=True)
    for row, patient in enumerate(examples):
        data = patients[patient]
        source = int(bp[("population", 10, 42)]["source_indices"][bp[("population", 10, 42)]["subjects"] == patient][0])
        index = int(np.flatnonzero(data["source_indices"] == source)[0])
        p = bp[("population", 10, 42)]
        prediction_index = int(np.flatnonzero((p["subjects"] == patient) & (p["source_indices"] == source))[0])
        if not np.array_equal(data["y"][index], p["y"][prediction_index]):
            raise ValueError("Example waveform/label alignment mismatch")
        for channel, name in enumerate(("ECG", "PPG")):
            axes[row, channel].plot(np.arange(1250) / 125, data["x"][index, channel], lw=.75)
            axes[row, channel].set_title(f"{patient.split(':')[-1]}: observed {name}")
            axes[row, channel].set_xlabel("Seconds in observed window")
            axes[row, channel].set_ylabel("Per-window min-max amplitude")
        label = p["y"][prediction_index].tolist()
        predictions = {m: bp[(m, 10, 42)]["pred"][prediction_index].tolist() for m in ("population", "bias", "full")}
        example_metadata.append({"patient": patient, "source_index": source, "start": float(data["start"][index]),
                                 "reference_sbp_dbp": label, "predictions_sbp_dbp": predictions})
    fig.savefig(out / "figures/observed_waveforms.pdf", bbox_inches="tight")
    plt.close(fig)
    (out / "results/example_windows.json").write_text(json.dumps(example_metadata, indent=2) + "\n")
    rows = [[x["patient"].split(":")[-1], str(x["source_index"]),
             "/".join(fmt(v) for v in x["reference_sbp_dbp"])] +
            ["/".join(fmt(v) for v in x["predictions_sbp_dbp"][m]) for m in ("population", "bias", "full")]
            for x in example_metadata]
    tables.append(table(["Patient", "Source row", "Reference S/D", "Population", "Bias", "Full"], rows,
        "Scalar SBP/DBP values (mmHg) paired with the observed-waveform figure. These are the first later query windows "
        "of the upper-median and worst patients under seed-42 population MAE, not windows selected for a favorable prediction."))
    architecture(out)
    return "\n\n".join(tables), "\n\n".join(notes)
