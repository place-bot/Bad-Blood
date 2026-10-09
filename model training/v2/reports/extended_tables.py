"""Compact ACL tables from verified experimental summaries only.

No files are read or written and no strategy is selected here. Call only after
the report inventory verifier has bound the input summaries to real artifacts.
"""
from __future__ import annotations

import numpy as np

SEEDS = (42, 43, 44)
BUDGETS = (1, 2, 3, 5, 10)
METHODS = ("population", "bias", "affine", "head", "last_block", "full", "last_reference", "calibration_mean")
NEURAL_METHODS = ("head", "last_block", "full")
DISPLAY = {"population": "Population", "bias": "Bias", "affine": "Affine", "head": "Head",
           "last_block": "Last block", "full": "Full", "last_reference": "Last reference",
           "calibration_mean": "Reference mean", "baseline": "CNN", "se_last": "SE-last",
           "temporal_last": "Temporal", "cbam_last": "CBAM1D", "additive_last": "Additive",
           "se_all": "SE-all", "baseline_ecg": "ECG only", "baseline_ppg": "PPG only", "eca_last": "ECA"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _table(headers, rows, caption, *, size="scriptsize"):
    _require(all(len(row) == len(headers) for row in rows), "Table row/header column mismatch")
    return "\n".join([
        f"\\begin{{table*}}[t]\\centering\\{size}\\setlength{{\\tabcolsep}}{{3pt}}",
        "\\begin{tabular}{l" + "r" * (len(headers) - 1) + r"}\toprule",
        " & ".join(headers) + r"\\\midrule",
        *[" & ".join(row) + r"\\" for row in rows],
        r"\bottomrule\end{tabular}", f"\\caption{{{caption}}}\\end{{table*}}",
    ])


def paired_interval_cell(result, target):
    estimate = np.asarray(result["estimate"], dtype=float)
    lower = np.asarray(result["ci_lower"], dtype=float)
    upper = np.asarray(result["ci_upper"], dtype=float)
    _require(all(x.shape == (2,) and np.isfinite(x).all() for x in (estimate, lower, upper)),
             "Paired interval must contain finite SBP/DBP vectors")
    _require(result["n_patients"] == 6 and result["resamples"] == 2000,
             "Full-cohort personal intervals require six patients and 2,000 resamples")
    _require(np.all(lower <= upper), "Reversed paired interval")
    # Braces prevent the interval's '[' from becoming LaTeX's optional row skip.
    return (r"\shortstack{" + f"{estimate[target]:.2f}" + r"\\{" +
            f"[{lower[target]:.2f}, {upper[target]:.2f}]" + "}}")


def per_patient_seed_values(summaries, method, patient, metric, target):
    values = []
    for seed in SEEDS:
        row = summaries[(method, 10, seed)]["per_patient"][patient]
        value = row[metric][target]
        _require(value is not None and np.isfinite(value), "Missing mandatory absolute patient error")
        values.append(float(value))
    return np.asarray(values)


def selected_setting_cell(method, budget, selected):
    if method == "affine" and budget < 3:
        return "--"
    setting = selected[f"{method}/K{budget}"]
    if method == "affine":
        _require(set(setting) == {"penalty"} and setting["penalty"] in (1, 10, 100),
                 "Affine selection is outside the frozen grid")
        return f"$\\lambda={setting['penalty']}$"
    _require(method in NEURAL_METHODS and set(setting) == {"lr", "steps"}
             and setting["lr"] in (1e-4, 1e-3) and setting["steps"] in (20, 100),
             "Neural selection is outside the frozen grid")
    exponent = -4 if setting["lr"] == 1e-4 else -3
    return f"\\shortstack{{$10^{{{exponent}}}$\\\\{setting['steps']} steps}}"


def latency_run_statistics(aruns):
    """Summarize medians from separate measured runs, not pooled timing trials."""
    variants = sorted({run["config"]["variant"] for run in aruns.values()},
                      key=lambda value: list(DISPLAY).index(value))
    output = {}
    for variant in variants:
        expected_seeds = (42,) if variant in ("baseline_ecg", "baseline_ppg") else SEEDS
        runs = [run for run in aruns.values() if run["config"]["variant"] == variant]
        _require(sorted(run["config"]["seed"] for run in runs) == list(expected_seeds),
                 f"Latency summary needs every matched seed exactly once: {variant}")
        output[variant] = {"n_runs": len(runs), "measurements": {}}
        for device in ("cpu", "mps"):
            for batch in (1, 256):
                values = []
                for run in runs:
                    matches = [row for row in run["costs"]["latency"]
                               if row["device"] == device and row["batch_size"] == batch]
                    _require(len(matches) <= 1, f"Duplicated timing condition: {variant}/{device}/{batch}")
                    if matches:
                        row = matches[0]
                        value = row["median_batch_ms"]
                        _require(row["repetitions"] == 20 and np.isfinite(value) and value > 0,
                                 "Latency measurement must have 20 finite positive measured repetitions")
                        _require(np.isclose(row["median_per_example_ms"], value / batch, rtol=1e-9, atol=1e-9),
                                 "Batch/per-example timing units disagree")
                        values.append(float(value))
                _require(not values or len(values) == len(runs), "A latency condition is missing for some seeds")
                _require(device != "cpu" or bool(values), "Every run must include CPU latency")
                _require(device != "mps" or values or all(run["config"]["device"] != "mps" for run in runs),
                         "An MPS run is missing its measured MPS latency")
                output[variant]["measurements"][(device, batch)] = (
                    {"median": float(np.median(values)), "minimum": min(values), "maximum": max(values)}
                    if values else None)
    return output


def build_extended_tables(aruns, bs, bc, selection):
    """Return all-budget CIs, absolute patient errors, actual settings and costs."""
    tables = []
    for metric in ("mae", "rmse"):
        for q, target in enumerate(("SBP", "DBP")):
            rows = []
            for method in METHODS[1:]:
                cells = []
                for budget in BUDGETS:
                    cells.append("--" if method == "affine" and budget < 3 else
                                 paired_interval_cell(bc[(method, budget)][metric], q))
                rows.append([DISPLAY[method]] + cells)
            tables.append(_table(["Method minus population"] + [f"K={k}" for k in BUDGETS], rows,
                f"All-budget paired {target} {metric.upper()} changes in mmHg: estimate above descriptive 95\\% patient-bootstrap interval. "
                "For each patient, differences are averaged over the three matched seeds before resampling the six patients 2,000 times. "
                "Negative changes favor the method. These are not seed-SD intervals or simultaneous multiplicity-adjusted intervals. "
                "Affine calibration is unavailable below three references."))
    patients = sorted(bs[("population", 10, 42)]["per_patient"])
    _require(len(patients) == 6, "Absolute patient tables require all six final targets")
    for method in METHODS:
        for seed in SEEDS:
            _require(sorted(bs[(method, 10, seed)]["per_patient"]) == patients, "Personal tables require identical patient membership")
            for patient in patients:
                _require(bs[(method, 10, seed)]["per_patient"][patient]["n_windows"] ==
                         bs[("population", 10, seed)]["per_patient"][patient]["n_windows"],
                         "Absolute patient table query counts differ")
    abbreviated = ["Population", "Bias", "Affine", "Head", r"\shortstack{Last\\block}",
                   "Full", r"\shortstack{Last\\reference}", r"\shortstack{Reference\\mean}"]
    for metric in ("mae", "rmse"):
        for q, target in enumerate(("SBP", "DBP")):
            rows = []
            for patient in patients:
                cells = []
                for method in METHODS:
                    values = per_patient_seed_values(bs, method, patient, metric, q)
                    cells.append(f"\\shortstack{{{values.mean():.2f}\\\\({values.std(ddof=1):.2f})}}")
                rows.append([_escape(patient.split(":")[-1])] + cells)
            tables.append(_table(["Patient"] + abbreviated, rows,
                f"Absolute {target} {metric.upper()} in mmHg for every final patient at K=10. "
                "Cells show the mean across the three population seeds, with sample seed SD in parentheses; each cell describes one patient, not a patient confidence interval. "
                "The same later queries are used for all methods. Absolute errors expose residual failures that an improvement-only heatmap can conceal."))
    selected = selection["selected"]
    expected = {f"{method}/K{budget}" for method in (*NEURAL_METHODS, "affine") for budget in BUDGETS
                if method != "affine" or budget >= 3}
    _require(set(selected) == expected, "Selected adaptation-setting inventory is incomplete")
    rows = [[DISPLAY[method]] + [selected_setting_cell(method, k, selected) for k in BUDGETS]
            for method in (*NEURAL_METHODS, "affine")]
    tables.append(_table(["Strategy"] + [f"K={k}" for k in BUDGETS], rows,
        "Actual frozen development-selected settings. Neural cells show learning rate above update steps; affine cells show the slope-shrinkage penalty. "
        "One setting per strategy and budget was chosen using four separate development patients averaged across three seeds, and then used unchanged for every final target. "
        "No final-target query determined these settings."))
    timing = latency_run_statistics(aruns)
    rows = []
    for variant, data in timing.items():
        cells = []
        for condition in (("cpu", 1), ("cpu", 256), ("mps", 1), ("mps", 256)):
            stats = data["measurements"][condition]
            if stats is None:
                cells.append("--")
            elif data["n_runs"] == 1:
                cells.append(f"{stats['median']:.3f}")
            else:
                cells.append(r"\shortstack{" + f"{stats['median']:.3f}" + r"\\{" +
                             f"[{stats['minimum']:.3f}, {stats['maximum']:.3f}]" + "}}")
        rows.append([DISPLAY[variant] + f" ({data['n_runs']})"] + cells)
    tables.append(_table(["Model (runs)", "CPU, B=1", "CPU, B=256", "MPS, B=1", "MPS, B=256"], rows,
        "Measured warm forward-pass batch latency in milliseconds. Each run times 20 repetitions after five warmups with device synchronization. "
        "Cells show the median of run-specific medians, with the range across runs below; single-seed ablations show their measured median. "
        "B=256 entries are whole-batch times, not per-window latency (divide by 256 for amortized time per example). "
        "Inputs already reside on the measured device; these exclude signal acquisition, preprocessing and input/output transfers, and are not Raspberry Pi timings. "
        "Run-to-run timing spread is measurement variability, not an accuracy confidence interval; unavailable devices are marked --."))
    return "\n\n".join(tables)
