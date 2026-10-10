"""Outcome-specific interpretation of complete, already verified Track B results.

Ranks in this module are descriptive final-test observations, never deployment
choices or feedback to adaptation. No file access, fitting or default scores.
"""
from __future__ import annotations

import numpy as np

from shared.metrics import paired_patient_change

METHODS = ("population", "bias", "affine", "head", "last_block", "full", "last_reference", "calibration_mean")
NEURAL = ("head", "last_block", "full")
CONTROLS = ("bias", "last_reference", "calibration_mean")
BUDGETS, SEEDS = (1, 2, 3, 5, 10), (42, 43, 44)
TARGETS = ("SBP", "DBP")
DISPLAY = {"population": "Population", "bias": "Bias", "affine": "Affine", "head": "Head",
           "last_block": "Last block", "full": "Full", "last_reference": "Last reference",
           "calibration_mean": "Reference mean"}
BIN_BOUNDS = ((0, 15), (15, 30), (30, 60), (60, None))
TOLERANCE = 1e-9


def _tex(text):
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in str(text))


def _available(budget):
    return tuple(method for method in METHODS if method != "affine" or budget >= 3)


def _patient_values(rows, metric="mae"):
    patients = sorted(rows[0]["per_patient"])
    if any(sorted(row["per_patient"]) != patients for row in rows):
        raise ValueError("Patient membership differs across matched seeds")
    values = np.array([[row["per_patient"][p][metric] for p in patients] for row in rows], float)
    if values.shape != (3, len(patients), 2) or not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite patient metrics")
    return patients, values.mean(axis=0)


def _validate_summaries(bs):
    expected = {(method, budget, seed) for budget in BUDGETS for method in _available(budget) for seed in SEEDS}
    if set(bs) != expected:
        raise ValueError("Interpretation requires the exact complete fixed Track B summary matrix")
    reference = bs[("population", 1, 42)]
    patients = sorted(reference["per_patient"])
    if len(patients) != 6 or reference["n_patients"] != 6:
        raise ValueError("The frozen final cohort must contain six patients")
    for key, row in bs.items():
        if sorted(row["per_patient"]) != patients or row["n_patients"] != 6 or row["n_windows"] != reference["n_windows"]:
            raise ValueError(f"Final query coverage differs: {key}")
        if row["n_windows"] != sum(p["n_windows"] for p in row["per_patient"].values()) or any(
            p["n_windows"] <= 0 for p in row["per_patient"].values()):
            raise ValueError(f"Invalid final patient/window counts: {key}")
        for patient in patients:
            if row["per_patient"][patient]["n_windows"] != reference["per_patient"][patient]["n_windows"]:
                raise ValueError(f"Patient query coverage differs: {key}/{patient}")
        for metric in ("mae", "rmse"):
            patient_values = np.asarray([row["per_patient"][p][metric] for p in patients], float)
            macro = np.asarray(row["macro"][metric], float)
            if not np.isfinite(patient_values).all() or np.any(patient_values < 0) or not np.isfinite(macro).all() or not np.allclose(macro, patient_values.mean(0), atol=TOLERANCE, rtol=0):
                raise ValueError(f"Patient-macro metric mismatch: {key}/{metric}")
    return patients


def _check_baseline_pair(bs, bc, method, budget):
    provided = bc[(method, budget)]["mae"]
    patients, model = _patient_values([bs[(method, budget, seed)] for seed in SEEDS])
    _, baseline = _patient_values([bs[("population", budget, seed)] for seed in SEEDS])
    changes = model - baseline
    actual = np.asarray([provided["patient_changes"][p] for p in patients], float)
    if set(provided["patient_changes"]) != set(patients) or not np.allclose(actual, changes, atol=TOLERANCE, rtol=0):
        raise ValueError("Paired patient changes disagree with summary metrics")
    if not np.allclose(provided["estimate"], changes.mean(0), atol=TOLERANCE, rtol=0):
        raise ValueError("Paired estimate disagrees with patient-macro difference")
    lower, upper = np.asarray(provided["ci_lower"], float), np.asarray(provided["ci_upper"], float)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("Invalid paired descriptive interval")
    return provided


def _drift_entries(bd, method, seed):
    entries = [row for row in bd[(method, 10, seed)] if row["axis"] == "elapsed_since_reference_min"]
    mapping = {(row["lower"], row["upper"]): row for row in entries}
    if len(entries) != 4 or set(mapping) != set(BIN_BOUNDS):
        raise ValueError("Missing or duplicated primary reference-age bins")
    return mapping


def _drift_analysis(bd, full_reference):
    maps = {(method, seed): _drift_entries(bd, method, seed) for method in METHODS for seed in SEEDS}
    output = []
    for bounds in BIN_BOUNDS:
        reference = maps[("population", 42)][bounds]
        n_windows, n_patients = int(reference["n_windows"]), int(reference["n_patients"])
        if (n_windows == 0) != (n_patients == 0) or (reference["metrics"] is None) != (n_windows == 0):
            raise ValueError("Inconsistent empty-bin coverage")
        for method in METHODS:
            for seed in SEEDS:
                entry = maps[(method, seed)][bounds]
                if (entry["n_windows"], entry["n_patients"]) != (n_windows, n_patients):
                    raise ValueError("Drift comparison coverage differs across methods/seeds")
                if (entry["metrics"] is None) != (n_windows == 0):
                    raise ValueError("Drift comparison has inconsistent unavailable metrics")
                if n_windows:
                    metric = entry["metrics"]
                    if metric["n_windows"] != n_windows or metric["n_patients"] != n_patients:
                        raise ValueError("Drift metrics disagree with reported coverage")
                    ref_patients = reference["metrics"]["per_patient"]
                    if set(metric["per_patient"]) != set(ref_patients) or any(
                        metric["per_patient"][p]["n_windows"] != ref_patients[p]["n_windows"] for p in ref_patients):
                        raise ValueError("Drift patient-level pairing differs")
        paired = {}
        for method in METHODS:
            if method != "population":
                paired[method] = paired_patient_change(
                    [maps[(method, seed)][bounds]["metrics"] for seed in SEEDS],
                    [maps[("population", seed)][bounds]["metrics"] for seed in SEEDS], "mae") if n_windows else None
        output.append({"lower_minutes_inclusive": bounds[0], "upper_minutes_exclusive": bounds[1],
                       "n_windows": n_windows, "n_patients": n_patients,
                       "patient_ids": sorted(reference["metrics"]["per_patient"]) if n_windows else [],
                       "paired_mae_changes": paired, "n_seeds": 3})
    if sum(row["n_windows"] for row in output) != full_reference["n_windows"]:
        raise ValueError("Reference-age bins do not cover all final query windows")
    full_counts = {p: row["n_windows"] for p, row in full_reference["per_patient"].items()}
    binned_counts = {p: 0 for p in full_counts}
    for bounds in BIN_BOUNDS:
        metrics = maps[("population", 42)][bounds]["metrics"]
        if metrics is not None:
            for patient, row in metrics["per_patient"].items():
                if patient not in binned_counts:
                    raise ValueError("Drift bin contains a patient outside the final cohort")
                binned_counts[patient] += row["n_windows"]
    if binned_counts != full_counts:
        raise ValueError("Reference-age bins do not preserve per-patient query coverage")
    return output


def build_personal_interpretation(bs, bc, bd):
    """Return LaTeX and JSON-ready details only from a complete final matrix."""
    patients = _validate_summaries(bs)
    for budget in BUDGETS:
        for method in _available(budget):
            if method != "population":
                _check_baseline_pair(bs, bc, method, budget)
    details = {"ranking_role": "descriptive final-test ranking, not deployment selection",
               "difference_sign": "method minus comparator; negative MAE change is better",
               "uncertainty": "patient bootstrap after averaging matched seeds; descriptive, six-person pilot, no rank-selection correction",
               "budgets": {}, "k10_patient_counts": {}, "worst_full_adaptation_change": {},
               "budget_caveat": "Nested reference budgets jointly change count and recency; not pure sample complexity"}
    paragraphs = [r"\paragraph{Budget-specific outcomes.}",
        "All rankings below describe the fixed methods on final queries; they are not used to select a deployment model. "
        "Each MAE gives six patients equal weight and then averages three matched seeds. Differences are method minus comparator, "
        "so negative values favor the named neural method. Lowest neural denotes the smallest observed MAE among Head, Last block and Full, "
        "not a new tuned strategy; a tied neural rank uses the first of these prespecified names solely to display comparisons."]
    comparisons = {}
    for budget in BUDGETS:
        outcomes, sentences = {}, []
        for q, target in enumerate(TARGETS):
            scores = {method: float(np.mean([bs[(method, budget, seed)]["macro"]["mae"][q] for seed in SEEDS]))
                      for method in _available(budget)}
            minimum = min(scores.values())
            lowest = [method for method in _available(budget) if abs(scores[method] - minimum) <= TOLERANCE]
            neural_minimum = min(scores[method] for method in NEURAL)
            lowest_neural = [method for method in NEURAL if abs(scores[method] - neural_minimum) <= TOLERANCE]
            representative = lowest_neural[0]
            control_results = {}
            for control in CONTROLS:
                key = (representative, control, budget)
                if key not in comparisons:
                    comparisons[key] = paired_patient_change(
                        [bs[(representative, budget, seed)] for seed in SEEDS],
                        [bs[(control, budget, seed)] for seed in SEEDS], "mae")
                comparison = comparisons[key]
                control_results[control] = {"comparator_mae": scores[control],
                    "mae_change": comparison["estimate"][q], "ci_lower": comparison["ci_lower"][q],
                    "ci_upper": comparison["ci_upper"][q],
                    "patient_changes": {p: comparison["patient_changes"][p][q] for p in patients}}
            outcomes[target] = {"all_method_mae": scores, "lowest_observed_methods": lowest,
                                "lowest_observed_mae": minimum, "lowest_neural_methods": lowest_neural,
                                "representative_lowest_neural": representative, "lowest_neural_mae": neural_minimum,
                                "neural_vs_controls": control_results}
            deltas = "/".join(f"{control_results[control]['mae_change']:+.2f}" for control in CONTROLS)
            sentences.append(f"{target}: {' / '.join(DISPLAY[m] for m in lowest)} has the lowest observed MAE "
                f"({minimum:.2f} mmHg); the lowest neural MAE is {' / '.join(DISPLAY[m] for m in lowest_neural)} "
                f"({neural_minimum:.2f}), with changes {deltas} mmHg versus Bias/Last reference/Reference mean, respectively.")
        details["budgets"][str(budget)] = outcomes
        paragraphs.append(f"\\textbf{{K={budget}.}} " + " ".join(sentences))

    full_reference_key = ("full", "calibration_mean", 10)
    if full_reference_key not in comparisons:
        comparisons[full_reference_key] = paired_patient_change(
            [bs[("full", 10, seed)] for seed in SEEDS],
            [bs[("calibration_mean", 10, seed)] for seed in SEEDS], "mae")
    full_reference = comparisons[full_reference_key]
    details["k10_full_minus_reference_mean"] = full_reference
    full_reference_text = "; ".join(
        f"{target} {full_reference['estimate'][q]:+.3f} "
        f"[{full_reference['ci_lower'][q]:+.3f}, {full_reference['ci_upper'][q]:+.3f}]"
        for q, target in enumerate(TARGETS))
    paragraphs.append(f"At K=10, Full minus Reference mean patient-macro MAE is {full_reference_text} mmHg "
                      r"(descriptive paired 95\% patient intervals). Positive changes favor Reference mean. "
                      "Matched-seed differences are averaged within each of the six patients before resampling patients; "
                      "these intervals are not adjusted for multiple comparisons and do not select a deployment method.")

    paragraphs.append(r"\paragraph{Whose prediction error improves or worsens at K=10?}")
    for q, target in enumerate(TARGETS):
        counts_text = []
        for method in METHODS:
            if method == "population":
                continue
            pair = bc[(method, 10)]["mae"]
            changes = np.array([pair["patient_changes"][patient][q] for patient in patients], float)
            counts = {"helped": int(np.sum(changes < -TOLERANCE)), "harmed": int(np.sum(changes > TOLERANCE)),
                      "unchanged": int(np.sum(np.abs(changes) <= TOLERANCE)), "n_patients": 6}
            details["k10_patient_counts"].setdefault(method, {})[target] = counts
            counts_text.append(f"{DISPLAY[method]} {counts['helped']}/{counts['harmed']}/{counts['unchanged']}")
        paragraphs.append(f"{target} patient counts (prediction error improves/worsens/is unchanged relative to population, after averaging matched seeds) are "
                          + "; ".join(counts_text) + ".")
        _, full = _patient_values([bs[("full", 10, seed)] for seed in SEEDS])
        _, population = _patient_values([bs[("population", 10, seed)] for seed in SEEDS])
        deltas = full[:, q] - population[:, q]
        maximum = float(deltas.max())
        worst_indices = np.flatnonzero(np.abs(deltas - maximum) <= TOLERANCE)
        worst = [{"patient": patients[i], "population_mae": float(population[i, q]),
                  "full_mae": float(full[i, q]), "mae_change": float(deltas[i])} for i in worst_indices]
        details["worst_full_adaptation_change"][target] = {"positive_harm_observed": maximum > TOLERANCE,
                                                          "largest_mae_change": maximum, "patients": worst}
        label = "largest increase in prediction error with full adaptation" if maximum > TOLERANCE else "largest full-adaptation change (no patient's prediction error worsened)"
        description = "; ".join(f"{_tex(row['patient'])}: {row['full_mae']:.2f} versus {row['population_mae']:.2f} mmHg, "
                                 f"change {row['mae_change']:+.2f}" for row in worst)
        paragraphs.append(f"For {target}, the {label} is {description}.")

    drift = _drift_analysis(bd, bs[("population", 10, 42)])
    details["k10_reference_age_bins"] = drift
    paragraphs.append(r"\paragraph{Within-case reference age at K=10.}")
    for q, target in enumerate(TARGETS):
        available = [row for row in drift if row["n_windows"]]
        below = sum(row["paired_mae_changes"]["full"]["estimate"][q] < -TOLERANCE for row in available)
        above = sum(row["paired_mae_changes"]["full"]["estimate"][q] > TOLERANCE for row in available)
        negative_intervals = sum(row["paired_mae_changes"]["full"]["ci_upper"][q] < -TOLERANCE for row in available)
        positive_intervals = sum(row["paired_mae_changes"]["full"]["ci_lower"][q] > TOLERANCE for row in available)
        details.setdefault("full_model_drift_outcomes", {})[target] = {
            "nonempty_bins": len(available), "below_population_bins": int(below), "above_population_bins": int(above),
            "intervals_wholly_below_zero": int(negative_intervals), "intervals_wholly_above_zero": int(positive_intervals)}
        paragraphs.append(f"Full adaptation has lower {target} MAE than population in {below}/{len(available)} nonempty "
                          f"reference-age bins and higher MAE in {above}/{len(available)}; the descriptive paired intervals lie "
                          f"wholly below/above zero in {negative_intervals}/{positive_intervals} bins, respectively.")
    headers = []
    for row in drift:
        interval = f"{row['lower_minutes_inclusive']}--{row['upper_minutes_exclusive']}" if row["upper_minutes_exclusive"] is not None else "60+"
        headers.append(r"\shortstack{" + interval + f" min\\\\n={row['n_patients']}, N={row['n_windows']}" + "}")
    table = [r"\begin{table*}[t]\centering\scriptsize\renewcommand{\arraystretch}{1.15}", r"\begin{tabular}{lrrrr}\toprule",
             "Method, outcome & " + " & ".join(headers) + r"\\\midrule"]
    for target_index, target in enumerate(TARGETS):
        for method in METHODS:
            if method == "population":
                continue
            cells = []
            for row in drift:
                pair = row["paired_mae_changes"][method]
                cells.append("--" if pair is None else r"\shortstack{" + f"{pair['estimate'][target_index]:+.2f}" + r"\\{" +
                             f"[{pair['ci_lower'][target_index]:+.2f}, {pair['ci_upper'][target_index]:+.2f}]" + "}}")
            table.append(f"{DISPLAY[method]}, {target}" + " & " + " & ".join(cells) + r"\\")
        if target_index == 0:
            table.append(r"\midrule")
    table += [r"\bottomrule\end{tabular}",
              r"\caption{K=10 paired patient-macro MAE changes (method minus population; negative is better), in mmHg, with descriptive 95\% patient-bootstrap intervals. Bins use query end minus actual last-reference end. Each header reports unique patients ($n$) and windows ($N$), not counts multiplied by seeds. Patients are paired within each bin and seed before averaging seeds. Empty bins are unavailable.}\end{table*}"]
    paragraphs.append("\n".join(table))
    paragraphs.append("Patient membership and window coverage change across age bins; a difference between bins is not an isolated "
                      "within-person decay effect. These six-patient intervals and outcome-specific ranks are descriptive, with no "
                      "correction for choosing the lowest observed rank. Nested budgets jointly change reference count and recency. "
                      "Neither the curves nor the ranks establish a universal durability time, a minimum cuff count, or cross-visit persistence.")
    return "\n\n".join(paragraphs), details
