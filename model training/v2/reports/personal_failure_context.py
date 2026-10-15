"""Pure, descriptive reference-distribution context for final personal cases.

Call only after complete inventory verification. Inputs are six already loaded
population/K10/seed42 unit records and their saved prediction/reference arrays.
This helper neither reads files nor modifies arrays, models or model choices.
"""
from __future__ import annotations

import numpy as np

TARGETS = ("SBP", "DBP")
EXPECTED_PATIENTS = {"vitaldb:" + patient for patient in
                     ("p005879", "p005885", "p005977", "p005984", "p006009", "p006052")}


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite, aligned SBP/DBP references")
    return {"n_windows": len(values), "targets": {
        target: {"mean": float(values[:, q].mean()), "sample_sd": float(values[:, q].std(ddof=1)),
                 "minimum": float(values[:, q].min()), "maximum": float(values[:, q].max()),
                 "range": [float(values[:, q].min()), float(values[:, q].max())]}
        for q, target in enumerate(TARGETS)}}


def _scalar(value):
    return np.asarray(value).item()


def build_personal_failure_context(patient_units):
    """Return (LaTeX table + caveat, JSON-ready context), with no side effects."""
    if len(patient_units) != 6:
        raise ValueError("The complete final cohort requires six patient units")
    patients, rows = {}, []
    for entry in patient_units:
        record, arrays = entry["record"], entry["predictions"]
        if (record["phase"], record["role"], record["strategy"], record["budget"], record["seed"]) != (
                "test", "target", "population", 10, 42):
            raise ValueError("Failure context requires final population/K10/seed42 artifacts")
        patient = record["patient"]
        if patient not in EXPECTED_PATIENTS or patient in patients:
            raise ValueError("Unexpected or duplicate final patient")
        if str(_scalar(arrays["case_id"])) != record["case"] or str(_scalar(arrays["config_hash"])) != record["config_hash"] or str(_scalar(arrays["checkpoint_sha256"])) != record["checkpoint_sha256"]:
            raise ValueError("Reference arrays do not match their unit record")
        support_y, query_y = np.asarray(arrays["support_y"]), np.asarray(arrays["y"])
        if support_y.shape != (10, 2):
            raise ValueError("K10 support references must have shape (10,2)")
        if query_y.ndim != 2 or query_y.shape[1] != 2 or query_y.shape[0] != record["summary"]["n_windows"]:
            raise ValueError("Query references do not match recorded window count")
        if not np.array_equal(arrays["subjects"], np.repeat(patient, len(query_y))):
            raise ValueError("Query references include a different patient")
        for prefix, count in (("support_", 10), ("", len(query_y))):
            indices = np.asarray(arrays[prefix + "source_indices"])
            start, end = np.asarray(arrays[prefix + "start"]), np.asarray(arrays[prefix + "end_exclusive"])
            if any(a.shape != (count,) for a in (indices, start, end)) or len(np.unique(indices)) != count:
                raise ValueError("Reference row identity/timestamps are misaligned")
            if not np.isfinite(start).all() or not np.isfinite(end).all() or np.any(end <= start):
                raise ValueError("Invalid reference timestamps")
        if np.intersect1d(arrays["support_source_indices"], arrays["source_indices"]).size:
            raise ValueError("Support/query reference overlap")
        boundary = float(_scalar(arrays["calibration_boundary"]))
        if np.max(arrays["support_end_exclusive"]) > boundary + 1e-6 or np.min(arrays["start"]) < boundary + 10 - 1e-6:
            raise ValueError("Reference windows violate chronological support/guard boundaries")
        support, query = _stats(support_y), _stats(query_y)
        shifts = {target: query["targets"][target]["mean"] - support["targets"][target]["mean"] for target in TARGETS}
        patients[patient] = {"case": record["case"], "support": support, "query": query,
                             "query_minus_support_mean": shifts,
                             "support_elapsed_span_seconds": float(np.max(arrays["support_end_exclusive"]) - np.min(arrays["support_start"])),
                             "query_elapsed_span_seconds": float(np.max(arrays["end_exclusive"]) - np.min(arrays["start"])),
                             "calibration_boundary": boundary, "config_hash": record["config_hash"],
                             "checkpoint_sha256": record["checkpoint_sha256"], "unit_path": record["unit_path"]}
    if set(patients) != EXPECTED_PATIENTS:
        raise ValueError("Final reference context is missing a patient")
    for patient in sorted(patients):
        context = patients[patient]
        for target in TARGETS:
            support, query = context["support"]["targets"][target], context["query"]["targets"][target]
            rows.append(f"{patient.split(':')[-1]} & {target} & {context['query']['n_windows']} & "
                        f"{support['mean']:.1f} ({support['sample_sd']:.1f}) & "
                        f"{query['mean']:.1f} ({query['sample_sd']:.1f}) & "
                        f"{context['query_minus_support_mean'][target]:+.1f}" + r"\\")
    caveat = ("These are descriptive differences between ten sparse early ABP-derived references and all later query references, "
              "not evidence that a particular physiological mechanism or overfitting caused an error increase. "
              "SD is variability among the retained reference windows, not cuff uncertainty. Unequal sample sizes and serial dependence "
              "also affect SD and observed ranges; a wider query range alone is not proof of temporal drift.")
    table = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{llrrrr}\toprule",
             r"Patient & Pressure & Query $N$ & Support mean (SD) & Query mean (SD) & Query $-$ support\\\midrule",
             *rows, r"\bottomrule\end{tabular}",
             r"\caption{Reference-pressure context for all six final patients at K=10, using the saved seed-42 population artifact once per patient. Pressure values and mean shifts are in mmHg; each support set has ten references. SD uses $n-1$. Signed shifts are later query mean minus early support mean. Full minima/maxima and elapsed spans are retained in the companion JSON. These are reference distributions, not prediction scores.}\end{table*}", caveat]
    details = {"budget": 10, "seed": 42, "artifact_strategy": "population", "n_patients": 6,
               "target_order": list(TARGETS), "units": "mmHg", "sd_ddof": 1,
               "shift_sign": "query mean minus support mean", "patients": patients,
               "interpretation_limit": caveat, "inference_claim": "descriptive, not causal or clinical harm"}
    return "\n".join(table), details
