"""Pressure metrics in mmHg; patient-aware aggregation, not clinical certification.

All prediction/reference arrays are N x 2 in SBP, DBP order. Fractions, not
percentages, are returned for within_* metrics. Undefined values become None
in JSON-facing output. No test metric is used to choose training settings.
"""
from __future__ import annotations

import numpy as np

TARGETS = ("SBP", "DBP")


def _arrays(y, pred, subjects):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    subjects = np.asarray(subjects).astype(str)
    if y.ndim != 2 or y.shape[1] != 2 or pred.shape != y.shape:
        raise ValueError("Expected matching N x 2 SBP/DBP arrays")
    if subjects.shape != (len(y),) or not len(y):
        raise ValueError("Need one patient identifier per nonempty prediction")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions/references must not be silently dropped")
    return y, pred, subjects


def _safe(value):
    array = np.asarray(value)
    if array.ndim:
        return [_safe(item) for item in array]
    return float(array) if np.isfinite(array) else None


def _vector_metrics(y, pred):
    error = pred - y
    absolute = np.abs(error)
    corr, r2 = [], []
    for q in range(2):
        sst = np.sum((y[:, q] - y[:, q].mean()) ** 2)
        r2.append(1 - np.sum(error[:, q] ** 2) / sst if sst > 0 else np.nan)
        if len(y) > 1 and np.ptp(y[:, q]) > 0 and np.ptp(pred[:, q]) > 0:
            corr.append(np.corrcoef(y[:, q], pred[:, q])[0, 1])
        else:
            corr.append(np.nan)
    return {
        "mae": absolute.mean(0),
        "rmse": np.sqrt(np.mean(error ** 2, axis=0)),
        "bias": error.mean(0),
        "error_sd": error.std(0, ddof=1) if len(y) > 1 else np.full(2, np.nan),
        "median_ae": np.median(absolute, axis=0),
        "q90_ae": np.quantile(absolute, .9, axis=0),
        "q95_ae": np.quantile(absolute, .95, axis=0),
        "within_5": np.mean(absolute <= 5, axis=0),
        "within_10": np.mean(absolute <= 10, axis=0),
        "within_15": np.mean(absolute <= 15, axis=0),
        "pearson_r": np.asarray(corr),
        "r2": np.asarray(r2),
    }


def quick_score(y, pred, subjects):
    """Validation-only selection score; equal patient weighting is explicit."""
    y, pred, subjects = _arrays(y, pred, subjects)
    error = pred - y
    _, inverse = np.unique(subjects, return_inverse=True)
    counts = np.bincount(inverse)
    macro = [np.mean(np.bincount(inverse, weights=np.abs(error[:, q])) / counts)
             for q in range(2)]
    return {"mae": np.abs(error).mean(0).tolist(),
            "rmse": np.sqrt((error ** 2).mean(0)).tolist(),
            "bias": error.mean(0).tolist(), "macro_mae": macro}


def summarize(y, pred, subjects):
    """Complete pooled, patient-macro, per-patient, range and BA summaries."""
    y, pred, subjects = _arrays(y, pred, subjects)
    unique, inverse = np.unique(subjects, return_inverse=True)
    vectors, per_patient = [], {}
    for index, subject in enumerate(unique):
        selected = inverse == index
        raw = _vector_metrics(y[selected], pred[selected])
        vectors.append(raw)
        per_patient[str(subject)] = {"n_windows": int(selected.sum()),
                                    **{k: _safe(v) for k, v in raw.items()}}
    macro, defined = {}, {}
    for key in vectors[0]:
        values = np.stack([v[key] for v in vectors])
        count = np.isfinite(values).sum(0)
        defined[key] = count.tolist()
        result = np.divide(np.nansum(values, axis=0), count,
                           out=np.full(2, np.nan), where=count > 0)
        macro[key] = _safe(result)
    pressure_ranges = {}
    for q, target in enumerate(TARGETS):
        cuts = [-np.inf, 100, 140, np.inf] if q == 0 else [-np.inf, 60, 90, np.inf]
        names = ["below_100", "100_to_below_140", "140_or_above"] if q == 0 else [
            "below_60", "60_to_below_90", "90_or_above"]
        strata = []
        for name, low, high in zip(names, cuts[:-1], cuts[1:]):
            selected = (y[:, q] >= low) & (y[:, q] < high)
            row = {"range": name, "n_windows": int(selected.sum()),
                   "n_patients": len(np.unique(subjects[selected]))}
            if selected.any():
                raw = _vector_metrics(y[selected], pred[selected])
                row.update({key: _safe(value[q]) for key, value in raw.items()})
            strata.append(row)
        pressure_ranges[target] = strata
    return {
        "target_order": list(TARGETS), "units": "mmHg",
        "within_metrics_unit": "fraction", "error_sd_ddof": 1,
        "n_windows": len(y), "n_patients": len(unique),
        "pooled": {k: _safe(v) for k, v in _vector_metrics(y, pred).items()},
        "macro": macro, "macro_defined_patients": defined,
        "per_patient": per_patient, "pressure_ranges": pressure_ranges,
        "repeated_measurement_bland_altman": repeated_bland_altman(y, pred, subjects),
    }


def repeated_bland_altman(y, pred, subjects):
    """Descriptive random-intercept error variance, with unbalanced group sizes.

    Intercepts represent between-patient error and residuals within-patient
    error. Serial correlation remains a limitation: limits are descriptive,
    not confidence intervals or device-validation claims.
    """
    y, pred, subjects = _arrays(y, pred, subjects)
    error = pred - y
    unique, inverse = np.unique(subjects, return_inverse=True)
    sizes = np.bincount(inverse).astype(float)
    means = np.stack([error[inverse == i].mean(0) for i in range(len(unique))])
    within_ss = np.sum((error - means[inverse]) ** 2, axis=0)
    grand = error.mean(0)
    between_ss = np.sum(sizes[:, None] * (means - grand) ** 2, axis=0)
    J, N = len(unique), len(y)
    within = within_ss / (N - J) if N > J else np.full(2, np.nan)
    if J > 1 and N > J:
        n0 = (N - np.sum(sizes ** 2) / N) / (J - 1)
        between = np.maximum(0, (between_ss / (J - 1) - within) / n0)
        half_width = 1.96 * np.sqrt(within + between)
    else:
        between, half_width = np.full(2, np.nan), np.full(2, np.nan)
    return {"method": "unbalanced one-way random-intercept error decomposition",
            "interpretation": "descriptive; serial correlation and sampling uncertainty remain",
            "pooled_bias": _safe(grand), "patient_mean_bias": _safe(means.mean(0)),
            "between_patient_variance": _safe(between), "within_patient_variance": _safe(within),
            "loa_lower": _safe(grand - half_width), "loa_upper": _safe(grand + half_width),
            "patient_mean_errors": {str(s): _safe(v) for s, v in zip(unique, means)}}


def patient_bootstrap(values, n_boot=2000, seed=20261009):
    """Resample patients, never windows; input may be patients x outputs."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Patient bootstrap requires finite patient-level values")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), (n_boot, len(values)))
    estimates = values[sampled].mean(1)
    return {"estimate": _safe(values.mean(0)),
            "ci_lower": _safe(np.quantile(estimates, .025, axis=0)),
            "ci_upper": _safe(np.quantile(estimates, .975, axis=0)),
            "n_patients": len(values), "resamples": n_boot,
            "method": "patient bootstrap percentile, descriptive 95% interval"}


def paired_patient_change(model_summaries, baseline_summaries, metric="mae"):
    """Average matched-seed per-patient model-minus-baseline before bootstrap."""
    if len(model_summaries) != len(baseline_summaries) or not model_summaries:
        raise ValueError("Need paired nonempty seed summaries")
    patients = sorted(model_summaries[0]["per_patient"])
    seed_differences = []
    for model, baseline in zip(model_summaries, baseline_summaries):
        if sorted(model["per_patient"]) != patients or sorted(baseline["per_patient"]) != patients:
            raise ValueError("Paired comparison requires identical patient membership")
        rows = []
        for patient in patients:
            m, b = model["per_patient"][patient], baseline["per_patient"][patient]
            if m["n_windows"] != b["n_windows"]:
                raise ValueError("Paired patient windows differ; verify exact source indices too")
            rows.append(np.asarray(m[metric]) - np.asarray(b[metric]))
        seed_differences.append(rows)
    values = np.asarray(seed_differences)
    patient_mean = values.mean(0)
    result = patient_bootstrap(patient_mean)
    result.update({"sign": "model minus baseline; negative is improvement",
                   "metric": metric, "seed_macro_changes": _safe(values.mean(1)),
                   "patient_changes": {p: _safe(v) for p, v in zip(patients, patient_mean)}})
    return result
