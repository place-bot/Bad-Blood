"""Participant-weighted BP metrics; intervals resample people, not windows."""

import numpy as np


def participant_errors(labels, predictions, subjects):
    errors = np.abs(np.asarray(predictions) - np.asarray(labels))
    ids, inverse = np.unique(subjects, return_inverse=True)
    totals = np.zeros((len(ids), 2), dtype=np.float64)
    np.add.at(totals, inverse, errors)
    return ids, totals / np.bincount(inverse)[:, None]


def metrics(labels, predictions, subjects, bootstrap=0, seed=20260923):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("Expected matching (N,2) labels and predictions")
    if not len(labels) or len(subjects) != len(labels) or not np.isfinite(predictions).all():
        raise ValueError("Invalid evaluation records")
    e = predictions.astype(np.float64) - labels
    ids, per_person = participant_errors(labels, predictions, subjects)
    result = {"records": len(e), "participants": len(ids),
              "mae": np.abs(e).mean(0).tolist(), "macro_mae": per_person.mean(0).tolist(),
              "primary": float(per_person.mean()), "bias": e.mean(0).tolist(),
              "error_sd": e.std(0).tolist(), "rmse": np.sqrt((e * e).mean(0)).tolist(),
              "participant_p90_mae": np.quantile(per_person, .9, axis=0).tolist(),
              "within_mmHg": {str(k): (np.abs(e) <= k).mean(0).tolist() for k in (5, 10, 15)},
              "coverage_of_eligible_rows": 1.0,
              "physically_ordered_fraction": float(((predictions[:, 0] > predictions[:, 1]) &
                                                      (predictions[:, 1] > 0)).mean()),
              "pressure_strata": {}}
    # Offline metrics include all predictions, even those a live UI would suppress.
    for target, edges in enumerate(((90, 110, 130, 150, 170), (60, 80, 100))):
        assigned = np.digitize(labels[:, target], edges)
        for group in range(len(edges) + 1):
            mask = assigned == group
            if not mask.any():
                continue
            members, values = participant_errors(labels[mask], predictions[mask], np.asarray(subjects)[mask])
            key = f'{("SBP", "DBP")[target]}_bin_{group}'
            result["pressure_strata"][key] = {
                "lower": None if group == 0 else edges[group - 1],
                "upper": None if group == len(edges) else edges[group],
                "participants": len(members), "records": int(mask.sum()),
                "macro_mae": values.mean(0).tolist(), "inconclusive": len(members) < 10}
    if bootstrap:
        rng = np.random.default_rng(seed)
        values = np.stack([per_person[rng.integers(len(ids), size=len(ids))].mean(0)
                           for _ in range(bootstrap)])
        result["macro_mae_ci95"] = np.quantile(values, [.025, .975], axis=0).tolist()
    return result


def paired_difference(labels, baseline, candidate, subjects, bootstrap=2000, seed=20260923):
    _, a = participant_errors(labels, baseline, subjects)
    _, b = participant_errors(labels, candidate, subjects)
    d = b - a
    rng = np.random.default_rng(seed)
    samples = np.stack([d[rng.integers(len(d), size=len(d))].mean(0) for _ in range(bootstrap)])
    return {"candidate_minus_baseline_macro_mae": d.mean(0).tolist(),
            "paired_participant_ci95": np.quantile(samples, [.025, .975], axis=0).tolist()}
