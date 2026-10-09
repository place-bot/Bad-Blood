"""Lossless compressed results and a small overview of verified metric summaries.

Neither helper verifies experimental completeness or selects a model. Call the
overview only on outputs of the authoritative inventory/aggregation checks.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile


def write_compressed_json(path, data) -> Path:
    """Atomically write complete JSON as deterministic gzip, returning its path.

    JSON must already be serializable: no truncation, rounding, NumPy coercion,
    or nonfinite-to-null conversion is performed. Invalid data leave an existing
    file unchanged. Gzip excludes the destination filename and uses mtime=0;
    identical JSON produces identical bytes with the same compression runtime.
    """
    path = Path(path)
    if not path.name.endswith(".json.gz"):
        raise ValueError("Compressed result path must end with .json.gz")
    payload = (json.dumps(data, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream,
                               compresslevel=9, mtime=0) as compressed:
                compressed.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=1):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
             f"Invalid {name}")
    return value


def _vector(value, name):
    _require(isinstance(value, (list, tuple)) and len(value) == 2,
             f"{name} must have two values in SBP/DBP order")
    _require(all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)
                              and math.isfinite(v)) for v in value),
             f"{name} contains nonfinite or nonnumeric values; undefined metrics must be None")
    return value


def _summarize_seeds(rows):
    """Average existing scalar statistics, never concatenate seed predictions."""
    seeds = sorted(rows)
    _require(bool(seeds), "Need at least one verified seed summary")
    first = rows[seeds[0]]
    metrics = sorted(first["pooled"])
    _require(bool(metrics), "Missing pooled metrics")
    windows = _integer(first["n_windows"], "window count")
    patients = _integer(first["n_patients"], "patient count")
    result = {"seeds": seeds, "n_seeds": len(seeds), "n_windows": windows,
              "n_patients": patients, "pooled": {}, "macro": {},
              "macro_defined_patients_by_seed": {}}
    for seed, row in rows.items():
        _integer(seed, "seed", minimum=0)
        _require(row["target_order"] == ["SBP", "DBP"] and row["units"] == "mmHg"
                 and row["within_metrics_unit"] == "fraction" and row["error_sd_ddof"] == 1,
                 "Metric units, pressure order, or error-SD convention changed")
        _require(row["n_windows"] == windows and row["n_patients"] == patients,
                 "Patient/window coverage differs across seeds")
        _require(set(row["pooled"]) == set(row["macro"]) == set(metrics)
                 == set(row["macro_defined_patients"]), "Metric inventory differs across seeds")
        defined = {}
        for metric in metrics:
            counts = row["macro_defined_patients"][metric]
            _require(isinstance(counts, (list, tuple)) and len(counts) == 2,
                     "Defined-patient counts must have SBP/DBP shape")
            defined[metric] = [_integer(v, "defined-patient count", minimum=0) for v in counts]
            _require(all(v <= patients for v in counts), "Defined-patient count exceeds cohort size")
            macro = _vector(row["macro"][metric], f"macro/{metric}")
            _require(all((count == 0) == (value is None) for count, value in zip(counts, macro)),
                     "Undefined macro metric and defined-patient count disagree")
        result["macro_defined_patients_by_seed"][str(seed)] = defined
    for aggregation in ("pooled", "macro"):
        for metric in metrics:
            vectors = [_vector(rows[seed][aggregation][metric], f"{aggregation}/{metric}") for seed in seeds]
            values = [[v[q] for v in vectors if v[q] is not None] for q in range(2)]
            result[aggregation][metric] = {
                "seed_mean": [mean(v) if v else None for v in values],
                "defined_seed_counts": [len(v) for v in values],
            }
    return result


def build_metric_overview(aruns, bs, validation_selections):
    """Return JSON-ready, seed-mean pooled/macro metrics without patient details.

    Inputs are verified aggregate_a run dictionaries, aggregate_b tuple-keyed
    summaries, and already-made validation selection metadata. This function
    copies selections without recomputing them; it performs no outcome ranking.
    All supplied pooled/macro metric keys are retained, including future keys.
    """
    _require(bool(aruns) and bool(bs), "Overview requires both verified track summaries")
    _require(isinstance(validation_selections, dict), "Validation selections must be a dictionary")
    # A JSON round trip prevents returning a mutable alias and rejects NaN or
    # unsupported metadata before any report writer is called.
    selections = json.loads(json.dumps(validation_selections, allow_nan=False))
    population = defaultdict(dict)
    for key, run in aruns.items():
        config, done = run["config"], run["completion"]
        variant, seed = config["variant"], _integer(config["seed"], "seed", minimum=0)
        _require(key == f"{variant}_seed{seed}", "Population run key and identity disagree")
        _require(done["status"] == "complete" and done["toy_test_only"] is False
                 and done["variant"] == variant and done["seed"] == seed,
                 "Population run is incomplete, synthetic, or has different identity")
        _require(set(run["metrics"]) == {"validation", "test", "aami"},
                 "Population overview requires validation, test, and aami separately")
        _require(seed not in population[variant], "Duplicate population seed")
        population[variant][seed] = run
    attention = {}
    for variant, runs in sorted(population.items()):
        parameter_counts = {_integer(r["completion"]["parameters"], "parameter count") for r in runs.values()}
        _require(len(parameter_counts) == 1, "Parameter count differs across seeds")
        epochs, selected = {}, {}
        for seed, run in sorted(runs.items()):
            done = run["completion"]
            epochs[str(seed)] = _integer(done["epochs_completed"], "completed epoch count")
            selected[str(seed)] = _integer(done["selected_epoch"], "selected epoch")
            _require(selected[str(seed)] <= epochs[str(seed)], "Selected epoch exceeds completed epochs")
            if "actual_epochs" in done:
                _require(done["actual_epochs"] == epochs[str(seed)], "Actual/completed epoch counts differ")
        attention[variant] = {
            "seeds": sorted(runs), "n_seeds": len(runs), "parameters": next(iter(parameter_counts)),
            "selected_epochs_by_seed": selected, "epochs_completed_by_seed": epochs,
            "splits": {split: _summarize_seeds({seed: run["metrics"][split] for seed, run in runs.items()})
                       for split in ("validation", "test", "aami")},
        }
    personal = defaultdict(dict)
    for key, row in bs.items():
        _require(isinstance(key, tuple) and len(key) == 3, "Personal keys must be method/budget/seed tuples")
        method, budget, seed = key
        _require(isinstance(method, str) and bool(method) and "/" not in method, "Invalid strategy name")
        _integer(budget, "reference budget")
        _integer(seed, "seed", minimum=0)
        personal[(method, budget)][seed] = row
    return {
        "schema_version": 1, "target_order": ["SBP", "DBP"],
        "definitions": {
            "scope": "Overview of supplied verified runs; this helper does not establish complete experimental inventory",
            "aggregation": "Arithmetic mean of already-computed per-seed statistics, not pooled predictions across seeds; RMSE and quantiles are not recomputed from concatenated errors",
            "pooled": "Each retained query window has equal weight within its seed and split",
            "macro": "Each patient with a defined metric has equal weight within its seed and split",
            "undefined": "None remains unavailable; seed means use only defined seeds, counted separately for each pressure; all undefined gives None and count zero",
            "coverage": "Patient/window counts are per split or method/budget and are not summed over seeds",
            "pressure_error_metrics_unit": "mmHg",
            "fraction_metrics": "within_5, within_10, within_15 are fractions in [0,1], not percentages; thresholds are inclusive absolute error in mmHg",
            "dimensionless_metrics": ["pearson_r", "r2"],
            "bias_sign": "prediction minus reference", "error_sd_ddof": 1,
            "full_detail": "Per-patient, pressure-stratum, paired, and Bland-Altman details remain in the complete compressed result files",
        },
        "population_run_count": len(aruns), "personal_summary_group_count": len(bs),
        "validation_selections": selections, "attention": attention,
        "personalization": {f"{method}/K{budget}": {"strategy": method, "budget": budget,
                            **_summarize_seeds(rows)} for (method, budget), rows in sorted(personal.items())},
    }
