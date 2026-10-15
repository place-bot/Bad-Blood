"""Temporary synthetic report-helper fixtures; never real report results."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
from reports import build_report
from reports.personal_details import (
    BUDGETS, METHODS, SEEDS, _hash, build_personal_details, compact_unit,
    prediction_seconds, seed_statistics,
)


def personal_fixture():
    provenance = {"unit_test_only": True, "source_hashes": {"test_source": "fixture-source-sha"}}
    selection = {"provenance_hash": _hash(provenance), "selected": {}, "grid": []}
    selection["selection_hash"] = _hash(selection)
    summaries, units = {}, []
    neural_counts = {"head": 1634, "last_block": 25730, "full": 37666}
    for method in METHODS:
        for budget in BUDGETS:
            if method == "affine" and budget < 3:
                continue
            for seed in SEEDS:
                value = seed - 41
                summaries[(method, budget, seed)] = {
                    "macro": {"mae": [value, value + 1], "rmse": [value + 2, value + 3]},
                    "n_windows": value * 100, "n_patients": 6}
                for patient in range(6):
                    neural = method in neural_counts
                    adaptation = {"trainable_parameters": neural_counts.get(method, 0), "total_parameters": 37666,
                                  "trainable_parameter_names": ["head.1.weight"] if neural else [],
                                  "fitted_scalar_coefficients": 0 if neural or method == "population" else (4 if method == "affine" else 2),
                                  "changed_parameter_l2": {"head.1.weight": 0.5} if neural else {},
                                  "changed_parameter_count": 1 if neural else 0,
                                  "adaptation_seconds": 0.2 if neural else 0.001}
                    if neural:
                        adaptation["total_adaptation_seconds"] = 0.3
                        latency = {"prediction_seconds": 0.8, "seconds_per_window": 0.002}
                    else:
                        shared = None if method in ("last_reference", "calibration_mean") else {
                            "query": {"prediction_seconds": 0.8}, "all_ten_support_references": {"prediction_seconds": 0.02}}
                        latency = {"scalar_transform_seconds": 0.004, "shared_population_inference": shared}
                    units.append({"patient": f"unit-test-patient-{patient}", "case": "unit-test-case", "role": "target",
                                  "strategy": method, "budget": budget, "seed": seed, "setting": {},
                                  "unit_path": f"fixture/{method}/{budget}/{seed}/{patient}", "summary": {"n_windows": 400},
                                  "last_reference_end": 100.0, "calibration_boundary": 120.0, "population_epoch": 4,
                                  "checkpoint_sha256": "unit-test-checkpoint", "config_hash": "unit-test-config",
                                  "provenance_hash": _hash(provenance), "selection_hash": selection["selection_hash"],
                                  "artifact_hashes": {"predictions.npz": "unit-test-prediction-sha"},
                                  "adaptation": adaptation, "prediction_latency": latency,
                                  "process_peak_rss_native_units": 123456, "peak_rss_units": "fixture units"})
    return summaries, units, selection, provenance


class ReportTests(unittest.TestCase):
    def test_number_and_exact_pairing(self):
        self.assertEqual(build_report.number(None), "--")
        self.assertEqual(build_report.number(np.nan), "--")
        self.assertEqual(build_report.number(float("inf")), "--")
        self.assertEqual(build_report.number(1.234), "1.23")
        a = {"y": np.array([[120., 80.], [130., 90.]]), "pred": np.zeros((2, 2)),
             "subjects": np.array(["patient-a", "patient-b"]), "source_indices": np.array([4, 9])}
        b = copy.deepcopy(a)
        b["pred"] += 1
        build_report.assert_pair(a, b)  # Predictions should differ; references must not.
        for key in ("y", "subjects", "source_indices"):
            b = copy.deepcopy(a)
            b[key] = b[key][::-1]
            with self.assertRaises(ValueError):
                build_report.assert_pair(a, b)

    def test_seed_statistics_are_macro_then_equal_seed_and_sample_sd(self):
        summaries, _, _, _ = personal_fixture()
        stats = seed_statistics(summaries, "head", 1, "mae", 0)
        self.assertEqual(stats["mean"], 2.0)
        self.assertEqual(stats["seed_sd"], 1.0)
        self.assertEqual(stats["seed_values"], [1.0, 2.0, 3.0])
        self.assertIsNone(seed_statistics(summaries, "affine", 1, "mae", 0))
        summaries[("head", 1, 42)]["macro"]["mae"][0] = None
        with self.assertRaises(ValueError):
            seed_statistics(summaries, "head", 1, "mae", 0)

    def test_costs_separate_reference_only_from_shared_population_inference(self):
        _, units, _, _ = personal_fixture()
        records = {record["strategy"]: record for record in units if record["budget"] == 10}
        self.assertAlmostEqual(prediction_seconds(records["head"]), 0.8)
        self.assertAlmostEqual(prediction_seconds(records["bias"]), 0.804)
        self.assertAlmostEqual(prediction_seconds(records["last_reference"]), 0.004)
        compact = compact_unit(records["head"])
        self.assertAlmostEqual(compact["milliseconds_per_query"], 2.0)
        self.assertEqual(compact["observed_reference_signal_seconds"], 100)
        self.assertEqual(compact["changed_parameter_l2"], {"head.1.weight": 0.5})
        self.assertEqual(compact["total_adaptation_seconds"], 0.3)

    def test_full_budget_tables_and_compact_source_traceability(self):
        summaries, units, selection, provenance = personal_fixture()
        with tempfile.TemporaryDirectory(prefix="bad-blood-report-unit-fixture-") as temp:
            out = Path(temp)
            latex = build_personal_details(out, summaries, units, selection, provenance=provenance)
            self.assertEqual(latex.count(r"\begin{table*}"), 6)
            self.assertIn(r"$2.00\pm1.00$", latex)
            self.assertIn("not a pure sample-complexity", latex)
            self.assertIn("1,634", latex)
            self.assertIn("25,730", latex)
            self.assertIn("37,666", latex)
            result = json.loads((out / "results" / "personal_details.json").read_text())
            self.assertEqual(result["n_units"], 684)
            self.assertEqual(result["provenance"]["source_hashes"], provenance["source_hashes"])
            self.assertEqual(result["parameter_masks"]["head"]["trainable_parameters"], 1634)
            self.assertEqual(result["k10_measured_costs"]["head"]["n_patient_seed_runs"], 18)
            self.assertIsNone(result["full_budget_patient_macro_mean_seed_sd"]["affine/K1/SBP/mae"])
            self.assertIn("checkpoint_sha256", result["units"][0])
            self.assertIn("config_hash", result["units"][0])
            self.assertIn("artifact_hashes", result["units"][0])

    def test_details_reject_provenance_selection_and_mask_changes(self):
        summaries, units, selection, provenance = personal_fixture()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "provenance"):
                build_personal_details(temp, summaries, units, selection, provenance={**provenance, "changed": True})
            with self.assertRaisesRegex(ValueError, "selection hash"):
                build_personal_details(temp, summaries, units, {**selection, "selected": {"tampered": True}}, provenance=provenance)
            changed = copy.deepcopy(units)
            changed[0]["adaptation"]["trainable_parameters"] = 1
            with self.assertRaisesRegex(ValueError, "mask"):
                build_personal_details(temp, summaries, changed, selection, provenance=provenance)


if __name__ == "__main__":
    unittest.main()
