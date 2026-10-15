"""In-memory synthetic table fixtures only; never generate a report/PDF."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from reports.extended_tables import (
    BUDGETS, METHODS, SEEDS, build_extended_tables, latency_run_statistics,
    paired_interval_cell, per_patient_seed_values, selected_setting_cell,
)


def fixture():
    aruns, bs, bc = {}, {}, {}
    for variant, seeds in (("baseline", SEEDS), ("baseline_ecg", (42,))):
        for seed in seeds:
            value = seed - 41.
            timing = [{"device": device, "batch_size": batch, "repetitions": 20,
                       "median_batch_ms": value * factor * batch,
                       "median_per_example_ms": value * factor}
                      for device, factor in (("cpu", 1.), ("mps", 2.)) for batch in (1, 256)]
            aruns[f"{variant}_seed{seed}"] = {"config": {"variant": variant, "seed": seed, "device": "mps"},
                                               "costs": {"latency": timing}}
    for method in METHODS:
        for seed in SEEDS:
            bs[(method, 10, seed)] = {"per_patient": {
                f"vitaldb:fixture_{i}": {"n_windows": i + 1,
                                       "mae": [seed - 41. + i, seed - 40. + i],
                                       "rmse": [seed - 39. + i, seed - 38. + i]}
                for i in range(6)}}
    for method in METHODS[1:]:
        for budget in BUDGETS:
            if method == "affine" and budget < 3:
                continue
            bc[(method, budget)] = {metric: {
                "estimate": [-1., -2.], "ci_lower": [-3., -4.], "ci_upper": [1., 0.],
                "n_patients": 6, "resamples": 2000}
                for metric in ("mae", "rmse")}
    selected = {f"{method}/K{k}": {"lr": 1e-4 if k < 5 else 1e-3, "steps": 20 if k < 5 else 100}
                for method in ("head", "last_block", "full") for k in BUDGETS}
    selected.update({f"affine/K{k}": {"penalty": 100} for k in (3, 5, 10)})
    return aruns, bs, bc, {"selected": selected}


class ExtendedTableTests(unittest.TestCase):
    def test_interval_and_absolute_values_preserve_targets_and_seed_weight(self):
        _, bs, bc, _ = fixture()
        self.assertEqual(paired_interval_cell(bc[("bias", 1)]["mae"], 0),
                         r"\shortstack{-1.00\\{[-3.00, 1.00]}}")
        self.assertEqual(paired_interval_cell(bc[("bias", 1)]["mae"], 1),
                         r"\shortstack{-2.00\\{[-4.00, 0.00]}}")
        values = per_patient_seed_values(bs, "head", "vitaldb:fixture_0", "mae", 0)
        np.testing.assert_array_equal(values, [1, 2, 3])
        self.assertEqual(values.mean(), 2)
        self.assertEqual(values.std(ddof=1), 1)
        bad = copy.deepcopy(bc[("bias", 1)]["mae"])
        bad["ci_lower"][0] = 5
        with self.assertRaises(ValueError):
            paired_interval_cell(bad, 0)

    def test_actual_selected_settings_and_affine_unavailability(self):
        selected = fixture()[3]["selected"]
        self.assertEqual(selected_setting_cell("head", 1, selected), r"\shortstack{$10^{-4}$\\20 steps}")
        self.assertEqual(selected_setting_cell("full", 10, selected), r"\shortstack{$10^{-3}$\\100 steps}")
        self.assertEqual(selected_setting_cell("affine", 3, selected), r"$\lambda=100$")
        self.assertEqual(selected_setting_cell("affine", 1, selected), "--")
        selected["head/K1"] = {"lr": .01, "steps": 20}
        with self.assertRaises(ValueError):
            selected_setting_cell("head", 1, selected)

    def test_latency_batch_units_and_matched_seed_inventory(self):
        aruns = fixture()[0]
        measured = latency_run_statistics(aruns)
        self.assertEqual(measured["baseline"]["n_runs"], 3)
        self.assertEqual(measured["baseline_ecg"]["n_runs"], 1)
        self.assertEqual(measured["baseline"]["measurements"][("cpu", 256)],
                         {"median": 512, "minimum": 256, "maximum": 768})
        bad = copy.deepcopy(aruns)
        bad["baseline_seed42"]["costs"]["latency"][1]["median_per_example_ms"] = 256
        with self.assertRaisesRegex(ValueError, "units"):
            latency_run_statistics(bad)
        bad = copy.deepcopy(aruns)
        del bad["baseline_seed44"]
        with self.assertRaisesRegex(ValueError, "seed"):
            latency_run_statistics(bad)

    def test_complete_tex_tables_have_compact_rows_and_explicit_uncertainty(self):
        latex = build_extended_tables(*fixture())
        self.assertEqual(latex.count(r"\begin{table*}"), 10)
        self.assertEqual(latex.count(r"\end{table*}"), 10)
        self.assertEqual(latex.count("All-budget paired"), 4)
        self.assertEqual(latex.count("Absolute SBP"), 2)
        self.assertEqual(latex.count("Absolute DBP"), 2)
        self.assertIn(r"\shortstack{2.00\\(1.00)}", latex)
        self.assertIn(r"fixture\_5", latex)
        self.assertIn(r"\shortstack{512.000\\{[256.000, 768.000]}}", latex)
        self.assertNotIn(r"\\[", latex)  # Never parse a CI as an optional row-spacing dimension.
        self.assertIn("whole-batch times, not per-window latency", latex)
        self.assertIn("not seed-SD intervals", latex)
        self.assertIn("No final-target query determined", latex)
        for block in latex.split(r"\begin{table*}")[1:]:
            self.assertLessEqual(block.count(" & "), 80)
        aruns, bs, bc, selection = fixture()
        del bc[("full", 1)]
        with self.assertRaises(KeyError):
            build_extended_tables(aruns, bs, bc, selection)
        aruns, bs, bc, selection = fixture()
        bs[("head", 10, 42)]["per_patient"]["vitaldb:fixture_0"]["n_windows"] += 1
        with self.assertRaisesRegex(ValueError, "query counts"):
            build_extended_tables(aruns, bs, bc, selection)


if __name__ == "__main__":
    unittest.main()
