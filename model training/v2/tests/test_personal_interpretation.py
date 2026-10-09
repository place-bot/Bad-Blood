"""Outcome interpretation tests use constructed metrics, not study results."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
from reports.personal_interpretation import (
    BIN_BOUNDS, BUDGETS, METHODS, SEEDS, build_personal_interpretation,
)
from shared.metrics import paired_patient_change


def interpretation_fixture(*, all_full_better=False, neural_tie=False, empty_last=False):
    """Six unequal-size people, seed shifts, heterogeneous harm, changing coverage."""
    patients = [f"unit_patient_{i}" for i in range(6)]
    baseline = np.column_stack((10. + np.arange(6), 6. + np.arange(6) / 2))
    early_counts, middle_counts = np.arange(12, 18), np.array([10, 11, 12, 13, 0, 0])
    late_counts = np.zeros(6, int) if empty_last else np.array([0, 0, 0, 0, 12, 13])
    bin_counts = [early_counts, middle_counts, np.zeros(6, int), late_counts]
    total_counts = sum(bin_counts)
    bs, bd = {}, {}

    def summary(method, budget, seed, counts):
        offset = {"population": [0, 0], "bias": [-1, -.5], "affine": [-2 if budget == 10 else -1.4, -.8],
                  "head": [-1.5, -.75], "last_block": [-1.2, -1.1], "full": [0, -.2],
                  "last_reference": [-1.7, -.6], "calibration_mean": [-.8, -1.3]}[method]
        if neural_tie and method in ("head", "last_block", "full"):
            offset = [-1.5, -.75]
        values = baseline + np.asarray(offset) + (seed - 43) * .2
        if method == "full" and not neural_tie:
            values = values + np.column_stack(([-3, -2, -1, 0, 1, 5], [-2, -1, 0, .2, 1, 1.8]))
        if method == "full" and all_full_better:
            values = baseline + [-2, -1] + (seed - 43) * .2
        selected = np.flatnonzero(counts)
        if not len(selected):
            return None
        rows = {patients[i]: {"n_windows": int(counts[i]), "mae": values[i].tolist(),
                              "rmse": np.sqrt(values[i] ** 2 + 4).tolist()} for i in selected}
        return {"n_windows": int(counts.sum()), "n_patients": len(selected), "per_patient": rows,
                "macro": {metric: np.mean([row[metric] for row in rows.values()], axis=0).tolist()
                          for metric in ("mae", "rmse")}}

    for budget in BUDGETS:
        for method in METHODS:
            if method == "affine" and budget < 3:
                continue
            for seed in SEEDS:
                bs[(method, budget, seed)] = summary(method, budget, seed, total_counts)
                if budget == 10:
                    entries = []
                    for axis in ("elapsed_since_reference_min", "elapsed_since_boundary_min"):
                        for (lo, hi), counts in zip(BIN_BOUNDS, bin_counts):
                            entries.append({"axis": axis, "lower": lo, "upper": hi, "n_windows": int(counts.sum()),
                                            "n_patients": int(np.count_nonzero(counts)),
                                            "metrics": summary(method, budget, seed, counts)})
                    bd[(method, budget, seed)] = entries
    bc = {(method, budget): {metric: paired_patient_change([bs[(method, budget, seed)] for seed in SEEDS],
                [bs[("population", budget, seed)] for seed in SEEDS], metric) for metric in ("mae", "rmse")}
          for budget in BUDGETS for method in METHODS if method != "population" and (method != "affine" or budget >= 3)}
    return bs, bc, bd


class PersonalInterpretationTests(unittest.TestCase):
    def test_each_budget_separate_outcomes_and_incremental_control_comparisons(self):
        tex, details = build_personal_interpretation(*interpretation_fixture())
        self.assertEqual(set(details["budgets"]), {"1", "2", "3", "5", "10"})
        for budget in BUDGETS:
            self.assertIn(f"\\textbf{{K={budget}.}}", tex)
            outcomes = details["budgets"][str(budget)]
            self.assertEqual(set(outcomes), {"SBP", "DBP"})
            sbp, dbp = outcomes["SBP"], outcomes["DBP"]
            self.assertEqual(sbp["lowest_observed_methods"], ["affine"] if budget == 10 else ["last_reference"])
            self.assertEqual(sbp["lowest_neural_methods"], ["head"])
            self.assertEqual(dbp["lowest_observed_methods"], ["calibration_mean"])
            self.assertEqual(dbp["lowest_neural_methods"], ["last_block"])
            self.assertAlmostEqual(sbp["neural_vs_controls"]["bias"]["mae_change"], -.5)
            self.assertAlmostEqual(sbp["neural_vs_controls"]["last_reference"]["mae_change"], .2)
            self.assertAlmostEqual(sbp["neural_vs_controls"]["calibration_mean"]["mae_change"], -.7)
            self.assertAlmostEqual(dbp["neural_vs_controls"]["calibration_mean"]["mae_change"], .2)
        self.assertIn("not used to select a deployment model", tex)
        self.assertIn("reference count and recency", tex)
        json.dumps(details, allow_nan=False)  # All results must remain machine-readable.

    def test_patient_counts_and_absolute_worst_harm(self):
        tex, details = build_personal_interpretation(*interpretation_fixture())
        counts = details["k10_patient_counts"]["full"]
        self.assertEqual(counts["SBP"], {"helped": 3, "harmed": 2, "unchanged": 1, "n_patients": 6})
        self.assertEqual(counts["DBP"], {"helped": 3, "harmed": 2, "unchanged": 1, "n_patients": 6})
        worst = details["worst_full_adaptation_change"]["SBP"]
        self.assertTrue(worst["positive_harm_observed"])
        self.assertEqual(worst["patients"][0]["patient"], "unit_patient_5")
        self.assertAlmostEqual(worst["patients"][0]["population_mae"], 15)
        self.assertAlmostEqual(worst["patients"][0]["full_mae"], 20)
        self.assertAlmostEqual(worst["largest_mae_change"], 5)
        self.assertIn(r"unit\_patient\_5", tex)
        self.assertIn("20.00 versus 15.00", tex)
        self.assertIn("Whose prediction error improves or worsens", tex)
        self.assertIn("largest increase in prediction error", tex)
        self.assertNotIn("harmed", tex)
        self.assertNotIn("adaptation harm", tex)

    def test_drift_pairs_within_bins_and_does_not_multiply_counts_by_seeds(self):
        tex, details = build_personal_interpretation(*interpretation_fixture())
        bins = details["k10_reference_age_bins"]
        self.assertEqual([(b["n_patients"], b["n_windows"]) for b in bins], [(6, 87), (4, 46), (0, 0), (2, 25)])
        self.assertIsNone(bins[2]["paired_mae_changes"]["full"])
        self.assertAlmostEqual(bins[3]["paired_mae_changes"]["full"]["estimate"][0], 3.0)
        self.assertAlmostEqual(bins[1]["paired_mae_changes"]["full"]["estimate"][0], -1.5)
        self.assertEqual(details["full_model_drift_outcomes"]["SBP"]["below_population_bins"], 1)
        self.assertEqual(details["full_model_drift_outcomes"]["SBP"]["above_population_bins"], 1)
        self.assertIn("n=2, N=25", tex)
        self.assertIn("coverage change across age bins", tex)
        self.assertIn("cross-visit persistence", tex)
        self.assertNotIn(r"\\[", tex)
        self.assertIn(r"\\{[", tex)

    def test_ties_and_all_patients_improving_do_not_fabricate_harm_or_unique_winner(self):
        tex, details = build_personal_interpretation(*interpretation_fixture(all_full_better=True))
        self.assertFalse(details["worst_full_adaptation_change"]["SBP"]["positive_harm_observed"])
        self.assertAlmostEqual(details["worst_full_adaptation_change"]["SBP"]["largest_mae_change"], -2)
        self.assertEqual(len(details["worst_full_adaptation_change"]["SBP"]["patients"]), 6)
        self.assertIn("no patient's prediction error worsened", tex)
        _, tied = build_personal_interpretation(*interpretation_fixture(neural_tie=True))
        self.assertEqual(tied["budgets"]["1"]["SBP"]["lowest_neural_methods"], ["head", "last_block", "full"])
        self.assertEqual(tied["budgets"]["1"]["SBP"]["representative_lowest_neural"], "head")

    def test_empty_late_bins_are_unavailable_not_claimed_as_durability(self):
        tex, details = build_personal_interpretation(*interpretation_fixture(empty_last=True))
        self.assertEqual(details["k10_reference_age_bins"][-1]["n_windows"], 0)
        self.assertIsNone(details["k10_reference_age_bins"][-1]["paired_mae_changes"]["head"])
        self.assertEqual(details["full_model_drift_outcomes"]["SBP"]["nonempty_bins"], 2)
        self.assertIn("Empty bins are unavailable", tex)

    def test_rejects_missing_matrix_corrupt_pairing_and_bin_coverage(self):
        bs, bc, bd = interpretation_fixture()
        incomplete = copy.deepcopy(bs)
        incomplete.pop(("head", 1, 42))
        with self.assertRaisesRegex(ValueError, "complete"):
            build_personal_interpretation(incomplete, bc, bd)
        corrupt = copy.deepcopy(bc)
        corrupt[("full", 10)]["mae"]["patient_changes"]["unit_patient_5"][0] += 1
        with self.assertRaisesRegex(ValueError, "Paired patient"):
            build_personal_interpretation(bs, corrupt, bd)
        mismatched = copy.deepcopy(bd)
        mismatched[("full", 10, 42)][0]["n_windows"] += 1
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            build_personal_interpretation(bs, bc, mismatched)
        missing_bin = copy.deepcopy(bd)
        missing_bin[("full", 10, 42)].pop(0)
        with self.assertRaisesRegex(ValueError, "Missing"):
            build_personal_interpretation(bs, bc, missing_bin)


if __name__ == "__main__":
    unittest.main()
