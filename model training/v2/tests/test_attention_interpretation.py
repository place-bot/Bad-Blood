"""In-memory synthetic summaries; never generate a final report or PDF."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from reports.attention_interpretation import (
    ATTENTION_SET, MODALITIES, PARAMETERS, REQUIRED_THREE_SEED, SEEDS,
    build_attention_interpretation, interval_relation,
)


def fixture(include_eca=False):
    """Exact frozen count dimensions with clearly fictional patient identities."""
    runs = {}
    val_deltas = {"baseline": 0., "se_last": -.4, "temporal_last": -1.,
                  "cbam_last": -.7, "additive_last": -3., "se_all": -.2,
                  "baseline_ecg": 2., "baseline_ppg": 1., "eca_last": -5.}
    test_deltas = {"baseline": [0., 0.], "temporal_last": [1.5, -.2],
                   "cbam_last": [-.5, -.1], "additive_last": [-1., -.8],
                   "se_all": [-.2, -.4], "baseline_ecg": [2., 1.],
                   "baseline_ppg": [1., 2.], "eca_last": [-2., -1.]}
    variants = (*REQUIRED_THREE_SEED, *MODALITIES, *(("eca_last",) if include_eca else ()))
    for variant in variants:
        for seed in ((42,) if variant in MODALITIES else SEEDS):
            run = {"config": {"variant": variant, "seed": seed},
                   "completion": {"status": "complete", "toy_test_only": False,
                                  "variant": variant, "seed": seed, "parameters": PARAMETERS[variant]},
                   "metrics": {}}
            for split, n, count in (("validation", 192, 360), ("test", 141, 400)):
                people = {}
                for i in range(n):
                    if split == "validation":
                        mae = np.array([30., 20.]) + val_deltas[variant] + .1 * (seed - 43)
                    else:
                        baseline = np.array([12. + i % 4, 7. + i % 2]) + .1 * (seed - 43)
                        delta = (np.array([-4. if i < 50 else 1., -.5]) + .25 * (seed - 43)
                                 if variant == "se_last" else np.asarray(test_deltas[variant]))
                        mae = baseline + delta
                    # Unequal row counts preserve total size and expose any
                    # accidental weighting by windows instead of people.
                    n_windows = count + (10 if i == 0 else -10 if i == 1 else 0)
                    people[f"synthetic_{split}_{i:03d}"] = {
                        "n_windows": n_windows, "mae": mae.tolist(), "rmse": (mae + 2).tolist()}
                run["metrics"][split] = {
                    "target_order": ["SBP", "DBP"], "n_patients": n, "n_windows": n * count,
                    "per_patient": people,
                    "macro": {metric: np.mean([p[metric] for p in people.values()], 0).tolist()
                              for metric in ("mae", "rmse")}}
            runs[f"{variant}_seed{seed}"] = run
    return runs


def change_summary(run, split, transform):
    """Consistent in-memory endpoint edits for reporting edge cases only."""
    summary = run["metrics"][split]
    for index, patient in enumerate(summary["per_patient"].values()):
        patient["mae"] = np.asarray(transform(index, np.asarray(patient["mae"])), float).tolist()
        patient["rmse"] = (np.asarray(patient["mae"]) + 2).tolist()
    summary["macro"] = {metric: np.mean([p[metric] for p in summary["per_patient"].values()], 0).tolist()
                        for metric in ("mae", "rmse")}


class AttentionInterpretationTests(unittest.TestCase):
    def test_validation_roles_exclude_optional_model_and_never_use_test_rank(self):
        runs = fixture(include_eca=True)
        latex, data = build_attention_interpretation(runs, "additive_last")
        selection = data["selection"]
        self.assertEqual(selection["attention_candidates"], list(ATTENTION_SET))
        self.assertEqual(selection["attention_winner"], "temporal_last")
        self.assertEqual(selection["required_winner"], "additive_last")
        self.assertFalse(selection["same_winner"])
        self.assertNotIn("eca_last", selection["validation_mean_macro_mae"])
        self.assertIn("Supplementary ECA", latex)
        self.assertIn("The choices differ", latex)
        # Make an ablation arbitrarily best on the test: selection must not move.
        altered = copy.deepcopy(runs)
        row = altered["se_all_seed42"]["metrics"]["test"]
        for patient in row["per_patient"].values():
            patient["mae"] = [0., 0.]
            patient["rmse"] = [0., 0.]
        row["macro"] = {"mae": [0., 0.], "rmse": [0., 0.]}
        _, altered_data = build_attention_interpretation(altered, "additive_last")
        self.assertEqual(selection, altered_data["selection"])
        with self.assertRaisesRegex(ValueError, "validation choice"):
            build_attention_interpretation(runs, "eca_last")

    def test_contrasts_preserve_patient_weight_seeds_targets_and_capacity_roles(self):
        latex, data = build_attention_interpretation(fixture(), "additive_last")
        pair = data["comparisons"]["se_last-minus-baseline"]["mae"]
        expected_sbp = (-4 * 50 + 91) / 141
        np.testing.assert_allclose(pair["estimate"], [expected_sbp, -.5], atol=1e-12)
        self.assertEqual(pair["n_patients"], 141)
        self.assertEqual(pair["resamples"], 2000)
        self.assertEqual(pair["target_details"]["SBP"]["patients_lower_error"], 50)
        self.assertEqual(pair["target_details"]["SBP"]["patients_higher_error"], 91)
        self.assertEqual(pair["target_details"]["DBP"]["patients_lower_error"], 141)
        np.testing.assert_allclose(pair["seed_macro_changes"],
                                   [[expected_sbp - .25, -.75], [expected_sbp, -.5], [expected_sbp + .25, -.25]])
        capacity = data["comparisons"]["se_last-minus-additive_last"]
        np.testing.assert_allclose(capacity["mae"]["estimate"], [expected_sbp + 1, .3])
        self.assertEqual(capacity["seeds"], [42, 43, 44])
        modality = data["comparisons"]["baseline_ecg-minus-baseline"]
        self.assertEqual(modality["seeds"], [42])
        self.assertEqual(modality["n_seeds"], 1)
        np.testing.assert_allclose(modality["mae"]["estimate"], [2., 1.])
        self.assertIn("374 more parameters", latex)
        self.assertIn("not a factorial test", latex)
        self.assertIn("cannot establish seed-robust sensor importance", latex)
        self.assertIn("not missing-sensor tests", latex)
        self.assertIn("no minimum clinically useful effect was prespecified", latex)
        json.dumps(data, allow_nan=False)

    def test_complete_real_inventory_and_summary_consistency_are_mandatory(self):
        mutations = (
            lambda r: r.pop("baseline_seed44"),
            lambda r: r["baseline_seed42"]["completion"].update(toy_test_only=True),
            lambda r: r["se_last_seed42"]["config"].update(seed=43),
            lambda r: r["se_last_seed42"]["completion"].update(parameters=37666),
            lambda r: r["se_last_seed42"]["metrics"]["test"]["macro"].update(mae=[1., 2.]),
            lambda r: r["se_last_seed42"]["metrics"]["test"].update(target_order=["DBP", "SBP"]),
            lambda r: r["se_last_seed42"]["metrics"]["test"]["per_patient"]["synthetic_test_000"].update(n_windows=400),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                runs = fixture()
                mutate(runs)
                with self.assertRaises(ValueError):
                    build_attention_interpretation(runs)
        partial_eca = fixture(include_eca=True)
        del partial_eca["eca_last_seed44"]
        with self.assertRaisesRegex(ValueError, "inventory"):
            build_attention_interpretation(partial_eca)

    def test_uncertainty_language_and_table_tex_are_safe(self):
        self.assertEqual(interval_relation(-2, -1), "wholly below zero")
        self.assertEqual(interval_relation(1, 2), "wholly above zero")
        self.assertEqual(interval_relation(-1, 1), "includes or touches zero")
        self.assertEqual(interval_relation(0, 2), "includes or touches zero")
        self.assertEqual(interval_relation(-2, 0), "includes or touches zero")
        for bounds in ((2, 1), (np.nan, 1), (-1, np.inf)):
            with self.assertRaises(ValueError):
                interval_relation(*bounds)
        runs = fixture()
        before = copy.deepcopy(runs)
        latex, _ = build_attention_interpretation(runs)
        self.assertEqual(runs, before)
        self.assertEqual(latex.count(r"\begin{table*}"), 3)
        self.assertEqual(latex.count(r"\end{table*}"), 3)
        self.assertNotIn(r"\\[", latex)
        self.assertIn("not evidence of equivalence", latex)
        self.assertIn("not multiplicity-adjusted", latex)

    def test_validation_table_separates_targets_single_seeds_and_architecture_mean(self):
        latex, data = build_attention_interpretation(fixture(include_eca=True))
        rows = data["validation_table"]
        self.assertEqual(tuple(rows), REQUIRED_THREE_SEED)
        self.assertEqual(rows["baseline"]["seeds"], [42, 43, 44])
        np.testing.assert_allclose(rows["baseline"]["sbp_dbp_by_seed"],
                                   [[29.9, 19.9], [30., 20.], [30.1, 20.1]])
        np.testing.assert_allclose(rows["baseline"]["sbp_dbp_seed_mean"], [30., 20.])
        np.testing.assert_allclose(rows["baseline"]["selection_score_by_seed"], [24.9, 25., 25.1])
        self.assertAlmostEqual(rows["baseline"]["selection_score_mean"], 25.)
        for variant, row in rows.items():
            self.assertEqual(row["selection_score_mean"], data["selection"]["validation_mean_macro_mae"][variant])
        table = latex.split(r"\begin{table*}", 1)[1].split(r"\end{table*}", 1)[0]
        self.assertIn(r"CNN & 30.000 & 20.000 & 24.900 & 25.000 & 25.100 & 25.000\\", table)
        self.assertIn(r"Additive & 27.000 & 17.000 & 21.900 & 22.000 & 22.100 & 22.000\\", table)
        self.assertIn(r"$Q_{42}$ & $Q_{43}$ & $Q_{44}$ & $\overline{Q}$", table)
        self.assertNotIn("ECA", table)
        self.assertNotIn("ECG only", table)
        self.assertIn("not primary-test error", table)
        self.assertIn("selection uses unrounded values", table)
        self.assertIn("not predictions in an ensemble", latex)
        self.assertIn("a seed-42 score is one run, not the three-seed mean", latex)

    def test_observed_test_minima_are_descriptive_target_specific_and_do_not_reselect(self):
        runs = fixture(include_eca=True)
        _, data = build_attention_interpretation(runs)
        observed = data["observed_test_evidence"]
        self.assertEqual(observed["candidate_models"], list(REQUIRED_THREE_SEED))
        self.assertFalse(observed["selection_uses_test"])
        self.assertNotIn("eca_last", observed["test_mean_macro_mae"])
        self.assertEqual(observed["lowest_observed_by_target"]["SBP"]["variants"], ["additive_last"])
        self.assertEqual(observed["lowest_observed_by_target"]["DBP"]["variants"], ["additive_last"])
        for seed in SEEDS:
            change_summary(runs[f"se_all_seed{seed}"], "test", lambda i, y: [1., y[1]])
            change_summary(runs[f"cbam_last_seed{seed}"], "test", lambda i, y: [y[0], 1.])
        _, updated = build_attention_interpretation(runs)
        self.assertEqual(data["selection"], updated["selection"])
        observed = updated["observed_test_evidence"]
        self.assertEqual(observed["lowest_observed_by_target"]["SBP"]["variants"], ["se_all"])
        self.assertEqual(observed["lowest_observed_by_target"]["DBP"]["variants"], ["cbam_last"])
        self.assertIn("SBP 1.000 mmHg for SE-all (placement ablation)", observed["abstract_latex"])
        self.assertIn("DBP 1.000 mmHg for CBAM1D", observed["abstract_latex"])
        self.assertIn("not a deployment-selection decision", observed["discussion_latex"])
        self.assertIn("not adjusted for multiple comparisons or selecting an observed minimum", observed["discussion_latex"])

    def test_validation_selected_baseline_does_not_suppress_se_test_improvement_or_uncertainty(self):
        runs = fixture()
        for seed in SEEDS:
            change_summary(runs[f"baseline_seed{seed}"], "validation", lambda i, y: [1., 1.])
            baseline = list(runs[f"baseline_seed{seed}"]["metrics"]["test"]["per_patient"].values())
            change_summary(runs[f"se_last_seed{seed}"], "test", lambda i, y:
                           np.asarray(baseline[i]["mae"]) + [-1. if i < 72 else 1., -.5])
        _, data = build_attention_interpretation(runs, "baseline")
        self.assertEqual(data["selection"]["attention_winner"], "baseline")
        observed = data["observed_test_evidence"]
        pair = data["comparisons"]["se_last-minus-baseline"]["mae"]
        self.assertLess(pair["estimate"][0], 0)
        self.assertLess(pair["ci_lower"][0], 0)
        self.assertGreater(pair["ci_upper"][0], 0)
        self.assertLess(pair["ci_upper"][1], 0)
        self.assertIn("SE-last has primary-test SBP/DBP MAE", observed["abstract_latex"])
        self.assertIn("descriptive paired 95\\% patient intervals", observed["abstract_latex"])
        self.assertIn("SE-last's observed SBP MAE is lower than CNN's", observed["discussion_latex"])
        self.assertIn("its paired interval includes or touches zero", observed["discussion_latex"])
        self.assertIn("its paired interval is wholly below zero", observed["discussion_latex"])

    def test_observed_baseline_minimum_has_no_fabricated_self_comparison(self):
        runs = fixture()
        for variant in REQUIRED_THREE_SEED[1:]:
            for seed in SEEDS:
                change_summary(runs[f"{variant}_seed{seed}"], "test", lambda i, y: [100., 100.])
        _, data = build_attention_interpretation(runs)
        observed = data["observed_test_evidence"]
        for target in ("SBP", "DBP"):
            self.assertEqual(observed["lowest_observed_by_target"][target]["variants"], ["baseline"])
            self.assertEqual(observed["lowest_observed_by_target"][target]["baseline_contrasts"], {})
        self.assertNotIn("CNN minus CNN", observed["discussion_latex"])
        self.assertIn("SE-last's observed SBP MAE is higher", observed["discussion_latex"])
        json.dumps(data, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
