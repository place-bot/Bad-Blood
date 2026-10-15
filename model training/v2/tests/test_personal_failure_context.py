"""Memory-only synthetic fixtures for descriptive reference context."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
from reports.personal_failure_context import EXPECTED_PATIENTS, build_personal_failure_context


def context_fixture():
    entries = []
    for patient in sorted(EXPECTED_PATIENTS):
        record = {"phase": "test", "role": "target", "strategy": "population", "budget": 10, "seed": 42,
                  "patient": patient, "case": "unit-test-case", "config_hash": "unit-config", "checkpoint_sha256": "unit-checkpoint",
                  "summary": {"n_windows": 4}, "unit_path": "unit-fixture/only"}
        support = np.column_stack((np.arange(100., 110.), np.arange(60., 70.)))
        query = np.array([[110., 60.], [112., 62.], [114., 64.], [116., 66.]])
        arrays = {"support_y": support, "y": query, "subjects": np.repeat(patient, 4),
                  "case_id": np.array("unit-test-case"), "config_hash": np.array("unit-config"), "checkpoint_sha256": np.array("unit-checkpoint"),
                  "support_source_indices": np.arange(10), "source_indices": np.arange(20, 24),
                  "support_start": np.arange(10) * 60., "support_end_exclusive": np.arange(10) * 60. + 10,
                  "start": np.arange(4) * 10. + 610, "end_exclusive": np.arange(4) * 10. + 620,
                  "calibration_boundary": np.array(600.)}
        entries.append({"record": record, "predictions": arrays})
    return entries


class PersonalFailureContextTests(unittest.TestCase):
    def test_exact_reference_statistics_signed_shift_and_full_ranges(self):
        tex, details = build_personal_failure_context(context_fixture())
        patient = details["patients"][sorted(EXPECTED_PATIENTS)[0]]
        self.assertEqual(patient["support"]["n_windows"], 10)
        self.assertEqual(patient["query"]["n_windows"], 4)
        self.assertEqual(patient["support"]["targets"]["SBP"]["mean"], 104.5)
        self.assertAlmostEqual(patient["support"]["targets"]["SBP"]["sample_sd"], np.sqrt(82.5 / 9))
        self.assertEqual(patient["query"]["targets"]["SBP"]["range"], [110., 116.])
        self.assertEqual(patient["support"]["targets"]["DBP"]["range"], [60., 69.])
        self.assertEqual(patient["query_minus_support_mean"], {"SBP": 8.5, "DBP": -1.5})
        self.assertEqual(patient["support_elapsed_span_seconds"], 550.)
        self.assertEqual(patient["query_elapsed_span_seconds"], 40.)
        self.assertIn("+8.5", tex)
        self.assertIn("-1.5", tex)
        self.assertEqual(tex.count(r"\begin{table*}"), 1)
        self.assertIn("not prediction scores", tex)
        self.assertIn("not evidence that a particular physiological mechanism or overfitting caused", tex)
        json.dumps(details, allow_nan=False)

    def test_function_is_pure_and_does_not_use_predictions(self):
        entries = context_fixture()
        before = copy.deepcopy(entries)
        original = build_personal_failure_context(entries)
        for entry in entries:
            entry["predictions"]["pred"] = np.full((4, 2), np.nan)
        repeated = build_personal_failure_context(entries)
        self.assertEqual(original, repeated)
        for entry, old in zip(entries, before):
            for key, value in old["predictions"].items():
                np.testing.assert_array_equal(entry["predictions"][key], value)

    def test_rejects_partial_duplicate_or_wrong_phase_budget_seed(self):
        entries = context_fixture()
        with self.assertRaises(ValueError):
            build_personal_failure_context(entries[:-1])
        with self.assertRaises(ValueError):
            build_personal_failure_context(entries[:-1] + [entries[0]])
        for key, value in (("phase", "development"), ("seed", 43), ("budget", 5), ("strategy", "full")):
            changed = copy.deepcopy(entries)
            changed[0]["record"][key] = value
            with self.assertRaises(ValueError):
                build_personal_failure_context(changed)

    def test_rejects_label_identity_temporal_overlap_and_nonfinite_references(self):
        entries = context_fixture()
        for key, value in (("config_hash", np.array("wrong")), ("case_id", np.array("wrong")),
                           ("support_y", np.zeros((9, 2))), ("y", np.full((4, 2), np.nan)),
                           ("source_indices", np.arange(4)), ("start", np.arange(4) * 10. + 590)):
            changed = copy.deepcopy(entries)
            changed[0]["predictions"][key] = value
            with self.assertRaises(ValueError):
                build_personal_failure_context(changed)


if __name__ == "__main__":
    unittest.main()
