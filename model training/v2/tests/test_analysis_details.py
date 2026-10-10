"""Synthetic helper fixtures only; never generate or replace a final report."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
from reports import analysis_details
from shared.metrics import paired_patient_change, summarize


def analysis_fixture():
    """Unequal patient row counts expose window-vs-patient weighting errors."""
    cohort = {"target": [], "development": [], "manifest_hash": "synthetic-unit-fixture"}
    arrays = {key: [] for key in ("y", "subjects", "source_indices", "start", "end_exclusive")}
    for i in range(6):
        n = i + 1
        source = np.arange(n, dtype=np.int64) * 3 + 101
        start = 900 + i * 900 + np.arange(n) * 10.
        y = np.column_stack((np.full(n, 110 + i * 2.), np.full(n, 70 + i))).astype(np.float32)
        patient = f"synthetic-patient-{i}"
        record = {"patient_id": patient, "case_id": f"synthetic-case-{i}",
                  "x": np.tile(np.linspace(0, 1, 1250, dtype=np.float32), (n, 2, 1)),
                  "y": y, "source_indices": source, "start": start, "end_exclusive": start + 10,
                  "query_indices": np.arange(n), "support_indices": np.arange(n),
                  "calibration_boundary": 0., "subject_id": patient}
        cohort["target"].append(record)
        for key in arrays:
            arrays[key].append(np.repeat(patient, n) if key == "subjects" else record[key])
    joined = {key: np.concatenate(value) for key, value in arrays.items()}
    joined["elapsed_since_reference_min"] = joined["end_exclusive"] / 60
    joined["elapsed_since_boundary_min"] = joined["end_exclusive"] / 60
    variants = ["baseline", "se_last", "additive_last", "se_all"]
    errors = {"baseline": [10, 5], "se_last": [6, 3], "additive_last": [7, 4], "se_all": [5, 2]}
    aruns = {}
    for variant in variants:
        for seed in (42, 43, 44):
            pred = joined["y"] + np.asarray(errors[variant]) + seed - 43
            summary = summarize(joined["y"], pred, joined["subjects"])
            aruns[f"{variant}_seed{seed}"] = {"metrics": {"test": summary, "validation": summary}}
    apaired = {v: {metric: paired_patient_change(
        [aruns[f"{v}_seed{s}"]["metrics"]["test"] for s in (42, 43, 44)],
        [aruns[f"baseline_seed{s}"]["metrics"]["test"] for s in (42, 43, 44)], metric)
        for metric in ("mae", "rmse")} for v in variants if v != "baseline"}
    bs, bp, bd = {}, {}, {}
    errors = {"population": [10, 5], "bias": [4, 2], "head": [7, 3],
              "last_block": [6, 2.5], "full": [2, 1], "calibration_mean": [8, 4]}
    for method, error in errors.items():
        for budget in (1, 5, 10):
            for seed in (42, 43, 44):
                key = (method, budget, seed)
                p = {**joined, "pred": joined["y"] + np.asarray(error) + seed - 43}
                bp[key] = p
                bs[key] = summarize(p["y"], p["pred"], p["subjects"])
                bd[key] = []
                for lower, upper in ((0, 15), (15, 30), (30, 60), (60, np.inf)):
                    keep = (p["elapsed_since_reference_min"] >= lower) & (p["elapsed_since_reference_min"] < upper)
                    bd[key].append({"n_windows": int(keep.sum()), "n_patients": int(len(np.unique(p["subjects"][keep]))),
                                    "metrics": summarize(p["y"][keep], p["pred"][keep], p["subjects"][keep]) if keep.any() else None})
    return cohort, variants, aruns, apaired, bs, bp, bd


class AnalysisDetailTests(unittest.TestCase):
    def test_scalar_formatting_and_table_column_contract(self):
        self.assertEqual(analysis_details.fmt(None), "--")
        self.assertEqual(analysis_details.fmt(np.nan), "--")
        self.assertEqual(analysis_details.fmt(1.234), "1.23")
        self.assertEqual(analysis_details.spread([1., 2., 3.]), r"2.00 $\pm$ 1.00")
        self.assertEqual(analysis_details.spread([2.]), "2.00")
        latex = analysis_details.table(["A", "B", "C"], [["x", "1", "2"]], "Fixture only.")
        self.assertIn(r"\begin{tabular}{lrr}", latex)
        self.assertIn(r"A & B & C\\\midrule", latex)
        self.assertIn(r"x & 1 & 2\\", latex)

    def test_gap_breaks_keep_real_time_and_do_not_bridge_missing_observations(self):
        x = np.array([10., 20., 110., 120.])
        y = np.array([1., 2., 3., 4.])
        start = np.array([0., 10., 100., 110.])
        end = start + 10
        gap_x, gap_y = analysis_details.insert_gap_breaks(x, y, start, end)
        np.testing.assert_array_equal(gap_x, [10, 20, np.nan, 110, 120])
        np.testing.assert_array_equal(gap_y, [1, 2, np.nan, 3, 4])
        # Boundary-touching, tiny floating-point deviations and audited small
        # overlaps must not add spurious missing intervals.
        close_start = np.array([0., 10. + 5e-7, 19.992])
        x2, y2 = analysis_details.insert_gap_breaks(np.arange(3.), np.ones(3), close_start, close_start + 10)
        np.testing.assert_array_equal(x2, np.arange(3.))
        np.testing.assert_array_equal(y2, np.ones(3))
        for bad_x, bad_y, bad_start, bad_end in (
            (x, y[:-1], start, end), (x, y, start[::-1], end[::-1]),
            (x, y, start, start), (x, y * np.nan, start, end),
        ):
            with self.assertRaises(ValueError):
                analysis_details.insert_gap_breaks(bad_x, bad_y, bad_start, bad_end)

    def _call_details(self, out, fixture, expected_hash="synthetic-unit-fixture"):
        cohort, variants, aruns, apaired, bs, bp, bd = fixture
        return analysis_details.scientific_details(
            out, variants, aruns, {}, apaired, bs, bp, bd, {}, out / "synthetic-input",
            ["synthetic-patient-1", "synthetic-patient-5"], "se_last",
            expected_manifest_hash=expected_hash)

    def test_counts_paired_drift_and_actual_waveform_row_alignment(self):
        fixture = analysis_fixture()
        plots = []
        figures = {}
        original_plot = Axes.plot

        def capture_plot(axis, *args, **kwargs):
            if kwargs.get("label") == "Bias" and len(args) >= 2 and len(args[0]) == 4:
                plots.append(np.asarray(args[1], dtype=float))
            return original_plot(axis, *args, **kwargs)

        def capture_figure(fig, path, **kwargs):
            # Keep an in-memory figure for axis/legend assertions, never write
            # a synthetic scientific PDF that could be mistaken for a result.
            figures[Path(path).stem] = fig

        with tempfile.TemporaryDirectory(prefix="bad-blood-analysis-unit-only-") as temp:
            out = Path(temp)
            (out / "figures").mkdir()
            (out / "results").mkdir()
            with patch.object(analysis_details, "load_personal_cohort", return_value=fixture[0]), \
                    patch.object(Figure, "savefig", new=capture_figure), patch.object(Axes, "plot", new=capture_plot):
                latex, notes = self._call_details(out, fixture)
            # All 21 rows occupy the middle range, and six patients—not 63
            # rows or 18 patients after repeating three seeds—are counted.
            self.assertIn("CNN: 100 to below 140 & 21 & 6 & 10.00 & 10.00 & 10.00", latex)
            self.assertIn("CNN: below 100 & 0 & 0 & -- & -- & --", latex)
            self.assertIn("Three-seed comparisons average seeds 42/43/44 within patient", latex)
            self.assertIn("ECG-only/PPG-only ablations compare seed 42 only", latex)
            self.assertIn("The bootstrap resamples 141 patients, not windows", latex)
            self.assertEqual(notes, "")
            self.assertEqual(len(plots), 2)
            np.testing.assert_allclose(plots[0], [np.nan, -6, -6, -6], equal_nan=True)
            np.testing.assert_allclose(plots[1], [np.nan, -3, -3, -3], equal_nan=True)
            saved = json.loads((out / "results/example_windows.json").read_text())
            self.assertEqual([row["source_index"] for row in saved], [101, 101])
            self.assertEqual(saved[0]["reference_sbp_dbp"], [112., 71.])
            self.assertEqual(saved[1]["reference_sbp_dbp"], [120., 75.])
            self.assertEqual(saved[0]["predictions_sbp_dbp"]["bias"], [115., 72.])
            ba_axes = figures["repeated_bland_altman"].axes
            for row in (0, 3):
                self.assertTrue(ba_axes[row].get_shared_x_axes().joined(ba_axes[row], ba_axes[row + 2]))
                self.assertTrue(ba_axes[row].get_shared_y_axes().joined(ba_axes[row], ba_axes[row + 2]))
            legend = figures["repeated_bland_altman"].legends[0]
            self.assertEqual([text.get_text() for text in legend.get_texts()], [f"synthetic-patient-{i}" for i in range(6)])
            self.assertFalse((out / "final_report.tex").exists())
            self.assertFalse((out / "final_report.pdf").exists())

    def test_example_inputs_require_verified_manifest_and_matching_labels(self):
        for altered_hash, altered_labels in ((True, False), (False, True)):
            fixture = list(analysis_fixture())
            if altered_labels:
                fixture[0] = copy.deepcopy(fixture[0])
                fixture[0]["target"][1]["y"][0] += 5
            with tempfile.TemporaryDirectory(prefix="bad-blood-analysis-unit-only-") as temp:
                out = Path(temp)
                (out / "figures").mkdir()
                (out / "results").mkdir()
                with patch.object(analysis_details, "load_personal_cohort", return_value=fixture[0]), \
                        patch.object(Figure, "savefig"):
                    with self.assertRaises(ValueError):
                        self._call_details(out, tuple(fixture), "wrong-source" if altered_hash else "synthetic-unit-fixture")
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
