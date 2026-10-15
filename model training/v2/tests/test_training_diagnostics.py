"""Synthetic in-memory diagnostics fixtures; all figure writes are intercepted."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from reports.training_diagnostics import (
    FIGURES, PRIMARY, REQUIRED_THREE_SEED, SEEDS, SINGLE_SEED,
    build_training_diagnostics, summarize_training_histories,
)


class NeverReadMetrics:
    def __getitem__(self, key):
        raise AssertionError("Report-only history diagnostics must not read test/summary metrics")


def fixture(include_eca=False):
    """Small fictional histories with valid complete-inventory identity metadata."""
    runs = {}
    variants = (*REQUIRED_THREE_SEED, *SINGLE_SEED, *(("eca_last",) if include_eca else ()))
    for variant_index, variant in enumerate(variants):
        for seed in ((42,) if variant in SINGLE_SEED else SEEDS):
            selected = seed - 41
            actual = selected + 8
            history = []
            for epoch in range(1, actual + 1):
                score = 8. + .25 * variant_index + .1 * (epoch - selected) ** 2
                history.append({"epoch": epoch, "train_normalized_MSE": 1. / epoch + .01 * variant_index,
                                "validation": {"macro_mae": [score + 2., score - 2.],
                                               "mae": [score + 1., score - 1.]},
                                "selection_macro_mae": score, "seconds": 1.5 + .01 * epoch,
                                "lr": .001 if epoch < 6 else .0005})
            runs[f"{variant}_seed{seed}"] = {
                "config": {"variant": variant, "seed": seed, "toy_test_only": False},
                "completion": {"variant": variant, "seed": seed, "status": "complete", "toy_test_only": False,
                               "actual_epochs": actual, "epochs_completed": actual,
                               "selected_epoch": selected, "best_epoch": selected, "early_stopped": True,
                               "best_validation_macro_mae": history[selected - 1]["selection_macro_mae"],
                               "config_hash": "synthetic-config", "manifest_hash": "synthetic-manifest",
                               "best_checkpoint_sha256": "synthetic-checkpoint"},
                "history": history, "metrics": NeverReadMetrics(),
            }
    return runs


class TrainingDiagnosticTests(unittest.TestCase):
    def test_summary_keeps_exact_source_values_and_different_objective_units(self):
        runs = fixture()
        before = copy.deepcopy(runs["baseline_seed42"]["history"])
        details = summarize_training_histories(runs)
        self.assertEqual(details["required_completed_runs"], 20)
        self.assertEqual(details["total_completed_runs"], 20)
        record = details["runs"]["baseline_seed42"]
        self.assertEqual(record["actual_epochs"], 9)
        self.assertEqual(record["selected_epoch"], 1)
        self.assertEqual(record["history"], before)
        self.assertEqual(record["source_history_relative_path"], "attention/baseline_seed42/history.json")
        self.assertEqual(len(record["canonical_history_sha256"]), 64)
        self.assertIn("not the source", details["history_digest_definition"])
        self.assertIn("Dimensionless", details["training_objective"])
        self.assertIn("mmHg", details["validation_objective"])
        self.assertIn("no test metrics", details["source_scope"])
        record["history"][0]["validation"]["macro_mae"][0] = -100
        self.assertEqual(runs["baseline_seed42"]["history"], before)
        json.dumps(summarize_training_histories(runs), allow_nan=False)

    def test_incomplete_toy_misaligned_or_nonfinite_histories_fail_before_writing(self):
        mutations = (
            lambda runs: runs.pop("baseline_ppg_seed42"),
            lambda runs: runs["baseline_seed42"]["config"].update(toy_test_only=True),
            lambda runs: runs["baseline_seed42"]["completion"].update(status="running"),
            lambda runs: runs["baseline_seed42"]["completion"].update(toy_test_only=True),
            lambda runs: runs["baseline_seed42"]["config"].update(seed=43),
            lambda runs: runs["baseline_seed42"]["completion"].update(actual_epochs=8),
            lambda runs: runs["baseline_seed42"]["completion"].update(selected_epoch=2, best_epoch=2),
            lambda runs: runs["baseline_seed42"]["history"][2].update(epoch=2),
            lambda runs: runs["baseline_seed42"]["history"][2].update(train_normalized_MSE=np.nan),
            lambda runs: runs["baseline_seed42"]["history"][2].update(train_normalized_MSE=-1),
            lambda runs: runs["baseline_seed42"]["history"][2].update(selection_macro_mae=99),
            lambda runs: runs["baseline_seed42"]["completion"].update(best_validation_macro_mae=99),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix="training-diagnostic-test-") as directory:
                runs = fixture()
                mutation(runs)
                out = Path(directory) / "must-not-exist"
                with patch.object(Figure, "savefig") as saved:
                    with self.assertRaises(ValueError):
                        build_training_diagnostics(out, runs)
                    saved.assert_not_called()
                self.assertFalse(out.exists())

    def test_raw_epoch_lines_and_validation_markers_are_exact_and_readable(self):
        runs = fixture()
        captured = {}

        def capture(fig, path, **kwargs):
            captured[Path(path).name] = fig

        with tempfile.TemporaryDirectory(prefix="training-diagnostic-test-") as directory:
            with patch.object(Figure, "savefig", new=capture):
                latex, details = build_training_diagnostics(directory, runs)
            self.assertEqual(set(captured), set(FIGURES.values()))
            self.assertEqual(list((Path(directory) / "figures").iterdir()), [])
            self.assertIn("CNN 9/10/11 completed, 1/2/3 selected", latex)
            self.assertIn("not a directly comparable training-validation gap", latex)
            self.assertIn("without smoothing or extrapolation", latex)
            for kind, name in FIGURES.items():
                fig = captured[name]
                np.testing.assert_allclose(fig.get_size_inches(), [7, 5.2])
                self.assertEqual(len(fig.axes), 4)
                self.assertEqual(len(fig.legends[0].get_texts()), 3 if kind == "training" else 4)
                first_axes = fig.axes[0]
                for axis, variant in zip(fig.axes, PRIMARY):
                    self.assertTrue(first_axes.get_shared_x_axes().joined(first_axes, axis))
                    self.assertTrue(first_axes.get_shared_y_axes().joined(first_axes, axis))
                    self.assertEqual(len(axis.lines), 3 if kind == "training" else 6)
                    for seed_index, seed in enumerate(SEEDS):
                        record = details["runs"][f"{variant}_seed{seed}"]
                        line = axis.lines[seed_index if kind == "training" else seed_index * 2]
                        field = "train_normalized_MSE" if kind == "training" else "selection_macro_mae"
                        np.testing.assert_array_equal(line.get_xdata(), [row["epoch"] for row in record["history"]])
                        np.testing.assert_array_equal(line.get_ydata(), [row[field] for row in record["history"]])
                        if kind == "validation":
                            point = axis.lines[seed_index * 2 + 1]
                            selected = record["selected_epoch"]
                            np.testing.assert_array_equal(point.get_xdata(), [selected])
                            np.testing.assert_array_equal(point.get_ydata(), [record["history"][selected - 1][field]])
                # Exercise actual canvas layout without saving a synthetic PDF.
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                renderer = canvas.get_renderer()
                artists = [fig.legends[0]]
                for axis in fig.axes:
                    artists.extend([axis.title, axis.xaxis.label, axis.yaxis.label])
                    # Locators create extra ticks outside view limits; those
                    # artists are not drawn and do not count as clipped labels.
                    xmin, xmax = axis.get_xlim()
                    ymin, ymax = axis.get_ylim()
                    artists.extend(label for label in axis.get_xticklabels()
                                   if xmin <= label.get_position()[0] <= xmax)
                    artists.extend(label for label in axis.get_yticklabels()
                                   if ymin <= label.get_position()[1] <= ymax)
                for artist in artists:
                    if artist.get_visible() and (not hasattr(artist, "get_text") or artist.get_text()):
                        bounds = artist.get_window_extent(renderer)
                        self.assertGreaterEqual(bounds.x0, -1)
                        self.assertGreaterEqual(bounds.y0, -1)
                        self.assertLessEqual(bounds.x1, fig.bbox.width + 1)
                        self.assertLessEqual(bounds.y1, fig.bbox.height + 1)
            self.assertFalse((Path(directory) / "final_report.pdf").exists())
        self.assertEqual(plt.get_fignums(), [])

    def test_optional_eca_is_complete_only_and_cannot_change_primary_panels(self):
        base = summarize_training_histories(fixture())
        extra = summarize_training_histories(fixture(include_eca=True))
        self.assertTrue(extra["supplementary_eca_completed"])
        self.assertEqual(extra["total_completed_runs"], 23)
        self.assertEqual(extra["figure_variant_order"], list(PRIMARY))
        for variant in PRIMARY:
            for seed in SEEDS:
                key = f"{variant}_seed{seed}"
                self.assertEqual(extra["runs"][key], base["runs"][key])
        for seed in SEEDS:
            self.assertFalse(extra["runs"][f"eca_last_seed{seed}"]["shown_in_primary_figures"])
        partial = fixture(include_eca=True)
        del partial["eca_last_seed44"]
        with self.assertRaisesRegex(ValueError, "optional inventory"):
            summarize_training_histories(partial)


if __name__ == "__main__":
    unittest.main()
