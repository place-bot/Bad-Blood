"""Lossless storage and clearly weighted overviews, using synthetic fixtures."""
import copy
import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reports.result_storage import build_metric_overview, write_compressed_json
from shared.metrics import summarize


def fixtures():
    y = np.column_stack((np.arange(6) + 110., np.arange(6) + 65.))
    subjects = np.array(["a", "a", "b", "b", "b", "b"])
    rows = {}
    for seed, errors in ((42, [1., 1., 7., 7., 7., 7.]),
                         (43, [2., 2., 10., 10., 10., 10.])):
        error = np.array(errors)
        rows[seed] = summarize(y, y + np.column_stack((error, error / 2)), subjects)
    aruns = {f"baseline_seed{seed}": {
        "config": {"variant": "baseline", "seed": seed},
        "completion": {"status": "complete", "toy_test_only": False, "variant": "baseline", "seed": seed,
                       "parameters": 37666, "selected_epoch": seed - 41,
                       "epochs_completed": seed - 33, "actual_epochs": seed - 33},
        "metrics": {split: copy.deepcopy(row) for split in ("validation", "test", "aami")},
    } for seed, row in rows.items()}
    bs = {("head", 10, seed): copy.deepcopy(row) for seed, row in rows.items()}
    return aruns, bs, {"principal_attention": {"winner": "baseline", "role": "validation-only"}}


class CompressedResultTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="bad-blood-storage-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_exact_round_trip_without_metric_or_array_truncation(self):
        data = {"values": list(range(10000)), "metrics": [1.2345678901234567, -0.0, None, True],
                "nested": {"unicode": "µ and 中文", "fractions": [0.123456789, 1.0]}}
        before = copy.deepcopy(data)
        path = self.root / "results/complete.json.gz"
        self.assertEqual(write_compressed_json(path, data), path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            loaded = json.load(handle)
        self.assertEqual(loaded, data)
        self.assertEqual(len(loaded["values"]), 10000)
        self.assertEqual(math.copysign(1, loaded["metrics"][1]), -1)
        raw = gzip.decompress(path.read_bytes())
        self.assertEqual(raw, (json.dumps(data, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
        self.assertLess(path.stat().st_size, len(raw))
        self.assertEqual(data, before)

    def test_deterministic_bytes_ignore_timestamp_filename_and_key_order(self):
        first, second = self.root / "first.json.gz", self.root / "other/named_differently.json.gz"
        write_compressed_json(first, {"b": [2, 3], "a": 1})
        original = first.read_bytes()
        write_compressed_json(second, {"a": 1, "b": [2, 3]})
        write_compressed_json(first, {"b": [2, 3], "a": 1})
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(second.read_bytes(), original)
        self.assertEqual(original[4:8], b"\x00\x00\x00\x00")
        self.assertEqual(original[3] & 0x08, 0)  # Gzip FNAME is absent.

    def test_invalid_suffix_and_nonfinite_or_unsupported_data_preserve_destination(self):
        path = self.root / "complete.json.gz"
        write_compressed_json(path, {"before": 1})
        original = path.read_bytes()
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                write_compressed_json(path, {"invalid": value})
            self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(TypeError):
            write_compressed_json(path, {"invalid": object()})
        with self.assertRaisesRegex(ValueError, r"\.json\.gz"):
            write_compressed_json(self.root / "not_gzip.json", {})
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.root / "not_gzip.json").exists())

    def test_failed_atomic_replace_keeps_existing_data_and_cleans_owned_temporary(self):
        path = self.root / "complete.json.gz"
        write_compressed_json(path, {"before": 1})
        original = path.read_bytes()
        with patch("reports.result_storage.os.replace", side_effect=OSError("fixture replacement failure")):
            with self.assertRaisesRegex(OSError, "fixture replacement failure"):
                write_compressed_json(path, {"after": 2})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [path])


class MetricOverviewTests(unittest.TestCase):
    def test_all_metrics_splits_counts_epochs_and_patient_then_seed_weighting(self):
        aruns, bs, selections = fixtures()
        overview = build_metric_overview(aruns, bs, selections)
        baseline = overview["attention"]["baseline"]
        self.assertEqual(baseline["parameters"], 37666)
        self.assertEqual(baseline["seeds"], [42, 43])
        self.assertEqual(baseline["n_seeds"], 2)
        self.assertEqual(baseline["selected_epochs_by_seed"], {"42": 1, "43": 2})
        self.assertEqual(baseline["epochs_completed_by_seed"], {"42": 9, "43": 10})
        self.assertEqual(set(baseline["splits"]), {"validation", "test", "aami"})
        row = overview["personalization"]["head/K10"]
        self.assertEqual(row["n_windows"], 6)  # Not multiplied by two seeds.
        self.assertEqual(row["n_patients"], 2)
        self.assertEqual(row["macro"]["mae"]["seed_mean"], [5.0, 2.5])
        self.assertAlmostEqual(row["pooled"]["mae"]["seed_mean"][0], (5 + 22 / 3) / 2)
        expected = (math.sqrt(33) + math.sqrt(68)) / 2
        self.assertAlmostEqual(row["pooled"]["rmse"]["seed_mean"][0], expected)
        self.assertNotAlmostEqual(expected, math.sqrt((33 + 68) / 2))
        expected_keys = set(bs[("head", 10, 42)]["pooled"])
        for aggregation in ("pooled", "macro"):
            self.assertEqual(set(row[aggregation]), expected_keys)
            self.assertEqual(set(baseline["splits"]["test"][aggregation]), expected_keys)
        self.assertNotIn("per_patient", json.dumps(overview))
        self.assertIn("fractions", overview["definitions"]["fraction_metrics"])
        json.dumps(overview, allow_nan=False)

    def test_undefined_correlation_defined_seed_counts_and_future_metrics(self):
        aruns, bs, selections = fixtures()
        for seed in (42, 43):
            row = bs[("head", 10, seed)]
            row["macro"]["pearson_r"] = [None, .6 if seed == 42 else None]
            row["macro_defined_patients"]["pearson_r"] = [0, 2 if seed == 42 else 0]
            row["pooled"]["pearson_r"] = [None, .2 if seed == 42 else None]
            row["macro"]["future_metric"] = [float(seed), 2.]
            row["pooled"]["future_metric"] = [float(seed), 3.]
            row["macro_defined_patients"]["future_metric"] = [2, 2]
        row = build_metric_overview(aruns, bs, selections)["personalization"]["head/K10"]
        self.assertEqual(row["macro"]["pearson_r"], {"seed_mean": [None, .6], "defined_seed_counts": [0, 1]})
        self.assertEqual(row["pooled"]["pearson_r"], {"seed_mean": [None, .2], "defined_seed_counts": [0, 1]})
        self.assertEqual(row["macro"]["future_metric"]["seed_mean"], [42.5, 2.])
        self.assertEqual(row["macro_defined_patients_by_seed"]["43"]["pearson_r"], [0, 0])
        json.dumps(row, allow_nan=False)

    def test_nonfinite_inconsistent_units_coverage_and_identity_fail(self):
        mutations = (
            lambda a, b: b[("head", 10, 42)]["pooled"]["mae"].__setitem__(0, float("nan")),
            lambda a, b: b[("head", 10, 42)].__setitem__("within_metrics_unit", "percentage"),
            lambda a, b: b[("head", 10, 42)].__setitem__("n_windows", 99),
            lambda a, b: b[("head", 10, 42)]["macro_defined_patients"]["mae"].__setitem__(0, 0),
            lambda a, b: a["baseline_seed42"]["completion"].__setitem__("seed", 44),
            lambda a, b: a["baseline_seed42"]["completion"].__setitem__("status", "running"),
            lambda a, b: a["baseline_seed42"]["completion"].__setitem__("parameters", 1),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                aruns, bs, selections = fixtures()
                mutate(aruns, bs)
                with self.assertRaises(ValueError):
                    build_metric_overview(aruns, bs, selections)

    def test_inputs_are_not_modified_and_selections_are_not_aliased(self):
        aruns, bs, selections = fixtures()
        before = copy.deepcopy((aruns, bs, selections))
        result = build_metric_overview(aruns, bs, selections)
        self.assertEqual((aruns, bs, selections), before)
        result["validation_selections"]["principal_attention"]["winner"] = "changed-return-only"
        self.assertEqual(selections, before[2])


if __name__ == "__main__":
    unittest.main()
