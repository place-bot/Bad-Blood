"""Consequential identity, temporal, and input/label isolation checks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import data


def population(tmp_path):
    exclusions = tmp_path / "excluded.json"
    exclusions.write_text(json.dumps(["p000003_1"]))
    (tmp_path / "split.json").write_text(json.dumps({
        "train": ["p000001_1", "p000003_1"], "validation": ["p000002_1"]}))
    for name, subjects in {"train": ["p000001_1"] * 2 + ["p000002_1"] * 2 + ["p000003_1"] * 2,
                           "test": ["p000004_1"] * 2, "aami": ["p000005_1"] * 2}.items():
        folder = tmp_path / name
        folder.mkdir()
        n = len(subjects)
        np.save(folder / "subjects.npy", np.array(subjects))
        np.save(folder / "valid.npy", np.ones(n, dtype=bool))
        np.save(folder / "waves.npy", np.tile(np.linspace(0, 1, 1250, dtype=np.float32), (n, 2, 1)))
        np.save(folder / "y.npy", np.array([[100 + 10 * i, 60 + 5 * i] for i in range(n)], dtype=np.float32))
        (folder / "complete.json").write_text(json.dumps({"rows": n, "subjects": len(set(subjects)), "invalid": 0}))
    return tmp_path


def _check_population_disjoint_and_train_only_target_statistics(population):
    loaded = data.load_population(population)
    assert isinstance(loaded["train"]["waves"], np.memmap)
    assert loaded["train"]["waves"] is loaded["validation"]["waves"]
    assert loaded["train"]["y"] is loaded["validation"]["y"]
    np.testing.assert_array_equal(loaded["train"]["indices"], [0, 1])
    np.testing.assert_allclose(loaded["center"], [105, 62.5])
    np.testing.assert_allclose(loaded["scale"], [5, 2.5])
    assert loaded["center"].dtype == np.float32
    assert "sha256" in loaded["manifest"]["source_identity"]["train/waves.npy"]
    labels = np.load(population / "train/y.npy")
    labels[2:] += 1000
    np.save(population / "train/y.npy", labels)
    changed = data.load_population(population)
    np.testing.assert_array_equal(changed["center"], loaded["center"])
    assert changed["manifest_hash"] != loaded["manifest_hash"]


def _check_population_rejects_patient_overlap(population):
    path = population / "split.json"
    split = json.loads(path.read_text())
    split["validation"].append("p000001_1")
    path.write_text(json.dumps(split))
    with unittest.TestCase().assertRaisesRegex(ValueError, "Patient overlap"):
        data.load_population(population)


def _check_population_rejects_other_source_and_misaligned_shape(population):
    path = population / "test/subjects.npy"
    np.save(path, np.array(["p000004_0"] * 2))
    with unittest.TestCase().assertRaisesRegex(ValueError, "VitalDB"):
        data.load_population(population)
    np.save(path, np.array(["p000004_1"] * 2))
    np.save(population / "test/waves.npy", np.zeros((2, 1250, 2), dtype=np.float32))
    with unittest.TestCase().assertRaisesRegex(ValueError, "axes"):
        data.load_population(population)


def personal(tmp_path):
    subject, case, n = "p000101", "c0101", 70
    path = tmp_path / (subject + ".mat")
    starts = np.arange(n) * 60.0 + .008
    ends = starts + 10
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Subj_Wins")
        refs = handle.create_group("references")
        for field in ("T", "ECG_F", "PPG_F", "SegSBP", "SegDBP", "SubjectID", "CaseID"):
            target = group.create_dataset(field, (1, n), dtype=h5py.ref_dtype)
            for index in range(n):
                if field == "T":
                    value = starts[index] + np.arange(1250) / 125
                elif field in ("ECG_F", "PPG_F"):
                    value = np.linspace(2 + index, 4 + index, 1250)
                elif field == "SegSBP":
                    value = [100 + index]
                elif field == "SegDBP":
                    value = [60 + index / 2]
                else:
                    value = [ord(c) for c in (subject if field == "SubjectID" else case)]
                target[0, index] = refs.create_dataset(f"{field}_{index}", data=value).ref
    boundary = starts[0] + .2 * (ends[-1] - starts[0])
    pool = np.flatnonzero(ends <= boundary)
    support = pool[np.rint(np.linspace(0, len(pool) - 1, 10)).astype(int)]
    query = np.flatnonzero(starts >= boundary + 10)
    record = {"patient_id": "vitaldb:" + subject, "subject_id": subject, "case_id": case,
              "role": "target", "total_retained_windows": n, "calibration_boundary_s": boundary,
              "support": [[int(i), starts[i], ends[i]] for i in support],
              "queries": [[int(i), starts[i], ends[i]] for i in query], "query_count": len(query)}
    audit = {"individual_files": {"files": [{"file": str(path), "status": "complete",
              "expected_bytes": path.stat().st_size, "actual_bytes": path.stat().st_size}]},
             "final_personalization_protocol": {"calibration_budgets": [1, 2, 3, 5, 10],
              "patients": [record], "target_subjects": [subject], "development_subjects": []}}
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    return tmp_path, audit_path, path


def _check_personal_alignment_normalization_and_pressure_input_isolation(personal):
    root, audit_path, path = personal
    first = data.load_personal_cohort(root, audit_path)
    patient = first["target"][0]
    assert patient["x"].shape == (70, 2, 1250)
    np.testing.assert_array_equal(patient["x"].min(axis=2), np.zeros((70, 2)))
    np.testing.assert_array_equal(patient["x"].max(axis=2), np.ones((70, 2)))
    np.testing.assert_array_equal(patient["y"][:, 0], 100 + patient["source_indices"])
    assert not set(patient["support_indices"]) & set(patient["query_indices"])
    with h5py.File(path, "r+") as handle:
        handle[handle["Subj_Wins/SegSBP"][0, 0]][...] = [999]
    changed = data.load_personal_cohort(root, audit_path)
    np.testing.assert_array_equal(changed["target"][0]["x"], patient["x"])
    assert changed["target"][0]["y"][0, 0] == 999
    assert changed["manifest_hash"] != first["manifest_hash"]


def _check_personal_rejects_temporal_or_record_corruption(personal, corruption):
    root, audit_path, path = personal
    if corruption == "different_case":
        with h5py.File(path, "r+") as handle:
            handle[handle["Subj_Wins/CaseID"][0, 0]][...] = [ord(c) for c in "c9999"]
    else:
        audit = json.loads(audit_path.read_text())
        record = audit["final_personalization_protocol"]["patients"][0]
        if corruption == "timestamp":
            record["queries"][0][1] += .008
        else:
            record["queries"][0] = record["support"][0]
        audit_path.write_text(json.dumps(audit))
    with unittest.TestCase().assertRaisesRegex(ValueError, "timestamps|leakage|mismatch"):
        data.load_personal_cohort(root, audit_path)


class PopulationDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, value in {
            "SOURCE_COUNTS": {"train": (6, 3), "test": (2, 1), "aami": (2, 1)},
            "POPULATION_COUNTS": {"train": (2, 1), "validation": (2, 1), "test": (2, 1), "aami": (2, 1)},
            "EXCLUDED_PATIENT_COUNT": 1,
            "EXCLUSION_PATH": self.root / "excluded.json",
        }.items():
            context = patch.object(data, name, value)
            context.start()
            self.addCleanup(context.stop)
        population(self.root)

    def test_train_only_statistics_and_identity_hash(self):
        _check_population_disjoint_and_train_only_target_statistics(self.root)

    def test_split_leakage(self):
        _check_population_rejects_patient_overlap(self.root)

    def test_source_and_shape_contract(self):
        _check_population_rejects_other_source_and_misaligned_shape(self.root)


class PersonalDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = personal(Path(temporary.name))

    def test_alignment_and_input_isolation(self):
        _check_personal_alignment_normalization_and_pressure_input_isolation(self.fixture)

    def test_timestamp_mismatch(self):
        _check_personal_rejects_temporal_or_record_corruption(self.fixture, "timestamp")

    def test_temporal_leakage(self):
        _check_personal_rejects_temporal_or_record_corruption(self.fixture, "support_in_query")

    def test_record_mismatch(self):
        _check_personal_rejects_temporal_or_record_corruption(self.fixture, "different_case")


if __name__ == "__main__":
    unittest.main()
