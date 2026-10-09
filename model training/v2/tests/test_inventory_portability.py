"""Relocation changes discovery only; byte/manifest/configuration checks remain.

All files generated here are small synthetic fixtures, not study artifacts.
"""
from contextlib import ExitStack
import copy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reports import verify_inventory as vi
from shared import data
from test_data import personal, population
from test_inventory import a_fixture, b_fixture


class InventoryPortabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="bad-blood-portability-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / "original"
        self.original.mkdir()
        self.relocated = self.root / "relocated"
        vi._POPULATION_CACHE.clear()
        vi._PERSONAL_CACHE.clear()
        vi._HASH_CACHE.clear()
        torch.set_num_threads(2)

    def population_context(self, stack):
        population(self.original)
        for name, value in {
            "SOURCE_COUNTS": {"train": (6, 3), "test": (2, 1), "aami": (2, 1)},
            "POPULATION_COUNTS": {"train": (2, 1), "validation": (2, 1), "test": (2, 1), "aami": (2, 1)},
            "EXCLUDED_PATIENT_COUNT": 1,
            "EXCLUSION_PATH": self.root / "excluded.json",
        }.items():
            stack.enter_context(patch.object(data, name, value))
        shutil.copy2(self.original / "excluded.json", self.root / "excluded.json")
        loaded = data.load_population(self.original)
        config = {"data_root": str(self.original)}
        shutil.copytree(self.original, self.relocated)
        self.original.rename(self.root / "retired")
        return config, loaded["manifest"]

    def personal_context(self, stack):
        _, audit_path, _ = personal(self.original)
        audit_copy = self.root / "audit.json"
        shutil.copy2(audit_path, audit_copy)
        loader = data.load_personal_cohort
        cohort = loader(self.original, audit_copy)
        patients = cohort["development"] + cohort["target"]
        manifest = {
            "individual_root": str(self.original), "manifest_hash": cohort["manifest_hash"],
            "data_manifest": cohort["manifest"],
            "patient_data_hashes": {p["patient_id"]: vi.array_hash(
                p["x"], p["y"], p["start"], p["end_exclusive"], p["source_indices"],
                p["support_indices"], p["query_indices"]) for p in patients},
        }
        stack.enter_context(patch.object(data, "DEFAULT_AUDIT_PATH", audit_copy))
        stack.enter_context(patch.object(data, "load_personal_cohort",
                                        side_effect=lambda root: loader(root, audit_copy)))
        stack.enter_context(patch.object(vi, "DEV_IDS", ()))
        stack.enter_context(patch.object(vi, "TARGET_IDS", ("vitaldb:p000101",)))
        shutil.copytree(self.original, self.relocated)
        self.original.rename(self.root / "retired")
        return manifest

    def test_relocated_population_preserves_exact_manifest_and_original_config(self):
        with ExitStack() as stack:
            config, manifest = self.population_context(stack)
            before = copy.deepcopy((config, manifest))
            with self.assertRaises(FileNotFoundError):
                vi._verify_population_inputs(config, manifest)
            verified = vi._verify_population_inputs(config, manifest, population_data_root=self.relocated)
            self.assertEqual(verified["manifest"], manifest)
            self.assertEqual(verified["manifest_hash"], vi.canonical_hash(manifest))
            self.assertEqual((config, manifest), before)

    def test_relocated_population_changed_source_bytes_still_fail(self):
        with ExitStack() as stack:
            config, manifest = self.population_context(stack)
            path = self.relocated / "test/waves.npy"
            waves = np.load(path, allow_pickle=False)
            waves[0, 0, 10] = .5
            np.save(path, waves)
            with self.assertRaisesRegex(ValueError, "Population manifest differs"):
                vi._verify_population_inputs(config, manifest, population_data_root=self.relocated)

    def test_relocated_personal_preserves_all_identity_and_provenance_checks(self):
        with ExitStack() as stack:
            manifest = self.personal_context(stack)
            before = copy.deepcopy(manifest)
            with self.assertRaises(FileNotFoundError):
                vi._verify_personal_inputs(manifest)
            verified = vi._verify_personal_inputs(manifest, individual_root=self.relocated)
            self.assertEqual(verified["manifest"], manifest["data_manifest"])
            self.assertEqual(verified["manifest_hash"], manifest["manifest_hash"])
            self.assertEqual(manifest, before)
            changed = copy.deepcopy(manifest)
            changed["patient_data_hashes"]["vitaldb:p000101"] = "tampered"
            with self.assertRaisesRegex(ValueError, "input/reference/timestamp"):
                vi._verify_personal_inputs(changed, individual_root=self.relocated)

    def test_relocated_personal_changed_raw_bytes_still_fail(self):
        with ExitStack() as stack:
            manifest = self.personal_context(stack)
            with h5py.File(self.relocated / "p000101.mat", "r+") as handle:
                handle[handle["Subj_Wins/SegSBP"][0, 0]][...] = [101]
            with self.assertRaisesRegex(ValueError, "Personal data manifest hash mismatch"):
                vi._verify_personal_inputs(manifest, individual_root=self.relocated)

    def test_public_population_apis_forward_overrides_without_mutating_config(self):
        folder = self.root / "attention/baseline_seed42"
        bundle = a_fixture(folder)
        config_before = (folder / "config.json").read_bytes()
        with patch.object(vi, "_verify_population_inputs", return_value=bundle) as inputs, \
                patch.object(vi, "_verify_source_hashes"):
            record = vi.verify_a_folder(folder, "baseline", 42, population_data_root=self.relocated)
            self.assertEqual(inputs.call_args.kwargs, {"population_data_root": self.relocated})
        self.assertEqual((folder / "config.json").read_bytes(), config_before)
        with patch.object(vi, "verify_a_folder", return_value=record) as verify:
            result = vi.verify_a_inventory(folder.parent, population_data_root=self.relocated)
        self.assertEqual(len(result), 20)
        self.assertEqual(verify.call_count, 20)
        self.assertTrue(all(call.kwargs == {"population_data_root": self.relocated}
                            for call in verify.call_args_list))

    def test_public_personal_api_forwards_both_roots_and_keeps_provenance(self):
        folder = self.root / "personalization"
        records, _, _, cohort = b_fixture(folder)
        before = (folder / "run_manifest.json").read_bytes()
        with ExitStack() as stack:
            personal_inputs = stack.enter_context(patch.object(vi, "_verify_personal_inputs", return_value=cohort))
            population_inputs = stack.enter_context(patch.object(vi, "verify_a_folder",
                side_effect=lambda folder, variant, seed, **kwargs:
                    {"completion": {"best_checkpoint_sha256": f"checkpoint-{seed}"}}))
            stack.enter_context(patch.object(vi, "_verify_source_hashes"))
            stack.enter_context(patch.object(vi, "file_hash", return_value="fixture-file-hash"))
            stack.enter_context(patch.object(vi, "_verify_unit",
                                            side_effect=lambda folder, item, *args: records[item["unit_path"]]))
            result = vi.verify_b_inventory(folder, population_data_root=self.relocated / "population",
                                           individual_root=self.relocated / "individual")
        self.assertEqual(len(result["test_records"]), 684)
        self.assertEqual(personal_inputs.call_args.kwargs, {"individual_root": self.relocated / "individual"})
        self.assertEqual(population_inputs.call_count, 3)
        self.assertTrue(all(call.kwargs == {"population_data_root": self.relocated / "population"}
                            for call in population_inputs.call_args_list))
        self.assertEqual((folder / "run_manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
