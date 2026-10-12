"""Contract checks for the frozen 20261012 new-participant follow-up."""
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fresh_validation',ROOT/'paired_se/fresh_20261012/validate.py')
fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
from paired_se.model import PairSE, chronological_pairs, predict_bank
from shared.models import BPNet

class FreshProtocol(unittest.TestCase):
    def test_fixed_config_and_identity_exclusion(self):
        manifest=json.loads((ROOT/'paired_se/fresh_20261012/candidate_manifest.json').read_text())
        old=json.loads((ROOT/'paired_se/results/manifest.json').read_text())['population']
        excluded=set(old['excluded_patients'])
        for p in old['partitions'].values():excluded.update(p['patients'])
        people=[r['patient'] for r in manifest['candidates']]
        self.assertEqual(len(people),len(set(people)))
        self.assertEqual(len(people),21)
        self.assertFalse(set(people)&excluded)
        self.assertEqual(set(manifest['excluded_identities']),excluded)
        self.assertEqual(fresh.CONFIG,{'scope':'head','lr':.0003,'steps':25})
        self.assertEqual(manifest['config'],fresh.CONFIG)

    def test_early_pairs_have_no_current_self_reference(self):
        start=np.arange(0,100,10.)
        q,r=chronological_pairs(start,start+10)
        self.assertTrue(np.all(start[r]+20<=start[q]))
        self.assertEqual(set(np.r_[q,r]),set(range(10)))

    def test_singleton_matlab_record_is_ineligible_not_a_loader_crash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'single.mat'
            with h5py.File(p,'w') as f:f.create_group('Subj_Wins').create_dataset('T',data=np.arange(1250)[None,:]/125)
            with self.assertRaisesRegex(AssertionError,'Single-window record'):
                fresh.read_person(p,'vitaldb:p000073')

    def test_pair_zero_head_equals_bias_correction(self):
        torch.manual_seed(20261012);torch.set_num_threads(1)
        model=PairSE(BPNet(variant='se_last')).eval()
        rng=np.random.default_rng(5)
        ref=rng.random((4,2,1250),dtype=np.float32)
        current=rng.random((3,2,1250),dtype=np.float32)
        y=np.array([[120,70],[125,75],[130,80],[115,65]],np.float32)
        c=np.array([120,70],np.float32);s=np.array([20,10],np.float32)
        out=predict_bank(model,current,ref,y,c,s,'cpu')
        with torch.inference_mode():
            pr=model.encoder(torch.from_numpy(ref)).numpy()*s+c
            pc=model.encoder(torch.from_numpy(current)).numpy()*s+c
        np.testing.assert_allclose(out,pc+(y-pr).mean(0),atol=2e-5,rtol=0)

if __name__=='__main__':unittest.main()
