"""Deterministic implementation checks, with no final cohort outcomes."""
import inspect
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from personal_hpo_20261015.core import split_indices,encode,trajectory,predict_features,select_person
from modality_se.personalize import restore
from paired_se.run import fit_personal
from paired_se.model import predict_bank


class Tests(unittest.TestCase):
    def test_splits(self):
        t=np.arange(100)*10.
        ix=split_indices(t,t+10)
        self.assertEqual(len(ix['support']),80)
        self.assertEqual(ix['query'].tolist(),list(range(81,100)))
        self.assertEqual(len(ix['inner_train']),56)
        self.assertEqual(ix['inner_valid'].tolist(),list(range(57,80)))
        self.assertFalse(set(ix['query']) & set(ix['support']))
        self.assertTrue(set(ix['inner_valid'])<=set(ix['support']))

    def test_small_data_rejected(self):
        with self.assertRaises(ValueError):split_indices(np.arange(9)*10.,np.arange(9)*10.+10)

    def test_selection_interface(self):
        names=list(inspect.signature(select_person).parameters)
        self.assertNotIn('test_y',names)
        self.assertNotIn('people',names)
        self.assertIn('development_y',names)

    def test_equivalent_training_and_prediction(self):
        torch.set_num_threads(4);torch.manual_seed(55)
        saved=torch.load(ROOT/'modality_se/models/dual_pair_population_seed42.pt',map_location='cpu',weights_only=False)
        m=restore(saved,'dual');rng=np.random.default_rng(99)
        x=rng.random((12,2,1250),dtype=np.float32)
        y=rng.normal([120,75],[15,8],(12,2)).astype(np.float32)
        t=np.arange(12)*20.;c=np.asarray(saved['center'],np.float32);s=np.asarray(saved['scale'],np.float32)
        h,p=encode(m,x)
        reference,a=fit_personal(m,x,y,t,t+10,c,s,dict(scope='head',lr=.0003,steps=25),'cpu')
        _,fitted,b=list(trajectory(m,h,p,y,t,t+10,c,s,.0003,{25}))[0]
        self.assertEqual(a['pair_count'],b['pair_count'])
        q=rng.random((5,2,1250),dtype=np.float32);hq,pq=encode(m,q)
        new=predict_features(fitted,hq,pq,h,p,y,c,s)
        old=predict_bank(reference,q,x,y,c,s,'cpu')
        np.testing.assert_allclose(new,old,rtol=0,atol=2e-4)
        for k,v in m.state_dict().items():
            if not k.startswith('change.'):self.assertTrue(torch.equal(v,fitted.state_dict()[k]))

    def test_ppg_does_not_see_ecg(self):
        saved=torch.load(ROOT/'modality_se/models/ppg_pair_population_seed42.pt',map_location='cpu',weights_only=False)
        m=restore(saved,'ppg');x=np.random.default_rng(8).random((3,2,1250),dtype=np.float32)
        a=encode(m,x);x[:,0,:]=1234.;b=encode(m,x)
        for aa,bb in zip(a,b):self.assertTrue(torch.equal(aa,bb))


if __name__=='__main__':unittest.main()
