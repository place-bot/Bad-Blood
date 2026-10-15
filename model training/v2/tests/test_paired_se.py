"""Synthetic algorithm tests; not blood-pressure experiment results."""
import inspect
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from shared.models import BPNet
from paired_se.model import PairSE,same_person_references,chronological_pairs,predict_bank
from paired_se.run import fit_personal,configurations
from paired_se.predict import load_predict


class PairedSETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1);self.model=PairSE(BPNet(variant='se_last')).eval()
        self.x=np.random.default_rng(1).random((5,2,1250)).astype(np.float32)
        self.y=np.array([[110,70],[115,73],[117,75],[120,78],[122,80]],np.float32)

    def test_pair_identity_and_membership(self):
        ids=np.repeat(['a','b','c'],5);r=same_person_references(ids,42)
        self.assertTrue(np.all(ids[r]==ids));self.assertTrue(np.all(r!=np.arange(len(r))))
        np.testing.assert_array_equal(r,same_person_references(ids,42))
        with self.assertRaises(ValueError):same_person_references(['a'],1)

    def test_chronological_guard(self):
        start=np.arange(5)*30.;q,r=chronological_pairs(start,start+10)
        self.assertTrue(np.all(start[r]+20<=start[q]));self.assertEqual(len(q),10)

    def test_identity_prediction(self):
        for param in self.model.change.parameters():torch.nn.init.normal_(param,std=.05)
        x=torch.tensor(self.x);y=torch.tensor(self.y)
        with torch.no_grad():pred=self.model(x,x,y)
        torch.testing.assert_close(pred,y)

    def test_initialization_equals_bias_baseline(self):
        with torch.no_grad():
            x=torch.tensor(self.x);p=self.model.encoder(x).numpy()*[20,10]+[100,70]
        got=predict_bank(self.model,self.x,self.x,self.y,[100,70],[20,10])
        np.testing.assert_allclose(got,p+(self.y-p).mean(0),atol=2e-5)

    def test_fit_only_accepts_support_and_updates(self):
        self.assertNotIn('query',inspect.signature(fit_personal).parameters)
        self.assertNotIn('current_y',inspect.signature(predict_bank).parameters)
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        for scope in ('head','full'):
            fitted,audit=fit_personal(self.model,self.x,self.y,np.arange(5)*30.,np.arange(5)*30.+10,
                np.array([100,70]),np.array([20,10]),dict(scope=scope,lr=.001,steps=2),'cpu')
            self.assertTrue(audit['changed']);self.assertFalse(audit['query_used'])
            if scope=='head':self.assertTrue(all(k.startswith('change.') for k in audit['changed']))
            for k,v in self.model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertEqual(len(configurations()),9)

    def test_reference_permutation_and_saved_inference(self):
        a=predict_bank(self.model,self.x,self.x,self.y,[100,70],[20,10])
        b=predict_bank(self.model,self.x,self.x[::-1].copy(),self.y[::-1].copy(),[100,70],[20,10])
        np.testing.assert_allclose(a,b,atol=2e-5)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save(dict(state_dict=self.model.state_dict(),center=[100,70],scale=[20,10]),path)
            got=load_predict(path,self.x,self.x,self.y)
            np.testing.assert_allclose(got,a,atol=1e-6)
            with self.assertRaises(ValueError):load_predict(path,self.x*2,self.x,self.y)
            with self.assertRaises(ValueError):load_predict(path,self.x,self.x,self.y[:,::-1])


if __name__=='__main__':unittest.main()
