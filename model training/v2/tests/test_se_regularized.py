"""Algorithm invariants using synthetic fixtures, not study results."""
import unittest
import numpy as np
import torch
from se_personalization.regularized_head import fit_head, apply_head, grid, extract
from shared.models import BPNet


class RegularizedHeadTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(41)
        self.h=rng.normal(size=(60,32))
        self.p=rng.normal(size=(60,2))
        self.y=self.p+self.h[:,:2]*3+np.array([110.,70.])

    def test_grid_and_centered_support(self):
        self.assertEqual(len(grid()),24)
        for anchor in ('population','mean'):
            state=fit_head(self.h,self.p,self.y,anchor,1)
            np.testing.assert_allclose(apply_head(state,self.h,self.p).mean(0),self.y.mean(0),atol=1e-10)

    def test_large_penalty_recovers_controls(self):
        for anchor in ('population','mean'):
            state=fit_head(self.h,self.p,self.y,anchor,1e14)
            expected=self.p+(self.y-self.p).mean(0) if anchor=='population' else np.tile(self.y.mean(0),(60,1))
            np.testing.assert_allclose(apply_head(state,self.h,self.p),expected,atol=1e-10)

    def test_order_and_affine_features(self):
        state=fit_head(self.h,self.p,self.y,'population',.1)
        rev=fit_head(self.h[::-1],self.p[::-1],self.y[::-1],'population',.1)
        np.testing.assert_allclose(apply_head(state,self.h,self.p),apply_head(rev,self.h,self.p),atol=1e-10)
        self.assertLess(np.mean(abs(apply_head(state,self.h,self.p)-self.y)),.5)

    def test_extraction_matches_pretrained_linear_head(self):
        model=BPNet(variant='se_last').eval()
        x=np.random.default_rng(1).random((3,2,1250)).astype(np.float32)
        features,p=extract(model,x,[100,70],[20,10],'cpu')
        self.assertEqual(features['pooled48'].shape,(3,48))
        self.assertEqual(features['penultimate32'].shape,(3,32))
        with torch.no_grad():
            expected=model.head[3](torch.tensor(features['penultimate32'])).numpy()*[20,10]+[100,70]
        np.testing.assert_allclose(p,expected,atol=1e-5)

    def test_input_rejection(self):
        with self.assertRaises(ValueError):fit_head(self.h,self.p,self.y,'population',0)
        with self.assertRaises(ValueError):fit_head(self.h,self.p,self.y[:-1],'mean',1)


if __name__=='__main__':unittest.main()
