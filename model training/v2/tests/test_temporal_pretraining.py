import unittest
import numpy as np
import torch
from temporal_pretraining.run import episode_indices, bank
from context_se.adapt import MaskedContext
from paired_se.model import PairSE
from shared.models import BPNet

class TemporalPretrainingTests(unittest.TestCase):
    def test_chronology_and_random_budget(self):
        d=dict(h=np.zeros((100,48)),support=np.arange(18),query=np.arange(21,100),start=np.arange(100)*10.,end=np.arange(100)*10.+10.)
        for seed in range(20):
            s,q=episode_indices(d,np.random.default_rng(seed),'chronological')
            rs,rq=episode_indices(d,np.random.default_rng(seed),'random')
            self.assertEqual((len(s),len(q)),(len(rs),len(rq)))
            self.assertLessEqual(max(d['end'][s])+10,min(d['start'][q]))
            self.assertFalse(set(s)&set(q));self.assertFalse(set(rs)&set(rq))

    def test_bank_initialization_and_permutation(self):
        torch.manual_seed(0);h=torch.randn(4,48);p=torch.randn(4,2);hs=torch.randn(70,48);ps=torch.randn(70,2);ys=torch.randn(70,2)
        for kind,model in [('pair',PairSE(BPNet(variant='se_last'))),('context',MaskedContext())]:
            fc,fs=torch.zeros(48),torch.ones(48)
            pred=bank(model,kind,h,p,ys,hs,ps,fc,fs)
            self.assertTrue(torch.allclose(pred,p+(ys-ps).mean(0),atol=1e-6))
            ix=torch.randperm(70)
            self.assertTrue(torch.allclose(pred,bank(model,kind,h,p,ys[ix],hs[ix],ps[ix],fc,fs),atol=1e-6))

if __name__=='__main__':unittest.main()
