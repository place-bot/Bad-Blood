import unittest
import torch
from modality_se.model import ModalitySE,make_model
from paired_se.model import PairSE
from shared.models import make_model as original_make,count_parameters

class ModalitySETests(unittest.TestCase):
    def test_initial_weights_and_count(self):
        p=make_model('se_ppg',42);d=original_make('se_last',42)
        self.assertEqual(count_parameters(p),38296)
        self.assertEqual(set(p.state_dict()),set(d.state_dict()))
        for k,v in p.state_dict().items():self.assertTrue(torch.equal(v,d.state_dict()[k]))
    def test_no_ecg_in_forward_or_pair_encoder(self):
        m=ModalitySE('ppg').eval();x=torch.rand(3,2,1250);other=x.clone();other[:,0]=torch.randn_like(other[:,0])*100
        with torch.inference_mode():
            self.assertTrue(torch.equal(m(x),m(other)))
            pair=PairSE(m).eval();a=pair.encode(x);b=pair.encode(other)
            self.assertTrue(all(torch.equal(u,v) for u,v in zip(a,b)))
    def test_dual_is_unchanged(self):
        d=ModalitySE('dual').eval();old=original_make('se_last',0).eval();old.load_state_dict(d.state_dict());x=torch.rand(3,2,1250)
        with torch.inference_mode():self.assertTrue(torch.equal(d(x),old(x)))

if __name__=='__main__':unittest.main()
