import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
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

    def test_population_resume_and_test_label_isolation(self):
        from attention_cnn import train
        from tests.test_training import toy_bundle
        import copy
        with tempfile.TemporaryDirectory() as tmp,patch.object(train,'VARIANTS',('se_ppg',)),patch.object(train,'make_model',make_model):
            root=Path(tmp);data=toy_bundle();settings=dict(max_epochs=2,batch_size=3,inference_batch_size=3)
            train.run_training('se_ppg',43,root,root/'full','cpu',bundle=data,test_settings=settings)
            train.run_training('se_ppg',43,root,root/'resume','cpu',bundle=data,test_settings=settings,stop_after_epoch=1)
            train.run_training('se_ppg',43,root,root/'resume','cpu',True,bundle=data,test_settings=settings)
            a=torch.load(root/'full/best.pt',weights_only=False);b=torch.load(root/'resume/best.pt',weights_only=False)
            for k in a['state_dict']:self.assertTrue(torch.equal(a['state_dict'][k],b['state_dict'][k]))
            changed=copy.deepcopy(data);changed['test']['y']+=50;changed['aami']['y']+=30
            train.run_training('se_ppg',43,root,root/'changed','cpu',bundle=changed,test_settings=settings)
            b=torch.load(root/'changed/best.pt',weights_only=False)
            self.assertEqual(a['epoch'],b['epoch'])
            for k in a['state_dict']:self.assertTrue(torch.equal(a['state_dict'][k],b['state_dict'][k]))

    def test_saved_pair_retains_ppg_mask(self):
        from modality_se.predict import load_predict
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pair.pt';model=PairSE(ModalitySE('ppg')).eval()
            torch.save(dict(modality='ppg',state_dict=model.state_dict(),center=[120.,75.],scale=[20.,10.]),p)
            q=np.random.default_rng(1).random((3,2,1250),dtype=np.float32);r=q[:2].copy();bp=np.array([[110,70],[130,80]],np.float32)
            a=load_predict(p,q,r,bp);q[:,0]+=100;r[:,0]-=100
            np.testing.assert_array_equal(a,load_predict(p,q,r,bp))

if __name__=='__main__':unittest.main()
