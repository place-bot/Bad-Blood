"""Synthetic-data guards: no cohort test outcomes influence these checks."""
import copy
import inspect
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rolling_personal_20261015.core import (rolling_folds,prefix,tail,configure,
    pair_backward,trajectory,predict,choose)
from modality_se.personalize import restore
from paired_se.model import chronological_pairs,predict_bank


def source(mode='dual'):
    s=torch.load(ROOT/'modality_se/models'/f'{mode}_pair_population_seed42.pt',weights_only=False,map_location='cpu')
    return restore(s,mode),np.asarray(s['center'],np.float32),np.asarray(s['scale'],np.float32)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_folds(self):
        for n in (30,96,1159):
            start=np.arange(n)*10.;end=start+10
            fs=rolling_folds(start,end)
            self.assertEqual(len(fs),3)
            for f in fs:
                self.assertTrue(end[f['train']].max()+10<=start[f['valid']].min())
                self.assertGreaterEqual(len(f['valid']),2)
                self.assertLess(f['valid'].max(),n)
        self.assertEqual(rolling_folds(np.arange(30)*10.,np.arange(30)*10.+10)[0]['valid'].tolist(),[16,17,18])

    def test_cached_encoder_and_mask(self):
        x=np.random.default_rng(4).random((9,2,1250),dtype=np.float32)
        for mode in ('ppg','dual'):
            m,c,s=source(mode)
            z=prefix(m,x)
            with torch.no_grad():
                for a,b in zip(tail(m,z),m.encode(torch.as_tensor(x))):
                    torch.testing.assert_close(a,b)
            if mode=='ppg':
                altered=x.copy();altered[:,0]=900
                torch.testing.assert_close(z,prefix(m,altered),rtol=0,atol=0)

    def test_exact_gradients_and_frozen_state(self):
        torch.manual_seed(13)
        m,c,s=source();rng=np.random.default_rng(5)
        x=rng.random((8,2,1250),dtype=np.float32)
        y=rng.normal([120,75],[12,7],(8,2)).astype(np.float32)
        t=np.arange(8)*20.;q,r=chronological_pairs(t,t+10)
        q=torch.as_tensor(q);r=torch.as_tensor(r);ys=torch.as_tensor((y-c)/s)
        z=prefix(m,x)
        for scope in ('head','tail'):
            for endpoint in (0,1):
                a=configure(copy.deepcopy(m),scope);b=configure(copy.deepcopy(m),scope)
                h,p=tail(a,z)
                if scope=='head':h,p=h.detach(),p.detach()
                loss=pair_backward(a,h,p,ys,q,r,endpoint,chunk=5)
                hb,pb=tail(b,z)
                pred=b.from_features(hb[q],pb[q],hb[r],pb[r],ys[r])
                direct=(pred[:,endpoint]-ys[q,endpoint]).square().mean();direct.backward()
                self.assertAlmostEqual(loss,float(direct.detach()),places=5)
                for (name,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
                    if pa.requires_grad:
                        torch.testing.assert_close(pa.grad,pb.grad,rtol=3e-4,atol=3e-5,msg=name)
                _,fit,audit=list(trajectory(m,z,y,t,t+10,c,s,scope,.0001,endpoint,(2,),chunk=5))[0]
                changed=[]
                for name,v in m.state_dict().items():
                    if not torch.equal(v,fit.state_dict()[name]):changed.append(name)
                allowed=('change.',) if scope=='head' else ('change.','encoder.body.6.','encoder.gates.6.')
                self.assertTrue(all(k.startswith(allowed) for k in changed))
                self.assertFalse(any('running_' in k or 'num_batches' in k for k in changed))
                if scope=='tail':self.assertTrue(any(k.startswith('encoder.body.6.') for k in changed))
                new=predict(fit,z[-2:],z[:6],y[:6],c,s)
                old=predict_bank(fit,x[-2:],x[:6],y[:6],c,s)
                np.testing.assert_allclose(new,old,rtol=0,atol=.001)

    def test_selection(self):
        folds=[dict(candidates=[dict(config=None,mae=4.),dict(config=dict(scope='head',lr=.001,steps=10),mae=2.),
                               dict(config=dict(scope='tail',lr=.0001,steps=25),mae=3.)]) for _ in range(3)]
        result=choose(folds,0)
        self.assertEqual(result['selected']['combined']['config']['scope'],'head')
        self.assertEqual(result['selected']['tail']['config']['steps'],25)
        self.assertEqual(list(inspect.signature(choose).parameters),['fold_results','endpoint'])


if __name__=='__main__':unittest.main()
