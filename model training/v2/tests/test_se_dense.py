import copy
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from se_personalization.run import dense_support,fit,grid,name,select,DEV_IDS,SEEDS
from se_personalization.predict import load_and_predict
from shared.models import BPNet
from personalized_forecasting.adaptation import predict
from test_personalization import toy_patient

class SEDenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_all_early_rows_and_query_unchanged(self):
        p=toy_patient();before=p['query_indices'].copy()
        s=dense_support(p)
        np.testing.assert_array_equal(s,np.arange(120))
        np.testing.assert_array_equal(p['query_indices'],before)
        self.assertGreater(len(s),len(p['support_indices']))
        self.assertGreater(p['start'][before[0]],p['end_exclusive'][s[-1]])

    def test_masks_centering_fresh_initialization_and_no_query(self):
        self.assertNotIn('query',str(inspect.signature(fit)))
        rng=np.random.default_rng(19)
        x=rng.uniform(0,1,(4,2,1250)).astype('float32')
        y=np.array([[100,60],[120,80],[110,70],[105,68]],'float32')
        base=BPNet(variant='se_last');before=copy.deepcopy(base.state_dict())
        for scope in ('head_gate','last_gate','full'):
            for objective in ('plain','centered'):
                setting=dict(scope=scope,objective=objective,lr=.001,steps=2)
                m,offset,a=fit(base,x,y,[120,80],[20,10],setting)
                self.assertTrue(any(k.startswith('gates.') for k in a['trainable_names']))
                self.assertTrue(any(k.startswith('gates.') for k in a['changed_names']))
                self.assertEqual(a['support_count'],4)
                for k,v in base.state_dict().items():
                    torch.testing.assert_close(v,before[k],rtol=0,atol=0)
                    if k not in a['trainable_names']:
                        torch.testing.assert_close(m.state_dict()[k],before[k],rtol=0,atol=0)
                if objective=='centered':
                    pred,_=predict(m,x,[120,80],[20,10])
                    np.testing.assert_allclose((pred+offset*[20,10]).mean(0),y.mean(0),atol=2e-5)
                else: np.testing.assert_array_equal(offset,[0,0])

    def test_selection_complete_and_beats_both(self):
        rows=[]
        for pid in DEV_IDS:
            for seed in SEEDS:
                for m,val in [('calibration_mean',10),('bias',9)]+[(name(s),8+i) for i,s in enumerate(grid())]:
                    rows.append(dict(patient=pid,seed=seed,method=m,metrics={'macro':{'mae':[val,val]}}))
        self.assertEqual(len(grid()),24)
        s=select(rows)
        self.assertTrue(s['success'])
        self.assertEqual(s['selected']['setting'],grid()[0])
        with self.assertRaises(ValueError): select(rows[:-1])
        with self.assertRaises(ValueError): select(rows+[rows[0]])

    def test_saved_centered_model_replay_and_invalid_inputs(self):
        rng=np.random.default_rng(7)
        x=rng.uniform(0,1,(5,2,1250)).astype('float32')
        y=np.array([[100,60],[120,80],[110,70]],'float32')
        model,offset,_=fit(BPNet(variant='se_last'),x[:3],y,[120,80],[20,10],
            dict(scope='full',objective='centered',lr=.0001,steps=2))
        expected,_=predict(model,x[3:],[120,80],[20,10])
        expected+=offset*[20,10]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.pt'
            torch.save(dict(variant='se_last',state_dict=model.state_dict(),center=[120,80],scale=[20,10],offset_standardized=offset),path)
            np.testing.assert_allclose(load_and_predict(path,x[3:]),expected,rtol=0,atol=1e-5)
            with self.assertRaises(ValueError):load_and_predict(path,np.zeros_like(x))
            with self.assertRaises(ValueError):load_and_predict(path,x*2)

if __name__=='__main__': unittest.main()
