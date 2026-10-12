import unittest
import numpy as np
import torch
from residual_calibration.model import candidates,fit,predict,forward_cv

class ResidualCalibrationTests(unittest.TestCase):
    def test_constant_residual_is_bias(self):
        rng=np.random.default_rng(1);h=rng.normal(size=(10,48));p=rng.normal(size=(10,2));y=p+[5.,-3.]
        for cfg in candidates():
            if cfg['family']=='mean': continue
            st=fit(cfg,h,p,y,[120,80],[20,10]);np.testing.assert_allclose(predict(st,h,p),y,atol=1e-10)
    def test_no_query_labels_and_batch_independence(self):
        rng=np.random.default_rng(1);h=rng.normal(size=(8,48));p=rng.normal(size=(8,2));y=rng.normal(size=(8,2));q=rng.normal(size=(3,48));qp=rng.normal(size=(3,2))
        for cfg in candidates():
            st=fit(cfg,h,p,y,[120,80],[20,10]); all_=predict(st,q,qp)
            np.testing.assert_allclose(all_[:1],predict(st,q[:1],qp[:1]),atol=1e-10)
    def test_forward_cv_guard_and_fallback(self):
        rng=np.random.default_rng(1);h=rng.normal(size=(20,48));p=rng.normal(size=(20,2));y=rng.normal(size=(20,2));start=np.arange(20)*20.;end=start+10
        _, audit=forward_cv(h,p,y,start,end,[120,80],[20,10])
        self.assertGreater(audit['count'],0)
        for f in audit['folds']: self.assertLessEqual(max(end[f['train']])+10,min(start[f['validation']]))
        _, small=forward_cv(h[:2],p[:2],y[:2],start[:2],end[:2],[120,80],[20,10])
        self.assertEqual(small['selected'],['bias','bias'])
    def test_scalar_slope_can_be_learned(self):
        p=np.column_stack([np.arange(30.),np.arange(30.)]);y=2*p+[100.,50.];h=np.zeros((30,48))
        cfg=dict(family='prediction_linear',width=1.,penalty=.01)
        st=fit(cfg,h,p,y,[0,0],[1,1]); err=abs(predict(st,h,p)-y).mean()
        self.assertLess(err,.01)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS replay check')
    def test_saved_pair_double_labels_and_ppg_mask(self):
        from modality_se.model import ModalitySE
        from paired_se.model import PairSE, predict_bank
        from residual_calibration.run import predict_saved
        model=PairSE(ModalitySE('ppg')).to('mps').eval()
        rng=np.random.default_rng(3);x=rng.random((3,2,1250),dtype=np.float32);refs=x[:2]
        y=np.array([[115,75],[125,85]],np.float32);c=np.array([120.,80.]);s=np.array([20.,10.])
        with torch.inference_mode():hr,pr=model.encode(torch.as_tensor(refs,device='mps'))
        saved=dict(modality='ppg',encoder=model.encoder.state_dict(),center=c,scale=s,
            feature_center=np.zeros(48),feature_scale=np.ones(48),selected=['prior_pair','prior_pair'],states=[None,None],
            prior=dict(state_dict=model.state_dict(),hr=hr.cpu().numpy(),pr=pr.cpu().numpy(),yr=(y-c)/s))
        expected=predict_bank(model,x,refs,y,c,s,'mps')
        np.testing.assert_allclose(predict_saved(saved,x),expected,atol=.001,rtol=0)
        other=x.copy();other[:,0]+=50
        np.testing.assert_array_equal(predict_saved(saved,x),predict_saved(saved,other))

if __name__=='__main__': unittest.main()
