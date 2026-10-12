import unittest
import numpy as np
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

if __name__=='__main__': unittest.main()
