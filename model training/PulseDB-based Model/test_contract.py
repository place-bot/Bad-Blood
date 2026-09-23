import unittest
import numpy as np
from common import normalize, extract, from_raw_125

class ContractTests(unittest.TestCase):
    def setUp(self):
        t=np.arange(1250)/125
        self.x=np.stack([np.sin(2*np.pi*1.2*t),np.sin(2*np.pi*1.2*t-.5)])
    def test_shapes(self):
        z=normalize(self.x)
        self.assertEqual(z.shape,(2,1250))
        np.testing.assert_allclose(z.min(1),0)
        np.testing.assert_allclose(z.max(1),1)
        f,n=extract(z,True)
        self.assertEqual(len(f),122)
        self.assertEqual(len(set(n)),122)
        self.assertTrue(np.isfinite(f).all())
    def test_reject(self):
        for a in [np.zeros((2,1250)),np.ones((2,1249)),np.full((2,1250),np.nan)]:
            with self.assertRaises(ValueError): normalize(a)
    def test_raw_adapter(self):
        self.assertTrue(np.isfinite(from_raw_125(self.x)).all())
    def test_normalization_affine_invariance(self):
        np.testing.assert_allclose(normalize(self.x),normalize(self.x*4+7),atol=1e-6)

if __name__=='__main__':unittest.main()
