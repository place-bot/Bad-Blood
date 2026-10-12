import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from context_se.model import ContextResidual

class ContextModels(unittest.TestCase):
    def test_all_start_at_personal_bias_correction(self):
        torch.manual_seed(7)
        q=torch.randn(2,5,48);s=torch.randn(2,9,48);r=torch.randn(2,9,2)
        for variant in ('attention','context','kernel'):
            model=ContextResidual(variant)
            torch.testing.assert_close(model(q,s,r),r.mean(1,keepdim=True).expand(-1,5,-1))

    def test_context_permutation_and_finite_gradient(self):
        torch.manual_seed(8)
        q=torch.randn(2,5,48);s=torch.randn(2,9,48);r=torch.randn(2,9,2);perm=torch.randperm(9)
        for variant in ('attention','context','kernel'):
            model=ContextResidual(variant)
            if hasattr(model,'gain'):model.gain.data.fill_(1.)
            else:torch.nn.init.normal_(model.decode[-1].weight,std=.01)
            y=model(q,s,r);torch.testing.assert_close(y,model(q,s[:,perm],r[:,perm]),rtol=1e-4,atol=1e-5)
            y.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

if __name__=='__main__':unittest.main()
