import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from context_se.model import ContextResidual
from context_se.adapt import MaskedContext

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

    def test_masked_training_ignores_forbidden_reference_labels(self):
        torch.manual_seed(9);model=MaskedContext()
        torch.nn.init.normal_(model.decode[-1].weight,std=.01)
        q=torch.randn(2,1,48);s=torch.randn(2,5,48);r=torch.randn(2,5,2)
        mask=torch.tensor([[1,1,0,0,0],[1,1,1,0,0]],dtype=torch.bool)
        changed=r.clone();changed[~mask]+=1000
        torch.testing.assert_close(model(q,s,r,mask),model(q,s,changed,mask),atol=1e-6,rtol=0)

    def test_all_valid_mask_agrees_with_original_context_forward(self):
        torch.manual_seed(10);model=MaskedContext()
        q=torch.randn(2,3,48);s=torch.randn(2,5,48);r=torch.randn(2,5,2)
        torch.testing.assert_close(model(q,s,r),model(q,s,r,torch.ones(2,5,dtype=torch.bool)))

if __name__=='__main__':unittest.main()
