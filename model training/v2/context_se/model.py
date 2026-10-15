"""Reference-set residual predictors on a fixed SE representation."""
import torch
from torch import nn
from torch.nn import functional as F

class ContextResidual(nn.Module):
    def __init__(self,variant):
        super().__init__()
        if variant not in ('attention','context','kernel'):raise ValueError(variant)
        self.variant=variant
        if variant in ('attention','kernel'):
            self.project=nn.Sequential(nn.Linear(48,32),nn.SiLU(),nn.Linear(32,16))
            self.log_scale=nn.Parameter(torch.tensor(0.))
            self.gain=nn.Parameter(torch.zeros(2))
        if variant=='context':
            self.context=nn.Sequential(nn.Linear(50,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
            self.decode=nn.Sequential(nn.Linear(80,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,2))
            nn.init.zeros_(self.decode[-1].weight);nn.init.zeros_(self.decode[-1].bias)
        if variant=='kernel':self.log_noise=nn.Parameter(torch.tensor(-2.))

    def forward(self,q,s,residual):
        # B x Q x 48, B x S x 48, B x S x 2. No query labels accepted.
        mean=residual.mean(1,keepdim=True)
        centered=residual-mean
        if self.variant=='context':
            z=self.context(torch.cat([s,centered],-1)).mean(1,keepdim=True)
            current=self.decode(torch.cat([q,z.expand(-1,q.shape[1],-1)],-1))
            observed=self.decode(torch.cat([s,z.expand(-1,s.shape[1],-1)],-1))
            return mean+current-observed.mean(1,keepdim=True)
        qz,sz=self.project(q),self.project(s)
        distance=((qz[:,:,None,:]-sz[:,None,:,:])**2).mean(-1)
        scale=F.softplus(self.log_scale)+1e-4
        if self.variant=='attention':
            weights=torch.softmax(-distance/scale,dim=-1)
            return mean+self.gain*(weights@centered)
        ss=((sz[:,:,None,:]-sz[:,None,:,:])**2).mean(-1)
        kernel=torch.exp(-ss/scale)
        noise=F.softplus(self.log_noise)+1e-4
        coefficients=torch.linalg.solve(kernel+noise*torch.eye(s.shape[1],device=s.device),centered)
        change=torch.exp(-distance/scale)@coefficients
        anchor=(kernel@coefficients).mean(1,keepdim=True)
        return mean+self.gain*(change-anchor)
