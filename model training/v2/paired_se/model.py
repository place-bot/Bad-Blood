"""Reference-conditioned pressure change with a shared pretrained CNN+SE."""
import copy
import numpy as np
import torch
from torch import nn


class PairSE(nn.Module):
    def __init__(self, encoder, feature_center=None, feature_scale=None):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.register_buffer('feature_center',torch.as_tensor(np.zeros(48) if feature_center is None else feature_center,dtype=torch.float32))
        self.register_buffer('feature_scale',torch.as_tensor(np.ones(48) if feature_scale is None else feature_scale,dtype=torch.float32))
        self.change = nn.Sequential(nn.Linear(100,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,2))
        nn.init.zeros_(self.change[-1].weight);nn.init.zeros_(self.change[-1].bias)

    def encode(self, x):
        if x.ndim!=3 or x.shape[1:]!=(2,1250):raise ValueError('Expected ECG/PPG N x 2 x 1250')
        for i,layer in enumerate(self.encoder.body):
            x=layer(x)
            if str(i) in self.encoder.gates:x=self.encoder.gates[str(i)](x)
        return x,self.encoder.head(x)

    def from_features(self,hc,pc,hr,pr,yr):
        hc=(hc-self.feature_center)/self.feature_scale
        hr=(hr-self.feature_center)/self.feature_scale
        delta=pc-pr
        given=torch.cat([hc-hr,hr,yr,delta],dim=-1)
        identity=torch.cat([torch.zeros_like(hr),hr,yr,torch.zeros_like(delta)],dim=-1)
        return yr+delta+self.change(given)-self.change(identity)

    def forward(self,current,reference,reference_bp):
        hc,pc=self.encode(current);hr,pr=self.encode(reference)
        return self.from_features(hc,pc,hr,pr,reference_bp)


def same_person_references(subjects,seed):
    """One non-self reference per current row; no pressure-dependent sampling."""
    subjects=np.asarray(subjects);rng=np.random.default_rng(seed)
    out=np.empty(len(subjects),dtype=np.int64)
    for person in np.unique(subjects):
        rows=np.flatnonzero(subjects==person)
        if len(rows)<2:raise ValueError('A person needs at least two windows')
        positions=np.arange(len(rows))
        out[rows]=rows[(positions+rng.integers(1,len(rows),len(rows)))%len(rows)]
    if np.any(out==np.arange(len(out))) or np.any(subjects[out]!=subjects):raise AssertionError('Bad pairs')
    return out


def chronological_pairs(start,end,guard=10.):
    start,end=np.asarray(start),np.asarray(end)
    if start.ndim!=1 or end.shape!=start.shape or not np.all(end>start):raise ValueError('Bad times')
    current,reference=np.where(end[None,:]+guard<=start[:,None]+1e-6)
    if not len(current) or len(np.unique(np.r_[current,reference]))!=len(start):
        raise ValueError('Every support row must participate in a separated pair')
    return current,reference


def predict_bank(model,current,reference,reference_y,center,scale,device='cpu'):
    """Average over ALL known support references. No query labels accepted."""
    if len(reference)!=len(reference_y) or not len(reference):raise ValueError('Invalid reference bank')
    model=model.to(device).eval()
    c,s=np.asarray(center,np.float32),np.asarray(scale,np.float32)
    outputs=[]
    with torch.inference_mode():
        hr,pr=model.encode(torch.as_tensor(reference,device=device))
        yr=torch.as_tensor((reference_y-c)/s,device=device)
        n=len(reference)
        for i in range(0,len(current),32):
            hc,pc=model.encode(torch.as_tensor(current[i:i+32],device=device));b=len(hc)
            pred=model.from_features(hc[:,None,:].expand(-1,n,-1).reshape(-1,48),
                pc[:,None,:].expand(-1,n,-1).reshape(-1,2),hr.repeat(b,1),pr.repeat(b,1),yr.repeat(b,1))
            outputs.append(pred.reshape(b,n,2).mean(1).cpu().numpy()*s+c)
    return np.concatenate(outputs)
