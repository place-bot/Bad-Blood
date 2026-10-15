"""SE-last with an explicit input mask applied even by PairSE.encode."""
import torch
from torch import nn
from shared.models import BPNet

class InputMask(nn.Module):
    def __init__(self,ppg_only):super().__init__();self.ppg_only=ppg_only
    def forward(self,x):
        if not self.ppg_only:return x
        return x*x.new_tensor([0.,1.])[None,:,None]

class ModalitySE(BPNet):
    def __init__(self,modality='dual'):
        if modality not in ('dual','ppg'):raise ValueError(modality)
        super().__init__(variant='se_last');self.modality=modality
        self.body[0]=nn.Sequential(InputMask(modality=='ppg'),self.body[0])
    @property
    def available_channels(self):return (1,) if self.modality=='ppg' else (0,1)

def make_model(variant,seed):
    if variant!='se_ppg':raise ValueError(variant)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return ModalitySE('ppg')
