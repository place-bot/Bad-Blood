import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,a,b,stride=1):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(a,b,7,stride,3,bias=False),nn.BatchNorm1d(b),nn.SiLU(),nn.Conv1d(b,b,5,1,2,bias=False),nn.BatchNorm1d(b))
        self.skip=nn.Identity() if a==b and stride==1 else nn.Sequential(nn.Conv1d(a,b,1,stride,bias=False),nn.BatchNorm1d(b))
    def forward(self,x):return torch.nn.functional.silu(self.net(x)+self.skip(x))

class BPNet(nn.Module):
    def __init__(self,pool=5):
        super().__init__()
        self.body=nn.Sequential(nn.AvgPool1d(pool),nn.Conv1d(2,8,7,2,3,bias=False),nn.BatchNorm1d(8),nn.SiLU(),Block(8,16,2),Block(16,32,2),Block(32,48,2),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Dropout(.15),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,2))
    def forward(self,x):return self.head(self.body(x))
