"""Versioned, allowlisted ECG/PPG architectures."""

import torch
from torch import nn


class V1Block(nn.Module):
    def __init__(self, a, b, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(a, b, 7, stride, 3, bias=False), nn.BatchNorm1d(b), nn.SiLU(),
            nn.Conv1d(b, b, 5, 1, 2, bias=False), nn.BatchNorm1d(b))
        self.skip = (nn.Identity() if a == b and stride == 1 else nn.Sequential(
            nn.Conv1d(a, b, 1, stride, bias=False), nn.BatchNorm1d(b)))

    def forward(self, x):
        return torch.nn.functional.silu(self.net(x) + self.skip(x))


class ResidualCNN(nn.Module):
    """V1-compatible keys by default; explicit wider channels provide the control."""

    def __init__(self, channels=(8, 16, 32, 48), hidden=32):
        super().__init__()
        a, b, c, d = channels
        self.body = nn.Sequential(
            nn.AvgPool1d(1), nn.Conv1d(2, a, 7, 2, 3, bias=False),
            nn.BatchNorm1d(a), nn.SiLU(), V1Block(a, b, 2), V1Block(b, c, 2),
            V1Block(c, d, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(d, hidden),
                                  nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.head(self.body(x))


class SEGate(nn.Module):
    def __init__(self, channels=128, hidden=16):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(),
                                  nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x.mean(-1)).unsqueeze(-1)


class TemporalPool(nn.Module):
    def __init__(self, channels=128, hidden=32):
        super().__init__()
        self.score = nn.Sequential(nn.Conv1d(channels, hidden, 1), nn.Tanh(),
                                   nn.Conv1d(hidden, 1, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x):
        # Float32 softmax/reduction also under mixed precision.
        weights = torch.softmax(self.score(x).float(), dim=-1)
        return (x.float() * weights).sum(-1)


class EncoderStage(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(a, b, 5, 2, 2, bias=False),
                                 nn.BatchNorm1d(b), nn.SiLU(),
                                 nn.Conv1d(b, b, 3, 1, 1, bias=False), nn.BatchNorm1d(b))
        self.skip = nn.Sequential(nn.Conv1d(a, b, 1, 2, bias=False), nn.BatchNorm1d(b))

    def forward(self, x):
        return torch.nn.functional.silu(self.net(x) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stems = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, 16, k, padding=k // 2, bias=False),
                          nn.BatchNorm1d(16), nn.SiLU()) for k in (5, 15, 31)])
        self.project = nn.Sequential(nn.Conv1d(48, 32, 1, bias=False),
                                     nn.BatchNorm1d(32), nn.SiLU())
        self.stages = nn.Sequential(EncoderStage(32, 64), EncoderStage(64, 96))

    def forward(self, x):
        return self.stages(self.project(torch.cat([s(x) for s in self.stems], dim=1)))


class TemporalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        layers = []
        for i in range(2):
            layers.extend([nn.Conv1d(128, 128, 5, padding=2 * dilation,
                                     dilation=dilation, groups=128, bias=False),
                           nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128)])
            if i == 0:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.nn.functional.silu(self.net(x) + x)


class DualEncoderTCN(nn.Module):
    def __init__(self, attention="none"):
        super().__init__()
        self.ecg, self.ppg = Encoder(), Encoder()
        self.fusion = nn.Sequential(nn.Conv1d(192, 128, 1, bias=False),
                                    nn.BatchNorm1d(128), nn.SiLU())
        self.gate = SEGate() if attention in ("se", "both") else nn.Identity()
        self.temporal = nn.Sequential(*[TemporalBlock(d) for d in (1, 2, 4, 8, 16, 32)])
        self.pool = (TemporalPool() if attention in ("pool", "both") else
                     nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten()))
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, x):
        x = self.fusion(torch.cat((self.ecg(x[:, :1]), self.ppg(x[:, 1:2])), dim=1))
        return self.head(self.pool(self.temporal(self.gate(x))))


ARCHITECTURES = ("v1", "wider_resnet", "dual_tcn")
ATTENTIONS = ("none", "pool", "se", "both")
ARCHITECTURE_VERSION = 1


def make_model(architecture, attention="none"):
    if architecture not in ARCHITECTURES or attention not in ATTENTIONS:
        raise ValueError("Unknown architecture or attention variant")
    if architecture != "dual_tcn" and attention != "none":
        raise ValueError("Attention variants currently require dual_tcn")
    if architecture == "dual_tcn":
        return DualEncoderTCN(attention)
    return ResidualCNN() if architecture == "v1" else ResidualCNN((28, 56, 108, 160), 96)
