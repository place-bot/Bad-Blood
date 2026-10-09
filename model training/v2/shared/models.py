"""Frozen Track A models; the backbone retains every V1 state-dict key."""
from __future__ import annotations

import torch
from torch import nn

VARIANTS = (
    "baseline", "se_last", "temporal_last", "cbam_last", "additive_last",
    "se_all", "eca_last", "baseline_ecg", "baseline_ppg",
)
EXPECTED_PARAMETERS = {
    "baseline": 37666, "se_last": 38296, "temporal_last": 37680,
    "cbam_last": 38310, "additive_last": 38296, "se_all": 38670,
    "eca_last": 37669, "baseline_ecg": 37666, "baseline_ppg": 37666,
}


class Block(nn.Module):
    """Numerically identical to the original V1 residual block."""

    def __init__(self, a: int, b: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(a, b, 7, stride, 3, bias=False), nn.BatchNorm1d(b),
            nn.SiLU(), nn.Conv1d(b, b, 5, 1, 2, bias=False), nn.BatchNorm1d(b),
        )
        self.skip = nn.Identity() if a == b and stride == 1 else nn.Sequential(
            nn.Conv1d(a, b, 1, stride, bias=False), nn.BatchNorm1d(b),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.net(x) + self.skip(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, combined: bool = False, additive: bool = False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 8), nn.ReLU(),
            nn.Linear(channels // 8, channels),
        )
        self.combined = combined
        self.additive = additive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = self.mlp(x.mean(dim=-1))
        if self.additive:
            return x + descriptor.unsqueeze(-1)
        if self.combined:
            descriptor = descriptor + self.mlp(x.amax(dim=-1))
        return x * descriptor.sigmoid().unsqueeze(-1)


class TemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return x * self.conv(pooled).sigmoid()


class EfficientChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This convolution acts on channels, never on the original time axis.
        descriptor = self.conv(x.mean(dim=-1).unsqueeze(1)).squeeze(1)
        return x * descriptor.sigmoid().unsqueeze(-1)


class BPNet(nn.Module):
    """V1 high-resolution CNN with optional post-block interventions.

    ``body`` and ``head`` are constructed before all gates so equal seeds give
    identical backbone initialization. Gates are external to ``body`` to keep
    existing checkpoint keys and Track B parameter masks unchanged.
    """

    def __init__(self, pool: int = 1, variant: str = "baseline"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        if pool != 1:
            raise ValueError("The frozen v2 protocol requires pool=1")
        self.variant = variant
        self.pool = pool
        self.body = nn.Sequential(
            nn.AvgPool1d(pool), nn.Conv1d(2, 8, 7, 2, 3, bias=False),
            nn.BatchNorm1d(8), nn.SiLU(), Block(8, 16, 2), Block(16, 32, 2),
            Block(32, 48, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 2))
        self.gates = nn.ModuleDict()
        if variant == "se_all":
            self.gates.update({str(i): ChannelAttention(c) for i, c in ((4, 16), (5, 32), (6, 48))})
        elif variant in ("se_last", "additive_last"):
            self.gates["6"] = ChannelAttention(48, additive=variant == "additive_last")
        elif variant == "temporal_last":
            self.gates["6"] = TemporalAttention()
        elif variant == "cbam_last":
            self.gates["6"] = nn.Sequential(ChannelAttention(48, combined=True), TemporalAttention())
        elif variant == "eca_last":
            self.gates["6"] = EfficientChannelAttention()
        if count_parameters(self) != EXPECTED_PARAMETERS[variant]:
            raise AssertionError(f"Unexpected parameter count for {variant}")

    @property
    def available_channels(self) -> tuple[int, ...]:
        return (0,) if self.variant == "baseline_ecg" else (1,) if self.variant == "baseline_ppg" else (0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (2, 1250):
            raise ValueError(f"Expected (batch, 2, 1250), got {tuple(x.shape)}")
        if len(self.available_channels) == 1:
            mask = x.new_tensor([1., 0.] if self.variant == "baseline_ecg" else [0., 1.])
            x = x * mask[None, :, None]
        for i, layer in enumerate(self.body):
            x = layer(x)
            if str(i) in self.gates:
                x = self.gates[str(i)](x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model(variant: str = "baseline", seed: int = 42) -> BPNet:
    # Do not change the caller's training RNG state while initializing a model.
    with torch.random.fork_rng(devices=[]):
        # Parameters are constructed on CPU; do not also reseed MPS/CUDA.
        torch.random.default_generator.manual_seed(seed)
        return BPNet(pool=1, variant=variant)
