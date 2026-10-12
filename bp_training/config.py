"""Strict JSON configuration; no implicit CPU fallback or arbitrary model imports."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    architecture: str = "v1"
    attention: str = "none"
    loss: str = "standardized_mse"
    loss_multiplier: float = 1.0
    huber_delta: float = 5.0
    balancing: bool = False
    balance_cap: float = 4.0
    balance_min_subjects: int = 10
    sbp_edges: tuple = (90, 110, 130, 150, 170)
    dbp_edges: tuple = (60, 80, 100)
    seed: int = 20260923
    device: str = "cuda"
    precision: str = "fp32"
    batch_size: int = 256
    validation_batch_size: int = 512
    epochs: int = 40
    learning_rate: float = .001
    weight_decay: float = .001
    noise_std: float = .003
    clip_norm: float = 5.0
    lr_factor: float = .5
    lr_patience: int = 3
    early_stopping: int = 8
    workers: int = 0
    threads: int = 4
    prefetch: int = 2
    deterministic: bool = True

    def __post_init__(self):
        if self.architecture not in ("v1", "wider_resnet", "dual_tcn"):
            raise ValueError("Unknown architecture")
        if self.attention not in ("none", "pool", "se", "both"):
            raise ValueError("Unknown attention")
        if self.architecture != "dual_tcn" and self.attention != "none":
            raise ValueError("Attention requires dual_tcn")
        if self.loss not in ("standardized_mse", "mmhg_mse", "mmhg_huber"):
            raise ValueError("Unknown loss")
        if self.device not in ("cpu", "cuda", "mps") or self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("Unsupported device/precision")
        if self.precision != "fp32" and self.device != "cuda":
            raise ValueError("Mixed precision currently requires CUDA")
        for name in ("batch_size", "validation_batch_size", "epochs", "early_stopping",
                     "threads", "prefetch", "balance_min_subjects"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "workers", "lr_patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed >= 2**32:
            raise ValueError("seed must fit NumPy's 32-bit seed")
        for name in ("loss_multiplier", "huber_delta", "learning_rate", "clip_norm", "balance_cap"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("noise_std", "weight_decay"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not 0 < self.lr_factor < 1 or not math.isfinite(self.lr_factor):
            raise ValueError("lr_factor must be between zero and one")
        if self.balance_cap < 1:
            raise ValueError("balance_cap must be at least one")
        for name in ("balancing", "deterministic"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("sbp_edges", "dbp_edges"):
            values = tuple(getattr(self, name))
            if not values or any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
                raise ValueError(f"Invalid {name}")
            if any(a >= b for a, b in zip(values, values[1:])):
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, values)

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))


def load_config(path, **overrides):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Config must be an object")
    unknown = set(value) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    value.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**value)
