"""Atomic, weights-only-loadable training checkpoints and RNG restoration."""

import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def save_checkpoint(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def rng_state(sampler, device):
    state = np.random.get_state()
    result = {"python": random.getstate(),
              "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
              "torch": torch.get_rng_state(), "sampler": sampler.bit_generator.state}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state, sampler, device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    sampler.bit_generator.state = state["sampler"]
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])
