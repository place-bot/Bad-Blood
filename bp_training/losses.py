"""Objectives with explicit target units and an explicit global multiplier."""

import torch.nn.functional as F


def objective(prediction, labels, center, scale, config):
    if config.loss == "standardized_mse":
        raw = F.mse_loss(prediction.float(), (labels - center) / scale)
    else:
        mmhg = prediction.float() * scale + center
        raw = (F.mse_loss(mmhg, labels) if config.loss == "mmhg_mse" else
               F.huber_loss(mmhg, labels, delta=config.huber_delta))
    return raw * config.loss_multiplier, raw.detach()
