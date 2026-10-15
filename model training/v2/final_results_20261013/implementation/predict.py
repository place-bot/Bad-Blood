"""Portable CPU inference for the V2 final package. Input is normalized signals."""
import argparse
from pathlib import Path
import numpy as np
import torch
from modality_se.model import ModalitySE
from paired_se.model import PairSE, predict_bank


def load_model(checkpoint, device='cpu'):
    # Only load trusted project checkpoints; torch pickle is not a safe file format.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    encoder = ModalitySE(saved['modality'])
    model = encoder if saved.get('kind') == 'population' else PairSE(encoder)
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), saved


def validate_input(x, modality):
    x = np.asarray(x, dtype=np.float32)
    if modality == 'ppg' and x.ndim == 3 and x.shape[1:] == (1, 1250):
        x = np.concatenate([np.zeros_like(x), x], axis=1)
    if x.ndim != 3 or x.shape[1:] != (2, 1250) or not len(x):
        raise ValueError('Expected nonempty N x 2 x 1250; PPG may use N x 1 x 1250')
    if not np.isfinite(x).all():
        raise ValueError('Signals must be finite and already filtered/normalized')
    if x.min() < -1e-5 or x.max() > 1.00001:
        raise ValueError('Expected per-window/channel min-max normalized [0,1] signals')
    return x


def predict(checkpoint, current, reference=None, reference_bp=None, device='cpu'):
    model, saved = load_model(checkpoint, device)
    x = validate_input(current, saved['modality'])
    center = np.asarray(saved['center'], np.float32)
    scale = np.asarray(saved['scale'], np.float32)
    if saved.get('kind') == 'population':
        out = []
        with torch.inference_mode():
            for i in range(0, len(x), 256):
                out.append(model(torch.as_tensor(x[i:i+256], device=device)).cpu().numpy())
        return np.concatenate(out) * scale + center
    if reference is None or reference_bp is None:
        raise ValueError('PairSE requires early reference waveforms and known SBP/DBP')
    r = validate_input(reference, saved['modality'])
    y = np.asarray(reference_bp, np.float32)
    if y.shape != (len(r), 2) or not np.isfinite(y).all():
        raise ValueError('reference_bp must be K x 2, finite SBP/DBP in mmHg')
    return predict_bank(model, x, r, y, center, scale, device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--references', type=Path)
    p.add_argument('--reference-bp', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    load = lambda f: None if f is None else np.load(f, allow_pickle=False)
    y = predict(a.checkpoint, load(a.input), load(a.references), load(a.reference_bp), a.device)
    np.save(a.output, y)
    print(f'Saved {len(y)} SBP/DBP predictions (mmHg) to {a.output}')


if __name__ == '__main__':
    main()
