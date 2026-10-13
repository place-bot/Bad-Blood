"""Adapt only the PairSE change head on the complete early reference bank."""
import argparse
from pathlib import Path
import numpy as np
import torch
from predict import load_model, validate_input
from paired_se.model import chronological_pairs


def adapt(checkpoint, x, y, start, end, output, device='cpu'):
    model, saved = load_model(checkpoint, device)
    if saved.get('kind') != 'pair_population':
        raise ValueError('Start from the matching source PairSE checkpoint')
    x = validate_input(x, saved['modality'])
    y = np.asarray(y, np.float32)
    if y.shape != (len(x), 2) or not np.isfinite(y).all():
        raise ValueError('Expected finite K x 2 early reference BP')
    if np.shape(start) != (len(x),) or np.shape(end) != (len(x),):
        raise ValueError('Expected one start/end time per reference')
    q, r = chronological_pairs(start, end, guard=10.)
    c, s = np.asarray(saved['center'], np.float32), np.asarray(saved['scale'], np.float32)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('change.'))
    model.eval()  # Freeze dropout and BatchNorm running state.
    before = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
    optimizer = torch.optim.Adam(model.change.parameters(), lr=0.0003)
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor((y-c)/s, device=device)
    qt, rt = torch.as_tensor(q, device=device), torch.as_tensor(r, device=device)
    with torch.no_grad():
        h, p = model.encode(xt)
    losses = []
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        pred = model.from_features(h[qt], p[qt], h[rt], p[rt], yt[rt])
        loss = torch.nn.functional.mse_loss(pred, yt[qt])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite personal loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.change.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert all(torch.equal(before[k], v.cpu()) for k, v in model.encoder.state_dict().items())
    result = dict(saved, kind='personal', state_dict={k: v.detach().cpu() for k,v in model.state_dict().items()},
                  adaptation=dict(scope='head', lr=.0003, steps=25, n_support=len(y),
                                  pairs=len(q), loss_history=losses, query_labels_used=False))
    torch.save(result, output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--support', type=Path, required=True, help='NPZ keys: x, y, start, end')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    a = parser.parse_args()
    with np.load(a.support, allow_pickle=False) as z:
        adapt(a.checkpoint, z['x'], z['y'], z['start'], z['end'], a.output, a.device)
    print(f'Saved personal PairSE checkpoint to {a.output}')
