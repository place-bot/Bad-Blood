"""Portable personal tail adaptation. Never supply held-out query labels to fit()."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from modality_se.model import ModalitySE
from paired_se.model import PairSE
from rolling_core import rolling_folds, prefix, trajectory, predict, choose


def load(path, device='cpu'):
    # Load trusted project checkpoints only: torch pickle is not a safe file format.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = PairSE(ModalitySE(saved['modality'])).to(device).eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    return model, saved


def inputs(x, mode):
    x = np.asarray(x, np.float32)
    if mode == 'ppg' and x.ndim == 3 and x.shape[1:] == (1, 1250):
        x = np.concatenate([np.zeros_like(x), x], axis=1)
    if x.ndim != 3 or x.shape[1:] != (2, 1250) or len(x) == 0:
        raise ValueError('Expected N x 2 x 1250, or N x 1 x 1250 for PPG only')
    active = x[:, 1:] if mode == 'ppg' else x
    if not np.isfinite(x).all() or active.min() < -1e-5 or active.max() > 1.00001:
        raise ValueError('Use finite, filtered, per-window/channel [0,1] normalized signals')
    if np.any(np.ptp(active, axis=2) < 1e-8):
        raise ValueError('Flat active signal channel')
    return x


def fit(checkpoint, support, output, device='cpu'):
    """Select SBP/DBP settings inside this person's already isolated development set."""
    source, saved = load(checkpoint, device)
    x = inputs(support['x'], saved['modality'])
    y = np.asarray(support['y'], np.float32)
    start, end = np.asarray(support['start']), np.asarray(support['end'])
    if y.shape != (len(x), 2) or not np.isfinite(y).all() or np.any(y[:, 0] <= y[:, 1]):
        raise ValueError('y must contain finite SBP/DBP pairs in mmHg')
    if start.shape != (len(x),) or end.shape != start.shape or not np.allclose(end-start, 10):
        raise ValueError('start/end must identify ordered ten-second windows')
    c, s = np.asarray(saved['center'], np.float32), np.asarray(saved['scale'], np.float32)
    z = prefix(source, x, device)
    folds = rolling_folds(start, end)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selections = []
    for endpoint in (0, 1):
        records = []
        for fold in folds:
            t, v = fold['train'], fold['valid']
            rows = []
            def score(cfg, model):
                yp = predict(model, z[v], z[t], y[t], c, s)[:, endpoint]
                rows.append(dict(config=cfg, mae=float(np.abs(yp.astype(float)-y[v, endpoint]).mean())))
            score(None, source)
            for lr in (.0001, .0003):
                for steps, fitted, _ in trajectory(source, z[t], y[t], start[t], end[t], c, s,
                                                   'tail', lr, endpoint):
                    score(dict(scope='tail', lr=lr, steps=steps), fitted)
            records.append(dict(candidates=rows))
        selection = choose(records, endpoint)['selected']['tail']
        selections.append(selection)
        cfg = selection['config']
        if cfg is None:
            fitted = copy.deepcopy(source)
        else:
            _, fitted, _ = next(trajectory(source, z, y, start, end, c, s, 'tail', cfg['lr'],
                                          endpoint, (cfg['steps'],)))
        torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in fitted.state_dict().items()},
                        center=c, scale=s, modality=saved['modality'], endpoint=endpoint,
                        scope='tail', config=cfg), output/f'tail_endpoint{endpoint}.pt')
    (output/'selection.json').write_text(json.dumps(dict(method='rolling_tail', endpoint_selections=selections,
        final_test_used=False, support_count=len(x), source_checkpoint=str(checkpoint)), indent=2)+'\n')
    np.savez_compressed(output/'reference_bank.npz', x=x, y=y, start=start, end=end)


def infer(model_dir, current, references=None, device='cpu'):
    """Two independently selected endpoint copies; no current BP argument exists."""
    model_dir = Path(model_dir)
    if references is None:
        references = np.load(model_dir/'reference_bank.npz', allow_pickle=False)
    outputs = []
    for d in (0, 1):
        model, saved = load(model_dir/f'tail_endpoint{d}.pt', device)
        if saved.get('endpoint') != d or saved.get('scope') != 'tail':
            raise ValueError('Wrong endpoint checkpoint or adaptation scope')
        x = inputs(current, saved['modality'])
        r = inputs(references['x'], saved['modality'])
        y = np.asarray(references['y'], np.float32)
        if y.shape != (len(r), 2) or not np.isfinite(y).all():
            raise ValueError('Reference labels must be K x 2 finite SBP/DBP')
        yp = predict(model, prefix(model, x, device), prefix(model, r, device), y,
                     np.asarray(saved['center'], np.float32), np.asarray(saved['scale'], np.float32))
        outputs.append(yp[:, d])
    return np.column_stack(outputs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--checkpoint', type=Path, required=True)
    f.add_argument('--support', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    i = sub.add_parser('predict')
    i.add_argument('--models', type=Path, required=True)
    i.add_argument('--input', type=Path, required=True)
    i.add_argument('--references', type=Path)
    i.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.command == 'fit':
        fit(a.checkpoint, np.load(a.support, allow_pickle=False), a.output)
    else:
        refs = None if a.references is None else np.load(a.references, allow_pickle=False)
        np.save(a.output, infer(a.models, np.load(a.input, allow_pickle=False), refs))


if __name__ == '__main__':
    main()
