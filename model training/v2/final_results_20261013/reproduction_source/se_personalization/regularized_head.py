"""Development-only, L2-SP-inspired frozen-feature personalization screen.

No target evaluation is implemented here. All personal support is used in the
final closed-form fit. Predictions require waveforms, never query labels.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.data import load_personal_cohort
from shared.models import BPNet
from shared.metrics import summarize
from se_personalization.run import dense_support, load_se, SEEDS, DEV_IDS
from personalized_forecasting.run import atomic_json, file_hash, array_hash, canonical_hash


def grid():
    return [dict(feature=feature, anchor=anchor, penalty=penalty)
            for feature in ('penultimate32', 'pooled48')
            for anchor in ('population', 'mean')
            for penalty in (.01, .1, 1., 10., 100., 1000.)]


def key(config):
    return f"{config['feature']}_{config['anchor']}_lambda{config['penalty']:g}"


def fit_head(features, population_predictions, labels, anchor, penalty):
    h, p, y = [np.asarray(v, dtype=np.float64) for v in (features, population_predictions, labels)]
    if h.ndim != 2 or p.shape != (len(h), 2) or y.shape != p.shape or len(h) < 2:
        raise ValueError('Invalid support shapes')
    if not all(np.isfinite(v).all() for v in (h, p, y)) or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Invalid values')
    if anchor not in ('population', 'mean'):
        raise ValueError('Unknown anchor')
    location = h.mean(0)
    scale = np.maximum(h.std(0), .01)  # Support only; near-constant features bounded.
    z = (h-location)/scale
    offset = (y-p).mean(0) if anchor == 'population' else y.mean(0)
    residual = y-p-offset if anchor == 'population' else y-offset
    weights = np.linalg.solve(z.T@z/len(z)+penalty*np.eye(z.shape[1]), z.T@residual/len(z))
    return dict(location=location, scale=scale, weights=weights, offset=offset,
                anchor=anchor, penalty=float(penalty), support_count=len(y))


def apply_head(state, features, population_predictions):
    h, p = np.asarray(features, dtype=np.float64), np.asarray(population_predictions, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != len(state['location']) or p.shape != (len(h), 2):
        raise ValueError('Invalid inference shapes')
    if not np.isfinite(h).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite inference data')
    base = p if state['anchor'] == 'population' else np.zeros_like(p)
    return base+state['offset']+(h-state['location'])/state['scale']@state['weights']


def extract(model, waves, center, scale, device):
    model = model.to(device).eval()
    captured = {}
    handles = [model.head.register_forward_pre_hook(lambda m,a: captured.update(pooled48=a[0].detach())),
               model.head[3].register_forward_pre_hook(lambda m,a: captured.update(penultimate32=a[0].detach()))]
    features = {'penultimate32': [], 'pooled48': []}; preds = []
    try:
        with torch.inference_mode():
            for start in range(0, len(waves), 256):
                x = torch.as_tensor(waves[start:start+256], device=device)
                out = model(x).cpu().numpy()
                preds.append(out*np.asarray(scale)+np.asarray(center))
                for name in features: features[name].append(captured[name].cpu().numpy())
    finally:
        for handle in handles: handle.remove()
    return {k:np.concatenate(v) for k,v in features.items()}, np.concatenate(preds)


def rank(rows, ids):
    methods = ['mean', 'bias']+[key(c) for c in grid()]
    maes = {}
    for method in methods:
        matches = [r for r in rows if r['method'] == method and r['patient'] in ids]
        if {(r['patient'],r['seed']) for r in matches} != {(p,s) for p in ids for s in SEEDS}:
            raise ValueError('Incomplete development matrix')
        maes[method] = np.mean([r['metrics']['macro']['mae'] for r in matches], axis=0)
    ranking = []
    for config in grid():
        m = key(config)
        ratios = np.concatenate([maes[m]/maes[c] for c in ('mean', 'bias')])
        ranking.append(dict(method=m, config=config, mae=maes[m].tolist(),
            worst_ratio=float(ratios.max()), mean_mae=float(maes[m].mean())))
    ranking.sort(key=lambda x:(x['worst_ratio'], x['mean_mae'], x['method']))
    return dict(selected=ranking[0], ranking=ranking,
                controls={k:maes[k].tolist() for k in ('mean','bias')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--population-root', type=Path, required=True)
    p.add_argument('--individual-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('Use a fresh output; do not overwrite a study')
    torch.set_num_threads(4)
    cohort = load_personal_cohort(a.individual_root)['development']
    assert {v['patient_id'] for v in cohort} == set(DEV_IDS)
    checkpoints = {s:a.population_root/f'se_last_seed{s}'/'best.pt' for s in SEEDS}
    saved = {s:load_se(path,s) for s,path in checkpoints.items()}
    manifest = dict(study='regularized-head-development-only', created_at=datetime.now(timezone.utc).isoformat(),
        target_evaluated=False, grid=grid(), seeds=list(SEEDS), device=a.device,
        source_hashes={str(f.relative_to(ROOT)):file_hash(f) for f in
            [Path(__file__),Path(__file__).with_name('LITERATURE.md'),ROOT/'shared/models.py',ROOT/'shared/data.py',ROOT/'shared/metrics.py']},
        checkpoint_hashes={str(s):file_hash(path) for s,path in checkpoints.items()},
        participants={v['patient_id']:dict(support_indices=dense_support(v).tolist(),
            query_indices=v['query_indices'].tolist(), data_hash=array_hash(v['x'],v['y'],v['start'])) for v in cohort})
    a.output.mkdir(parents=True)
    atomic_json(a.output/'manifest.json', manifest)
    rows = []
    for seed in SEEDS:
        model = BPNet(variant='se_last'); model.load_state_dict(saved[seed]['state_dict'], strict=True)
        for patient in cohort:
            support, query = dense_support(patient), patient['query_indices']
            indices = np.r_[support, query]
            features, predictions = extract(model,patient['x'][indices],saved[seed]['center'],saved[seed]['scale'],a.device)
            n = len(support); sp, qp = predictions[:n], predictions[n:]
            y, truth = patient['y'][support], patient['y'][query]
            outputs = dict(mean=np.tile(y.astype(float).mean(0),(len(query),1)),
                           bias=qp+(y.astype(float)-sp).mean(0), population=qp)
            states = {}
            for config in grid():
                h = features[config['feature']]
                state = fit_head(h[:n],sp,y,config['anchor'],config['penalty'])
                outputs[key(config)] = apply_head(state,h[n:],qp)
                states[key(config)] = state
            directory = a.output/patient['patient_id']/f'seed{seed}'
            directory.mkdir(parents=True)
            for method, prediction in outputs.items():
                row = dict(patient=patient['patient_id'], seed=seed, method=method,
                    n_support=n, metrics=summarize(truth,prediction,np.repeat(patient['patient_id'],len(query))))
                rows.append(row)
                np.savez_compressed(directory/f'{method}.npz',prediction=prediction,y=truth,
                    support_indices=support,query_indices=query)
            for method, state in states.items():
                np.savez_compressed(directory/f'{method}_state.npz',**state)
            print(json.dumps(dict(seed=seed,patient=patient['patient_id'],n_support=n,units=len(rows))),flush=True)
    result = rank(rows, DEV_IDS)
    # Cross-person stability diagnostic, not an independent test or a retuning step.
    result['leave_one_development_person_out'] = {}
    for held in DEV_IDS:
        fit_ids = [v for v in DEV_IDS if v != held]
        selection = rank(rows,fit_ids)['selected']['method']
        held_mae = np.mean([r['metrics']['macro']['mae'] for r in rows if r['patient']==held and r['method']==selection],0)
        result['leave_one_development_person_out'][held] = dict(selected=selection,mae=held_mae.tolist())
    result['manifest_hash'] = canonical_hash(manifest)
    result['target_evaluated'] = False
    atomic_json(a.output/'records.json', rows)
    atomic_json(a.output/'summary.json', result)
    log = ROOT.parents[1]/'autoresearch'/'classic-261012-se-regularized'
    log.mkdir(parents=True,exist_ok=True)
    (log/'classic-results.tsv').write_text('iteration\tmethod\tworst_development_ratio\n'+''.join(
        f"{i}\t{r['method']}\t{r['worst_ratio']:.8f}\n" for i,r in enumerate(result['ranking'],1)))
    atomic_json(log/'handoff.json',dict(mode='classic',status='development_complete',iterations=24,
        selected=result['selected'],target_evaluated=False,manifest_hash=result['manifest_hash']))
    print(json.dumps(result['selected']),flush=True)


if __name__ == '__main__': main()
