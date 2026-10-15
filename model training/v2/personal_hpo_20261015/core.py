"""Personal-only selection primitives; none accepts final-test labels."""
from __future__ import annotations
import copy
import numpy as np
import torch
from paired_se.model import chronological_pairs

LRS = (.0003, .001)
STEPS = (10, 25, 50, 100)
FIXED = dict(scope='head', lr=.0003, steps=25)


def split_indices(start, end):
    start, end = np.asarray(start), np.asarray(end)
    assert len(start) == len(end) and np.all(np.diff(start) > 0)
    outer = int(.8 * len(start))
    support = np.arange(outer)
    query = np.flatnonzero((np.arange(len(start)) >= outer) &
                          (start >= end[outer-1] + 10 - 1e-6))
    cut = int(.7 * len(support))
    train = support[:cut]
    valid = support[cut:][start[support[cut:]] >= end[train[-1]] + 10 - 1e-6]
    if len(valid) < 2 or len(query) < 2:
        raise ValueError('Need at least two inner-validation and final-test segments')
    chronological_pairs(start[train], end[train])
    chronological_pairs(start[support], end[support])
    return dict(support=support, query=query, inner_train=train, inner_valid=valid)


def encode(model, x):
    hs, ps = [], []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            h, p = model.encode(torch.as_tensor(x[i:i+128], dtype=torch.float32))
            hs.append(h.detach()); ps.append(p.detach())
    return torch.cat(hs), torch.cat(ps)


def design(model, h, p, y, start, end, center, scale):
    """Exact all-pair objective, with frozen features computed just once."""
    q, r = chronological_pairs(start, end)
    q, r = torch.as_tensor(q), torch.as_tensor(r)
    with torch.no_grad():
        z = (h-model.feature_center)/model.feature_scale
        yr = torch.as_tensor((np.asarray(y, np.float32)-center)/scale)
        delta = p[q]-p[r]
        given = torch.cat([z[q]-z[r], z[r], yr[r], delta], -1)
        identity = torch.cat([torch.zeros_like(z), z, yr, torch.zeros_like(p)], -1)
        base = yr[r]+delta
        target = yr[q]
    return given, identity, r, base, target


def trajectory(model, h, p, y, start, end, center, scale, lr, checkpoints):
    """Fresh source initialization; all support pairs in every gradient update."""
    fitted = copy.deepcopy(model).cpu().eval()
    for n, param in fitted.named_parameters():
        param.requires_grad_(n.startswith('change.'))
    given, identity, refs, base, target = design(fitted,h,p,y,start,end,center,scale)
    optimizer = torch.optim.Adam(fitted.change.parameters(), lr=lr)
    history = []
    for step in range(1, max(checkpoints)+1):
        optimizer.zero_grad(set_to_none=True)
        # Compute repeated identity terms once per reference, preserving the
        # exact sum of gradients. Chunked loss weights sum to one full-batch MSE.
        value = 0.
        for first in range(0, len(given), 65536):
            sl = slice(first, first+65536)
            pred = base[sl]+fitted.change(given[sl])-fitted.change(identity)[refs[sl]]
            loss = (pred-target[sl]).square().sum()/(len(given)*2)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite personal training loss')
            loss.backward(); value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(fitted.change.parameters(), 5., error_if_nonfinite=True)
        optimizer.step(); history.append(value)
        if step in checkpoints:
            yield step, fitted, dict(pair_count=len(given), loss_history=list(history),
                                    support_count=len(y), encoder_frozen=True)


def predict_features(model, hc, pc, hr, pr, reference_y, center, scale):
    """All references, no current labels; same formula as predict_bank."""
    yr = torch.as_tensor((np.asarray(reference_y,np.float32)-center)/scale)
    outputs=[]
    with torch.no_grad():
        for i in range(0,len(hc),32):
            h,p=hc[i:i+32],pc[i:i+32]; b,n=len(h),len(hr)
            out=model.from_features(h[:,None,:].expand(-1,n,-1).reshape(-1,48),
                p[:,None,:].expand(-1,n,-1).reshape(-1,2),
                hr.repeat(b,1),pr.repeat(b,1),yr.repeat(b,1))
            outputs.append(out.reshape(b,n,2).mean(1).numpy()*scale+center)
    return np.concatenate(outputs)


def candidate_score(truth, pred, controls):
    mae=np.abs(np.asarray(pred,float)-truth).mean(0)
    denom=np.stack([np.maximum(np.abs(c-truth).mean(0),.5) for c in controls])
    return dict(mae=mae.tolist(), worst_ratio=float((mae[None,:]/denom).max()),
                mean_mae=float(mae.mean()))


def select_person(model, h, p, development_y, start, end, indices, center, scale):
    """Only THIS person's development labels; no cohort/test-label argument."""
    t,v=indices['inner_train'],indices['inner_valid']
    y=np.asarray(development_y)
    original=p.numpy()*scale+center
    controls=[np.tile(y[t].astype(float).mean(0),(len(v),1)),
              original[v]+(y[t].astype(float)-original[t]).mean(0)]
    pred=predict_features(model,h[v],p[v],h[t],p[t],y[t],center,scale)
    rows=[dict(config=None, **candidate_score(y[v],pred,controls))]
    for lr in LRS:
        for step,fitted,_ in trajectory(model,h[t],p[t],y[t],start[t],end[t],center,scale,lr,STEPS):
            pred=predict_features(fitted,h[v],p[v],h[t],p[t],y[t],center,scale)
            rows.append(dict(config=dict(scope='head',lr=lr,steps=step),
                             **candidate_score(y[v],pred,controls)))
    chosen=min(range(len(rows)),key=lambda i:(rows[i]['worst_ratio'],rows[i]['mean_mae'],i))
    return dict(selected=rows[chosen], ranking=rows, validation_rows=len(v),
                training_rows=len(t), criterion='own-inner-validation worst control ratio')
