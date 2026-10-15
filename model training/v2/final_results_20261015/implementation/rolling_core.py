"""Rolling personal selection and exact all-pair endpoint adaptation."""
import copy
import numpy as np
import torch
from paired_se.model import chronological_pairs

STEPS = (10, 25, 50, 100)
LRS = {'head': (.0003, .001), 'tail': (.0001, .0003)}
FOLDS = ((.5, .65), (.65, .8), (.8, 1.))


def rolling_folds(start, end):
    start, end = np.asarray(start), np.asarray(end)
    assert len(start)==len(end) and np.all(np.diff(start)>0)
    n=len(start); out=[]
    for a,b in FOLDS:
        cut,stop=int(n*a),int(n*b)
        t=np.arange(cut)
        v=np.arange(cut,stop)
        v=v[start[v]>=end[t[-1]]+10-1e-6]
        if len(v)<2:raise ValueError('Insufficient rolling validation rows')
        chronological_pairs(start[t],end[t])
        out.append(dict(train=t,valid=v))
    return out


def configure(model, scope):
    model.eval()
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith('change.') or (scope=='tail' and
            name.startswith(('encoder.body.6.','encoder.gates.6.'))))
    return model


def prefix(model, x, device='cpu'):
    """Cache only frozen layers: raw masked input through residual block 5."""
    parts=[]
    with torch.no_grad():
        for i in range(0,len(x),128):
            z=torch.as_tensor(x[i:i+128],dtype=torch.float32,device=device)
            for k in range(6):
                z=model.encoder.body[k](z)
                if str(k) in model.encoder.gates:z=model.encoder.gates[str(k)](z)
            parts.append(z.detach())
    return torch.cat(parts)


def tail(model, z):
    parts=[];ps=[]
    for first in range(0,len(z),128):
        h=z[first:first+128]
        for k in range(6,len(model.encoder.body)):
            h=model.encoder.body[k](h)
            if str(k) in model.encoder.gates:h=model.encoder.gates[str(k)](h)
        parts.append(h);ps.append(model.encoder.head(h))
    return torch.cat(parts),torch.cat(ps)


def pair_backward(model, h, p, ystd, q, r, endpoint, chunk=65536):
    """Accumulate exact full-pair loss; encoder gradients are applied once.

    Leaf features isolate the repeated pair-head graphs from the CNN graph.
    The chain rule propagates their summed gradients to the tail afterwards.
    """
    train_encoder=h.requires_grad
    hl=h.detach().requires_grad_(train_encoder)
    pl=p.detach().requires_grad_(train_encoder)
    total=0.
    for first in range(0,len(q),chunk):
        qi,ri=q[first:first+chunk],r[first:first+chunk]
        zn=(hl-model.feature_center)/model.feature_scale
        delta=pl[qi]-pl[ri]
        given=torch.cat([zn[qi]-zn[ri],zn[ri],ystd[ri],delta],-1)
        identity=torch.cat([torch.zeros_like(zn),zn,ystd,torch.zeros_like(pl)],-1)
        pred=ystd[ri]+delta+model.change(given)-model.change(identity)[ri]
        loss=(pred[:,endpoint]-ystd[qi,endpoint]).square().sum()/len(q)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adaptation loss')
        loss.backward();total+=float(loss.detach().cpu())
    if train_encoder:
        torch.autograd.backward((h,p),(hl.grad,pl.grad))
    return total


def trajectory(source, z, y, start, end, center, scale, scope, lr, endpoint,
               checkpoints=STEPS, chunk=65536):
    model=configure(copy.deepcopy(source),scope)
    device=z.device
    q,r=chronological_pairs(start,end)
    q=torch.as_tensor(q,device=device);r=torch.as_tensor(r,device=device)
    ys=torch.as_tensor((np.asarray(y,np.float32)-center)/scale,device=device)
    optim=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
    history=[]
    if scope=='head':
        with torch.no_grad():cached=tail(model,z)
    for step in range(1,max(checkpoints)+1):
        optim.zero_grad(set_to_none=True)
        h,p=cached if scope=='head' else tail(model,z)
        value=pair_backward(model,h,p,ys,q,r,endpoint,chunk)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
        optim.step();history.append(value)
        if step in checkpoints:
            yield step,model,dict(loss_history=list(history),pair_count=len(q),support_count=len(y),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def predict(model,zcurrent,zreference,reference_y,center,scale):
    """Bank average with only known reference labels; no query labels."""
    device=zcurrent.device
    yr=torch.as_tensor((np.asarray(reference_y,np.float32)-center)/scale,device=device)
    outputs=[]
    with torch.no_grad():
        hr,pr=tail(model,zreference);hc,pc=tail(model,zcurrent);n=len(hr)
        for i in range(0,len(hc),32):
            h,p=hc[i:i+32],pc[i:i+32];b=len(h)
            pred=model.from_features(h[:,None,:].expand(-1,n,-1).reshape(-1,48),
                p[:,None,:].expand(-1,n,-1).reshape(-1,2),hr.repeat(b,1),pr.repeat(b,1),yr.repeat(b,1))
            outputs.append(pred.reshape(b,n,2).mean(1).cpu().numpy()*scale+center)
    return np.concatenate(outputs)


def key(config):
    return 'source' if config is None else f"{config['scope']}_lr{config['lr']}_steps{config['steps']}"


def rank_key(row):
    c=row['config']
    return (row['mean_validation_mae'],0 if c is None else c['steps'],
            0 if c is None or c['scope']=='head' else 1,0 if c is None else c['lr'])


def choose(fold_results,endpoint):
    """All arguments contain THIS person's development-fold results only."""
    bykey={}
    for fold in fold_results:
        for row in fold['candidates']:
            k=key(row['config'])
            bykey.setdefault(k,dict(config=row['config'],fold_mae=[]))['fold_mae'].append(row['mae'])
    ranking=[]
    for k,row in bykey.items():
        assert len(row['fold_mae'])==3
        ranking.append(dict(**row,mean_validation_mae=float(np.mean(row['fold_mae']))))
    selected={scope:min([r for r in ranking if r['config'] is None or r['config']['scope']==scope],key=rank_key)
              for scope in ('head','tail')}
    selected['combined']=min(ranking,key=rank_key)
    return dict(endpoint=endpoint,ranking=ranking,selected=selected,criterion='equal-fold endpoint MAE')
