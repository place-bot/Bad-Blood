"""Dense early-support SE adaptation; development selection precedes target evaluation."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.data import load_personal_cohort
from shared.models import BPNet
from shared.metrics import summarize
from personalized_forecasting.adaptation import predict, fit_calibration, apply_calibration
from personalized_forecasting.run import (atomic_json, canonical_hash, file_hash,
    array_hash, validate_patient, DEV_IDS, TARGET_IDS, SEEDS)

CONTROLS = ('population', 'calibration_mean', 'bias')

def grid():
    return [dict(scope=scope, objective=objective, lr=lr, steps=steps)
            for objective in ('plain', 'centered')
            for scope in ('head_gate', 'last_gate', 'full')
            for steps in (20, 100) for lr in (1e-4, 1e-3)]

def name(setting):
    return f"{setting['scope']}_{setting['objective']}_lr{setting['lr']:g}_s{setting['steps']}"

def load_se(path, seed):
    """Validate original population provenance without changing the frozen runner."""
    parent = path.parent
    config = json.loads((parent/'config.json').read_text())
    manifest = json.loads((parent/'manifest.json').read_text())
    done = json.loads((parent/'completion.json').read_text())
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if done.get('status') != 'complete' or done.get('toy_test_only') is not False or config.get('toy_test_only') is not False:
        raise ValueError('Require completed real-data population fit')
    if file_hash(path) != done['best_checkpoint_sha256']:
        raise ValueError('Checkpoint checksum differs')
    for item in (saved, config):
        if item['variant'] != 'se_last' or item['seed'] != seed or item['pool'] != 1:
            raise ValueError('Expected matched-seed SE-last checkpoint')
    if saved['config'] != config or saved['config_hash'] != canonical_hash(config) or done['config_hash'] != saved['config_hash']:
        raise ValueError('Checkpoint config mismatch')
    if saved['hashes']['manifest'] != canonical_hash(manifest) or done['manifest_hash'] != canonical_hash(manifest):
        raise ValueError('Checkpoint manifest mismatch')
    forbidden = set(DEV_IDS + TARGET_IDS)
    if len(manifest['excluded_patients']) != 31 or not forbidden <= set(manifest['excluded_patients']):
        raise ValueError('Missing personal exclusions')
    if any(forbidden & set(v['patients']) for v in manifest['partitions'].values()):
        raise ValueError('Personal subject leakage')
    if saved['epoch'] != done['selected_epoch']:
        raise ValueError('Wrong checkpoint epoch')
    for k, mk in [('center','target_center'), ('scale','target_scale')]:
        if not np.array_equal(saved[k], manifest[mk]):
            raise ValueError('Target normalization mismatch')
    return saved

def dense_support(patient):
    validate_patient(patient)  # Check original frozen query manifest first.
    support = np.flatnonzero(patient['end_exclusive'] <= patient['calibration_boundary'] + 1e-6)
    if not len(support) or np.intersect1d(support, patient['query_indices']).size:
        raise ValueError('Invalid dense support')
    return support

def fit(model, support_x, support_y, center, scale, setting, device='cpu'):
    """Only early support is accepted. No query inputs/labels in this API."""
    x, y = np.asarray(support_x, np.float32), np.asarray(support_y, np.float32)
    if x.shape != (len(y), 2, 1250) or y.shape != (len(y), 2) or not len(y):
        raise ValueError('Invalid support shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite support')
    if setting['scope'] not in ('head_gate','last_gate','full') or setting['objective'] not in ('plain','centered'):
        raise ValueError('Unknown adaptation configuration')
    if setting['lr'] <= 0 or setting['steps'] <= 0:
        raise ValueError('Invalid optimizer setting')
    adapted = copy.deepcopy(model).to(device).eval()
    before = {k:v.detach().cpu().clone() for k,v in adapted.state_dict().items()}
    trainable = []
    for k,p in adapted.named_parameters():
        enabled = (setting['scope']=='full' or k.startswith(('head.', 'gates.'))
                   or (setting['scope']=='last_gate' and k.startswith('body.6.')))
        p.requires_grad_(enabled)
        if enabled: trainable.append(k)
    xt = torch.as_tensor(x, device=device)
    c, s = np.asarray(center,np.float32), np.asarray(scale,np.float32)
    if c.shape != (2,) or s.shape != (2,) or not np.isfinite(c).all() or not np.isfinite(s).all() or np.any(s<=0):
        raise ValueError('Invalid population target normalization')
    yt = torch.as_tensor((y-c)/s, device=device)
    opt = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=setting['lr'], weight_decay=.001)
    loss_history=[]
    started=time.perf_counter()
    for _ in range(setting['steps']):
        opt.zero_grad(set_to_none=True)
        out=adapted(xt)
        if setting['objective']=='centered':
            out=out-out.mean(0)+yt.mean(0)
        loss=torch.nn.functional.mse_loss(out,yt)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite adaptation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in adapted.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
        opt.step()
        loss_history.append(float(loss.detach().cpu()))
    with torch.no_grad():
        raw_support=adapted(xt)
        offset=(yt.mean(0)-raw_support.mean(0)).cpu().numpy() if setting['objective']=='centered' else np.zeros(2,np.float32)
        final_loss=float(torch.nn.functional.mse_loss(raw_support+torch.as_tensor(offset,device=device),yt).cpu())
    changed=[]
    for k,v in adapted.state_dict().items():
        if not torch.equal(v.cpu(),before[k]):
            if k not in trainable: raise AssertionError('Frozen parameter or buffer changed: '+k)
            changed.append(k)
    audit=dict(setting=setting, support_count=len(y), full_support_used_every_step=True,
        trainable_names=trainable, trainable_parameters=sum(p.numel() for p in adapted.parameters() if p.requires_grad),
        changed_names=changed, batchnorm_buffers_frozen=True, dropout_active=False,
        loss_history=loss_history, final_support_mse=final_loss,
        update_seconds=time.perf_counter()-started, query_used=False)
    return adapted, offset, audit

def score(records, key):
    rows=[r for r in records if r['method']==key]
    expected={(p,s) for p in DEV_IDS for s in SEEDS}
    if {(r['patient'],r['seed']) for r in rows} != expected or len(rows)!=len(expected):
        raise ValueError('Incomplete development grid: '+key)
    return np.mean([r['metrics']['macro']['mae'] for r in rows],axis=0)

def select(records):
    means={m:score(records,m) for m in ('calibration_mean','bias')}
    ranking=[]
    for setting in grid():
        mae=score(records,name(setting))
        ratios={m:(mae/means[m]).tolist() for m in means}
        ranking.append(dict(setting=setting,method=name(setting),mae=mae.tolist(),ratios=ratios,
            worst_ratio=float(np.max(list(ratios.values()))),mean_mae=float(mae.mean())))
    best=min(ranking,key=lambda r:(r['worst_ratio'],r['mean_mae'],grid().index(r['setting'])))
    return dict(selected=best,ranking=ranking,controls={k:v.tolist() for k,v in means.items()},
        success=best['worst_ratio']<1,selection_scope='four development patients, three seeds only')

def summarize_phase(records):
    result={}
    for method in sorted({r['method'] for r in records}):
        rows=[r for r in records if r['method']==method]
        seeds=sorted({r['seed'] for r in rows})
        per_seed=[]
        for seed in seeds:
            match=[r for r in rows if r['seed']==seed]
            ys=[];preds=[];ids=[]
            for r in match:
                with np.load(r['_prediction_path']) as a:
                    ys.append(a['y']); preds.append(a['pred']); ids.extend([r['patient']]*len(a['y']))
            per_seed.append(summarize(np.concatenate(ys),np.concatenate(preds),ids))
        combined={}
        for scope in ('macro','pooled'):
            combined[scope]={}
            for metric in per_seed[0][scope]:
                vals=np.array([[np.nan if v is None else v for v in z[scope][metric]] for z in per_seed])
                count=np.isfinite(vals).sum(0)
                avg=np.divide(np.nansum(vals,0),count,out=np.full(2,np.nan),where=count>0)
                combined[scope][metric]=[float(v) if np.isfinite(v) else None for v in avg]
        combined['per_patient_mae']={p:np.mean([r['metrics']['macro']['mae'] for r in rows if r['patient']==p],0).tolist()
                                          for p in sorted({r['patient'] for r in rows})}
        combined['seeds']=seeds
        combined['n_patients']=len(combined['per_patient_mae'])
        combined['n_query_windows']=per_seed[0]['n_windows']
        result[method]=combined
    return result

def run_unit(patient, support, seed, base, saved, setting, method, directory, manifest_hash,
             selection_hash, sp, qp, device):
    directory.mkdir(parents=True,exist_ok=True)
    config=dict(patient=patient['patient_id'],seed=seed,method=method,setting=setting,
        manifest_hash=manifest_hash,selection_hash=selection_hash,
        support_hash=array_hash(support),query_hash=array_hash(patient['query_indices']))
    marker=directory/'unit.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        if record['config']!=config: raise ValueError('Changed unit config')
        for n,h in record['hashes'].items():
            if file_hash(directory/n)!=h: raise ValueError('Corrupt completed artifact')
        record['_prediction_path']=str(directory/'predictions.npz')
        return record
    q=patient['query_indices']; sx=patient['x'][support]; sy=patient['y'][support]
    center,scale=np.asarray(saved['center']),np.asarray(saved['scale'])
    if setting is None:
        state=fit_calibration(method,sp,sy,center,scale)
        prediction=apply_calibration(state,qp)
        atomic_json(directory/'calibration.json',state)
        artifact='calibration.json'; audit={'support_count':len(support),'query_used':False}
    else:
        adapted,offset,audit=fit(base,sx,sy,center,scale,setting,device)
        prediction,_=predict(adapted,patient['x'][q],center,scale,device)
        prediction=prediction+offset*scale
        artifact='adapted.pt'
        torch.save(dict(state_dict={k:v.cpu() for k,v in adapted.state_dict().items()},
            variant='se_last',center=center,scale=scale,offset_standardized=offset,config=config),directory/artifact)
    if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite query predictions')
    np.savez_compressed(directory/'predictions.npz',y=patient['y'][q],pred=prediction,
        support_indices=support,query_indices=q,support_y=sy,support_population_pred=sp,
        start=patient['start'][q],end_exclusive=patient['end_exclusive'][q])
    record=dict(config=config,patient=patient['patient_id'],seed=seed,method=method,
        n_support=len(support),metrics=summarize(patient['y'][q],prediction,np.repeat(patient['patient_id'],len(q))),
        audit=audit,hashes={n:file_hash(directory/n) for n in (artifact,'predictions.npz')},
        completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(marker,record)
    record['_prediction_path']=str(directory/'predictions.npz')
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('development','test'),required=True)
    parser.add_argument('--population-root',type=Path,required=True)
    parser.add_argument('--individual-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True,warn_only=True)
    cohort=load_personal_cohort(args.individual_root)
    supports={p['patient_id']:dense_support(p) for role in ('development','target') for p in cohort[role]}
    paths={seed:args.population_root/f'se_last_seed{seed}'/'best.pt' for seed in SEEDS}
    saved={s:load_se(p,s) for s,p in paths.items()}
    source=[Path(__file__),Path(__file__).with_name('PROTOCOL.md'),ROOT/'shared/data.py',ROOT/'shared/models.py',
            ROOT/'shared/metrics.py',ROOT/'personalized_forecasting/adaptation.py',ROOT/'personalized_forecasting/run.py']
    manifest=dict(protocol='se-dense-support-20261012',device=args.device,
        source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in source},
        checkpoint_hashes={str(s):file_hash(p) for s,p in paths.items()},
        source_manifest=cohort['manifest'],grid=grid(),torch_version=str(torch.__version__),
        patients={p['patient_id']:dict(role=role,n_support=len(supports[p['patient_id']]),
            support_source_indices=p['source_indices'][supports[p['patient_id']]].tolist(),
            query_source_indices=p['source_indices'][p['query_indices']].tolist(),
            boundary=float(p['calibration_boundary']),array_hash=array_hash(p['x'],p['y'],p['start']))
            for role in ('development','target') for p in cohort[role]})
    args.output.mkdir(parents=True,exist_ok=True)
    mp=args.output/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=manifest: raise ValueError('Changed run provenance; use new output')
    else:
        if args.phase=='test': raise ValueError('Development must precede test')
        atomic_json(mp,manifest)
    manifest_hash=canonical_hash(manifest)
    selection=None
    if args.phase=='test':
        selection=json.loads((args.output/'selection.json').read_text())
        unhashed={k:v for k,v in selection.items() if k!='selection_hash'}
        if selection['selection_hash']!=canonical_hash(unhashed) or selection['manifest_hash']!=manifest_hash:
            raise ValueError('Selection provenance mismatch')
    records=[]
    for seed in SEEDS:
        base=BPNet(variant='se_last'); base.load_state_dict(saved[seed]['state_dict'],strict=True);base.eval()
        for patient in cohort['development' if args.phase=='development' else 'target']:
            support=supports[patient['patient_id']]
            sp,_=predict(base,patient['x'][support],saved[seed]['center'],saved[seed]['scale'],args.device)
            qp,_=predict(base,patient['x'][patient['query_indices']],saved[seed]['center'],saved[seed]['scale'],args.device)
            candidates=[(m,None) for m in CONTROLS]
            configs=grid() if selection is None else [selection['selected']['setting']]
            candidates += [(name(s),s) for s in configs]
            for method,setting in candidates:
                directory=args.output/args.phase/patient['patient_id']/f'seed{seed}'/method
                record=run_unit(patient,support,seed,base,saved[seed],setting,method,directory,
                    manifest_hash,selection['selection_hash'] if selection else None,sp,qp,args.device)
                records.append(record)
                print(json.dumps(dict(phase=args.phase,patient=patient['patient_id'],seed=seed,method=method,
                    n_support=len(support),mae=record['metrics']['macro']['mae'],completed=len(records))),flush=True)
    if args.phase=='development':
        selection=select(records);selection['manifest_hash']=manifest_hash
        selection['selection_hash']=canonical_hash(selection)
        sf=args.output/'selection.json'
        if sf.exists() and json.loads(sf.read_text())!=selection: raise ValueError('Cannot change frozen selection')
        atomic_json(sf,selection)
    summary=summarize_phase(records)
    result=dict(complete=True,phase=args.phase,n_units=len(records),manifest_hash=manifest_hash,
        selection_hash=selection['selection_hash'],selected_method=selection['selected']['method'],methods=summary)
    if args.phase=='test':
        selected=summary[result['selected_method']]
        selected_mae=np.array(selected['macro']['mae'])
        result['success_both_pressures_both_controls']=all(np.all(selected_mae < summary[m]['macro']['mae']) for m in ('bias','calibration_mean'))
        rng=np.random.default_rng(20261012)
        result['paired_differences']={}
        for m in ('bias','calibration_mean'):
            delta=np.array([np.array(selected['per_patient_mae'][p])-summary[m]['per_patient_mae'][p] for p in sorted(selected['per_patient_mae'])])
            boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            result['paired_differences'][m]=dict(estimate=delta.mean(0).tolist(),
                ci95=np.quantile(boot,[.025,.975],axis=0).tolist(),
                improved_patients=(delta<0).sum(0).tolist(),n_patients=len(delta),
                note='Exploratory patient bootstrap after averaging seeds; model minus control')
    atomic_json(args.output/f'{args.phase}_summary.json',result)
    print(json.dumps({'phase':args.phase,'done':True,'selected':result['selected_method'],
        'success':result.get('success_both_pressures_both_controls',selection['success'])}),flush=True)

if __name__=='__main__':
    main()
