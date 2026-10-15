"""One development-selected residual-calibration experiment; no target retuning."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from residual_calibration.model import candidates, fit, predict, forward_cv, predict_endpoint_states
from modality_se.model import ModalitySE
from modality_se.personalize import normalize_person
from modality_se.report import differences
from paired_se.model import PairSE, predict_bank
from paired_se.fresh_20261012.validate import read_person
from shared.data import load_personal_cohort
from shared.metrics import summarize
from se_personalization.run import summarize_phase
from personalized_forecasting.run import atomic_json, file_hash, canonical_hash, array_hash

SEEDS = (42, 43, 44)


def encode(model, x, center, scale):
    hs = []; ps = []
    with torch.inference_mode():
        for i in range(0, len(x), 256):
            h, p = model.encode(torch.as_tensor(x[i:i+256], device='mps'))
            hs.append(h.cpu().numpy()); ps.append(p.cpu().numpy())
    return np.concatenate(hs), np.concatenate(ps)*scale+center


def save_record(folder, v, seed, method, pred, config, audit, states=None):
    folder.mkdir(parents=True, exist_ok=True)
    q, s = v['query'], v['support']
    np.savez_compressed(folder/'predictions.npz', y=v['y'][q], pred=pred, support_indices=s, query_indices=q)
    files = ['predictions.npz']
    if states is not None:
        torch.save(states, folder/'model.pt'); files.append('model.pt')
    row = dict(patient=v['patient'], seed=seed, method=method, config=config,
        n_support=len(s), audit=audit, metrics=summarize(v['y'][q], pred, [v['patient']]*len(q)),
        hashes={f:file_hash(folder/f) for f in files}, _prediction_path=str(folder/'predictions.npz'))
    atomic_json(folder/'unit.json', row)
    return row


def predict_saved(saved, x):
    """Replay a self-contained, endpoint-selected pipeline on normalized windows."""
    base = PairSE(ModalitySE(saved['modality'])).to('mps').eval()
    base.encoder.load_state_dict(saved['encoder'])
    h, p = encode(base, x, saved['center'], saved['scale'])
    h = (h-saved['feature_center'])/saved['feature_scale']
    results = []
    for j, method in enumerate(saved['selected']):
        if method == 'prior_pair':
            prior = PairSE(ModalitySE(saved['modality'])).to('mps').eval()
            prior.load_state_dict(saved['prior']['state_dict'])
            # Only encoded support features and known labels, no raw bank needed.
            with torch.inference_mode():
                hc, pc = prior.encode(torch.as_tensor(x, device='mps'))
                hr, pr, yr = [torch.as_tensor(saved['prior'][k], dtype=torch.float32, device='mps') for k in ('hr', 'pr', 'yr')]
                out = []
                for first in range(0, len(x), 32):
                    ch, cp = hc[first:first+32], pc[first:first+32]; b, n = len(ch), len(hr)
                    y = prior.from_features(ch[:,None,:].expand(-1,n,-1).reshape(-1,48),
                        cp[:,None,:].expand(-1,n,-1).reshape(-1,2), hr.repeat(b,1), pr.repeat(b,1), yr.repeat(b,1))
                    out.append(y.reshape(b,n,2).mean(1).cpu().numpy()*np.asarray(saved['scale'],np.float32)+np.asarray(saved['center'],np.float32))
            results.append(np.concatenate(out)[:,j])
        else:
            results.append(predict(saved['states'][j], h, p)[:,j])
    return np.column_stack(results)


def phase_run(people, phase, locks, args, manifest_hash):
    rows = {'dual':[], 'ppg':[]}; choices = {c['name']:c for c in candidates()}
    prior_choices = json.loads((args.prior/'personalization_complete.json').read_text())['selections']
    for modality in ('dual', 'ppg'):
        for seed in SEEDS:
            saved = torch.load(ROOT/f'modality_se/models/{modality}_population_seed{seed}.pt', map_location='cpu', weights_only=False)
            norm = torch.load(ROOT/f'modality_se/models/{modality}_pair_population_seed{seed}.pt', map_location='cpu', weights_only=False)
            c, scale = np.asarray(saved['center']), np.asarray(saved['scale'])
            fc, fs = [norm['state_dict'][k].numpy() for k in ('feature_center','feature_scale')]
            base = PairSE(ModalitySE(modality)).to('mps').eval(); base.encoder.load_state_dict(saved['state_dict'])
            for v in people:
                s, q = v['support'], v['query']
                assert len(set(s)&set(q)) == 0 and max(v['end'][s])+10 <= min(v['start'][q])+1e-6
                h, p = encode(base, v['x'], c, scale); h = (h-fc)/fs
                hs, ps, ys = h[s], p[s], v['y'][s]
                destination = args.output/modality/phase/v['patient']/f'seed{seed}'
                prior_method = prior_choices[modality]['selected']['method']
                prior_folder = args.prior/modality/phase/v['patient']/f'seed{seed}'/prior_method
                with np.load(prior_folder/'predictions.npz') as f:
                    np.testing.assert_array_equal(f['support_indices'], s); np.testing.assert_array_equal(f['query_indices'], q)
                    prior_pred = f['pred'].copy()
                predictions = dict(population=p[q], prior_pair=prior_pred); states = {}; audits = {}
                wanted = list(choices) + ['forward_cv'] if phase == 'development' else list(dict.fromkeys(['bias','calibration_mean']+locks[modality]['selected']))
                for method in wanted:
                    if method == 'prior_pair': continue
                    if method == 'forward_cv':
                        st, audit = forward_cv(hs, ps, ys, v['start'][s], v['end'][s], c, scale)
                        predictions[method] = predict_endpoint_states(st, h[q], p[q]); states[method] = st; audits[method] = audit
                    else:
                        st = fit(choices[method], hs, ps, ys, c, scale)
                        predictions[method] = predict(st, h[q], p[q]); states[method] = st
                artifact = None
                if phase != 'development':
                    selected = locks[modality]['selected']
                    predictions['selected'] = np.column_stack([predictions[m][:,j] for j,m in enumerate(selected)])
                    endpoint_states = [None if m=='prior_pair' else states[m][j] if m=='forward_cv' else states[m] for j,m in enumerate(selected)]
                    artifact = dict(modality=modality, encoder=saved['state_dict'], center=c, scale=scale,
                        feature_center=fc, feature_scale=fs, selected=selected, states=endpoint_states,
                        support_indices=s, query_indices=q, manifest_hash=manifest_hash, seed=seed, patient=v['patient'])
                    if 'prior_pair' in selected:
                        prior_saved = torch.load(prior_folder/'model.pt', map_location='cpu', weights_only=False)
                        prior_model = PairSE(ModalitySE(modality)).to('mps').eval(); prior_model.load_state_dict(prior_saved['state_dict'])
                        with torch.inference_mode():
                            hr, pr = prior_model.encode(torch.as_tensor(v['x'][s], device='mps'))
                        artifact['prior'] = dict(state_dict=prior_saved['state_dict'], hr=hr.cpu().numpy(), pr=pr.cpu().numpy(), yr=(ys-c)/scale)
                    keep = ('population','calibration_mean','bias','prior_pair','selected')
                else: keep = tuple(predictions)
                for method in keep:
                    config = dict(manifest_hash=manifest_hash, modality=modality, phase=phase,
                        support_hash=array_hash(s), query_hash=array_hash(q),
                        selection=locks[modality] if locks else None)
                    row = save_record(destination/method, v, seed, method, predictions[method], config,
                        audits.get(method, dict(query_labels_used=False)), artifact if method=='selected' else None)
                    rows[modality].append(row)
                print('DONE',phase,modality,seed,v['patient'],flush=True)
            del base; torch.mps.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ('output','prior','personal','fresh'): parser.add_argument('--'+name,type=Path,required=True)
    a = parser.parse_args(); a.output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4)
    if (a.output/'complete.json').exists():
        print((a.output/'complete.json').read_text()); return
    manifest = dict(protocol_sha256=file_hash(ROOT/'residual_calibration/PROTOCOL.md'),
        source_hashes={n:file_hash(ROOT/n) for n in ('residual_calibration/model.py','residual_calibration/run.py')},
        prior_manifest_sha256=file_hash(a.prior/'personal_manifest.json'),
        weights_manifest=json.loads((ROOT/'modality_se/models/manifest.json').read_text()), candidates=candidates(), seeds=list(SEEDS))
    for name, expected in manifest['weights_manifest'].items(): assert file_hash(ROOT/'modality_se/models'/name)==expected
    mh = canonical_hash(manifest); atomic_json(a.output/'manifest.json',manifest)
    audit = json.loads((ROOT/'audit_snapshot.json').read_text()); proto=audit['final_personalization_protocol']
    proto['patients']=[v for v in proto['patients'] if v['role']=='development']; proto['target_subjects']=[]
    atomic_json(a.output/'development_only.json',audit)
    dev = [normalize_person(v) for v in load_personal_cohort(a.personal,a.output/'development_only.json')['development']]
    development = phase_run(dev,'development',None,a,mh); locks={}
    development_summary = {}
    for modality in ('dual','ppg'):
        summary=summarize_phase(development[modality]); development_summary[modality]=summary
        names=[c['name'] for c in candidates()]+['forward_cv','prior_pair']
        selected=[min(names,key=lambda m:summary[m]['macro']['mae'][j]) for j in (0,1)]
        locks[modality]=dict(selected=selected, seed_averaged_development_mae={m:summary[m]['macro']['mae'] for m in names},
                             query_targets_used=False)
    if (a.output/'selection.json').exists():
        assert json.loads((a.output/'selection.json').read_text())==locks, 'Development selection changed on replay'
    atomic_json(a.output/'selection.json',locks)
    print('LOCKED',json.dumps(locks),flush=True)
    # Only after both locks are on disk do we load target files.
    audit=json.loads((ROOT/'audit_snapshot.json').read_text()); proto=audit['final_personalization_protocol']
    proto['patients']=[v for v in proto['patients'] if v['role']=='target']; proto['development_subjects']=[]
    atomic_json(a.output/'target_only.json',audit)
    historical=[normalize_person(v) for v in load_personal_cohort(a.personal,a.output/'target_only.json')['target']]
    recent=[]
    for r in json.loads((ROOT/'paired_se/fresh_20261012/results/cohort_audit.json').read_text()):
        if r['status']!='included': continue
        path=a.fresh/(r['patient'].split(':')[1]+'.mat'); assert file_hash(path)==r['file_sha256']
        v=read_person(path,r['patient']); np.testing.assert_array_equal(v['support'],r['support_indices']); np.testing.assert_array_equal(v['query'],r['query_indices']); recent.append(v)
    assert len(historical)==6 and len(recent)==16
    result=dict(manifest=manifest,selections=locks,development=development_summary,targets={})
    units=sum(len(v) for v in development.values()); replays=0; maxerr=0.
    for phase,people in [('historical_six',historical),('previous_sixteen',recent)]:
        records=phase_run(people,phase,locks,a,mh); result['targets'][phase]={}
        lookup={v['patient']:v for v in people}
        for modality, rows in records.items():
            units+=len(rows); summary=summarize_phase(rows)
            paired={control:differences(summary['selected']['per_patient_mae'],summary[control]['per_patient_mae']) for control in ('calibration_mean','bias','prior_pair')}
            per_seed={}
            for seed in SEEDS: per_seed[str(seed)]=summarize_phase([r for r in rows if r['seed']==seed])
            result['targets'][phase][modality]=dict(methods=summary,paired=paired,per_seed=per_seed)
            for row in rows:
                path=Path(row['_prediction_path']); v=lookup[row['patient']]
                with np.load(path) as f:
                    np.testing.assert_array_equal(f['support_indices'],v['support']); np.testing.assert_array_equal(f['query_indices'],v['query'])
                    metric=summarize(f['y'],f['pred'],[row['patient']]*len(f['y']))
                    assert metric==row['metrics']
                    if row['method']=='selected':
                        stored=torch.load(path.parent/'model.pt',map_location='cpu',weights_only=False)
                        replay=predict_saved(stored,v['x'][v['query']]); err=float(abs(replay-f['pred']).max()); maxerr=max(maxerr,err)
                        np.testing.assert_allclose(replay,f['pred'],atol=.001,rtol=0); replays+=len(replay)
    result['verification']=dict(units=units,replayed_predictions=replays,max_replay_error=maxerr,
                               fixed_support_query=True,independent_new_test=False)
    atomic_json(a.output/'comparison.json',result)
    atomic_json(a.output/'complete.json',dict(status='verified_complete',units=units,result=str(a.output/'comparison.json')))
    print('COMPLETE',json.dumps(result['verification']),flush=True)


if __name__=='__main__': main()
