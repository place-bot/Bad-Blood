"""Independent post-run metric, model replay, and matched-budget checks."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from personal_hpo_20261015.run import read_all,score,SEEDS,MODALITIES,METHODS,clean
from modality_se.personalize import restore
from paired_se.model import predict_bank
from personalized_forecasting.run import file_hash,canonical_hash,atomic_json


def verify(out):
    torch.set_num_threads(4)
    lock=json.loads((out/'lock.json').read_text());lh=canonical_hash(lock)
    eligible=sorted(r['patient'] for r in lock['cohort']['candidate_audit'] if r['status']=='eligible')
    expected=sorted(np.random.default_rng(20261015).choice(eligible,30,replace=False).tolist())
    assert expected==[r['patient'] for r in lock['cohort']['selected']]
    assert not set(expected)&set(lock['cohort']['excluded_population_or_development_ids'])
    complete=json.loads((out/'complete.json').read_text())
    assert complete['status']=='complete' and complete['units']==180
    assert complete['lock_hash']==lh
    assert file_hash(out/'summary.json')==complete['summary_sha256']
    for f,h in lock['spec']['files'].items():assert file_hash(ROOT/f)==h
    summary=json.loads((out/'summary.json').read_text())
    errors=[];count=0;rows=[];selection_counts={m:{} for m in MODALITIES}
    for entry in lock['cohort']['selected']:
        path=Path(entry['file']);assert file_hash(path)==entry['sha256']
        person=read_all(path);pid=person['patient'];same=None
        s,q=person['indices']['support'],person['indices']['query']
        assert person['end'][s].max()+10<=person['start'][q].min()+1e-6
        assert np.intersect1d(s,q).size==0
        for mode in MODALITIES:
            for seed in SEEDS:
                folder=out/'units'/mode/f'seed{seed}'/pid.replace(':','_')
                record=json.loads((folder/'unit.json').read_text())
                assert record['lock_hash']==lh
                for f,h in record['hashes'].items():assert file_hash(folder/f)==h
                selection=json.loads((folder/'selection.json').read_text())
                ranking=selection['ranking']
                ix=min(range(len(ranking)),key=lambda i:(ranking[i]['worst_ratio'],ranking[i]['mean_mae'],i))
                assert ranking[ix]==selection['selected']
                cfg=json.dumps(selection['selected']['config'],sort_keys=True)
                selection_counts[mode][cfg]=selection_counts[mode].get(cfg,0)+1
                with np.load(folder/'predictions.npz') as z:
                    for k,v in person['indices'].items():np.testing.assert_array_equal(z[k],v)
                    np.testing.assert_array_equal(z['y'],person['y'][q])
                    if same is None:same=z['personal_mean'].copy()
                    else:np.testing.assert_array_equal(z['personal_mean'],same)
                    expected_mean=np.tile(person['y'][s].astype(float).mean(0),(len(q),1))
                    np.testing.assert_array_equal(z['personal_mean'],expected_mean)
                    residual=z['bias']-z['population']
                    np.testing.assert_allclose(residual,np.tile(residual[0],(len(q),1)),atol=1e-10,rtol=0)
                    for method in METHODS:
                        recomputed=score(z['y'],z[method])
                        for metric,value in recomputed.items():
                            np.testing.assert_allclose(value,record['metrics'][method][metric],atol=1e-10,rtol=0)
                        listed=summary['modality_results'][mode]['methods'][method]['per_seed'][str(seed)]['per_patient'][pid]
                        for metric,value in recomputed.items():
                            np.testing.assert_allclose(value,listed[metric],atol=1e-10,rtol=0)
                    for method in ('pair_fixed','pair_personal_hpo'):
                        checkpoint=torch.load(folder/(method+'.pt'),map_location='cpu',weights_only=False)
                        model=restore(checkpoint,mode)
                        source=torch.load(ROOT/'modality_se/models'/f'{mode}_pair_population_seed{seed}.pt',map_location='cpu',weights_only=False)
                        population=torch.load(ROOT/'modality_se/models'/f'{mode}_population_seed{seed}.pt',map_location='cpu',weights_only=False)
                        for k,v in population['state_dict'].items():
                            assert torch.equal(v,source['state_dict']['encoder.'+k])
                        for k,v in source['state_dict'].items():
                            if not k.startswith('change.'):assert torch.equal(v,checkpoint['state_dict'][k])
                        # Original, noncached public inference implementation.
                        prediction=predict_bank(model,person['x'][q[:2]],person['x'][s],person['y'][s],
                            np.asarray(checkpoint['center'],np.float32),np.asarray(checkpoint['scale'],np.float32),'cpu')
                        maximum=float(np.max(np.abs(prediction-z[method][:2])))
                        assert maximum<.001,(pid,mode,seed,method,maximum)
                        errors.append(maximum)
                count+=1
        rows.append(dict(patient=pid,n_support=len(s),n_test=len(q)))
    assert count==180 and len(errors)==360
    record=dict(status='passed',units=count,model_replays=len(errors),max_replay_difference_mmHg=max(errors),
        metric_checks='All methods, people, seeds; sample SD ddof=1',
        matched_mean_and_split_checks=True,all_frozen_encoders_identical=True,
        chosen_config_counts=selection_counts,participants=rows,lock_hash=lh)
    atomic_json(out/'verification.json',clean(record))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.output),indent=2))
