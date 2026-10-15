"""Independent artifact/metric replay checks, without model selection changes."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rolling_personal_20261015.core import choose,rolling_folds
from rolling_personal_20261015.report import METHODS,METRICS
from personal_hpo_20261015.run import read_all,score,clean
from modality_se.personalize import restore
from paired_se.model import predict_bank
from personalized_forecasting.run import atomic_json,file_hash,canonical_hash


def verify(out):
    torch.set_num_threads(4)
    lock=json.loads((out/'lock.json').read_text());lh=canonical_hash(lock)
    trained=json.loads((out/'training_complete.json').read_text())
    assert trained['lock_hash']==lh and len(trained['records'])==180
    for k,h in lock['spec']['code'].items():assert file_hash(ROOT/k)==h
    summary=json.loads((out/'summary.json').read_text())
    replay=[];units=0;folds=0;scope_counts={};endpoint_records=[]
    for entry in lock['people']:
        assert file_hash(Path(entry['file']))==entry['sha256']
        person=read_all(Path(entry['file']));s,q=person['indices']['support'],person['indices']['query']
        assert clean(rolling_folds(person['start'][s],person['end'][s]))==entry['folds']
        assert person['end'][s].max()+10<=person['start'][q].min()+1e-6
        for mode in ('ppg','dual'):
            for seed in (42,43,44):
                folder=out/'units'/mode/f'seed{seed}'/entry['patient'].replace(':','_')
                training=json.loads((folder/'trained.json').read_text());assert training['lock_hash']==lh
                for k,h in training['hashes'].items():assert file_hash(folder/k)==h
                sel=json.loads((folder/'selection.json').read_text())
                assert sel['selected_utc']<=training['completed_utc']<=trained['completed_utc']
                for endpoint in (0,1):
                    records=[]
                    for fi in range(3):
                        record=json.loads((folder/f'fold{fi}_endpoint{endpoint}.json').read_text())
                        assert record['lock_hash']==lh and file_hash(folder/record['predictions_file'])==record['predictions_sha256']
                        t,v=entry['folds'][fi]['train'],entry['folds'][fi]['valid']
                        assert t==record['train'] and v==record['valid']
                        with np.load(folder/record['predictions_file']) as z:
                            np.testing.assert_array_equal(z['y'],person['y'][s[v],endpoint])
                            for candidate in record['candidates']:
                                c=candidate['config'];name='source' if c is None else f"{c['scope']}_lr{c['lr']}_steps{c['steps']}"
                                mae=float(np.abs(z[name].astype(float)-z['y']).mean())
                                assert abs(mae-candidate['mae'])<1e-9
                        records.append(record);folds+=1
                    assert choose(records,endpoint)==sel['endpoint_selections'][endpoint]
                    config=sel['endpoint_selections'][endpoint]['selected']['combined']['config']
                    key=f'{mode}_{endpoint}_'+('source' if config is None else config['scope'])
                    scope_counts[key]=scope_counts.get(key,0)+1
                    endpoint_records.append(dict(person=entry['patient'],mode=mode,seed=seed,endpoint=endpoint,selected=config))
                evaluation=json.loads((folder/'evaluation.json').read_text())
                assert evaluation['evaluated_utc']>=trained['completed_utc']
                assert file_hash(folder/'predictions.npz')==evaluation['predictions_sha256']
                with np.load(folder/'predictions.npz') as z:
                    np.testing.assert_array_equal(z['support'],s);np.testing.assert_array_equal(z['query'],q)
                    np.testing.assert_array_equal(z['y'],person['y'][q])
                    expected=np.tile(person['y'][s].astype(float).mean(0),(len(q),1))
                    np.testing.assert_array_equal(z['personal_mean'],expected)
                    for method in METHODS:
                        metrics=score(z['y'],z[method])
                        listed=summary['modality_results'][mode]['methods'][method]['per_seed'][str(seed)]['per_patient'][entry['patient']]
                        for metric in METRICS:
                            np.testing.assert_allclose(metrics[metric],evaluation['metrics'][method][metric],rtol=0,atol=1e-10)
                            np.testing.assert_allclose(metrics[metric],listed[metric],rtol=0,atol=1e-10)
                    source=torch.load(ROOT/'modality_se/models'/f'{mode}_pair_population_seed{seed}.pt',map_location='cpu',weights_only=False)
                    for scope in ('head','tail'):
                        for endpoint in (0,1):
                            saved=torch.load(folder/f'{scope}_endpoint{endpoint}.pt',map_location='cpu',weights_only=False)
                            assert saved['config']==sel['endpoint_selections'][endpoint]['selected'][scope]['config']
                            allowed=('change.',) if scope=='head' else ('change.','encoder.body.6.','encoder.gates.6.')
                            for k,v in source['state_dict'].items():
                                if not k.startswith(allowed) or 'running_' in k or 'num_batches_tracked' in k:
                                    assert torch.equal(v,saved['state_dict'][k]),k
                            model=restore(saved,mode)
                            pred=predict_bank(model,person['x'][q[:2]],person['x'][s],person['y'][s],saved['center'],saved['scale'])[:,endpoint]
                            error=float(np.max(np.abs(pred-z['rolling_'+scope][:2,endpoint])))
                            assert error<.001,(entry['patient'],mode,seed,scope,endpoint,error)
                            replay.append(error)
                    for endpoint in (0,1):
                        c=sel['endpoint_selections'][endpoint]['selected']['combined']['config']
                        scope='head' if c is None else c['scope']
                        np.testing.assert_array_equal(z['rolling_selected'][:,endpoint],z['rolling_'+scope][:,endpoint])
                units+=1
    assert units==180 and folds==1080 and len(replay)==720
    record=dict(status='passed',lock_hash=lh,units=units,validation_folds=folds,model_replays=len(replay),
        max_replay_difference_mmHg=max(replay),selected_scopes=scope_counts,endpoint_selections=endpoint_records,
        full_budget_matched=True,metric_checks='all methods/people/seeds; sample SD ddof=1',
        final_test_after_all_choices_and_weights=True)
    atomic_json(out/'verification.json',clean(record))
    atomic_json(out/'complete.json',dict(status='complete',lock_hash=lh,units=units,
        summary_sha256=file_hash(out/'summary.json'),verification_sha256=file_hash(out/'verification.json')))
    print(json.dumps({k:v for k,v in record.items() if k!='endpoint_selections'},indent=2))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    verify(p.parse_args().output)
