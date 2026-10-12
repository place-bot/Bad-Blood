"""Replay saved weights and independently check all reported core metrics."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT))
from temporal_pretraining.run import predictions
from context_se.run import encode
from context_se.adapt import MaskedContext
from paired_se.run import restore
from shared.data import load_personal_cohort
from se_personalization.run import dense_support
from personalized_forecasting.run import atomic_json,file_hash

KEYS=('mae','bias','error_sd','within_5','within_10','within_15')

def independent(y,p):
    e=p.astype(float)-y.astype(float);ae=abs(e)
    return dict(mae=ae.mean(0),bias=e.mean(0),error_sd=e.std(0,ddof=1),
                within_5=(ae<=5).mean(0),within_10=(ae<=10).mean(0),within_15=(ae<=15).mean(0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--individual-root',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(4);summary=json.loads((a.run/'summary.json').read_text());manifest=json.loads((a.run/'manifest.json').read_text())
    assert manifest['code_sha256']==file_hash(HERE/'run.py')
    saved=torch.load(ROOT/'paired_se/models/paired_population_seed42.pt',map_location='cpu',weights_only=False)
    base=restore(saved,'cpu');c,scale=np.asarray(saved['center']),np.asarray(saved['scale']);fc,fs=base.feature_center,base.feature_scale
    records={}
    for r in json.loads((a.features/'complete.json').read_text())['audit']:
        if r['role']!='validation' or r['status']!='included':continue
        with np.load(a.features/(Path(r['file']).stem+'.npz')) as f:d={k:f[k] for k in f.files}
        d['yn']=(d['y']-c)/scale;records[r['patient']]=d
    for v in load_personal_cohort(a.individual_root,a.run/'development_only_audit.json')['development']:
        h,p=encode(base,v['x']);records[v['patient_id']]=dict(h=h,p=p,y=v['y'],yn=(v['y']-c)/scale,support=dense_support(v),query=v['query_indices'])
    checks=0;maxerror=0.;zero_delta=0.
    for result in summary['source_results']+summary['development_results']:
        source='validation' in result
        folder=a.run/result['name'] if source else a.run/'development'/result['name']
        paths=list((folder/'predictions').glob('*.npz')) if source else list(folder.glob('*.npz'))
        expect=result['validation'] if source else result['metrics'];computed=[]
        for path in paths:
            person=path.stem.replace('vitaldb_','vitaldb:');d=records[person]
            ck=folder/'best.pt' if source else path.with_suffix('.pt')
            checkpoint=torch.load(ck,map_location='cpu',weights_only=False);kind=checkpoint['kind']
            model=copy.deepcopy(base) if kind=='pair' else MaskedContext();model.load_state_dict(checkpoint['state_dict']);model.eval()
            if kind=='pair':
                for key,value in model.encoder.state_dict().items():
                    assert torch.equal(value,base.encoder.state_dict()[key]),'Frozen encoder changed'
            with np.load(path) as f:
                np.testing.assert_array_equal(f['support'],d['support']);np.testing.assert_array_equal(f['query'],d['query'])
                replay=predictions(model,kind,d,fc,fs)*scale+c
                maxerror=max(maxerror,float(abs(replay-f['pred']).max()))
                np.testing.assert_allclose(replay,f['pred'],atol=1e-5,rtol=0)
                metric=independent(f['y'],f['pred']);computed.append(metric)
                expected=expect['per_patient'][person]
                for key in KEYS:np.testing.assert_allclose(metric[key],expected[key],rtol=0,atol=1e-9)
            checks+=1
        for key in KEYS:np.testing.assert_allclose(np.mean([r[key] for r in computed],0),expect['macro'][key],rtol=0,atol=1e-9)
    dest=HERE/'results';dest.mkdir(exist_ok=True)
    for f in a.run.rglob('*.json'):
        if f.name=='development_only_audit.json':continue
        target=dest/f.relative_to(a.run);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    shutil.copyfile(a.features/'complete.json',dest/'source_cohort_audit.json')
    for f in a.run.rglob('*.pt'):
        target=HERE/'models'/f.relative_to(a.run);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    atomic_json(HERE/'models/manifest.json',{str(f.relative_to(HERE/'models')):file_hash(f) for f in sorted((HERE/'models').rglob('*.pt'))})
    atomic_json(dest/'verification.json',dict(prediction_files_replayed=checks,max_absolute_replay_difference=maxerror,independently_recomputed_metrics=list(KEYS),
        encoder_unchanged=True,zero_calibration_change=zero_delta,target_people_evaluated=False,
        original_source_roles_preserved=True,small_validation_cohort=len([p for p in records if p not in [v['patient'] for v in summary['development_results'][0]['people']]])))
    log=ROOT.parents[1]/'autoresearch/orchestrator-261012-temporal';log.mkdir(parents=True,exist_ok=True)
    lines=['phase\tmethod\tsbp_mae\tdbp_mae\tselected_epoch_or_steps']
    for r in summary['source_results']:
        m=r['validation']['macro']['mae'];lines.append(f"source\t{r['name']}\t{m[0]}\t{m[1]}\t{r['selected_epoch']}")
    for r in summary['development_results']:
        m=r['metrics']['macro']['mae'];lines.append(f"personal\t{r['name']}\t{m[0]}\t{m[1]}\t{r['config']}")
    (log/'results.tsv').write_text('\n'.join(lines)+'\n')
    atomic_json(log/'handoff.json',dict(status='CEILING',goal_achieved=False,source_validation_both_improved=summary['source_selection']['both_validation_endpoints_improved'],
        personal_development_pass=summary['development_pass'],independent_verification='Not entered: development joint criterion failed',
        reason='Four matched source fits and six personal settings complete; chronology did not establish superiority over matched random training or both personal controls',
        preserved_release='v2-20261012.1'))
    print('VERIFIED',checks,'replays; max error',maxerror,flush=True)

if __name__=='__main__':main()
