"""Verify published artifacts and replay portable inference on real held-out data."""
import argparse
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
V2=HERE.parent
sys.path.insert(0,str(V2))
sys.path.insert(0,str(HERE/'implementation'))
from personalize import load, infer, inputs
from personal_hpo_20261015.run import read_all


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run):
    torch.set_num_threads(4)
    complete=json.loads((HERE/'metrics/complete.json').read_text())
    for key,name in [('summary_sha256','summary.json'),('verification_sha256','verification.json')]:
        assert complete[key]==sha(HERE/'metrics'/name)
    verification=json.loads((HERE/'metrics/verification.json').read_text())
    assert verification['status']=='passed'
    lock=json.loads((HERE/'metrics/lock.json').read_text())
    for rel,digest in lock['spec']['code'].items():assert sha(V2/rel)==digest,rel
    selections=json.loads((HERE/'metrics/tail_selections.json').read_text())
    assert len(selections)==360
    byfile={r['file']:r for r in selections}
    checkpoints=0
    for path in sorted((HERE/'02_personalization/personal_weights').glob('*.tar.gz')):
        with tarfile.open(path) as tar:
            for item in tar.getmembers():
                if not item.name.endswith('.pt'):continue
                payload=tar.extractfile(item).read()
                row=byfile[f'{path.relative_to(HERE)}::{item.name}']
                assert hashlib.sha256(payload).hexdigest()==row['sha256']
                saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
                assert saved['scope']=='tail' and saved['endpoint']==row['endpoint']
                assert saved['config']==row['selection']['config']
                assert all(torch.isfinite(x).all() for x in saved['state_dict'].values())
                checkpoints+=1
    for path in (HERE/'02_personalization/source_weights').glob('*.pt'):load(path)
    maxdiff=0.;replayed=0;ppgdiff=0.;example={}
    people=sorted(lock['people'],key=lambda r:r['patient'])[:2]
    for entry in people:
        p=read_all(Path(entry['file']))
        support=np.array(entry['support']);query=np.array(entry['query'])[:8]
        refs=dict(x=p['x'][support],y=p['y'][support])
        pid=entry['patient'].replace(':','_')
        for mode in ['ppg','dual']:
            for seed in [42,43,44]:
                # Use the exact exported weights, not the local source of the archive.
                with tempfile.TemporaryDirectory(prefix='bad-blood-replay-') as tmp:
                    archive=HERE/'02_personalization/personal_weights'/f'{mode}_seed{seed}.tar.gz'
                    with tarfile.open(archive) as tar:
                        for d in [0,1]:
                            member=f'{pid}/tail_endpoint{d}.pt'
                            Path(tmp,f'tail_endpoint{d}.pt').write_bytes(tar.extractfile(member).read())
                    yp=infer(tmp,p['x'][query],refs)
                    unit=run/'units'/mode/f'seed{seed}'/pid
                    with np.load(unit/'predictions.npz') as z:expected=z['rolling_tail'][:len(query)]
                    difference=float(np.max(np.abs(yp-expected)));assert difference<5e-5
                    maxdiff=max(maxdiff,difference);replayed+=len(query)
                    if mode=='ppg':
                        altered=p['x'][query].copy();altered[:,0]=1-altered[:,0]
                        changed=infer(tmp,altered,refs)
                        single=infer(tmp,p['x'][query,1:,:],dict(x=refs['x'][:,1:,:],y=refs['y']))
                        assert np.array_equal(yp,changed) and np.array_equal(yp,single)
                        ppgdiff=max(ppgdiff,float(np.max(np.abs(yp-changed))))
                    if entry==people[0] and mode=='dual' and seed==42:
                        with np.load(unit/'predictions.npz') as z:
                            example=dict(patient=entry['patient'],query_index=int(query[0]),support_count=len(support),
                                known_first_bp=refs['y'][0].tolist(),truth=p['y'][query[0]].tolist(),
                                prediction=yp[0].tolist(),personal_mean=z['personal_mean'][0].tolist(),
                                bias=z['bias'][0].tolist())
    (HERE/'metrics/worked_example.json').write_text(json.dumps(example,indent=2)+'\n')
    result=dict(status='passed',personal_endpoint_checkpoints=checkpoints,source_checkpoints=6,
        portable_predictions_replayed=replayed,max_portable_replay_difference_mmHg=maxdiff,
        ppg_ecg_invariance_difference=ppgdiff,frozen_experiment_hashes_match=True,
        original_experiment_verification=verification['status'],adopted_method='rolling_tail')
    (HERE/'release_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
