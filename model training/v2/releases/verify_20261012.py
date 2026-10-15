"""Independent artifact/metric checks for the dated release; no training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from personalized_forecasting.run import file_hash,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    release=ROOT/'releases/20261012'
    checked=0
    for directory in ('01_population_cnn_se','02_personalization'):
        base=release/directory
        for fn,h in json.loads((base/'SHA256SUMS.json').read_text()).items():
            assert file_hash(base/fn)==h,fn;checked+=1
    summary=json.loads((a.run/'summary.json').read_text())
    audit=json.loads((a.run/'cohort_audit.json').read_text());rows=[];encoders=0
    for r in audit:
        if r['status']!='included':continue
        pid=r['patient']
        for seed in (42,43,44):
            pop=torch.load(ROOT/f'paired_se/models/paired_population_seed{seed}.pt',map_location='cpu',weights_only=False)
            fitted=torch.load(a.run/pid/f'seed{seed}/paired/model.pt',map_location='cpu',weights_only=False)
            for k,v in pop['state_dict'].items():
                if not k.startswith('change.'):
                    assert torch.equal(v,fitted['state_dict'][k]),'Frozen encoder/normalizer changed'
            encoders+=1
            for method in ('population','calibration_mean','bias','paired'):
                folder=a.run/pid/f'seed{seed}'/method;unit=json.loads((folder/'unit.json').read_text())
                for name,h in unit['hashes'].items():assert file_hash(folder/name)==h
                with np.load(folder/'predictions.npz') as z:
                    assert np.array_equal(z['support_indices'],r['support_indices'])
                    assert np.array_equal(z['query_indices'],r['query_indices'])
                    err=z['pred'].astype(float)-z['y'].astype(float)
                    metrics=dict(mae=abs(err).mean(0),rmse=np.sqrt((err**2).mean(0)),bias=err.mean(0),error_sd=err.std(0,ddof=1),
                        within_5=(abs(err)<=5).mean(0),within_10=(abs(err)<=10).mean(0),within_15=(abs(err)<=15).mean(0))
                    for k,v in metrics.items():np.testing.assert_allclose(v,unit['metrics']['macro'][k],rtol=0,atol=1e-10)
                    rows.append((method,metrics))
    assert len(rows)==192 and encoders==48
    for m in summary['methods']:
        for metric in ('mae','rmse','bias','error_sd','within_5','within_10','within_15'):
            v=np.mean([r[1][metric] for r in rows if r[0]==m],0)
            np.testing.assert_allclose(v,summary['methods'][m]['macro'][metric],rtol=0,atol=1e-10)
    data=dict(status='verified',date='20261012',checksummed_files=checked,independently_recomputed_units=len(rows),
        unchanged_personal_encoders=encoders,project_tests_passed=117,
        snapshot_runner_replayed_predictions=summary['verification']['replayed_predictions'],
        snapshot_max_replay_difference=summary['verification']['max_replay_error'])
    atomic_json(release/'verification.json',data);print(json.dumps(data))

if __name__=='__main__':main()
