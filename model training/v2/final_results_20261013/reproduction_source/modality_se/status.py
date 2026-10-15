import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
if (a.run/'study_complete.json').exists():
    print((a.run/'study_complete.json').read_text())
else:
    result=dict(status='incomplete',waveform_fits=[],pair_fits=[],personal_units=len(list(a.run.glob('*/*/*/*/*/unit.json'))),expected_personal_units=816)
    for seed in (42,43,44):
        folder=a.run/f'population/se_ppg_seed{seed}';history=folder/'history.json'
        h=json.loads(history.read_text()) if history.exists() else []
        result['waveform_fits'].append(dict(seed=seed,epochs=len(h),complete=(folder/'completion.json').exists(),best_validation=min([r['selection_macro_mae'] for r in h],default=None)))
        folder=a.run/f'pair_population/ppg_seed{seed}'
        result['pair_fits'].append(dict(seed=seed,complete=(folder/'complete.json').exists()))
    print(json.dumps(result,indent=2))
