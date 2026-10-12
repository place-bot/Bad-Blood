"""Preserve bounded development results, including failures, without target selection."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from personalized_forecasting.run import file_hash,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,required=True);p.add_argument('--adapt',type=Path,required=True);a=p.parse_args()
    here=Path(__file__).parent;results=here/'results';results.mkdir(exist_ok=True)
    screen=json.loads((a.screen/'summary.json').read_text());adapt=json.loads((a.adapt/'summary.json').read_text());checks=0
    for run in (a.screen,a.adapt):
        for f in run.glob('*/*.npz'):
            with np.load(f) as z:
                person=f.stem.replace('vitaldb_','vitaldb:')
                metrics=np.abs(z['pred'].astype(float)-z['y'].astype(float)).mean(0)
            meta=f.parent/('development.json' if run==a.screen else 'result.json')
            records=json.loads(meta.read_text())['people']
            row=next(r for r in records if r.get('patient',r.get('person'))==person)
            np.testing.assert_allclose(metrics,row['metrics']['macro']['mae'],rtol=0,atol=1e-10);checks+=1
    for label,run in (('screen',a.screen),('adaptation',a.adapt)):
        dest=results/label;dest.mkdir(exist_ok=True)
        for name in ('summary.json','manifest.json'):shutil.copyfile(run/name,dest/name)
        for folder in run.iterdir():
            if not folder.is_dir():continue
            out=dest/folder.name;out.mkdir(exist_ok=True)
            for f in folder.glob('*.json'):shutil.copyfile(f,out/f.name)
            for f in folder.glob('*.pt'):
                target=here/'models'/label/folder.name/f.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    atomic_json(results/'verification.json',dict(verified_person_level_prediction_files=checks,development_only=True,target_evaluation=False,
        screen_pass=screen['development_pass'],adaptation_pass=adapt['selected']['development_pass']))
    atomic_json(here/'models/manifest.json',{str(f.relative_to(here/'models')):file_hash(f) for f in sorted((here/'models').rglob('*.pt'))})
    log=ROOT.parents[1]/'autoresearch/orchestrator-261012-context';log.mkdir(parents=True,exist_ok=True)
    (log/'results.tsv').write_text('phase\tmethod\tsbp_mae\tdbp_mae\tworst_control_ratio\n'+''.join(
        f"screen\t{r['variant']}\t{r['mae'][0]}\t{r['mae'][1]}\t{r['worst_control_ratio']}\n" for r in screen['outcomes'])+''.join(
        f"adaptation\t{r['config']}\t{r['mae'][0]}\t{r['mae'][1]}\t{r['worst_ratio']}\n" for r in adapt['outcomes']))
    atomic_json(log/'handoff.json',dict(status='CEILING',goal_achieved=False,reason='Three architecture families and nine follow-up personal adaptation settings failed the joint development criterion',
        independent_verification='Not entered; no development candidate passed',next_step='Design temporal source-calibration episodes or expand development cohort in a separately frozen experiment',results=str(results)))
    print('Preserved and verified',checks,'development prediction files')

if __name__=='__main__':main()
