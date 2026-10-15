"""Run portable inference parity checks, adaptation equivalence and file hashing."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import re
import subprocess
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'implementation'),str(HERE.parent)]
from predict import load_model, predict
from adapt import adapt
from modality_se.predict import load_predict as original_predict
from paired_se.run import fit_personal


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    x = rng.uniform(0,1,(3,2,1250)).astype('float32')
    r = rng.uniform(0,1,(4,2,1250)).astype('float32')
    y = np.array([[120,75],[122,76],[118,73],[124,77]],np.float32)
    files=sorted(HERE.glob('0*/weights/**/*.pt'))
    differences=[]; ppg_errors=[]
    for f in files:
        new=predict(f,x,r,y)
        old=original_predict(f,x,r,y,'cpu')
        diff=float(np.max(np.abs(new-old)));differences.append(diff)
        assert new.shape==(3,2) and np.isfinite(new).all() and diff<=1e-4,(f,diff)
        saved=torch.load(f,map_location='cpu',weights_only=False)
        if saved['modality']=='ppg':
            x2=x.copy();x2[:,0]=1-x2[:,0];r2=r.copy();r2[:,0]=1-r2[:,0]
            ignored=predict(f,x2,r2,y)
            single=predict(f,x[:,1:2],r[:,1:2],y)
            err=float(max(np.max(np.abs(new-ignored)),np.max(np.abs(new-single))))
            assert err==0,(f,err)
            ppg_errors.append(err)
    adaptation=[]
    with tempfile.TemporaryDirectory(prefix='bad-blood-release-check-') as temp:
        for mode in ('dual','ppg'):
            f=HERE/f'02_personalization/weights/{mode}_pair_population_seed42.pt'
            model,saved=load_model(f)
            start=np.arange(4)*30.;end=start+10.
            expected,_=fit_personal(model,r,y,start,end,saved['center'],saved['scale'],
                dict(scope='head',lr=.0003,steps=25),'cpu')
            actual=adapt(f,r,y,start,end,Path(temp)/f'{mode}.pt')
            errors=[float((v.cpu()-actual['state_dict'][k]).abs().max()) for k,v in expected.state_dict().items()]
            assert max(errors)==0,(mode,max(errors))
            adaptation.append(dict(modality=mode,max_state_difference=max(errors)))
    suite=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],
                         cwd=HERE.parent,capture_output=True,text=True,check=True)
    match=re.search(r'Ran (\d+) tests',suite.stderr)
    assert match and suite.stderr.rstrip().endswith('OK')
    verification=dict(date='20261013',status='passed',loaded_checkpoints=len(files),
        repository_unit_tests_passed=int(match.group(1)),
        synthetic_interface_test=True,not_new_accuracy_evaluation=True,
        max_original_inference_difference_mmHg=max(differences),
        ppg_checkpoint_checks=len(ppg_errors),max_ecg_invariance_difference=max(ppg_errors),
        selected_adaptation_equivalence=adaptation)
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    excluded={'.aux','.log','.out','.blg','.pyc'}
    manifest={}
    for f in sorted(HERE.rglob('*')):
        if f.is_file() and f!=HERE/'manifest.json' and f.name!='main.pdf' and f.suffix not in excluded and '__pycache__' not in f.parts:
            manifest[str(f.relative_to(HERE))]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(verification,indent=2))
    print('Hashed',len(manifest),'artifacts')


if __name__=='__main__':main()
