"""Resumable bounded three-seed waveform training, followed by personalization."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modality_se.train import configure,original
from personalized_forecasting.run import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--personal',type=Path,required=True);p.add_argument('--fresh',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);configure()
    for seed in (42,43,44):
        directory=a.output/f'population/se_ppg_seed{seed}'
        atomic_json(a.output/'status.json',dict(stage='population',seed=seed,status='running'))
        result=original.run_training('se_ppg',seed,a.data,directory,'mps',resume=(directory/'config.json').exists())
        print('POPULATION COMPLETE',seed,json.dumps(result),flush=True)
    atomic_json(a.output/'status.json',dict(stage='personalization',status='running'))
    cmd=[sys.executable,str(ROOT/'modality_se/personalize.py'),'--data',str(a.data),'--output',str(a.output),'--prior',str(a.prior),'--personal',str(a.personal),'--fresh',str(a.fresh)]
    subprocess.run(cmd,check=True)
    atomic_json(a.output/'status.json',dict(stage='verification_pending',status='training_complete'))

if __name__=='__main__':main()
