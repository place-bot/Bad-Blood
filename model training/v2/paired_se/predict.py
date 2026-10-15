"""Predict current BP using a saved pair model and an observed reference bank."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paired_se.run import restore
from paired_se.model import predict_bank


def load_predict(checkpoint,current,reference,reference_bp,device='cpu'):
    for waves in (current,reference):
        if waves.ndim!=3 or waves.shape[1:]!=(2,1250) or waves.dtype!=np.float32:
            raise ValueError('Use float32 N x 2 x 1250 normalized ECG/PPG')
        if not np.isfinite(waves).all() or waves.min() < -1e-6 or waves.max()>1+1e-6 or np.any(np.ptp(waves,axis=-1)<1e-8):
            raise ValueError('Nonfinite, flat or unnormalized waveforms')
    reference_bp=np.asarray(reference_bp,dtype=np.float32)
    if reference_bp.shape!=(len(reference),2) or not np.isfinite(reference_bp).all() or np.any(reference_bp[:,0]<=reference_bp[:,1]):
        raise ValueError('Invalid observed reference SBP/DBP')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=restore(saved,device)
    return predict_bank(model,current,reference,reference_bp,saved['center'],saved['scale'],device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--current',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--reference-bp',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(4)
    result=load_predict(a.checkpoint,np.load(a.current,allow_pickle=False),np.load(a.reference,allow_pickle=False),
        np.load(a.reference_bp,allow_pickle=False),a.device)
    with a.output.open('x',newline='') as handle:
        w=csv.writer(handle);w.writerow(['predicted_sbp_mmHg','predicted_dbp_mmHg']);w.writerows(result)


if __name__=='__main__':main()
