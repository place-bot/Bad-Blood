"""Run saved SE personal/population weights on normalized ECG/PPG arrays."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shared.models import BPNet
from personalized_forecasting.adaptation import predict

def load_and_predict(checkpoint, waves, device='cpu'):
    # Only load trusted project checkpoints; torch checkpoint containers may use pickle.
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['variant']!='se_last': raise ValueError('Expected SE-last model')
    x=np.asarray(waves,np.float32)
    if x.ndim!=3 or x.shape[1:]!=(2,1250) or not len(x): raise ValueError('Expected N x 2 x 1250')
    if not np.isfinite(x).all() or x.min() < -1e-5 or x.max() > 1+1e-5 or np.any(np.ptp(x,axis=2)<1e-8):
        raise ValueError('Require finite nonflat filtered per-window [0,1] normalized inputs')
    model=BPNet(variant='se_last');model.load_state_dict(saved['state_dict'],strict=True)
    y,_=predict(model,x,saved['center'],saved['scale'],device)
    offset=np.asarray(saved.get('offset_standardized',[0,0]))
    y=y+offset*np.asarray(saved['scale'])
    if not np.isfinite(y).all(): raise ValueError('Nonfinite output')
    return y

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--waves',type=Path,required=True,help='N x 2 x 1250 .npy; filtered, per-window normalized ECG then PPG')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu',choices=('cpu','mps','cuda'))
    a=p.parse_args()
    torch.set_num_threads(4)
    y=load_and_predict(a.checkpoint,np.load(a.waves,allow_pickle=False),a.device)
    with a.output.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['segment_index','pred_SBP_mmHg','pred_DBP_mmHg'])
        writer.writerows((i,float(v[0]),float(v[1])) for i,v in enumerate(y))

if __name__=='__main__':main()
