"""Inference for compact modality-aware population and personal checkpoints."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modality_se.model import ModalitySE
from modality_se.personalize import restore
from paired_se.model import predict_bank
from personalized_forecasting.adaptation import predict as population_predict

def load_predict(checkpoint,current,reference=None,reference_bp=None,device='cpu'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    modality=saved['modality']
    if saved.get('kind')=='population':
        model=ModalitySE(modality);model.load_state_dict(saved['state_dict'])
        return population_predict(model,current,saved['center'],saved['scale'],device)[0]
    if reference is None or reference_bp is None:raise ValueError('Known calibration waveforms and BP are required')
    model=restore(saved,modality,device)
    return predict_bank(model,np.asarray(current,np.float32),np.asarray(reference,np.float32),np.asarray(reference_bp,np.float32),saved['center'],saved['scale'],device)
