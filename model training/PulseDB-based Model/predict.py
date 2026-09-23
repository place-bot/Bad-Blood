import argparse,csv
from pathlib import Path
import numpy as np
from inference import Predictor

def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',required=True);p.add_argument('--format',choices=['filtered','raw125'],required=True);p.add_argument('--model-dir',default=str(Path(__file__).parent/'models'));a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('Choose a new output filename')
    model=Predictor(a.model_dir);rows=[]
    with open(a.input,newline='') as f:
        for r in csv.DictReader(f):
            result={'participant_id':r.get('participant_id',''),'segment_id':r.get('segment_id',''),'pred_SBP_mmHg':'','pred_DBP_mmHg':'','status':'','detail':''}
            try:
                if float(r['fs_hz'])!=125:raise ValueError('fs_hz must be 125')
                waves=np.array([[float(r[f'{c}_{i:04d}']) for i in range(1250)] for c in ['ecg','ppg']],dtype='float32')
                pred=model.predict(waves[None],input_format=a.format)[0]
                if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
                result.update(pred_SBP_mmHg=f'{pred[0]:.3f}',pred_DBP_mmHg=f'{pred[1]:.3f}',status='ok' if pred[0]>pred[1] else 'review_prediction')
            except (ValueError,KeyError) as e:result.update(status='invalid_input',detail=str(e))
            rows.append(result)
    with open(a.output,'x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['participant_id','segment_id','pred_SBP_mmHg','pred_DBP_mmHg','status','detail']);w.writeheader();w.writerows(rows)
    print(f'{len(rows)} records written to {a.output}')

if __name__=='__main__':main()
