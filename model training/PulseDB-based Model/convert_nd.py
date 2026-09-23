"""Join transcribed cuff records to electronic waveform segments.

Input segment CSV: time_s,ecg_raw,ppg_raw (1250 rows at 125 Hz).
Reference log: participant_id,visit,cycle,segment_id,waveform_file,SBP,DBP,status.
Output: same wide filtered waveform schema as the public-data examples.
"""
import argparse,csv
from pathlib import Path
import numpy as np
from common import from_raw_125

HEADER=['dataset','participant_id','segment_id','fs_hz','SBP','DBP']+[f'{c}_{i:04d}' for c in ['ecg','ppg'] for i in range(1250)]

def convert(log,output):
    log=Path(log);output=Path(output)
    if output.exists():raise FileExistsError(output)
    converted=[];seen=set()
    with log.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['status']!='accepted':continue
            pid=row['participant_id'].strip();sid=row['segment_id'].strip()
            if not pid or not sid or sid in seen:raise ValueError('Missing ID or duplicate segment_id')
            seen.add(sid)
            sbp,dbp=float(row['SBP']),float(row['DBP'])
            if not np.isfinite([sbp,dbp]).all() or sbp<=dbp:raise ValueError(f'{sid}: check transcribed cuff values')
            path=log.parent/row['waveform_file']
            with path.open(newline='') as g:
                samples=list(csv.DictReader(g))
            t=np.array([float(s['time_s']) for s in samples])
            if len(t)!=1250 or not np.isfinite(t).all() or not np.allclose(t-t[0],np.arange(1250)/125,atol=.0005):
                raise ValueError(f'{sid}: expected 1250 regularly sampled points at 125 Hz')
            x=np.array([[float(s[k]) for s in samples] for k in ['ecg_raw','ppg_raw']])
            filtered=from_raw_125(x)
            converted.append(['ND',pid,sid,125,sbp,dbp,*filtered.reshape(-1).tolist()])
    if not converted:raise ValueError('No accepted complete records')
    with output.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(HEADER);w.writerows(converted)
    return len(converted)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log');p.add_argument('--output',required=True);a=p.parse_args()
    print(f'{convert(a.log,a.output)} segments exported')
