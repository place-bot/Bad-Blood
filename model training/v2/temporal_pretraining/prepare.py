"""Freeze original source roles, then retrieve timestamped public records."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
from urllib.request import urlopen
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from paired_se.fresh_20261012.validate import FOLDER, download, read_person
from paired_se.run import restore
from context_se.run import encode
from personalized_forecasting.run import atomic_json, file_hash

def freeze():
    target=HERE/'sources.json'
    if target.exists():raise ValueError('Source manifest already frozen')
    manifest=json.loads((ROOT/'paired_se/results/manifest.json').read_text())['population']
    page=urlopen(FOLDER,timeout=60).read().decode()
    decoded=page.replace('\\x22','"').replace('\\x5b','[').replace('\\x5d',']').replace('\\/','/')
    entries=dict((n,fid) for fid,n in re.findall(r'\["([\w-]+)",\["[\w-]+"\],"([^"]+)"',decoded) if re.fullmatch(r'p\d{6}\.mat',n))
    rows=[]
    for role in ('train','validation'):
        for name,fid in sorted(entries.items()):
            patient='vitaldb:'+name[:-4]
            if patient in manifest['partitions'][role]['patients']:
                rows.append(dict(file=name,drive_id=fid,patient=patient,role=role))
    assert all(r['patient'] not in manifest['excluded_patients'] for r in rows)
    atomic_json(target,dict(date='20261012',folder=FOLDER,selection='all original train/validation IDs from observed public first page',
        page_unique_files=len(entries),protocol_sha256=file_hash(HERE/'PROTOCOL.md'),records=rows))
    print('FROZEN',len(rows),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--raw',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.freeze:freeze();return
    lock=json.loads((HERE/'sources.json').read_text());assert lock['protocol_sha256']==file_hash(HERE/'PROTOCOL.md')
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    saved=torch.load(ROOT/'paired_se/models/paired_population_seed42.pt',map_location='cpu',weights_only=False)
    base=restore(saved,'cpu');audit=[]
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures={ex.submit(download,r,a.raw):r for r in lock['records']}
        for future in as_completed(futures):
            row=futures[future]
            try:path=future.result()
            except Exception as e:
                audit.append(dict(**row,status='download_failed',reason=str(e)));atomic_json(a.output/'audit.json',audit);continue
            stamp=dict(**row,sha256=file_hash(path))
            try:record=read_person(path,row['patient'])
            except (ValueError,AssertionError) as e:
                audit.append(dict(**stamp,status='ineligible',reason=str(e)));atomic_json(a.output/'audit.json',audit);print('EXCLUDE',row['patient'],str(e),flush=True);continue
            h,pr=encode(base,record['x'])
            target=a.output/(row['file'][:-4]+'.npz')
            np.savez_compressed(target,h=h,p=pr,y=record['y'],start=record['start'],end=record['end'],support=record['support'],query=record['query'])
            audit.append(dict(**stamp,status='included',n=len(h),support=len(record['support']),query=len(record['query']),features_sha256=file_hash(target),case=record['case']))
            atomic_json(a.output/'audit.json',audit);print('PREPARED',row['patient'],row['role'],len(h),flush=True)
    atomic_json(a.output/'complete.json',dict(sources_sha256=file_hash(HERE/'sources.json'),checkpoint_sha256=file_hash(ROOT/'paired_se/models/paired_population_seed42.pt'),audit=audit))
    print('DONE', {role:sum(r['status']=='included' and r['role']==role for r in audit) for role in ('train','validation')},flush=True)

if __name__=='__main__':main()
