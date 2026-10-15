"""One immediate, resumable local job: train, evaluate, verify. No scheduler."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
DEFAULT=Path('/Users/jguo9/Desktop/26fall/LLM/output/bad_blood_v2_runs/20261015_rolling_personal')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args()
    out=a.output;out.mkdir(exist_ok=True,parents=True)
    def status(stage,**extra):
        record=dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        tmp=out/'job_status.json.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out/'job_status.json')
        print(json.dumps(record),flush=True)
    status('preflight')
    subprocess.run([sys.executable,str(HERE/'test_rolling.py')],check=True)
    subprocess.run([sys.executable,str(HERE/'run.py'),'prepare','--output',str(out)],check=True)
    workers=[];handles=[]
    try:
        for i in range(a.workers):
            handle=(out/f'worker{i}.log').open('a');handles.append(handle)
            proc=subprocess.Popen([sys.executable,str(HERE/'run.py'),'train','--output',str(out),
                '--worker',str(i),'--workers',str(a.workers),'--threads','2'],stdout=handle,stderr=subprocess.STDOUT)
            workers.append(proc)
        last=-1
        while any(p.poll() is None for p in workers):
            failed=[dict(pid=p.pid,returncode=p.returncode) for p in workers if p.poll() not in (None,0)]
            if failed:raise RuntimeError('Worker failed: '+json.dumps(failed))
            count=sum(1 for _ in out.glob('units/*/*/*/trained.json'))
            if count!=last:status('training',completed=count,total=180,workers=[p.pid for p in workers]);last=count
            time.sleep(5)
        if any(p.returncode for p in workers):raise RuntimeError('A training worker failed')
        status('evaluating',completed=180,total=180)
        with (out/'evaluation.log').open('a') as handle:
            subprocess.run([sys.executable,str(HERE/'run.py'),'evaluate','--output',str(out)],check=True,stdout=handle,stderr=subprocess.STDOUT)
        status('verifying',completed=180,total=180)
        with (out/'verification.log').open('a') as handle:
            subprocess.run([sys.executable,str(HERE/'verify.py'),'--output',str(out)],check=True,stdout=handle,stderr=subprocess.STDOUT)
        result=json.loads((out/'summary.json').read_text())
        counts={mode:{m:r['seed_average_all_four_count'] for m,r in v['comparisons'].items()}
                for mode,v in result['modality_results'].items()}
        status('complete',completed=180,total=180,counts=counts)
        ledger=HERE/'autoresearch/classic-261015-rolling'
        (ledger/'results.tsv').write_text('iteration\tstatus\tppg_joint_wins\tdual_joint_wins\tdescription\n'
            '0\tbaseline\t6\t7\tPrevious single-window joint-endpoint head HPO\n'
            f"1\tverified\t{counts['ppg']['rolling_selected']}\t{counts['dual']['rolling_selected']}\tOne frozen comparison; full results retained regardless of outcome\n")
        (ledger/'handoff.json').write_text(json.dumps(dict(version='2.1.0',source='loop',status='COMPLETE',
            results_tsv=str(ledger/'results.tsv'),findings=counts,output=str(out),published=False),indent=2)+'\n')
    except BaseException as error:
        for p in workers:
            if p.poll() is None:p.terminate()
        status('failed',error=str(error))
        raise
    finally:
        for h in handles:h.close()


if __name__=='__main__':main()
