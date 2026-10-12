"""Personal gradient adaptation of the context network; development only."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from context_se.model import ContextResidual
from context_se.run import encode
from paired_se.run import restore
from shared.data import load_personal_cohort
from shared.metrics import summarize
from se_personalization.run import dense_support
from personalized_forecasting.run import atomic_json,file_hash

class MaskedContext(ContextResidual):
    def __init__(self):super().__init__('context')
    def forward(self,q,s,residual,mask=None):
        if mask is None:return super().forward(q,s,residual)
        weight=mask[:,:,None].to(s.dtype)
        denominator=weight.sum(1,keepdim=True)
        if (denominator<=0).any():raise ValueError('Empty reference context')
        mean=(residual*weight).sum(1,keepdim=True)/denominator
        z=(self.context(torch.cat([s,residual-mean],-1))*weight).sum(1,keepdim=True)/denominator
        current=self.decode(torch.cat([q,z.expand(-1,q.shape[1],-1)],-1))
        observed=self.decode(torch.cat([s,z.expand(-1,s.shape[1],-1)],-1))
        return mean+current-(observed*weight).sum(1,keepdim=True)/denominator

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);p.add_argument('--individual-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
    saved=torch.load(ROOT/'paired_se/models/paired_population_seed42.pt',weights_only=False,map_location='cpu');base=restore(saved,'cpu')
    center,scale=np.asarray(saved['center']),np.asarray(saved['scale']);fc,fs=base.feature_center.numpy(),base.feature_scale.numpy()
    checkpoint=torch.load(a.source_run/'context/best.pt',weights_only=False,map_location='cpu')
    initial=MaskedContext();initial.load_state_dict(checkpoint['state_dict'])
    dev=load_personal_cohort(a.individual_root,a.source_run/'development_only_audit.json')['development']
    data=[]
    for v in dev:
        s=dense_support(v);q=v['query_indices'];h,pred=encode(base,v['x'])
        mask=(v['end_exclusive'][s][None,:]+10<=v['start'][s][:,None]+1e-6)
        trainq=np.flatnonzero(mask.any(1));mask=mask[trainq]
        assert set(np.r_[trainq,np.flatnonzero(mask.any(0))])==set(range(len(s)))
        data.append(dict(person=v['patient_id'],h=(h-fc)/fs,p=pred,y=(v['y']-center)/scale,raw_y=v['y'],support=s,query=q,trainq=trainq,mask=mask))
    settings=[dict(lr=lr,steps=n) for lr in (.0001,.0003,.001) for n in (10,25,100)]
    manifest=dict(protocol_sha256=file_hash(Path(__file__).with_name('ADAPTATION_PROTOCOL.md')),source_checkpoint_sha256=file_hash(a.source_run/'context/best.pt'),
        code_sha256=file_hash(Path(__file__)),settings=settings,people=[v['person'] for v in data],seed=42,target_used=False)
    mp=a.output/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:atomic_json(mp,manifest)
    controls=json.loads((a.source_run/'summary.json').read_text())['controls'];outcomes=[]
    for config in settings:
        name=f"lr{config['lr']:g}_s{config['steps']}";directory=a.output/name;directory.mkdir(exist_ok=True)
        if (directory/'result.json').exists():outcomes.append(json.loads((directory/'result.json').read_text()));continue
        rows=[]
        for v in data:
            model=copy.deepcopy(initial);opt=torch.optim.Adam(model.parameters(),lr=config['lr'])
            s,q=v['support'],v['query'];h,pr,y=v['h'],v['p'],v['y'];tq=v['trainq'];m=len(tq)
            qt=torch.from_numpy(h[s[tq]][:,None]);st=torch.from_numpy(h[s][None]).expand(m,-1,-1)
            rt=torch.from_numpy((y[s]-pr[s])[None]).expand(m,-1,-1);mask=torch.from_numpy(v['mask']);yt=torch.from_numpy(y[s[tq]]);pt=torch.from_numpy(pr[s[tq]])
            losses=[]
            for step in range(config['steps']):
                opt.zero_grad(set_to_none=True)
                out=pt+model(qt,st,rt,mask)[:,0];loss=torch.nn.functional.mse_loss(out,yt)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adaptation')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();losses.append(loss.item())
            outputs=[]
            with torch.inference_mode():
                for i in range(0,len(q),128):
                    idx=q[i:i+128];res=model(torch.from_numpy(h[idx][None]),torch.from_numpy(h[s][None]),torch.from_numpy((y[s]-pr[s])[None]))[0].numpy()
                    outputs.append((pr[idx]+res)*scale+center)
            pred=np.concatenate(outputs);person=v['person'];metrics=summarize(v['raw_y'][q],pred,np.repeat(person,len(q)))
            np.savez_compressed(directory/(person.replace(':','_')+'.npz'),y=v['raw_y'][q],pred=pred,support=s,query=q)
            torch.save(dict(state_dict=model.state_dict(),variant='context',config=config,person=person),directory/(person.replace(':','_')+'.pt'))
            rows.append(dict(person=person,metrics=metrics,loss_history=losses,all_early_rows_used=True,query_labels_used=False))
        mae=np.mean([r['metrics']['macro']['mae'] for r in rows],0);ratio=float(max(np.r_[mae/controls['bias'],mae/controls['calibration_mean']]))
        result=dict(config=config,mae=mae.tolist(),worst_ratio=ratio,development_pass=ratio<1,people=rows)
        atomic_json(directory/'result.json',result);outcomes.append(result);print(name,mae.tolist(),ratio,flush=True)
    chosen=min(outcomes,key=lambda r:(r['worst_ratio'],np.mean(r['mae']),settings.index(r['config'])))
    atomic_json(a.output/'summary.json',dict(status='complete',controls=controls,outcomes=outcomes,selected=chosen,independent_validation_complete=False))
    print('COMPLETE',chosen['config'],chosen['mae'],chosen['development_pass'],flush=True)

if __name__=='__main__':main()
