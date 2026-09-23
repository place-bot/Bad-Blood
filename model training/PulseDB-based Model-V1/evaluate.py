from paths import DATA
import json,time
from pathlib import Path
import numpy as np
from train_models import ROOT,load_split,score

def full_metrics(y,p,s):
    result=score(y,p,s);e=p-y
    result['error_SD']=np.std(e,axis=0,ddof=1).tolist()
    result['r2']=(1-(e**2).sum(0)/((y-y.mean(0))**2).sum(0)).tolist()
    subjects=np.unique(s);individual=np.array([np.abs(e[s==a]).mean(0) for a in subjects]);rng=np.random.default_rng(20260923)
    means=np.array([individual[rng.integers(0,len(individual),len(individual))].mean(0) for _ in range(2000)])
    result['macro_MAE_95CI']=np.quantile(means,[.025,.975],axis=0).T.tolist();result['participants']=len(subjects);result['segments']=len(y)
    return result

def main():
    reportfile=ROOT/'reports/final_evaluation.json'
    if reportfile.exists():raise RuntimeError('Final test already evaluated. Preserve the locked evaluation; do not tune on test.')
    ti,vi,y,s=load_split();tree_scores=json.loads((ROOT/'reports/tree_validation.json').read_text())
    candidates=[]
    for cnn_name in ['cnn','cnn_full']:
        cnn=np.load(ROOT/f'models/{cnn_name}_validation.npy')
        metrics=score(y[vi],cnn,s[vi])
        candidates.append({'tree_name':None,'cnn_name':cnn_name,'tree_weight':0,'validation':metrics})
    for name in ['xgb_d3','xgb_d5']:
        trees=np.load(ROOT/f'models/{name}_validation.npy')
        metrics=score(y[vi],trees,s[vi])
        candidates.append({'tree_name':name,'cnn_name':None,'tree_weight':1,'validation':metrics})
    selected=min(candidates,key=lambda a:np.mean(a['validation']['macro_mae']))
    selected['preprocessing']='pulsedb_filtered_minmax_v2';selected['targets']=['SBP','DBP'];selected['selection_metric']='participant_macro_MAE_mean_over_targets'
    selected['fs_hz']=125;selected['samples_per_channel']=1250
    (ROOT/'models/selection.json').write_text(json.dumps(selected,indent=2))
    (ROOT/'reports/validation_selection.json').write_text(json.dumps(candidates,indent=2))
    from inference import Predictor
    predictor=Predictor(ROOT/'models');outputs={}
    prior=set(json.loads((ROOT/'reports/prior_exploratory_subjects.json').read_text()))
    baseline=np.load(ROOT/'models/train_mean.npy')
    for name in ['test','aami']:
        d=DATA/name;waves=np.load(d/'waves.npy',mmap_mode='r');labels=np.load(d/'y.npy');sub=np.load(d/'subjects.npy');valid=np.load(d/'valid.npy')
        ids=np.flatnonzero(valid);pred=[];start=time.time()
        for pos in range(0,len(ids),512):pred.append(predictor.predict(waves[ids[pos:pos+512]]))
        pred=np.concatenate(pred);np.save(ROOT/f'reports/{name}_predictions.npy',pred)
        outputs[name]={'model':full_metrics(labels[ids],pred,sub[ids]),'baseline':full_metrics(labels[ids],np.tile(baseline,(len(ids),1)),sub[ids]),'seconds':time.time()-start}
        clean=~np.isin(sub[ids],list(prior))
        outputs[name]['previous_exploratory_subject_overlap']=sorted(set(sub[ids])&prior)
        outputs[name]['excluding_prior_subjects']=full_metrics(labels[ids][clean],pred[clean],sub[ids][clean])
        print(name,outputs[name],flush=True)
    report={'selected':selected,'evaluation':outputs,'training_participants':len(np.unique(s[ti])),'validation_participants':len(np.unique(s[vi])),'training_segments':len(ti),'validation_segments':len(vi),'limitations':['Source-domain evaluation only; ND transfer remains unmeasured.','No clinical validation claim.','Raw adapter is a documented deployment preprocessing approximation; bench and ND validation required.']}
    reportfile.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
