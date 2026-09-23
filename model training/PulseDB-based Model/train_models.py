from paths import DATA
import argparse,json,time,os
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parent

def load_split():
    s=np.load(DATA/'train/subjects.npy');y=np.load(DATA/'train/y.npy');ok=np.load(DATA/'train/valid.npy');split=json.loads((DATA/'split.json').read_text())
    ti=np.flatnonzero(np.isin(s,split['train'])&ok);vi=np.flatnonzero(np.isin(s,split['validation'])&ok)
    return ti,vi,y,s

def score(y,p,s):
    e=p-y
    return {'mae':abs(e).mean(0).tolist(),'rmse':np.sqrt((e**2).mean(0)).tolist(),'bias':e.mean(0).tolist(),'macro_mae':np.mean([abs(e[s==a]).mean(0) for a in np.unique(s)],0).tolist()}

def train_trees():
    import xgboost as xgb
    from threadpoolctl import threadpool_limits
    ti,vi,y,s=load_split();X=np.load(DATA/'train/features.npy',mmap_mode='r')
    trainX=np.asarray(X[ti]);validX=np.asarray(X[vi]);results={}
    results['mean']=score(y[vi],np.tile(y[ti].mean(0),(len(vi),1)),s[vi])
    np.save(ROOT/'models/train_mean.npy',y[ti].mean(0))
    for depth in [3,5]:
        models=[];pred=[]
        for target in range(2):
            model=xgb.XGBRegressor(n_estimators=1800,max_depth=depth,learning_rate=.035,min_child_weight=30,subsample=.8,colsample_bytree=.85,reg_lambda=20,reg_alpha=.1,tree_method='hist',max_bin=128,n_jobs=4,random_state=20260923,early_stopping_rounds=80)
            with threadpool_limits(limits=4):model.fit(trainX,y[ti,target],eval_set=[(validX,y[vi,target])],verbose=False)
            model.save_model(ROOT/f'models/xgb_d{depth}_{target}.ubj');pred.append(model.predict(validX));models.append(model)
            print('xgb',depth,target,'best iteration',model.best_iteration,flush=True)
        pred=np.stack(pred,1);results[f'xgb_d{depth}']=score(y[vi],pred,s[vi])
        np.save(ROOT/f'models/xgb_d{depth}_validation.npy',pred)
        (ROOT/'reports/tree_validation.json').write_text(json.dumps(results,indent=2));print(results,flush=True)

def train_cnn(epochs,pool=5):
    import torch
    from network import BPNet
    torch.set_num_threads(4);torch.manual_seed(20260923);np.random.seed(20260923)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    ti,vi,y,s=load_split();waves=np.load(DATA/'train/waves.npy',mmap_mode='r')
    center=y[ti].mean(0);scale=y[ti].std(0);target=(y-center)/scale
    name='cnn' if pool==5 else 'cnn_full'
    model=BPNet(pool=pool).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3)
    best=float('inf');stale=0;history=[];batch=256;rng=np.random.default_rng(20260923)
    for epoch in range(1,epochs+1):
        start=time.time();model.train();order=rng.permutation(ti);loss_total=0
        for pos in range(0,len(order),batch):
            ids=order[pos:pos+batch];xx=torch.from_numpy(np.array(waves[ids],copy=True)).to(device);yy=torch.from_numpy(target[ids]).to(device)
            xx=xx+torch.randn_like(xx)*.003
            opt.zero_grad(set_to_none=True);pred=model(xx);loss=torch.nn.functional.mse_loss(pred,yy);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();loss_total+=float(loss.detach().cpu())*len(ids)
        model.eval();preds=[]
        with torch.inference_mode():
            for pos in range(0,len(vi),512):
                ids=vi[pos:pos+512];preds.append(model(torch.from_numpy(np.array(waves[ids],copy=True)).to(device)).cpu().numpy()*scale+center)
        pred=np.concatenate(preds);metrics=score(y[vi],pred,s[vi]);value=np.mean(metrics['macro_mae']);scheduler.step(value)
        row={'epoch':epoch,'train_normalized_MSE':loss_total/len(ti),'validation':metrics,'seconds':time.time()-start,'lr':opt.param_groups[0]['lr']};history.append(row)
        (ROOT/f'reports/{name}_history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
        if value<best:
            best=value;stale=0
            torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'center':center.tolist(),'scale':scale.tolist(),'epoch':epoch,'validation':metrics,'pool':pool},ROOT/f'models/{name}.pt')
            np.save(ROOT/f'models/{name}_validation.npy',pred)
        else:stale+=1
        if stale>=8:break

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['trees','cnn','cnn_full']);p.add_argument('--epochs',type=int,default=40);a=p.parse_args()
    train_trees() if a.kind=='trees' else train_cnn(a.epochs,1 if a.kind=='cnn_full' else 5)
