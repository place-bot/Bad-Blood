import json
from pathlib import Path
import numpy as np
from common import normalize,extract,from_raw_125

class Predictor:
    def __init__(self,model_dir):
        self.root=Path(model_dir);self.config=json.loads((self.root/'selection.json').read_text());self.trees=None;self.cnn=None
        if self.config['tree_weight']>0:
            import xgboost as xgb
            self.trees=[]
            for target in range(2):
                m=xgb.XGBRegressor();m.load_model(self.root/f"{self.config['tree_name']}_{target}.ubj");self.trees.append(m)
        if self.config['tree_weight']<1:
            import torch
            torch.set_num_threads(4)
            from network import BPNet
            bundle=torch.load(self.root/(self.config.get('cnn_name','cnn')+'.pt'),map_location='cpu',weights_only=True)
            self.cnn=BPNet(pool=bundle.get('pool',5)).eval();self.cnn.load_state_dict(bundle['state_dict']);self.center=np.array(bundle['center']);self.scale=np.array(bundle['scale'])
    def predict(self,waves,input_format='filtered'):
        if input_format not in ('filtered','raw125'):
            raise ValueError('input_format must be filtered or raw125')
        x=np.stack([from_raw_125(a) if input_format=='raw125' else normalize(a) for a in waves])
        w=self.config['tree_weight'];yp=0
        if self.trees:
            features=np.stack([extract(a) for a in x]);yp=w*np.stack([m.predict(features) for m in self.trees],1)
        if self.cnn:
            import torch
            with torch.inference_mode():p=self.cnn(torch.from_numpy(x)).numpy()*self.scale+self.center
            yp=yp+(1-w)*p
        return np.asarray(yp)
