from paths import DATA
import argparse,json,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py
from common import normalize,extract

ROOT=Path(__file__).resolve().parent

def convert(path,name):
    dest=DATA/name;dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    with h5py.File(path,'r') as f:
        g=f['Subset'];ds=g['Signals'];print(name,ds.shape,ds.chunks,flush=True)
        if ds.shape[:2]!=(1250,3):raise ValueError(f'Unexpected MATLAB signal axes {ds.shape}')
        n=ds.shape[2]
        waves=np.lib.format.open_memmap(dest/'waves.npy',mode='w+',dtype='float32',shape=(n,2,1250))
        valid=np.ones(n,dtype=bool)
        for start in range(0,n,1024):
            end=min(start+1024,n);x=np.asarray(ds[:,0:2,start:end],dtype='float32').transpose(2,1,0)
            span=np.ptp(x,axis=2,keepdims=True);ok=np.isfinite(x).all((1,2)) & (span[:,:,0]>1e-8).all(1)
            valid[start:end]=ok
            waves[start:end]=np.nan_to_num((x-x.min(axis=2,keepdims=True))/np.maximum(span,1e-8))
            if start%51200==0:print(name,'waveforms',end,'/',n,flush=True)
        waves.flush()
        y=np.stack([np.asarray(g[k]).reshape(-1) for k in ['SBP','DBP']],axis=1)
        refs=np.asarray(g['Subject']).reshape(-1)
        subjects=[]
        for ref in refs:
            subjects.append(''.join(chr(int(c)) for c in np.asarray(f[ref]).reshape(-1)))
        subjects=np.asarray(subjects)
        valid &= np.isfinite(y).all(1)&(y[:,0]>y[:,1])
        np.save(dest/'y.npy',y.astype('float32'));np.save(dest/'subjects.npy',subjects);np.save(dest/'valid.npy',valid)
        (dest/'complete.json').write_text(json.dumps({'file':path.name,'rows':n,'subjects':len(np.unique(subjects)),'invalid':int((~valid).sum())},indent=2))
        print(name,'converted',n,'subjects',len(np.unique(subjects)),flush=True)

def chunk_features(args):
    path,start,end=args;x=np.load(path,mmap_mode='r')
    output=[]
    for row in x[start:end]:
        try:output.append(extract(row))
        except ValueError:output.append(np.zeros(122,dtype='float32'))
    return start,np.array(output)

def featurize(name):
    dest=DATA/name
    if (dest/'features_complete.json').exists():return
    path=dest/'waves.npy';w=np.load(path,mmap_mode='r');n=len(w)
    example,names=extract(w[0],True);print('features',len(example),flush=True)
    X=np.lib.format.open_memmap(dest/'features.npy',mode='w+',dtype='float32',shape=(n,len(example)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for start,xx in pool.map(chunk_features,[(path,s,min(s+256,n)) for s in range(0,n,256)]):
            X[start:start+len(xx)]=xx
            if start%25600==0:print(name,'features',start,'/',n,flush=True)
    X.flush();(DATA/'feature_names.json').write_text(json.dumps(names,indent=2))
    (dest/'features_complete.json').write_text(json.dumps({'rows':n,'columns':len(example)}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',action='store_true');a=p.parse_args()
    matdir=DATA/'official';matdir.mkdir(exist_ok=True)
    mapping={'VitalDB_Train_Subset.mat':'train','VitalDB_CalFree_Test_Subset.mat':'test','VitalDB_AAMI_Test_Subset.mat':'aami'}
    with zipfile.ZipFile(DATA/'pulsedb_official_subsets.zip') as z:
        for member in z.infolist():
            basename=Path(member.filename).name
            if basename not in mapping:continue
            target=matdir/basename
            if not target.exists() or target.stat().st_size!=member.file_size:
                print('extracting',basename,flush=True)
                with z.open(member) as src,target.open('wb') as out:
                    import shutil;shutil.copyfileobj(src,out,1024*1024*8)
            convert(target,mapping[basename])
    subjects=np.load(DATA/'train/subjects.npy');unique=np.unique(subjects)
    rng=np.random.default_rng(20260923);rng.shuffle(unique);nv=round(len(unique)*.15)
    split={'validation':unique[:nv].tolist(),'train':unique[nv:].tolist(),'seed':20260923}
    (DATA/'split.json').write_text(json.dumps(split,indent=2))
    test=np.load(DATA/'test/subjects.npy');aami=np.load(DATA/'aami/subjects.npy')
    assert not set(test)&set(subjects);assert not set(aami)&set(subjects)
    if a.features:
        for name in ['train','test','aami']:featurize(name)

if __name__=='__main__':main()
