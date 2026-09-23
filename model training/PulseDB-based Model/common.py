"""Versioned waveform interface shared by training and inference."""
import numpy as np
from scipy.signal import butter,cheby2,sosfiltfilt,find_peaks,peak_widths
from scipy.stats import skew,kurtosis

FS=125
N=1250
VERSION='pulsedb_filtered_minmax_v2'

def normalize(x):
    x=np.asarray(x,dtype=np.float32)
    if x.shape!=(2,N) or not np.isfinite(x).all():raise ValueError('Expected finite ECG/PPG array shaped (2,1250)')
    spread=np.ptp(x,axis=1,keepdims=True)
    if np.any(spread<1e-8):raise ValueError('Flat waveform')
    return (x-x.min(axis=1,keepdims=True))/spread

def from_raw_125(x):
    """ND adapter; review against hardware/approved acquisition chain before use."""
    x=np.asarray(x,dtype=np.float64)
    if x.shape!=(2,N) or not np.isfinite(x).all():raise ValueError('Expected (2,1250) raw samples at 125 Hz')
    ecg=sosfiltfilt(butter(4,[.5,40],btype='bandpass',fs=FS,output='sos'),x[0])
    ppg=sosfiltfilt(cheby2(4,20,[.5,8],btype='bandpass',fs=FS,output='sos'),x[1])
    return normalize(np.stack([ecg,ppg]))

def extract(x,return_names=False):
    x=normalize(x);vals=[];names=[]
    def add(prefix,vs,ns):
        vals.extend(vs);names.extend(prefix+'_'+n for n in ns)
    for ch,a in zip(['ecg','ppg'],x):
        d=np.diff(a);f=np.fft.rfftfreq(N,1/FS);power=abs(np.fft.rfft(a-a.mean()))**2;total=power.sum()+1e-12
        band=(f>=.5)&(f<=3.5);dom=f[band][np.argmax(power[band])]
        add(ch,[a.mean(),a.std(),skew(a),kurtosis(a),*np.quantile(a,[.05,.1,.25,.5,.75,.9,.95]),np.mean(abs(d)),np.std(d),dom],
            ['mean','std','skew','kurtosis','q05','q10','q25','q50','q75','q90','q95','mean_abs_diff','std_diff','dominant_hz'])
        add(ch,[power[(f>=lo)&(f<hi)].sum()/total for lo,hi in [(.5,1),(1,2),(2,3),(3,4),(4,6),(6,8),(8,15),(15,40)]],
            ['power05_1','power1_2','power2_3','power3_4','power4_6','power6_8','power8_15','power15_40'])
        peaks,_=find_peaks(a,distance=30,prominence=.15)
        rr=np.diff(peaks)/FS
        widths=peak_widths(a,peaks,rel_height=.5)[0]/FS if len(peaks) else np.array([0.])
        add(ch,[len(peaks)/10,np.mean(rr) if len(rr) else 0,np.std(rr) if len(rr) else 0,np.median(widths),np.std(widths)],
            ['peak_rate','rr_mean','rr_std','width50_median','width50_std'])
        beats=[]
        if ch=='ppg':
            # Minima between peaks define full-cycle templates; no ABP-derived landmarks.
            feet=[l+int(np.argmin(a[l:r])) for l,r in zip(peaks[:-1],peaks[1:])]
            for l,r in zip(feet[:-1],feet[1:]):
                if r-l>=25:
                    b=a[l:r+1];b=(b-b.min())/(np.ptp(b)+1e-8)
                    beats.append(np.interp(np.linspace(0,1,32),np.linspace(0,1,len(b)),b))
        else:
            for q in peaks:
                if q>=25 and q+50<N:
                    b=a[q-25:q+50];beats.append(np.interp(np.linspace(0,1,32),np.linspace(0,1,len(b)),b))
        template=np.median(beats,axis=0) if beats else np.zeros(32)
        add(ch,template,[f'template{i:02d}' for i in range(32)])
        add(ch,[len(beats),np.mean(np.std(beats,axis=0)) if beats else 0],['template_count','template_variability'])
    result=np.nan_to_num(np.asarray(vals,dtype=np.float32),nan=0,posinf=0,neginf=0)
    return (result,names) if return_names else result
