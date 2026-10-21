"""Single feature schema. Raw quality failures are errors; undefined statistics are NaN."""
import numpy as np
from scipy import signal, stats

RELEASE_VERSION = 'fog-unified-1.2-no-dwt'
BASE = ['Mean','RMS','Std','Variance','Min','Max','SMA','Dominant_Frequency',
        'Total_Power','Loco_Power','Freeze_Power','Freeze_Index','Low_Freq_Ratio','PSE']
GROUPS = {
    'jerk': 'Jerk RMS (g/s)', 'sampen': 'Sample Entropy',
    'apen': 'Approximate Entropy', 'autocorr': 'Autocorrelation peak + lag',
    'kurtosis': 'Excess kurtosis', 'crest': 'Centered crest factor',
    'corr': 'Within-sensor Pearson correlation (lag 0)',
    'crosscorr': 'Between-sensor magnitude cross-correlation',
    'ater': 'Log Ankle-to-Trunk energy ratio (loco / freeze)',
}
AXES = ('Forward','Vertical','Lateral')
SENSORS = ('Ankle','Thigh','Trunk')
PER_AXIS = {'jerk':['Jerk_RMS'], 'sampen':['Sample_Entropy'],
            'apen':['Approx_Entropy'], 'autocorr':['Autocorr_Peak','Autocorr_Lag_s'],
            'kurtosis':['Kurtosis_Excess'], 'crest':['Crest_Factor_Centered']}
DEFAULTS = dict(entropy_m=2, entropy_r_ratio=0.2, autocorr_min_s=0.35,
                autocorr_max_s=1.0, crosscorr_max_s=0.25, ater_epsilon_g2=1e-8)

def options(config):
    v=config.get('feature_options', {})
    if not isinstance(v,dict) or set(v)-set(GROUPS)-set(DEFAULTS):
        raise ValueError('Feature options không hợp lệ; chỉ hỗ trợ các nhóm trong registry.')
    out=dict(DEFAULTS)
    for k in GROUPS:
        if k in v and not isinstance(v[k],bool): raise ValueError(k+' cần true/false')
        out[k]=v.get(k,False)
    out.update({k:v[k] for k in DEFAULTS if k in v})
    if not isinstance(out['entropy_m'],int) or not 1<=out['entropy_m']<=3: raise ValueError('entropy_m phải 1..3')
    for k in DEFAULTS:
        if not np.isfinite(out[k]): raise ValueError('Feature parameter không hữu hạn: '+k)
    if not 0<out['entropy_r_ratio']<=1: raise ValueError('Entropy tolerance phải trong (0,1]')
    if not 0<out['autocorr_min_s']<out['autocorr_max_s']: raise ValueError('Autocorr lag không hợp lệ')
    if not 0<out['crosscorr_max_s']<=1 or out['ater_epsilon_g2']<=0: raise ValueError('Cross-correlation / ATER parameter không hợp lệ')
    return out

def axis_names(config):
    o=options(config)
    return BASE+[f for k,fs in PER_AXIS.items() if o[k] for f in fs]

def pairs(config):
    ss=config['sensors']
    return [(a,b) for i,a in enumerate(ss) for b in ss[i+1:]]

def window_names(config):
    o=options(config); out=[]
    if o['corr']:
        for s in config['sensors']:
            aa=config['axes']
            out.extend(f'{s}_Corr_{a}_{b}' for i,a in enumerate(aa) for b in aa[i+1:])
    if o['crosscorr']:
        if len(config['sensors'])<2 or set(config['axes'])!=set(AXES):
            raise ValueError('Cross-correlation cần >=2 vị trí và đủ 3 trục.')
        for a,b in pairs(config): out.extend([f'{a}_{b}_Mag_XCorr', f'{a}_{b}_Mag_XCorr_Lag_s'])
    if o['ater']:
        if not {'Ankle','Trunk'}.issubset(config['sensors']) or set(config['axes'])!=set(AXES):
            raise ValueError('ATER cần Ankle + Trunk và đủ 3 trục; dùng Batch Configuration.')
        out.extend(['Ankle_Trunk_LogATER_Loco','Ankle_Trunk_LogATER_Freeze'])
    return out

def feature_names(config):
    return [f'{s}_{a}_{f}' for s in config['sensors'] for a in config['axes']
            for f in axis_names(config)]+window_names(config)

def entropy(x,m,r_ratio,approx=False):
    sd=float(np.std(x)); n=len(x)
    if n<m+2 or sd<=1e-12: return np.nan
    # Standardize first: Chebyshev tolerance is r_ratio * window SD.
    x=(x-np.mean(x))/sd
    if approx:
        def phi(k):
            v=np.lib.stride_tricks.sliding_window_view(x,k)
            counts=np.zeros(len(v))
            for start in range(0,len(v),32):
                dist=np.max(np.abs(v[start:start+32,None,:]-v[None,:,:]),axis=2)
                counts[start:start+32]=np.sum(dist<=r_ratio,axis=1)
            return float(np.mean(np.log(counts/len(v))))
        return phi(m)-phi(m+1)
    v=np.lib.stride_tricks.sliding_window_view(x,m+1); B=A=0
    for i in range(len(v)-1):
        diff=np.abs(v[i+1:]-v[i]); match=np.max(diff[:,:m],axis=1)<=r_ratio
        B+=int(match.sum()); A+=int((match & (diff[:,m]<=r_ratio)).sum())
    return float(-np.log(A/B)) if A>0 and B>0 else np.nan

def axis_extra(x,fs,config):
    o=options(config); x=np.asarray(x,float)
    if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError('Raw window chứa NaN/Inf hoặc sai chiều.')
    xc=x-x.mean(); energy=float(xc@xc); flat=energy/len(x)<=1e-24; out={}
    if o['jerk']: out['Jerk_RMS']=float(np.sqrt(np.mean((np.diff(x)*fs)**2)))
    if o['sampen']: out['Sample_Entropy']=entropy(x,o['entropy_m'],o['entropy_r_ratio'])
    if o['apen']: out['Approx_Entropy']=entropy(x,o['entropy_m'],o['entropy_r_ratio'],True)
    if o['kurtosis']: out['Kurtosis_Excess']=float(stats.kurtosis(x,fisher=True,bias=False)) if not flat else np.nan
    if o['crest']: out['Crest_Factor_Centered']=float(np.max(np.abs(xc))/np.sqrt(energy/len(x))) if not flat else np.nan
    if o['autocorr']:
        lo=max(1,int(np.ceil(o['autocorr_min_s']*fs))); hi=min(len(x)//2,int(np.floor(o['autocorr_max_s']*fs)))
        peak=lag=np.nan
        if not flat and hi>=lo:
            ac=signal.correlate(xc,xc,mode='full',method='auto')[len(x)-1:]/energy
            k=lo+int(np.argmax(ac[lo:hi+1]));peak=float(ac[k]);lag=k/fs
        out.update(Autocorr_Peak=peak,Autocorr_Lag_s=lag)
    return out

def corr(u,v):
    u=np.asarray(u,float);v=np.asarray(v,float);u=u-u.mean();v=v-v.mean()
    den=np.sqrt((u@u)*(v@v))
    return float(np.clip((u@v)/den,-1,1)) if den>1e-24 else np.nan

def window_extra(signals,fs,config,channel_values):
    o=options(config);out={}
    if o['corr']:
        for s in config['sensors']:
            aa=config['axes']
            for i,a in enumerate(aa):
                for b in aa[i+1:]: out[f'{s}_Corr_{a}_{b}']=corr(signals[f'{s}_{a}_mg'],signals[f'{s}_{b}_mg'])
    if o['crosscorr']:
        mags={}
        for s in config['sensors']:
            x=np.column_stack([signals[f'{s}_{a}_mg'] for a in AXES]); x=x-x.mean(axis=0)
            mags[s]=np.linalg.norm(x,axis=1)
        for a,b in pairs(config):
            u=mags[a];v=mags[b]; lim=min(int(o['crosscorr_max_s']*fs),len(u)//2)
            ks=np.arange(-lim,lim+1);vals=[]
            # Positive lag pairs a[n] with b[n+k]. At least half a window overlaps.
            for k in ks:
                uu,vv=(u[:len(u)-k],v[k:]) if k>=0 else (u[-k:],v[:len(v)+k])
                vals.append(corr(uu,vv))
            vals=np.asarray(vals); peak=lag=np.nan
            if np.isfinite(vals).any():
                j=int(np.nanargmax(abs(vals)));peak=float(vals[j]);lag=float(ks[j]/fs)
            out.update({f'{a}_{b}_Mag_XCorr':peak,f'{a}_{b}_Mag_XCorr_Lag_s':lag})
    if o['ater']:
        for band,key in [('Loco','Loco_Power'),('Freeze','Freeze_Power')]:
            ea=sum(channel_values[('Ankle',a)][key] for a in AXES)
            et=sum(channel_values[('Trunk',a)][key] for a in AXES)
            ep=o['ater_epsilon_g2'];out[f'Ankle_Trunk_LogATER_{band}']=float(np.log((ea+ep)/(et+ep)))
    return out
