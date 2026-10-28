"""Single feature schema. Raw quality failures are errors; undefined statistics are NaN."""
import numpy as np
from scipy import signal, stats

RELEASE_VERSION = 'fog-unified-1.4-cwt'
BASE = ['Mean','RMS','Std','Variance','Min','Max','SMA','Dominant_Frequency',
        'Total_Power','Loco_Power','Freeze_Power','Freeze_Index','Low_Freq_Ratio','PSE']
GROUPS = {
    'skewness': 'Skewness', 'spectral_shape': 'Spectral centroid / spread / flatness',
    'harmonic': 'Spectral even / odd harmonic ratio', 'wavelet': 'DWT energy / std / entropy',
    'magnitude': 'Magnitude Mean / RMS / FI / Jerk', 'temporal': 'Temporal delta / RMS trend',
    'jerk': 'Jerk RMS (g/s)', 'sampen': 'Sample Entropy',
    'apen': 'Approximate Entropy', 'autocorr': 'Autocorrelation peak + lag',
    'kurtosis': 'Excess kurtosis', 'crest': 'Centered crest factor',
    'corr': 'Within-sensor Pearson correlation (lag 0)',
    'crosscorr': 'Between-sensor magnitude cross-correlation',
    'ater': 'Log Ankle-to-Trunk energy ratio (loco / freeze)',
    'cwt': 'CWT db4 FOG index (signed + absolute)',
}
AXES = ('Forward','Vertical','Lateral')
SENSORS = ('Ankle','Thigh','Trunk')
PER_AXIS = {'jerk':['Jerk_RMS'], 'sampen':['Sample_Entropy'],
            'apen':['Approx_Entropy'], 'autocorr':['Autocorr_Peak','Autocorr_Lag_s'],
            'skewness':['Skewness'], 'spectral_shape':['Spectral_Centroid','Spectral_Spread','Spectral_Flatness'],
            'harmonic':['Harmonic_Even_Odd_Ratio'], 'kurtosis':['Kurtosis_Excess'], 'crest':['Crest_Factor_Centered']}
DEFAULTS = dict(entropy_m=2, entropy_r_ratio=0.2, autocorr_min_s=0.35,
                autocorr_max_s=1.0, crosscorr_max_s=0.25, ater_epsilon_g2=1e-8, wavelet_name='db4', wavelet_level=4,
                wavelet_mode='symmetric', magnitude_mode='dynamic', harmonic_f0_hz=1.0,
                harmonic_count=8, temporal_history=3)

def options(config):
    v=config.get('feature_options', {})
    if not isinstance(v,dict) or set(v)-set(GROUPS)-set(DEFAULTS)-{'base_features'}:
        raise ValueError('Feature options không hợp lệ; chỉ hỗ trợ các nhóm trong registry.')
    out=dict(DEFAULTS)
    base=v.get('base_features',BASE)
    if not isinstance(base,list) or any(k not in BASE for k in base) or len(base)!=len(set(base)):
        raise ValueError('Danh sách đặc trưng cơ bản không hợp lệ.')
    out['base_features']=[k for k in BASE if k in base]
    for k in GROUPS:
        if k in v and not isinstance(v[k],bool): raise ValueError(k+' cần true/false')
        out[k]=v.get(k,False)
    out.update({k:v[k] for k in DEFAULTS if k in v})
    if not isinstance(out['entropy_m'],int) or not 1<=out['entropy_m']<=3: raise ValueError('entropy_m phải 1..3')
    for k in DEFAULTS:
        if isinstance(DEFAULTS[k],str): continue
        if not isinstance(out[k],(int,float)) or isinstance(out[k],bool) or not np.isfinite(out[k]): raise ValueError('Feature parameter không hữu hạn: '+k)
    if not 0<out['entropy_r_ratio']<=1: raise ValueError('Entropy tolerance phải trong (0,1]')
    if not 0<out['autocorr_min_s']<out['autocorr_max_s']: raise ValueError('Autocorr lag không hợp lệ')
    if not 0<out['crosscorr_max_s']<=1 or out['ater_epsilon_g2']<=0: raise ValueError('Cross-correlation / ATER parameter không hợp lệ')
    if out['wavelet_name'] not in ('db4','db8') or out['wavelet_mode'] not in ('symmetric','periodization'):
        raise ValueError('Wavelet / extension mode không hợp lệ.')
    for key,lo,hi in [('wavelet_level',1,5),('harmonic_count',2,20),('temporal_history',2,20)]:
        if type(out[key]) is not int or not lo<=out[key]<=hi: raise ValueError(key+' ngoài giới hạn.')
    if out['harmonic_f0_hz']<=0: raise ValueError('Harmonic f0 phải >0.')
    if out['magnitude_mode'] not in ('dynamic','input'): raise ValueError('Magnitude mode không hợp lệ.')
    return out

def axis_names(config):
    o=options(config)
    out=o['base_features']+[f for k,fs in PER_AXIS.items() if o[k] for f in fs]
    if o['wavelet']:
        out += [f'Wavelet_D{level}_{stat}' for level in range(1,o['wavelet_level']+1) for stat in ('Energy','Relative_Energy','Std','Entropy')]
        if o['wavelet_level']>=4: out.append('Wavelet_D3_D4_Energy_Ratio')
    if o['cwt']:
        from cwt_features import FEATURE_NAMES
        out += list(FEATURE_NAMES)
    return out

def pairs(config):
    ss=config['sensors']
    return [(a,b) for i,a in enumerate(ss) for b in ss[i+1:]]

def window_names(config):
    o=options(config); out=[]
    if o['magnitude']:
        if set(config['axes'])!=set(AXES): raise ValueError('Magnitude cần đủ 3 trục.')
        out.extend(f'{s}_Magnitude_{key}' for s in config['sensors'] for key in ('Mean','RMS','Freeze_Index','Jerk_RMS'))
    if o['temporal']:
        out.extend(f'{s}_{a}_{key}' for s in config['sensors'] for a in config['axes'] for key in ('Delta_RMS','Delta_Freeze_Index','RMS_Trend_g_s'))
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
    if len(x)<2 or not np.isfinite(fs) or fs<=0: raise ValueError('Window hoặc Fs không hợp lệ.')
    if o['skewness']: out['Skewness']=float(stats.skew(x,bias=False)) if not flat else np.nan
    if o['spectral_shape']:
        f,p=window_psd(x,fs);mask=(f>=.5)&(f<=min(16.,fs/2));f=f[mask];p=p[mask]
        total=float(p.sum())
        if total>1e-24:
            centroid=float(np.sum(f*p)/total)
            out.update(Spectral_Centroid=centroid,Spectral_Spread=float(np.sqrt(np.sum((f-centroid)**2*p)/total)),
                       Spectral_Flatness=float(np.exp(np.mean(np.log(np.maximum(p,1e-300))))/np.mean(p)))
        else: out.update(Spectral_Centroid=np.nan,Spectral_Spread=np.nan,Spectral_Flatness=np.nan)
    if o['harmonic']:
        f=np.fft.rfftfreq(len(x),1/fs);amp=np.abs(np.fft.rfft(xc));h=np.arange(1,o['harmonic_count']+1);freq=h*o['harmonic_f0_hz']
        valid=freq<fs/2;h=h[valid];freq=freq[valid]
        a=np.interp(freq,f,amp);odd=float(a[h%2==1].sum());even=float(a[h%2==0].sum())
        out['Harmonic_Even_Odd_Ratio']=even/odd if odd>1e-12 and np.any(h%2==0) and not flat else np.nan
    if o['wavelet']:
        pywt=wavelet_module();w=pywt.Wavelet(o['wavelet_name']);level=o['wavelet_level']
        if level>pywt.dwt_max_level(len(x),w.dec_len): raise ValueError(f'DWT {w.name} level {level} vượt mức hữu ích cho {len(x)} mẫu. Giảm level hoặc tăng window.')
        coeffs=pywt.wavedec(xc,w,mode=o['wavelet_mode'],level=level);den=sum(float(c@c) for c in coeffs);energies={}
        for j in range(1,level+1):
            c=coeffs[-j];detail_energy=float(c@c);energies[j]=detail_energy;prob=c*c/detail_energy if detail_energy>1e-24 else np.array([]);prob=prob[prob>0]
            prefix=f'Wavelet_D{j}_'
            out.update({prefix+'Energy':detail_energy,prefix+'Relative_Energy':detail_energy/den if den>1e-24 else np.nan,
                        prefix+'Std':float(np.std(c)),prefix+'Entropy':float(-np.sum(prob*np.log2(prob))) if len(prob) else np.nan})
        if level>=4: out['Wavelet_D3_D4_Energy_Ratio']=energies[3]/energies[4] if energies[4]>1e-24 else np.nan
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
    if o['cwt']:
        from cwt_features import features as cwt_features
        out.update(cwt_features(x,fs))
    return out

def corr(u,v):
    u=np.asarray(u,float);v=np.asarray(v,float);u=u-u.mean();v=v-v.mean()
    den=np.sqrt((u@u)*(v@v))
    return float(np.clip((u@v)/den,-1,1)) if den>1e-24 else np.nan

def window_extra(signals,fs,config,channel_values):
    o=options(config);out={}
    if o['magnitude']:
        for sensor in config['sensors']:
            x=np.column_stack([signals[f'{sensor}_{a}_mg'] for a in AXES])
            if o['magnitude_mode']=='dynamic': x=x-x.mean(axis=0)
            vm=np.linalg.norm(x,axis=1);f,p=window_psd(vm,fs)
            loco=band_power(f,p,.5,3.);freeze=band_power(f,p,3.,8.)
            values=dict(Mean=float(vm.mean()),RMS=float(np.sqrt(np.mean(vm*vm))),
                        Freeze_Index=freeze/loco if loco>0 else 0.,Jerk_RMS=float(np.sqrt(np.mean((np.diff(vm)*fs)**2))))
            out.update({f'{sensor}_Magnitude_{k}':v for k,v in values.items()})
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


def wavelet_module():
    try:
        import pywt
    except ImportError as exc:
        raise ValueError('DWT cần PyWavelets: python -m pip install PyWavelets') from exc
    return pywt


def window_psd(x,fs):
    x=np.asarray(x,float);x=x-x.mean();n=len(x);nperseg=min(64,n)
    return signal.welch(x,fs=fs,window='hann',nperseg=nperseg,noverlap=nperseg//2,nfft=max(256,n),detrend=False,scaling='density')


def band_power(f,p,lo,hi):
    mask=(f>=lo)&(f<=hi)
    if mask.sum()<2: return 0.
    return float(np.trapezoid(p[mask],f[mask]) if hasattr(np,'trapezoid') else np.trapz(p[mask],f[mask]))


def temporal_extra(history,current,start,config,fs):
    o=options(config);out={}
    if not o['temporal']: return out
    for sensor in config['sensors']:
        for axis in config['axes']:
            key=(sensor,axis);now=current[key];previous=history[-1][1][key] if history else None
            prefix=f'{sensor}_{axis}_'
            out[prefix+'Delta_RMS']=now['RMS']-previous['RMS'] if previous is not None else np.nan
            out[prefix+'Delta_Freeze_Index']=now['Freeze_Index']-previous['Freeze_Index'] if previous is not None else np.nan
            recent=(history+[(start,current)])[-o['temporal_history']:]
            if len(recent)<2: slope=np.nan
            else:
                t=np.asarray([h[0] for h in recent]);v=np.asarray([h[1][key]['RMS'] for h in recent]);t=t-t.mean()
                slope=float(t@(v-v.mean())/(t@t)) if t@t>0 else np.nan
            out[prefix+'RMS_Trend_g_s']=slope
    return out
