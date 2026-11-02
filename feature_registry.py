"""Single feature schema. Raw quality failures are errors; undefined statistics are NaN."""
import numpy as np
from scipy import signal, stats

RELEASE_VERSION = 'fog-unified-1.4-cwt'
BASE = ['Mean','RMS','Std','Variance','Min','Max','SMA','Dominant_Frequency',
        'Total_Power','Loco_Power','Freeze_Power','Freeze_Index','Low_Freq_Ratio','PSE']
RODRIGUEZ_AXIS_FEATURES = [
    'Mean_Locomotor', 'Std_Locomotor', 'Entropy_Locomotor',
    'Mean_Freeze', 'Std_Freeze', 'Entropy_Freeze',
    'Freq_Std_Low', 'Freq_Std_High',
    *[f'AR_{i}' for i in range(1, 13)],
    'Integrals',
    'Peak_Amp_Locomotor', 'Peak_Freq_Locomotor',
    'Peak_Amp_Freeze', 'Peak_Freq_Freeze',
]
RODRIGUEZ_WINDOW_FEATURES = [
    'Spectral_CoM', 'Increments_AP',
    'Diff_Inc_XY', 'Diff_Inc_XZ', 'Diff_Inc_YZ',
    'PCA_1', 'PCA_2', 'PCA_3',
]
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
    'cwt': 'CWT db4 FOG index — Forward/Anterior only (signed + absolute)',
    'rodriguez_axis': 'Rodriguez/Mazilu per-axis (Loco/Freeze stats, Freq Std, AR1-12, Integrals, Peaks)',
    'rodriguez_window': 'Rodriguez/Ahlrichs multi-axis (Spectral CoM 0.1-8Hz, Increments, PCA 1-3)',
}
AXES = ('Forward','Vertical','Lateral')
SENSORS = ('Ankle','Thigh','Trunk')
PER_AXIS = {'jerk':['Jerk_RMS'], 'sampen':['Sample_Entropy'],
            'apen':['Approx_Entropy'], 'autocorr':['Autocorr_Peak','Autocorr_Lag_s'],
            'skewness':['Skewness'], 'spectral_shape':['Spectral_Centroid','Spectral_Spread','Spectral_Flatness'],
            'harmonic':['Harmonic_Even_Odd_Ratio'], 'kurtosis':['Kurtosis_Excess'], 'crest':['Crest_Factor_Centered'],
            'rodriguez_axis': RODRIGUEZ_AXIS_FEATURES}
DEFAULTS = dict(
    entropy_m=2, entropy_r_ratio=0.2, autocorr_min_s=0.35,
    autocorr_max_s=1.0, crosscorr_max_s=0.25, ater_epsilon_g2=1e-8,
    wavelet_name='db4', wavelet_level=4, wavelet_mode='symmetric',
    magnitude_mode='dynamic', harmonic_f0_hz=1.0, harmonic_count=8,
    temporal_history=3, cfs=False, cfs_max_features=30
)

def options(config):
    v = config.get('feature_options', {})
    if not isinstance(v, dict) or set(v) - set(GROUPS) - set(DEFAULTS) - {'base_features'}:
        raise ValueError('Feature options không hợp lệ; chỉ hỗ trợ các nhóm trong registry.')
    out = dict(DEFAULTS)
    base = v.get('base_features', BASE)
    if not isinstance(base, list) or any(k not in BASE for k in base) or len(base) != len(set(base)):
        raise ValueError('Danh sách đặc trưng cơ bản không hợp lệ.')
    out['base_features'] = [k for k in BASE if k in base]
    for k in GROUPS:
        if k in v and not isinstance(v[k], bool):
            raise ValueError(k + ' cần true/false')
        out[k] = v.get(k, False)
    if 'cfs' in v and not isinstance(v['cfs'], bool):
        raise ValueError('cfs cần true/false')
    out.update({k: v[k] for k in DEFAULTS if k in v})
    if not isinstance(out['entropy_m'], int) or not 1 <= out['entropy_m'] <= 3:
        raise ValueError('entropy_m phải 1..3')
    for k in DEFAULTS:
        if isinstance(DEFAULTS[k], (str, bool)):
            continue
        if not isinstance(out[k], (int, float)) or isinstance(out[k], bool) or not np.isfinite(out[k]):
            raise ValueError('Feature parameter không hữu hạn: ' + k)
    if not 0 < out['entropy_r_ratio'] <= 1:
        raise ValueError('Entropy tolerance phải trong (0,1]')
    if not 0 < out['autocorr_min_s'] < out['autocorr_max_s']:
        raise ValueError('Autocorr lag không hợp lệ')
    if not 0 < out['crosscorr_max_s'] <= 1 or out['ater_epsilon_g2'] <= 0:
        raise ValueError('Cross-correlation / ATER parameter không hợp lệ')
    if out['wavelet_name'] not in ('db4', 'db8') or out['wavelet_mode'] not in ('symmetric', 'periodization'):
        raise ValueError('Wavelet / extension mode không hợp lệ.')
    for key, lo, hi in [('wavelet_level', 1, 5), ('harmonic_count', 2, 20), ('temporal_history', 2, 20), ('cfs_max_features', 1, 500)]:
        if type(out[key]) is not int or not lo <= out[key] <= hi:
            raise ValueError(key + ' ngoài giới hạn.')
    if out['harmonic_f0_hz'] <= 0:
        raise ValueError('Harmonic f0 phải >0.')
    if out['magnitude_mode'] not in ('dynamic', 'input'):
        raise ValueError('Magnitude mode không hợp lệ.')
    return out

def axis_names(config):
    o=options(config)
    out=o['base_features']+[f for k,fs in PER_AXIS.items() if o[k] for f in fs]
    if o['wavelet']:
        out += [f'Wavelet_D{level}_{stat}' for level in range(1,o['wavelet_level']+1) for stat in ('Energy','Relative_Energy','Std','Entropy')]
        if o['wavelet_level']>=4: out.append('Wavelet_D3_D4_Energy_Ratio')
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
    if o['cwt']:
        if 'Forward' not in config['axes']:
            raise ValueError('Nhánh CWT yêu cầu phải chọn trục Forward (Anterior-Posterior).')
        from cwt_features import FEATURE_NAMES
        out.extend(f'{s}_Forward_{f}' for s in config['sensors'] for f in FEATURE_NAMES)
    if o['rodriguez_window']:
        if set(config['axes']) != set(AXES):
            raise ValueError('Nhóm Rodriguez/Ahlrichs multi-axis cần chọn đủ 3 trục (Forward, Vertical, Lateral).')
        out.extend(f'{s}_{key}' for s in config['sensors'] for key in RODRIGUEZ_WINDOW_FEATURES)
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

def _ar_yule_walker(xc, order=12):
    n = len(xc)
    if n <= order:
        return [np.nan] * order
    r = np.array([np.dot(xc[:n - k], xc[k:]) / n for k in range(order + 1)], dtype=float)
    if r[0] <= 1e-24 or not np.isfinite(r).all():
        return [np.nan] * order
    a = np.zeros(order + 1, dtype=float)
    e = r[0]
    for k in range(1, order + 1):
        acc = r[k] - np.dot(a[1:k], r[k - 1:0:-1])
        if abs(e) <= 1e-24:
            return [np.nan] * order
        lam = acc / e
        if not np.isfinite(lam) or abs(lam) >= 1.0:
            return [np.nan] * order
        a_new = a.copy()
        a_new[k] = lam
        for j in range(1, k):
            a_new[j] = a[j] - lam * a[k - j]
        a = a_new
        e *= (1.0 - lam * lam)
    return [float(v) for v in a[1:]]

def _band_stats_and_peaks(f, p, lo, hi):
    mask = (f >= lo) & (f <= hi)
    fb, pb = f[mask], p[mask]
    if len(pb) == 0 or float(np.sum(pb)) <= 1e-24:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mean_v = float(np.mean(pb))
    std_v = float(np.std(pb))
    prob = pb / np.sum(pb)
    prob = prob[prob > 0]
    ent_v = float(-np.sum(prob * np.log2(prob))) if len(prob) else 0.0
    idx = int(np.argmax(pb))
    return mean_v, std_v, ent_v, float(pb[idx]), float(fb[idx])

def _freq_std_band(f, p, lo, hi):
    mask = (f >= lo) & (f <= hi)
    fb, pb = f[mask], p[mask]
    total = float(np.sum(pb))
    if len(fb) < 2 or total <= 1e-24:
        return np.nan
    centroid = float(np.sum(fb * pb) / total)
    return float(np.sqrt(np.sum(((fb - centroid) ** 2) * pb) / total))

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
    if o['rodriguez_axis']:
        f_psd, p_psd = window_psd(x, fs)
        m_loco, s_loco, e_loco, pa_loco, pf_loco = _band_stats_and_peaks(f_psd, p_psd, 0.5, 3.0)
        m_free, s_free, e_free, pa_free, pf_free = _band_stats_and_peaks(f_psd, p_psd, 3.0, 8.0)
        f_fft = np.fft.rfftfreq(len(xc), 1.0 / fs)
        p_fft = (np.abs(np.fft.rfft(xc)) ** 2) / len(xc)
        fstd_low = _freq_std_band(f_fft, p_fft, 0.1, 0.68)
        fstd_high = _freq_std_band(f_fft, p_fft, 8.0, min(20.0, fs / 2.0))
        ar_coeffs = _ar_yule_walker(xc, order=12) if not flat else [np.nan] * 12
        integ_fn = getattr(np, 'trapezoid', np.trapezoid)
        integral_val = float(integ_fn(np.abs(xc), dx=1.0 / fs))
        out.update(
            Mean_Locomotor=m_loco, Std_Locomotor=s_loco, Entropy_Locomotor=e_loco,
            Mean_Freeze=m_free, Std_Freeze=s_free, Entropy_Freeze=e_free,
            Freq_Std_Low=fstd_low, Freq_Std_High=fstd_high,
            **{f'AR_{i}': ar_coeffs[i - 1] for i in range(1, 13)},
            Integrals=integral_val,
            Peak_Amp_Locomotor=pa_loco, Peak_Freq_Locomotor=pf_loco,
            Peak_Amp_Freeze=pa_free, Peak_Freq_Freeze=pf_free,
        )
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
    if o['cwt']:
        if 'Forward' not in config['axes']:
            raise ValueError('Nhánh CWT yêu cầu phải chọn trục Forward (Anterior-Posterior).')
        from cwt_features import features as cwt_features
        for s in config['sensors']:
            cwt_vals = cwt_features(signals[f'{s}_Forward_mg'], fs)
            for k, v in cwt_vals.items():
                out[f'{s}_Forward_{k}'] = v
    if o['rodriguez_window']:
        if set(config['axes']) != set(AXES):
            raise ValueError('Nhóm Rodriguez/Ahlrichs multi-axis cần chọn đủ 3 trục (Forward, Vertical, Lateral).')
        for sensor in config['sensors']:
            ax_x = np.asarray(signals[f'{sensor}_Forward_mg'], float)
            ax_y = np.asarray(signals[f'{sensor}_Vertical_mg'], float)
            ax_z = np.asarray(signals[f'{sensor}_Lateral_mg'], float)
            fx, px = window_psd(ax_x, fs)
            _, py = window_psd(ax_y, fs)
            _, pz = window_psd(ax_z, fs)
            p_sum = px + py + pz
            mask_com = (fx >= 0.1) & (fx <= 8.0)
            tot_com = float(np.sum(p_sum[mask_com]))
            spec_com = float(np.sum(fx[mask_com] * p_sum[mask_com]) / tot_com) if tot_com > 1e-24 else np.nan
            inc_x = float(np.mean(np.abs(np.diff(ax_x))))
            inc_y = float(np.mean(np.abs(np.diff(ax_y))))
            inc_z = float(np.mean(np.abs(np.diff(ax_z))))
            mat = np.column_stack([ax_x - ax_x.mean(), ax_y - ax_y.mean(), ax_z - ax_z.mean()])
            cov = (mat.T @ mat) / max(len(ax_x) - 1, 1)
            eigvals = np.linalg.eigvalsh(cov)[::-1]
            eigvals = np.maximum(eigvals, 0.0)
            eig_sum = float(np.sum(eigvals))
            if eig_sum > 1e-24:
                pca1, pca2, pca3 = (float(eigvals[0] / eig_sum), float(eigvals[1] / eig_sum), float(eigvals[2] / eig_sum))
            else:
                pca1 = pca2 = pca3 = np.nan
            out.update({
                f'{sensor}_Spectral_CoM': spec_com,
                f'{sensor}_Increments_AP': inc_x,
                f'{sensor}_Diff_Inc_XY': inc_x - inc_y,
                f'{sensor}_Diff_Inc_XZ': inc_x - inc_z,
                f'{sensor}_Diff_Inc_YZ': inc_y - inc_z,
                f'{sensor}_PCA_1': pca1,
                f'{sensor}_PCA_2': pca2,
                f'{sensor}_PCA_3': pca3,
            })
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
    return float(np.trapezoid(p[mask],f[mask]) if hasattr(np,'trapezoid') else np.trapezoid(p[mask],f[mask]))


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
