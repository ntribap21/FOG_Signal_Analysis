"""Shared DAPHNET processing. No GUI imports; raw data are never modified."""
from dataclasses import dataclass, asdict, field
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy import signal

SENSORS = ('Ankle', 'Thigh', 'Trunk')
AXES = ('Forward', 'Vertical', 'Lateral')
CHANNELS = tuple(f'{s}_{a}_mg' for s in SENSORS for a in AXES)
COLUMNS = ('Time_ms', *CHANNELS, 'Label')
FEATURES = ('Mean', 'RMS', 'Std', 'Variance', 'Min', 'Max', 'SMA',
            'Dominant_Frequency', 'Total_Power', 'Loco_Power', 'Freeze_Power',
            'Freeze_Index', 'Low_Freq_Ratio', 'PSE')

@dataclass
class Config:
    source_fs: float = 64.0
    selection: str = 'experiment'  # experiment / time / all
    start_s: float = 0.0
    end_s: float = 0.0
    exclude_zero: bool = True
    gap_periods: float = 1.5
    resample: bool = False
    target_fs: float = 64.0
    filter_enabled: bool = False
    filter_kind: str = 'lowpass'
    order: int = 4
    cutoff_low: float = 10.0
    cutoff_high: float = 15.0
    filter_mode: str = 'zero_phase'
    detrend: str = 'none'
    sensors: list = field(default_factory=lambda: ['Ankle'])
    axes: list = field(default_factory=lambda: list(AXES))
    window_s: float = 2.0
    overlap: float = 50.0
    label_rule: str = 'majority'
    fog_threshold: float = 0.5

    def validate(self):
        if self.source_fs != 64: raise ValueError('DAPHNET input sampling rate must be 64 Hz.')
        if self.selection not in ('experiment', 'time', 'all'): raise ValueError('Invalid selection.')
        if self.selection == 'time' and (self.start_s < 0 or self.end_s <= self.start_s):
            raise ValueError('End time must exceed start time.')
        if not 1 < self.gap_periods <= 10: raise ValueError('Invalid gap threshold.')
        if not self.sensors or any(s not in SENSORS for s in self.sensors): raise ValueError('Select a sensor.')
        if not self.axes or any(a not in AXES for a in self.axes): raise ValueError('Select an axis.')
        if len(set(self.sensors)) != len(self.sensors) or len(set(self.axes)) != len(self.axes): raise ValueError('Duplicate channels.')
        fs = self.target_fs if self.resample else self.source_fs
        if not np.isfinite(fs) or fs <= 0: raise ValueError('Invalid target sampling rate.')
        if self.filter_mode not in ('zero_phase','causal'): raise ValueError('Invalid filter mode.')
        if self.filter_enabled:
            if self.filter_kind not in ('lowpass','highpass','bandpass'): raise ValueError('Invalid filter type.')
            if not 1 <= self.order <= 12: raise ValueError('Filter order must be 1–12.')
            if not 0 < self.cutoff_low < fs/2: raise ValueError('Cutoff must be below Nyquist.')
            if self.filter_kind == 'bandpass' and not self.cutoff_low < self.cutoff_high < fs/2:
                raise ValueError('Band-pass requires 0 < low < high < Nyquist.')
        if self.detrend not in ('none','constant','linear'): raise ValueError('Invalid detrend.')
        if not np.isfinite(self.window_s) or round(self.window_s*fs) < 2: raise ValueError('Window must contain at least 2 samples.')
        if not 0 <= self.overlap < 100: raise ValueError('Overlap must be 0–<100%.')
        if self.label_rule not in ('majority','threshold','strict'): raise ValueError('Invalid label rule.')
        if not 0 < self.fog_threshold <= 1: raise ValueError('FOG threshold must be >0 and <=1.')
        return self

    def dictionary(self):
        d = asdict(self)
        d.update(schema_version=2, feature_version='daphnet14-v3', units='g',
                 axis_order=list(AXES), label_mapping={'0':None,'1':0,'2':1},
                 causal_initialization='sosfilt_zi * first sample; reset per selected continuous run',
                 processing_order=['selection','split_runs','resample','filter','detrend','window','features'],
                 majority_tie='Non-FOG', feature_layout='one window per row',
                 detrend_scope='selected continuous run', standardized=False)
        # Canonicalize inactive settings so equivalent datasets share an ID.
        effective = dict(d)
        if self.selection != 'time': effective.update(start_s=None,end_s=None)
        if not self.resample: effective['target_fs'] = self.source_fs
        if not self.filter_enabled:
            for k in ('filter_kind','order','cutoff_low','cutoff_high','filter_mode','causal_initialization'): effective[k]=None
        elif self.filter_kind != 'bandpass': effective['cutoff_high']=None
        if self.label_rule != 'threshold': effective['fog_threshold']=None
        d['config_id'] = hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()[:16]
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**{k:v for k,v in d.items() if k in cls.__dataclass_fields__}).validate()

@dataclass
class Recording:
    path: str
    subject: str
    recording: str
    frame: pd.DataFrame

@dataclass
class Window:
    window_id: int
    segment_id: int
    time: np.ndarray
    values: np.ndarray  # samples x channels, g
    raw_values: np.ndarray  # sampled at the processed grid only for plot comparison
    label: int
    fog_ratio: float


def read_daphnet(path):
    path = Path(path)
    if path.suffix.lower() == '.txt':
        df = pd.read_csv(path, sep=r'\s+', header=None)
        if df.shape[1] != 11: raise ValueError(f'{path.name}: expected 11 columns, got {df.shape[1]}.')
        df.columns = COLUMNS
    elif path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        missing = set(COLUMNS)-set(df.columns)
        if missing: raise ValueError('CSV must contain raw DAPHNET columns: '+', '.join(sorted(missing)))
        df = df[list(COLUMNS)].copy()
    else: raise ValueError('Use a DAPHNET .txt or a raw .csv with matching column names.')
    if len(df)<2: raise ValueError('Recording must have at least two samples.')
    for col in COLUMNS: df[col] = pd.to_numeric(df[col],errors='raise')
    if not np.isfinite(df[list(COLUMNS)].to_numpy(float)).all(): raise ValueError('NaN/Inf found in raw recording.')
    if not df.Label.isin([0,1,2]).all(): raise ValueError('DAPHNET labels must be 0, 1, 2.')
    delta = np.diff(df.Time_ms.to_numpy(float))
    if np.any(delta<=0): raise ValueError('Timestamps must be strictly increasing.')
    regular=delta[delta<=1.5*1000/64]
    if not len(regular) or not 14<=np.median(regular)<=17: raise ValueError('Timestamp spacing does not match 64 Hz DAPHNET.')
    df['Original_Label'] = df.Label.astype(np.int8)
    df['Binary_Label'] = df.Label.map({1:0.0,2:1.0})
    df['Valid_Experiment'] = df.Label.isin([1,2])
    name=path.stem; match=re.match(r'^(S\d+)R\d+$',name,re.I)
    return Recording(str(path.resolve()), match.group(1).upper() if match else 'unknown', name, df)


def select_runs(recording, config):
    config.validate(); df=recording.frame
    t=df.Time_ms.to_numpy(float)/1000
    keep=np.ones(len(df),bool)
    if config.selection=='time':
        if config.start_s < t[0]-1e-9 or config.end_s > t[-1]+1e-9:
            raise ValueError(f'{recording.recording}: selected time range is outside the recording.')
        keep &= (t>=config.start_s)&(t<=config.end_s)
    if config.selection=='experiment' or config.exclude_zero: keep &= df.Valid_Experiment.to_numpy()
    idx=np.flatnonzero(keep)
    if not len(idx): raise ValueError('No samples remain after selection.')
    valid=df.Valid_Experiment.to_numpy()[idx]
    breaks=np.r_[True,(np.diff(idx)>1)|(np.diff(t[idx])>config.gap_periods/config.source_fs)|(valid[1:]!=valid[:-1])]
    boundaries=np.r_[np.flatnonzero(breaks),len(idx)]
    return [(i+1,df.iloc[idx[a:b]].copy().reset_index(drop=True)) for i,(a,b) in enumerate(zip(boundaries[:-1],boundaries[1:]))]


def segment_summary(runs):
    return pd.DataFrame([{'Segment_ID':sid,'Start_s':float(d.Time_ms.iloc[0]/1000),
                         'End_s':float(d.Time_ms.iloc[-1]/1000),'Samples':len(d),
                         'Duration_s':(len(d)/64), 'Outside':int((d.Label==0).sum()),
                         'Non_FOG':int((d.Label==1).sum()),'FOG':int((d.Label==2).sum())} for sid,d in runs])


def preprocess(runs, config):
    config.validate(); output=[]; fs=config.target_fs if config.resample else config.source_fs
    sos=None
    if config.filter_enabled:
        cut=[config.cutoff_low,config.cutoff_high] if config.filter_kind=='bandpass' else config.cutoff_low
        sos=signal.butter(config.order,cut,fs=fs,btype=config.filter_kind,output='sos')
    ratio=Fraction(float(fs/config.source_fs)).limit_denominator(1000)
    if not np.isclose(config.source_fs*ratio.numerator/ratio.denominator,fs): raise ValueError('Cannot represent resample ratio exactly.')
    for sid,frame in runs:
        d=frame.copy(); x=d[list(CHANNELS)].to_numpy(float)
        t=d.Time_ms.to_numpy(float)/1000
        if config.resample and not np.isclose(fs,config.source_fs):
            x=signal.resample_poly(x,ratio.numerator,ratio.denominator,axis=0)
            new_t=t[0]+np.arange(len(x))/fs
            keep=new_t<=t[-1]+1e-9; new_t=new_t[keep]; x=x[keep]
            # Nearest labels on the same continuous run, never interpolated as numbers.
            right=np.searchsorted(t,new_t).clip(0,len(t)-1); left=np.maximum(right-1,0)
            nearest=np.where(np.abs(t[left]-new_t)<=np.abs(t[right]-new_t),left,right)
            d=frame.iloc[nearest].copy().reset_index(drop=True);d['Time_ms']=new_t*1000
        if sos is not None:
            if config.filter_mode=='zero_phase':
                try: x=signal.sosfiltfilt(sos,x,axis=0)
                except ValueError as exc: raise ValueError(f'Segment {sid} too short for zero-phase; choose causal or disable filter.') from exc
            else:
                zi=signal.sosfilt_zi(sos)[:,:,None]*x[0][None,None,:]
                x,_=signal.sosfilt(sos,x,axis=0,zi=zi)
        if config.detrend!='none':
            if len(x)<2 and config.detrend=='linear': raise ValueError('Linear detrend requires at least 2 samples per run.')
            x=signal.detrend(x,axis=0,type=config.detrend)
        d[list(CHANNELS)]=x;d['Segment_ID']=sid;output.append((sid,d))
    return output,fs


def channel_names(config): return [f'{s}_{a}_mg' for s in config.sensors for a in config.axes]


def create_windows(processed_runs, raw_runs, config, fs):
    config.validate(); cols=channel_names(config)
    size=int(round(config.window_s*fs));step=max(1,int(round(size*(1-config.overlap/100))))
    windows=[]; raw_by_id=dict(raw_runs); counts={'excluded_zero_samples':0,'short_segments':0,'tail_samples':0,'rejected_windows':0}
    for sid,d in processed_runs:
        if not d.Label.isin([1,2]).all():
            counts['excluded_zero_samples']+=len(d);continue
        if len(d)<size: counts['short_segments']+=1;counts['tail_samples']+=len(d);continue
        values=d[cols].to_numpy(float)/1000;t=d.Time_ms.to_numpy(float)/1000
        source=raw_by_id[sid];rt=source.Time_ms.to_numpy(float)/1000
        raw=np.column_stack([np.interp(t,rt,source[c].to_numpy(float)/1000) for c in cols])
        last_end=0
        for start in range(0,len(d)-size+1,step):
            stop=start+size;xx=values[start:stop]
            if not np.isfinite(xx).all(): counts['rejected_windows']+=1;continue
            ratio=float((d.Label.iloc[start:stop]==2).mean())
            label=int(ratio>0.5) if config.label_rule=='majority' else int(ratio>=config.fog_threshold) if config.label_rule=='threshold' else int(ratio==1)
            windows.append(Window(len(windows)+1,sid,t[start:stop].copy(),xx.copy(),raw[start:stop].copy(),label,ratio));last_end=stop
        counts['tail_samples']+=len(d)-last_end
    return windows,counts


def compute_features(x,fs):
    x=np.asarray(x,float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all(): raise ValueError('Feature input requires finite 1D data with at least 2 samples.')
    mean=float(x.mean());xc=x-mean;n=len(x)
    freq=np.fft.rfftfreq(n,1/fs);mag=2*np.abs(np.fft.rfft(xc))/n;mag[0]*=.5
    if n%2==0:mag[-1]*=.5
    pf,pp=signal.welch(xc,fs=fs,window='hann',nperseg=min(256,n),detrend=False,scaling='density')
    integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    def band(lo,hi):
        if pf[-1]<hi: return 0.0
        f=np.r_[lo,pf[(pf>lo)&(pf<hi)],hi];p=np.interp(f,pf,pp)
        return float(integrate(p,f))
    total=float(integrate(pp,pf));loco=band(.5,3);freeze=band(3,8)
    prob=pp/pp.sum() if pp.sum()>0 else pp;prob=prob[prob>0]
    feature=dict(zip(FEATURES,[mean,float(np.sqrt(np.mean(x*x))),float(x.std()),float(x.var()),float(x.min()),float(x.max()),float(np.abs(x).mean()),
                  float(freq[1+np.argmax(mag[1:])]) if np.any(mag[1:]>0) else 0.,total,loco,freeze,freeze/loco if loco>1e-15 else 0.,loco/total if total>1e-15 else 0.,float(-np.sum(prob*np.log2(prob))) ]))
    return feature,(freq,mag,pf,pp)


def feature_table(recording, windows, config, fs):
    if fs<16: raise ValueError('Feature extraction requires Fs >=16 Hz for the 3–8 Hz freeze band.')
    rows=[];cols=channel_names(config);cid=config.dictionary()['config_id']
    for w in windows:
        row={'Subject_ID':recording.subject,'Recording_ID':recording.recording,'Segment_ID':w.segment_id,
             'Window_ID':w.window_id,'Start_Time':w.time[0],'End_Time':w.time[-1],
             'Label':w.label,'FOG_Ratio':w.fog_ratio,'Config_ID':cid}
        for j,col in enumerate(cols):
            values,_=compute_features(w.values[:,j],fs)
            row.update({f'{col.removesuffix("_mg")}_{key}':value for key,value in values.items()})
        rows.append(row)
    names=[f'{c.removesuffix("_mg")}_{f}' for c in cols for f in FEATURES]
    metadata=['Subject_ID','Recording_ID','Segment_ID','Window_ID','Start_Time','End_Time','Label','FOG_Ratio','Config_ID']
    return pd.DataFrame(rows,columns=metadata+names),names
