"""Single numerical pipeline shared by GUI, batch and CLI. Original formulas preserved."""
RELEASE_VERSION = 'fog-unified-1.2-no-dwt'
import copy
import hashlib
import json
import re
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
from scipy import signal
from feature_registry import BASE, GROUPS, options, axis_names, window_names, feature_names, axis_extra, window_extra
PIPELINE_VERSION = 'original14-unified-v1'
GUI_SENSORS = ('Ankle','Thigh','Trunk')
GUI_AXES = ('Forward','Vertical','Lateral')
GUI_COLUMNS = ['Time_ms']+[f'{s}_{a}_mg' for s in GUI_SENSORS for a in GUI_AXES]+['Label']
class FrozenControl:
    def __init__(self,value): self._value=value
    def value(self): return self._value
    def currentText(self): return self._value
    def isChecked(self): return bool(self._value)
    
class ProcessingEngine:
    def __init__(self,config):
        self.feature_config = config
        for name,value in config['controls'].items(): setattr(self,name,FrozenControl(value))

    def get_sensor_columns(self):

        return [
            "Ankle_Forward_mg",
            "Ankle_Vertical_mg",
            "Ankle_Lateral_mg",
            "Thigh_Forward_mg",
            "Thigh_Vertical_mg",
            "Thigh_Lateral_mg",
            "Trunk_Forward_mg",
            "Trunk_Vertical_mg",
            "Trunk_Lateral_mg"
        ]
    def get_dataframe_time_array(self, dataframe, sampling_rate):

        if "Time_ms" in dataframe.columns:
            return (
                dataframe["Time_ms"]
                .to_numpy(dtype=float) / 1000.0
            )

        if sampling_rate:
            return (
                np.arange(len(dataframe), dtype=float)
                / sampling_rate
            )

        return None
    def nearest_label_values(self, original_time, original_values, new_time):

        indices = np.searchsorted(
            original_time,
            new_time,
            side="left"
        )

        indices = np.clip(
            indices,
            0,
            len(original_time) - 1
        )

        left_indices = np.maximum(indices - 1, 0)

        choose_left = (
            np.abs(new_time - original_time[left_indices])
            <=
            np.abs(new_time - original_time[indices])
        )

        nearest_indices = np.where(
            choose_left,
            left_indices,
            indices
        )

        return original_values[nearest_indices]
    def resample_dataframe(self, dataframe, source_fs, target_fs):

        if source_fs is None or source_fs <= 0:
            raise ValueError(
                "Không xác định được Sampling Rate để Resample."
            )

        if target_fs <= 0:
            raise ValueError(
                "Target Sampling Rate phải lớn hơn 0."
            )

        if len(dataframe) < 2:
            return dataframe.copy()

        original_time = self.get_dataframe_time_array(
            dataframe, source_fs
        )

        if original_time is None:
            raise ValueError("Không tạo được Time array.")

        if np.isclose(source_fs, target_fs):
            return dataframe.copy()

        start_time = float(original_time[0])
        end_time = float(original_time[-1])
        duration = end_time - start_time

        new_count = max(
            2,
            int(round(duration * target_fs)) + 1
        )

        new_time = np.linspace(
            start_time,
            end_time,
            new_count
        )

        ratio = (
            Fraction(float(target_fs)).limit_denominator(1000)
            /
            Fraction(float(source_fs)).limit_denominator(1000)
        )

        up = ratio.numerator
        down = ratio.denominator

        output = pd.DataFrame(
            index=np.arange(new_count)
        )
        output["Time_ms"] = new_time * 1000.0

        for column in [
            c for c in self.get_sensor_columns()
            if c in dataframe.columns
        ]:
            values = dataframe[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Signal '{column}' chứa NaN/Inf."
                )

            resampled_values = signal.resample_poly(
                values,
                up,
                down
            )

            old_index = np.linspace(
                0, 1, len(resampled_values)
            )
            new_index = np.linspace(0, 1, new_count)

            output[column] = np.interp(
                new_index,
                old_index,
                resampled_values
            )

        for column in [
            "Label",
            "Original_Label",
            "Binary_Label"
        ]:
            if column in dataframe.columns:
                output[column] = self.nearest_label_values(
                    original_time,
                    dataframe[column].to_numpy(),
                    new_time
                )

        return output
    def apply_filter_to_dataframe(self, dataframe, fs):

        if fs is None or fs <= 0:
            raise ValueError(
                "Sampling Rate không hợp lệ cho Filter."
            )

        nyquist = fs / 2.0
        filter_type = self.filter_type_combo.currentText()
        order = int(self.filter_order_spinbox.value())
        cutoff_1 = self.cutoff_1_spinbox.value()

        if filter_type == "Low-pass":
            if cutoff_1 <= 0 or cutoff_1 >= nyquist:
                raise ValueError(
                    "Low-pass cutoff phải thỏa 0 < cutoff < Nyquist."
                )
            b, a = signal.butter(
                order, cutoff_1, btype="lowpass", fs=fs
            )

        elif filter_type == "High-pass":
            if cutoff_1 <= 0 or cutoff_1 >= nyquist:
                raise ValueError(
                    "High-pass cutoff phải thỏa 0 < cutoff < Nyquist."
                )
            b, a = signal.butter(
                order, cutoff_1, btype="highpass", fs=fs
            )

        else:
            cutoff_2 = self.cutoff_2_spinbox.value()
            if (
                cutoff_1 <= 0
                or cutoff_2 <= cutoff_1
                or cutoff_2 >= nyquist
            ):
                raise ValueError(
                    "Band-pass cần Cutoff 1 < Cutoff 2 < Nyquist."
                )
            b, a = signal.butter(
                order, [cutoff_1, cutoff_2],
                btype="bandpass", fs=fs
            )

        processed = dataframe.copy()

        for column in self.get_sensor_columns():
            if column not in processed.columns:
                continue

            values = processed[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Signal '{column}' chứa NaN/Inf."
                )

            minimum_length = max(len(a), len(b)) * 3

            if len(values) <= minimum_length:
                filtered = signal.lfilter(b, a, values)
            else:
                filtered = signal.filtfilt(b, a, values)

            processed[column] = filtered

        return processed
    def apply_detrend_to_dataframe(self, dataframe):

        processed = dataframe.copy()
        detrend_type = (
            "linear"
            if self.detrend_type_combo.currentText() == "Linear"
            else "constant"
        )

        for column in self.get_sensor_columns():
            if column not in processed.columns:
                continue

            values = processed[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Signal '{column}' chứa NaN/Inf."
                )

            processed[column] = signal.detrend(
                values,
                type=detrend_type
            )

        return processed
    def assign_window_label(self, binary_values):
        """
        Gán một Binary Label cho một window.

        binary_values phải là mảng 0/1.
        Hàm đồng thời trả về FOG Ratio để hiển thị/debug.
        """

        values = np.asarray(binary_values, dtype=float)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return 0, 0.0

        fog_ratio = float(np.mean(values >= 0.5))
        mode = self.window_label_mode_combo.currentText()

        if mode == "Majority":
            label = 1 if fog_ratio > 0.5 else 0

        elif mode == "FOG Threshold":
            threshold = (
                self.window_fog_threshold_spinbox.value()
                / 100.0
            )
            label = 1 if fog_ratio >= threshold else 0

        else:
            # Strict FOG: mọi sample phải là FOG.
            label = 1 if fog_ratio >= 1.0 else 0

        return label, fog_ratio
    def _compute_features(self,x,fs):
        x=np.asarray(x,dtype=float)
        if x.ndim != 1 or not np.isfinite(x).all(): raise ValueError("Signal chứa NaN/Inf; không xóa mẫu riêng từng trục.")
        if len(x)<2: raise ValueError("Window phải có ít nhất 2 sample.")
        mean=float(np.mean(x)); rms=float(np.sqrt(np.mean(x*x))); std=float(np.std(x)); var=float(np.var(x)); mn=float(np.min(x)); mx=float(np.max(x)); sma=float(np.mean(np.abs(x)))
        xc=x-mean; n=len(xc)
        freq=np.fft.rfftfreq(n,1.0/fs); mag=(2.0/n)*np.abs(np.fft.rfft(xc));
        if len(mag)>1: mag[0]*=0.5
        di=int(np.argmax(mag[1:])+1) if len(mag)>1 else 0
        dom=float(freq[di])

        nperseg=min(64,n);
        noverlap = nperseg // 2 if nperseg >= 2 else 0
        nfft = max(256, n);
        pf,pp=signal.welch(
            xc,fs=fs, window="hann",
            nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            detrend=False, scaling="density")
        integ=lambda y,z: float(np.trapezoid(y,z) if hasattr(np,"trapezoid") else np.trapz(y,z)) if len(y)>1 else 0.0
        total=integ(pp,pf); 
        lm=(pf >= 0.5) & (pf <= 3.0); 
        fm=(pf >= 3.0) & (pf <= 8.0); 
        
        loco=integ(pp[lm],pf[lm]); 
        freeze=integ(pp[fm],pf[fm]);
        
        fi=freeze/loco if loco>0 else 0.0; 
        low=loco/total if total>0 else 0.0

        ps=pp/np.sum(pp) if np.sum(pp)>0 else pp; ps=ps[ps>0]; pse=float(-np.sum(ps*np.log2(ps))) if len(ps) else 0.0
        result = {"Mean":mean,"RMS":rms,"Std":std,"Variance":var,"Min":mn,"Max":mx,"SMA":sma,"Dominant Frequency":dom,"Total Power":total,"Loco Power":loco,"Freeze Power":freeze,"Freeze Index":fi,"Low Freq Ratio":low,"PSE":pse,"FFT_Freq":freq,"FFT_Mag":mag,"PSD_Freq":pf,"PSD_Power":pp}
        result.update({k.replace("_", " "): v for k,v in axis_extra(x,fs,getattr(self,"feature_config",{})).items()})
        return result
    def _all_feature_rows(self):
        if not self.windowing_applied or not self.window_metadata or not self.window_signal_data:
            raise ValueError("Hãy APPLY WINDOWING trước khi Publish.")
        config=self.feature_config;fs=float(self.windowing_sampling_rate);rows=[]
        for item,meta in zip(self.window_signal_data,self.window_metadata):
            values={}
            for sensor in config['sensors']:
                for axis in config['axes']:
                    col=f'{sensor}_{axis}_mg'
                    if col not in item['Signals']: raise ValueError('Thiếu tín hiệu: '+col)
                    f=self._compute_features(item['Signals'][col],fs)
                    values[(sensor,axis)]={k: f[k.replace('_',' ')] for k in axis_names(config)}
            shared=window_extra(item['Signals'],fs,config,values)
            for (sensor,axis),f in values.items():
                row=dict(Window_ID=int(meta['Window_ID']),Segment_ID=int(meta['Segment_ID']),
                         Sensor=sensor,Axis=dict(zip(GUI_AXES,['X','Y','Z']))[axis],
                         Start_Time=float(meta['Start_Time']),End_Time=float(meta['End_Time']),
                         Binary_Label=int(meta['Binary_Label']),FOG_Ratio=float(meta['FOG_Ratio']))
                row.update(f);row.update(shared);rows.append(row)
        return pd.DataFrame(rows)

def gui_read_raw(path):
    path=Path(path)
    if path.suffix.lower()=='.txt':
        frame=pd.read_csv(path,sep=r'\s+',header=None)
        if frame.shape[1]!=11: raise ValueError('DAPHNET TXT cần đúng 11 cột.')
        frame.columns=GUI_COLUMNS
    else:
        frame=pd.read_csv(path)
        missing=set(GUI_COLUMNS)-set(frame.columns)
        if missing: raise ValueError('CSV raw thiếu cột: '+', '.join(sorted(missing)))
        frame=frame[GUI_COLUMNS].copy()
    if len(frame)<2: raise ValueError('Bản ghi cần ít nhất 2 mẫu.')
    for col in GUI_COLUMNS: frame[col]=pd.to_numeric(frame[col],errors='raise')
    if not np.isfinite(frame.to_numpy(float)).all(): raise ValueError('Raw chứa NaN/Inf.')
    if not frame.Label.isin([0,1,2]).all(): raise ValueError('Nhãn DAPHNET phải là 0, 1 hoặc 2.')
    dt=np.diff(frame.Time_ms.to_numpy(float)/1000)
    if not np.all(dt>0): raise ValueError('Timestamp phải tăng nghiêm ngặt.')
    if not np.isclose(np.median(dt),1/64,rtol=.04): raise ValueError('File raw không có bước thời gian DAPHNET 64 Hz.')
    return frame

def gui_labels(frame):
    frame=frame.copy();frame['Original_Label']=frame.Label
    frame['Binary_Label']=frame.Label.map({0:'Outside Experiment', 1:0., 2:1.})
    frame['Valid_Experiment']=frame.Label.isin([1,2])
    return frame

def gui_split(frame,fs=64.,valid_only=False):
    if frame is None or frame.empty:return []
    data=frame.loc[frame.Original_Label.isin([1,2])].copy() if valid_only else frame.copy()
    if data.empty:return []
    t=data.Time_ms.to_numpy(float)/1000
    valid=data.Original_Label.isin([1,2]).to_numpy()
    boundaries=np.flatnonzero((np.diff(t)>1.5/fs)|(valid[1:]!=valid[:-1]))+1
    return [(i+1,part.copy()) for i,part in enumerate([data.iloc[a:b] for a,b in zip(np.r_[0,boundaries],np.r_[boundaries,len(data)])]) if len(part)]

def _select_scope(frame,config):
    scope=config['selection'];mask=np.ones(len(frame),bool)
    if scope['mode']==0:mask &= frame.Original_Label.isin([1,2]).to_numpy()
    if scope['mode']==1:
        if scope['end']<=scope['start']:raise ValueError('End phải lớn hơn Start.')
        t=frame.Time_ms.to_numpy()/1000
        mask &= (t>=scope['start'])&(t<=scope['end'])
    if scope['exclude_zero']:mask &= frame.Original_Label.isin([1,2]).to_numpy()
    return gui_split(frame.loc[mask],64.)

def gui_process(runs,config):
    config=Config.from_dict(config)
    engine=ProcessingEngine(config);controls=config['controls'];fs=64.
    if controls['resample_checkbox']:fs=float(controls['target_fs_spinbox'])
    output=[]
    for sid,frame in runs:
        data=frame.copy()
        if controls['resample_checkbox']:data=engine.resample_dataframe(data,64.,fs)
        if controls['filter_checkbox']:data=engine.apply_filter_to_dataframe(data,fs)
        if controls['detrend_checkbox']:data=engine.apply_detrend_to_dataframe(data)
        data=gui_labels(data);output.append((sid,data))
    return output,fs

def gui_windows(runs,config,fs):
    config=Config.from_dict(config)
    engine=ProcessingEngine(config);c=config['controls']
    size=max(1,int(round(c['window_length_spinbox']*fs)))
    if size<2:raise ValueError('Window cần ít nhất 2 mẫu để phân tích feature.')
    step=max(1,int(round(size*(1-c['window_overlap_spinbox']/100))))
    axes=[a for a,flag in zip(GUI_AXES,['windowing_axis_x_checkbox','windowing_axis_y_checkbox','windowing_axis_z_checkbox']) if c[flag]]
    if not axes:raise ValueError('Hãy chọn ít nhất một axis.')
    sensor=c['windowing_sensor_combo'];columns=[f'{sensor}_{a}_mg' for a in axes]
    items=[];meta=[];sid=0
    for _,frame in runs:
        # Never join separate runs; outside-experiment samples cannot reach ML.
        for _,part in gui_split(frame,fs,True):
            sid+=1;t=part.Time_ms.to_numpy(float)/1000;y=part.Binary_Label.to_numpy(float)
            for start in range(0,len(part)-size+1,step):
                end=start+size;wid=len(items)+1
                label,ratio=engine.assign_window_label(y[start:end])
                items.append({'Window_ID':wid,'Time':t[start:end].copy(),
                              'Signals':{col:part[col].to_numpy(float)[start:end].copy()/1000 for col in columns}})
                meta.append({'Window_ID':wid,'Segment_ID':sid,'Start_Index':start,'End_Index':end-1,
                             'Start_Time':float(t[start]),'End_Time':float(t[end-1]),'Samples':size,
                             'Binary_Label':int(label),'FOG_Ratio':ratio})
    if not items:raise ValueError('Không có đoạn hợp lệ đủ dài để tạo window.')
    return items,meta,size,step

def gui_feature_rows(items,meta,fs,config):
    config=Config.from_dict(config)
    # Reuse original feature/export logic; schema stays one window/axis per row.
    engine=ProcessingEngine(config);engine.windowing_applied=True;engine.window_metadata=meta
    engine.window_signal_data=items;engine.windowing_sampling_rate=fs
    return engine._all_feature_rows()

# Configuration/exclusion additions; algorithms above retain their original formulas.
def recording_id(path):
    name=Path(path).stem.upper()
    match=re.fullmatch(r'(S\d+R\d+)(?:\(\d+\))?',name)
    if not match: raise ValueError('Tên file phải là SxxRxx.txt (chấp nhận hậu tố bản sao (1)).')
    return match.group(1)


def normalize_exclusions(value):
    if value is None: return {}
    if not isinstance(value,dict): raise ValueError('excluded_regions phải là object theo Recording_ID.')
    output={}
    for name,regions in value.items():
        rid=recording_id(str(name)+'.txt')
        if not isinstance(regions,list): raise ValueError('Danh sách đoạn loại trừ không hợp lệ.')
        output[rid]=[]
        for row in regions:
            start=float(row['start_s']);end=float(row['end_s']);enabled=row.get('enabled',True)
            if not isinstance(enabled,bool) or not np.isfinite([start,end]).all() or start<0 or end<=start:
                raise ValueError('Đoạn loại trừ cần 0 <= Start < End và enabled true/false.')
            output[rid].append(dict(start_s=start,end_s=end,reason=str(row.get('reason','')),enabled=enabled))
        output[rid].sort(key=lambda r:(r['start_s'],r['end_s']))
    return output


def exclusion_mask(frame,regions):
    t=frame.Time_ms.to_numpy(float)/1000;mask=np.zeros(len(frame),bool)
    for region in regions:
        if region.get('enabled',True):mask |= (t>=region['start_s'])&(t<=region['end_s'])
    return mask


def _canonicalize_dict(config):
    result=copy.deepcopy(dict(config));result.pop('config_id',None)
    active_recording=result.pop('recording_id',None)
    if result.get('schema') not in ('original-core-gui-v1','fog-unified-v1'):raise ValueError('Config không thuộc pipeline unified/original. Cần chuyển đổi rõ cấu hình cũ trước khi sử dụng.')
    result['excluded_regions']=normalize_exclusions(result.get('excluded_regions',{}))
    if result.get('schema') in ('original-core-gui-v1','fog-unified-v1'):
        c=result['controls']
        for key in ['resample_checkbox','filter_checkbox','detrend_checkbox','windowing_axis_x_checkbox','windowing_axis_y_checkbox','windowing_axis_z_checkbox']:
            if not isinstance(c[key],bool):raise ValueError(key+' cần true/false.')
        if result.get('source_fs',64)!=64:raise ValueError('DAPHNET source_fs phải là 64 Hz.')
        if c['detrend_type_combo'] not in ['Linear','Constant']:raise ValueError('Detrend type không hợp lệ.')
        if not 0<c['window_fog_threshold_spinbox']<=100:raise ValueError('FOG threshold phải >0 và <=100%.')
        if not np.isfinite(c['window_length_spinbox']):raise ValueError('Window length không hợp lệ.')
        fs=c['target_fs_spinbox'] if c['resample_checkbox'] else 64.
        if not np.isfinite(fs) or fs<16:raise ValueError('Export features cần Fs >= 16 Hz.')
        sensors=result.get('sensors',[c['windowing_sensor_combo']]);axes=result.get('axes',[a for a,k in zip(GUI_AXES,['windowing_axis_x_checkbox','windowing_axis_y_checkbox','windowing_axis_z_checkbox']) if c[k]])
        if not sensors or len(set(sensors))!=len(sensors) or any(s not in GUI_SENSORS for s in sensors):raise ValueError('Chọn 1–3 vị trí cảm biến.')
        if not axes or len(set(axes))!=len(axes) or any(a not in GUI_AXES for a in axes):raise ValueError('Chọn 1–3 trục.')
        result['sensors']=[s for s in GUI_SENSORS if s in sensors];result['axes']=[a for a in GUI_AXES if a in axes]
        if not 0<=c['window_overlap_spinbox']<100 or round(c['window_length_spinbox']*fs)<2:raise ValueError('Window/overlap không hợp lệ.')
        if c['filter_checkbox']:
            lo=c['cutoff_1_spinbox'];hi=c['cutoff_2_spinbox']
            if c['filter_type_combo'] not in ['Low-pass','High-pass','Band-pass'] or not 0<lo<fs/2 or not 1<=int(c['filter_order_spinbox'])<=12:raise ValueError('Filter không hợp lệ.')
            if c['filter_type_combo']=='Band-pass' and not lo<hi<fs/2:raise ValueError('Band-pass cần low < high < Nyquist.')
        scope=result['selection']
        if not isinstance(scope['exclude_zero'],bool):raise ValueError('exclude_zero cần true/false.')
        if scope['mode'] not in [0,1,2] or (scope['mode']==1 and not 0<=scope['start']<scope['end']):raise ValueError('Selection không hợp lệ.')
        if c['window_label_mode_combo'] not in ['Majority','FOG Threshold','Strict FOG']:raise ValueError('Quy tắc nhãn không hợp lệ.')
        result['feature_schema']='one-window-per-row'
    result['schema']='fog-unified-v1'
    result['feature_version'] = 'original14-welch64-bandfix-v2'
    fo=options(result)
    if any(fo[k] for k in GROUPS):
        result['feature_options']=fo
        result['feature_version']='extended-no-dwt-v1'
        result['feature_names']=feature_names(result)
        result['undefined_feature_policy']='nan; train-fold median imputation with indicators; empty features -> 0'
    else:
        for key in ['feature_options','feature_names','undefined_feature_policy']: result.pop(key,None)

    result['spectral_settings'] = {
        'method': 'welch',
        'window': 'hann',
        'nperseg_rule': 'min(64, n)',
        'noverlap_rule': 'nperseg // 2',
        'nfft_rule': 'max(256, n)',
        'detrend': False,
        'scaling': 'density',
        'loco_band_hz': [0.5, 3.0],
        'freeze_band_hz': [3.0, 8.0],
        'band_power_method': 'trapezoid_including_endpoints',
        'fi_zero_denominator_value': 0.0
    }
    result['processing_order']=['selection','recording_exclusions','split_runs','resample','filter','detrend','windowing','features']
    effective=copy.deepcopy(result)
    if effective['selection']['mode']!=1:effective['selection'].update(start=None,end=None)
    ec=effective['controls']
    if not ec['resample_checkbox']:ec['target_fs_spinbox']=64.
    if not ec['filter_checkbox']:
        for key in ['filter_type_combo','filter_order_spinbox','cutoff_1_spinbox','cutoff_2_spinbox']:ec[key]=None
    elif ec['filter_type_combo']!='Band-pass':ec['cutoff_2_spinbox']=None
    if not ec['detrend_checkbox']:ec['detrend_type_combo']=None
    if ec['window_label_mode_combo']!='FOG Threshold':ec['window_fog_threshold_spinbox']=None
    result['config_id']=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()[:16]
    if active_recording:result['recording_id']=active_recording
    return result


def merge_configs(configs):
    if not configs:raise ValueError('Không có config.')
    first=canonical_config(configs[0]);merged=copy.deepcopy(first)
    def common(c):
        c=copy.deepcopy(c)
        for k in ['excluded_regions','config_id','recording_id']:c.pop(k,None)
        if c.get('selection',{}).get('mode')!=1:
            c['selection'].pop('start',None);c['selection'].pop('end',None)
        return c
    for config in configs[1:]:
        current=canonical_config(config)
        if common(current)!=common(first):raise ValueError('Config chung khác nhau; không tự ghi đè filter/window/channels.')
        for rid,regions in current['excluded_regions'].items():
            target=merged['excluded_regions'].setdefault(rid,[])
            for region in regions:
                conflict=next((r for r in target if (r['start_s'],r['end_s'])==(region['start_s'],region['end_s'])),None)
                if conflict is not None and conflict!=region:raise ValueError('Điều kiện loại trừ mâu thuẫn: '+rid)
                if region not in target:target.append(region)
    return canonical_config(merged)




def gui_select(frame,config):
    config=Config.from_dict(config)
    rid=config.get('recording_id')
    regions=normalize_exclusions(config.get('excluded_regions',{})).get(rid,[])
    excluded=exclusion_mask(frame,regions)
    # Split on original index discontinuity as well as time gap; never bridge a removed sample.
    pieces=[];idx=np.flatnonzero(~excluded)
    boundaries=np.r_[0,np.flatnonzero(np.diff(idx)>1)+1,len(idx)]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if b>a:pieces.extend(_select_scope(frame.iloc[idx[a:b]],config))
    return [(i+1,d) for i,(_,d) in enumerate(pieces)]


def prepare_recording(path,config):
    config=canonical_config(config);rid=recording_id(path);local=copy.deepcopy(config);local['recording_id']=rid
    raw=gui_labels(gui_read_raw(path));regions=config['excluded_regions'].get(rid,[]);excluded=exclusion_mask(raw,regions)
    runs=gui_select(raw,local);processed,fs=gui_process(runs,local)
    # Build identical windows for every selected position, then combine channel features.
    items=None;meta=None
    for sensor in config['sensors']:
        per=copy.deepcopy(local);per['controls']['windowing_sensor_combo']=sensor
        for axis,key in zip(GUI_AXES,['windowing_axis_x_checkbox','windowing_axis_y_checkbox','windowing_axis_z_checkbox']):per['controls'][key]=axis in config['axes']
        channel_items,channel_meta,_,_=gui_windows(processed,per,fs)
        if items is None:items=channel_items;meta=channel_meta
        else:
            if channel_meta!=meta:raise ValueError('Cửa sổ giữa các vị trí không đồng bộ.')
            for item,new in zip(items,channel_items):item['Signals'].update(new['Signals'])
    engine=ProcessingEngine(local);rows=[];names=feature_names(config)
    for item,m in zip(items,meta):
        row=dict(Subject_ID=re.match(r'S\d+',rid).group(),Recording_ID=rid,Segment_ID=m['Segment_ID'],Window_ID=m['Window_ID'],Start_Time=m['Start_Time'],End_Time=m['End_Time'],Label=m['Binary_Label'],FOG_Ratio=m['FOG_Ratio'],Config_ID=config['config_id'])
        channel_values={}
        for sensor in config['sensors']:
            for axis in config['axes']:
                values=engine._compute_features(item['Signals'][f'{sensor}_{axis}_mg'],fs)
                f={key: values[key.replace('_',' ')] for key in axis_names(config)}
                channel_values[(sensor,axis)]=f
                row.update({f'{sensor}_{axis}_{key}':value for key,value in f.items()})
        row.update(window_extra(item['Signals'],fs,config,channel_values))
        rows.append(row)
    report=dict(Recording_ID=rid,Subject_ID=re.match(r'S\d+',rid).group(),Status='ok',Raw_Samples=len(raw),Excluded_Samples=int(excluded.sum()),Excluded_FOG_Samples=int((excluded & (raw.Label.to_numpy()==2)).sum()),Excluded_Non_FOG_Samples=int((excluded & (raw.Label.to_numpy()==1)).sum()),Excluded_Regions=json.dumps(regions,ensure_ascii=False),Selected_Samples=sum(len(d) for _,d in runs),Selected_Runs=len(runs),Windows=len(meta),FOG_Windows=sum(m['Binary_Label']==1 for m in meta),Non_FOG_Windows=sum(m['Binary_Label']==0 for m in meta))
    return pd.DataFrame(rows),names,report


class Config(dict):
    """The single validated configuration type shared by GUI, batch and CLI.

    JSON remains a mapping for compatibility; never dispatch algorithms by schema.
    original-core-gui-v1 is migrated to the same original14-unified-v1 algorithm.
    """
    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict):raise TypeError('Configuration must be a JSON object or Config.')
        return cls(_canonicalize_dict(value))
    def validate(self):
        value=_canonicalize_dict(self);self.clear();self.update(value);return self
    def dictionary(self):return copy.deepcopy(dict(self))

def canonical_config(value):return Config.from_dict(value)

def compute_features(x,fs):
    # Feature extraction needs no GUI state.
    return ProcessingEngine.__new__(ProcessingEngine)._compute_features(x,fs)
