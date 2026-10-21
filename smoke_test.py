import sys,importlib.util,tempfile,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import signal_processing as sp, data_prepare as dp, train_fog as tr
from feature_registry import *
c=dict(schema='fog-unified-v1',source_fs=64.,sensors=list(SENSORS),axes=list(AXES),selection=dict(mode=0,start=0.,end=1.,exclude_zero=True),controls=dict(resample_checkbox=True,target_fs_spinbox=64.,filter_checkbox=True,filter_type_combo='Low-pass',filter_order_spinbox=4,cutoff_1_spinbox=16.,cutoff_2_spinbox=12.,detrend_checkbox=True,detrend_type_combo='Linear',window_length_spinbox=2.,window_overlap_spinbox=50.,window_label_mode_combo='Majority',window_fog_threshold_spinbox=50.,windowing_sensor_combo='Ankle',windowing_axis_x_checkbox=True,windowing_axis_y_checkbox=True,windowing_axis_z_checkbox=True),excluded_regions={})
c=sp.canonical_config(c)
x=np.sin(2*np.pi*5*np.arange(128)/64)+.12*np.cos(2*np.pi*1.5*np.arange(128)/64)
assert len(feature_names(c))==126
cc=dict(c,feature_options={k:True for k in GROUPS});cc=sp.canonical_config(cc);assert sp.canonical_config(cc)==cc
assert np.isnan(entropy(np.ones(128),2,.2));assert np.isnan(corr(np.ones(128),x));assert abs(corr(x,x)-1)<1e-12
ac=axis_extra(np.sin(2*np.pi*2*np.arange(128)/64),64,cc);assert abs(ac['Autocorr_Lag_s']-.5)<.02,ac
assert np.isfinite(axis_extra(x,64,cc)['Sample_Entropy'])
print('PASS baseline feature count; entropy, correlation, autocorrelation edge cases')
with tempfile.TemporaryDirectory() as td:
 td=Path(td);paths=[]
 for si in range(1,5):
  n=640;t=np.arange(n)/64; signals=[]
  for ch in range(9):signals.append(np.rint(1000*(.15*np.sin(2*np.pi*(1.4+.08*ch)*t)+.025*np.sin(2*np.pi*5*t))+ch*15))
  mat=np.column_stack([np.rint(t*1000),*signals,np.where(t<5,1,2)]).astype(int);p=td/f'S{si:02d}R01.txt';np.savetxt(p,mat,fmt='%d');paths.append(p)
 table,names,report=dp.prepare_files(paths,cc);assert len(names)==len(set(names));assert names==feature_names(cc)
 outs=dp.export_dataset(table,names,cc,td/'features',report=report)
 data,loaded,meta=tr.load_datasets([td/'features.npz']);assert loaded==names;assert data['X'].shape[1]==len(names)
 data2,names2,_=tr.load_datasets([td/'features.csv']);assert names2==names;assert np.allclose(data['X'],data2['X'],equal_nan=True)
 local=dict(cc,recording_id='S01R01');raw=sp.gui_labels(sp.gui_read_raw(paths[0]));runs=sp.gui_select(raw,local);processed,fs=sp.gui_process(runs,local);items=met=None
 for sensor in SENSORS:
  per=dict(local,controls=dict(local['controls'],windowing_sensor_combo=sensor));it,mt,_,_=sp.gui_windows(processed,per,fs)
  if items is None:items,met=it,mt
  else:
   for i,j in zip(items,it):i['Signals'].update(j['Signals'])
 long=sp.gui_feature_rows(items,met,fs,local);wide,wn=dp.wide_from_gui(long,local)
 assert wn==names;assert np.allclose(wide[names],table[table.Recording_ID=='S01R01'][names],equal_nan=True)
 # Undefined statistics survive export and are handled inside the fit pipeline.
 X=data['X'];X[0,-1]=np.nan
 from sklearn.pipeline import Pipeline
 from sklearn.impute import SimpleImputer
 from sklearn.preprocessing import StandardScaler
 from sklearn.svm import SVC
 pipe=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scaler',StandardScaler()),('svm',SVC())]);pipe.fit(X,data['y']);assert np.isfinite(pipe.decision_function(X)).all()
 print('PASS synthetic batch -> CSV/NPZ -> trainer',table.shape,'feature_count',len(names))
 print('PASS GUI feature/export equals batch; missing-value fit/inference')
 tr.train_loso([td/'features.npz'],td/'loso',cache_size=64,save_models=False)
 print('PASS 4-subject LOSO synthetic smoke')
