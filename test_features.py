import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import feature_registry as fr
import signal_processing as sp
import data_prepare as dp


def configuration(extra=None):
    return sp.canonical_config(dict(schema='fog-unified-v1',source_fs=64.,sensors=list(fr.SENSORS),axes=list(fr.AXES),
        selection=dict(mode=0,start=0.,end=1.,exclude_zero=True),excluded_regions={},
        filter_implementation='original filtfilt; lfilter fallback on short segments',
        controls=dict(resample_checkbox=True,target_fs_spinbox=64.,filter_checkbox=True,filter_type_combo='Low-pass',
            filter_order_spinbox=4,cutoff_1_spinbox=16.,cutoff_2_spinbox=12.,detrend_checkbox=True,detrend_type_combo='Linear',
            window_length_spinbox=2.,window_overlap_spinbox=50.,window_label_mode_combo='Majority',window_fog_threshold_spinbox=50.,
            windowing_sensor_combo='Ankle',windowing_axis_x_checkbox=True,windowing_axis_y_checkbox=True,windowing_axis_z_checkbox=True),
        feature_options=extra or {},label_mapping={'0':None,'1':0,'2':1}))


def signals_and_metadata(n=5):
    t=np.arange(128)/64;items=[];meta=[]
    for i in range(n):
        signals={f'{s}_{a}_mg':(1+.1*i)*np.sin(2*np.pi*2*t+.1*j)+.1*np.sin(2*np.pi*6*t)
                 for j,(s,a) in enumerate((s,a) for s in fr.SENSORS for a in fr.AXES)}
        items.append(dict(Window_ID=i+1,Time=t+i,Signals=signals))
        meta.append(dict(Window_ID=i+1,Segment_ID=1 if i<3 else 2,Start_Time=float(i),End_Time=float(i+t[-1]),Binary_Label=i%2,FOG_Ratio=float(i%2)))
    return items,meta


class FeatureTests(unittest.TestCase):
    def test_schema_and_selection(self):
        self.assertEqual(len(fr.feature_names(configuration())),126)
        legacy={k:True for k in ['jerk','sampen','apen','autocorr','kurtosis','crest','corr','crosscorr','ater']}
        self.assertEqual(len(fr.feature_names(configuration(legacy))),206)
        self.assertEqual(len(fr.feature_names(configuration({k:True for k in fr.GROUPS if k!='cwt'}))),443)
        self.assertEqual(len(fr.feature_names(configuration({k:True for k in fr.GROUPS}))),461)
        self.assertEqual(len(fr.feature_names(configuration({'base_features':['RMS','Freeze_Index']}))),18)
        with self.assertRaises(ValueError):configuration({'base_features':[]})

    def test_wavelet_validation(self):
        for wave,level in [('db4',5),('db8',4)]:
            with self.assertRaises(ValueError):configuration(dict(wavelet=True,wavelet_name=wave,wavelet_level=level))
        configuration(dict(wavelet=True,wavelet_name='db8',wavelet_level=3))

    def test_spectral_and_skew(self):
        c=configuration(dict(spectral_shape=True,skewness=True));t=np.arange(128)/64
        f=fr.axis_extra(np.sin(2*np.pi*6*t),64,c)
        self.assertAlmostEqual(f['Spectral_Centroid'],6.,places=2)
        self.assertAlmostEqual(f['Skewness'],0.,places=10)
        f=fr.axis_extra(np.zeros(128),64,c)
        self.assertTrue(np.isnan(f['Skewness']))
        self.assertTrue(np.isnan(f['Spectral_Centroid']))

    def test_temporal_reset_and_causality(self):
        c=configuration(dict(temporal=True));items,meta=signals_and_metadata()
        before=list(sp.window_feature_records(items,meta,64,c))
        self.assertTrue(np.isnan(before[0][2]['Ankle_Forward_Delta_RMS']))
        self.assertTrue(np.isnan(before[3][2]['Ankle_Forward_Delta_RMS']))
        self.assertGreater(before[1][2]['Ankle_Forward_RMS_Trend_g_s'],0)
        changed=copy.deepcopy(items)
        changed[-1]['Signals']['Ankle_Forward_mg']*=10
        after=list(sp.window_feature_records(changed,meta,64,c))
        for a,b in zip(before[:-1],after[:-1]):
            for key in a[2]:np.testing.assert_allclose(a[2][key],b[2][key],equal_nan=True)
        gap=copy.deepcopy(meta);gap[2]['Start_Time']=20.
        r=list(sp.window_feature_records(items,gap,64,c))
        self.assertTrue(np.isnan(r[2][2]['Ankle_Forward_Delta_RMS']))

    def test_new_groups_preserve_existing_statistics(self):
        old_options={k:True for k in ['jerk','sampen','apen','autocorr','kurtosis','crest']}
        x=np.sin(2*np.pi*2*np.arange(128)/64)+.2*np.sin(2*np.pi*6*np.arange(128)/64)
        before=fr.axis_extra(x,64,configuration(old_options))
        after=fr.axis_extra(x,64,configuration({k:True for k in fr.GROUPS}))
        for key in before:np.testing.assert_allclose(before[key],after[key],equal_nan=True)
        self.assertLessEqual(abs(after['Autocorr_Peak']),1.)

    def test_magnitude_rotation(self):
        c=configuration(dict(magnitude=True));items,_=signals_and_metadata();a=items[0]['Signals'];b=copy.deepcopy(a)
        q,_=np.linalg.qr(np.random.default_rng(42).normal(size=(3,3)))
        for sensor in fr.SENSORS:
            x=np.column_stack([a[f'{sensor}_{axis}_mg'] for axis in fr.AXES])@q
            for j,axis in enumerate(fr.AXES):b[f'{sensor}_{axis}_mg']=x[:,j]
        f=fr.window_extra(a,64,c,{});g=fr.window_extra(b,64,c,{})
        for key in f:np.testing.assert_allclose(f[key],g[key],rtol=1e-10,atol=1e-10)

    def test_all_features_gui_export(self):
        c=configuration({k:True for k in fr.GROUPS});c['recording_id']='S01R01'
        items,meta=signals_and_metadata();preview=sp.gui_feature_rows(items,meta,64,c)
        wide,names=dp.wide_from_gui(preview,c)
        self.assertEqual((len(wide),len(names)),(5,461))
        with tempfile.TemporaryDirectory() as td:
            dp.export_dataset(wide,names,c,Path(td)/'test')
            with np.load(Path(td)/'test.npz',allow_pickle=False) as z:
                self.assertEqual(z['X'].shape,(5,461));self.assertTrue(np.isnan(z['X']).any());self.assertFalse(np.isinf(z['X']).any())

    def test_raw_batch_gui_agree(self):
        c=configuration({k:True for k in fr.GROUPS});c['excluded_regions']={'S01R01':[dict(start_s=15.,end_s=16.,enabled=True,reason='test')]};c=sp.canonical_config(c)
        t=np.arange(64*20)/64
        raw=np.column_stack([t*1000]+[1000*np.sin(2*np.pi*(1.5+.1*i)*t+.1*i) for i in range(9)]+[np.where((t>=12)&(t<13),0,np.where((t>=6)&(t<9),2,1))])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'S01R01.txt';np.savetxt(path,raw,fmt='%.9f')
            batch,names,report=sp.prepare_recording(path,c)
            local=copy.deepcopy(c);local['recording_id']='S01R01'
            processed,fs=sp.gui_process(sp.gui_select(sp.gui_labels(sp.gui_read_raw(path)),local),local)
            items=meta=None
            for sensor in c['sensors']:
                per=copy.deepcopy(local);per['controls']['windowing_sensor_combo']=sensor
                it,mt,_,_=sp.gui_windows(processed,per,fs)
                if items is None:items,meta=it,mt
                else:
                    self.assertEqual(meta,mt)
                    for old,new in zip(items,it):old['Signals'].update(new['Signals'])
            gui=sp.gui_feature_rows(items,meta,fs,local);wide,gnames=dp.wide_from_gui(gui,local)
            self.assertEqual(names,gnames)
            np.testing.assert_allclose(batch[names],wide[names],equal_nan=True)
            for start,end in batch[['Start_Time','End_Time']].to_numpy():
                self.assertFalse(start<13 and end>=12)
                self.assertFalse(start<=16 and end>=15)

if __name__=='__main__':unittest.main(verbosity=2)
