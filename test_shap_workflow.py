import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import shap_fog as sf


def make_fixture(folder,n_features=8):
    root=Path(folder)/'result';root.mkdir();rng=np.random.default_rng(12);x=rng.normal(size=(72,n_features))
    y=(x[:,0]+.5*x[:,1]+rng.normal(scale=.4,size=72)>0).astype(int);x[::5,2]=np.nan;x[:,-1]=np.nan
    subjects=np.repeat(['S01','S02','S03'],24);names=[f'Ankle_Forward_Test{i}' for i in range(n_features)]
    meta=pd.DataFrame(dict(Subject_ID=subjects,Recording_ID=[s+'R01' for s in subjects],Segment_ID=np.ones(72,int),
        Window_ID=np.tile(np.arange(1,25),3),Start_Time=np.tile(np.arange(24,dtype=float),3),End_Time=np.tile(np.arange(24,dtype=float)+1.984,3),Binary_Label=y))
    dataset=Path(folder)/'input.npz'
    np.savez_compressed(dataset,X=x,y=y,feature_names=np.asarray(names),subject_id=subjects,recording_id=meta.Recording_ID.to_numpy(str),
        segment_id=meta.Segment_ID.to_numpy(),window_id=meta.Window_ID.to_numpy(),window_start=meta.Start_Time.to_numpy(),window_end=meta.End_Time.to_numpy(),config_id=np.repeat('fixture',72))
    rows=[]
    for i,subject in enumerate(np.unique(subjects),1):
        directory=root/f'fold_{i:02d}_{subject}';directory.mkdir();test=np.flatnonzero(subjects==subject);train=np.flatnonzero(subjects!=subject)
        model=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True)),('scaler',StandardScaler()),('svm',SVC(C=1,gamma='scale',class_weight='balanced'))])
        model.fit(x[train],y[train]);scores=model.decision_function(x[test]);pred=(scores>0).astype(int)
        manifest=meta.copy();manifest['Split']=np.where(subjects==subject,'test','train');manifest.to_csv(directory/'split_manifest.csv',index=False)
        table=manifest.iloc[test].copy();table['Dataset_Row']=test;table['Outer_Fold']=i;table['Prediction']=pred;table['FOG_Decision_Score']=scores;table['Decision_Threshold']=0.
        table.to_csv(directory/'test_predictions.csv',index=False)
        joblib.dump(dict(pipeline=model,feature_names=names,train_subjects=sorted(set(subjects[train])),test_subjects=[subject]),directory/'fog_svm.joblib')
        counts={'TN':int(((y[test]==0)&(pred==0)).sum()),'FP':int(((y[test]==0)&(pred==1)).sum()),'FN':int(((y[test]==1)&(pred==0)).sum()),'TP':int(((y[test]==1)&(pred==1)).sum())}
        rows.append(counts)
    (root/'feature_names.json').write_text(json.dumps(names))
    (root/'dataset_info.json').write_text(json.dumps({'configurations':[{'config_id':'fixture'}]}))
    (root/'loso_summary.json').write_text(json.dumps({'status':'completed','pooled_metrics':{'confusion_counts':{k:sum(r[k] for r in rows) for k in rows[0]}}}))
    return root,dataset


def args_for(root,dataset):
    return sf.build_parser().parse_args(['--input',str(dataset),'--results',str(root),'--background','8','--explain','8','--nsamples','64','--l1-reg','0'])


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root,cls.dataset=make_fixture(cls.temp.name)
        cls.args=args_for(cls.root,cls.dataset);sf.run(cls.args)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_real_kernel_shap_with_nan_pipeline(self):
        for folder in self.root.glob('fold_*'):
            with np.load(folder/'shap/values.npz',allow_pickle=False) as z:
                self.assertEqual(z['shap_values'].shape,(8,8))
                np.testing.assert_allclose(z['base_value']+z['shap_values'].sum(axis=1),z['scores'],rtol=1e-5,atol=1e-5)
                self.assertTrue(np.isfinite(z['shap_values']).all());self.assertTrue(np.isnan(z['feature_values']).any())
            meta=json.loads((folder/'shap/manifest.json').read_text())
            self.assertFalse(set(meta['background_subjects'])&set(meta['test_subjects']))
            self.assertEqual(meta['scope'],'outer_test_explanation_only')
    def test_fold_metadata_protection(self):
        x,meta,names,ids=sf.read_dataset([self.dataset]);bad=meta.copy();bad.loc[0,'Window_ID']=999
        with self.assertRaisesRegex(ValueError,'không khớp'):sf.prepare_fold(self.root,next(self.root.glob('fold_*')),x,bad,names,ids)
    def test_feature_order_protection(self):
        x,meta,names,ids=sf.read_dataset([self.dataset])
        with self.assertRaisesRegex(ValueError,'feature_names'):sf.prepare_fold(self.root,next(self.root.glob('fold_*')),x,meta,names[::-1],ids)
    def test_altered_features_detected(self):
        x,meta,names,ids=sf.read_dataset([self.dataset]);x=x.copy();x[0,0]+=100
        folder=self.root/'fold_01_S01'
        with self.assertRaisesRegex(ValueError,'Score tính lại'):sf.prepare_fold(self.root,folder,x,meta,names,ids)
    def test_sampling_weights(self):
        outcomes=np.array(['TP']*40+['FP']*10+['FN']*20+['TN']*30)
        selected,w=sf.select_explained(np.arange(100),outcomes,20,'errors-balanced',np.random.default_rng(42))
        for label,n in [('TP',40),('FP',10),('FN',20),('TN',30)]:self.assertAlmostEqual(w[outcomes[selected]==label].sum(),n)
    def test_no_test_rows_in_background(self):
        x,meta,_,_=sf.read_dataset([self.dataset]);train=np.flatnonzero(meta.Subject_ID!='S01')
        selected=sf.select_balanced_background(train,meta,8,np.random.default_rng(42))
        self.assertTrue(set(selected).issubset(set(train)))
    def test_gui_reader_and_plots(self):
        import app_compare as ac
        runs=ac.load_runs(self.root);self.assertEqual(len(runs),1);self.assertEqual(len(runs[0].shap_data),3)
        data=runs[0].shap_data[0];idx=ac.shap_indices(data)
        figures=[ac.shap_bar(data,idx,5),ac.shap_beeswarm(data,idx,5),ac.shap_waterfall(data,0,5),ac.shap_stability(runs[0].shap_data,'All',5)]
        for i,figure in enumerate(figures):
            figure.savefig(Path(self.temp.name)/f'plot_{i}.png');figure.clear()
        path=Path(self.temp.name)/'result.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for p in self.root.rglob('*'):
                if p.is_file() and p.suffix!='.joblib':z.write(p,Path('result')/p.relative_to(self.root))
        loaded=ac.load_runs(path);self.assertEqual(len(loaded[0].shap_data),3)
    def test_gui_controls(self):
        import app_compare as ac
        self.__class__.app=ac.QApplication.instance() or ac.QApplication([])
        window=ac.MainWindow();window.loaded(ac.load_runs(self.root),[])
        self.assertEqual(window.tabs.count(),5);self.assertEqual(window.shapfold.count(),3)
        for mode in range(4):window.shapmode.setCurrentIndex(mode)
        self.assertGreater(window.shapwindow.count(),0)
        window.shapoutcome.setCurrentText('FP');window.draw_shap()
        window.close()

    def test_reload_updates_shap(self):
        import app_compare as ac
        self.__class__.app=ac.QApplication.instance() or ac.QApplication([])
        window=ac.MainWindow();old=ac.load_runs(self.root);old[0].shap_data=[];window.loaded(old,[])
        self.assertEqual(window.shapfold.count(),0)
        window.loaded(ac.load_runs(self.root),[])
        self.assertEqual(window.list.count(),1);self.assertEqual(window.shapfold.count(),3);window.close()

    def test_default_sparse_443_features(self):
        with tempfile.TemporaryDirectory() as td:
            root,dataset=make_fixture(td,n_features=443)
            args=sf.build_parser().parse_args(['--input',str(dataset),'--results',str(root),'--background','8','--explain','2','--nsamples','128','--folds','S01'])
            sf.run(args)
            with np.load(root/'fold_01_S01/shap/values.npz',allow_pickle=False) as z:
                self.assertEqual(z['shap_values'].shape,(2,443));self.assertTrue(np.isfinite(z['shap_values']).all())

    def test_training_wrapper_forwards_arguments(self):
        import shutil,subprocess,sys
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            for name in ['train_fog_shap.py','shap_fog.py']:shutil.copy2(Path(__file__).parent/name,folder/name)
            (folder/'train_fog.py').write_text("import json,sys\nfrom pathlib import Path\nPath(__file__).with_name('forwarded.json').write_text(json.dumps(sys.argv[1:]))\nraise SystemExit(0)\n")
            result=subprocess.run([sys.executable,str(folder/'train_fog_shap.py'),'--input',str(self.dataset),'--output',str(self.root),'--evaluation','loso','--C','1','--shap-folds','S01','--shap-explain','2'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            forwarded=json.loads((folder/'forwarded.json').read_text())
            self.assertIn('--evaluation',forwarded);self.assertIn('--C',forwarded)
            self.assertNotIn('--shap-folds',forwarded);self.assertNotIn('--shap-explain',forwarded)

    def test_tampered_shap_rejected(self):
        import app_compare as ac
        folder=self.root/'fold_01_S01/shap';manifest=json.loads((folder/'manifest.json').read_text());manifest['background_subjects']=['S01']
        expected=json.loads((self.root/'feature_names.json').read_text())
        with self.assertRaisesRegex(ValueError,'lẫn người test'):ac.read_shap_data('fold/shap/manifest.json',manifest,lambda n:(folder/Path(n).name).read_bytes(),expected)

if __name__=='__main__':unittest.main(verbosity=2)
