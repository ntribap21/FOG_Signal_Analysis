from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import warnings
import numpy as np
import pandas as pd

SCHEMA = 'fog-shap-v1'
KEYS = ['Subject_ID','Recording_ID','Segment_ID','Window_ID','Start_Time','End_Time','Binary_Label']


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)


def read_dataset(paths):
    matrices=[];metas=[];names=None;config_ids=set()
    aliases={'subject_id':'Subject_ID','recording_id':'Recording_ID','segment_id':'Segment_ID',
             'window_id':'Window_ID','window_start':'Start_Time','window_end':'End_Time'}
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            required={'X','y','feature_names',*aliases}
            if not required.issubset(z.files):raise ValueError(f'{path}: thiếu NPZ fields {sorted(required-set(z.files))}')
            current=z['feature_names'].astype(str).tolist();x=np.asarray(z['X'],float);y=np.asarray(z['y'])
            if x.ndim!=2 or y.ndim!=1 or x.shape!=(len(y),len(current)) or not len(y):raise ValueError('Kích thước X/y không hợp lệ.')
            if len(set(current))!=len(current) or not current:raise ValueError('Tên features rỗng hoặc trùng.')
            if names is not None and current!=names:raise ValueError('Tên / thứ tự features giữa NPZ không khớp.')
            if np.isinf(x).any() or not np.isin(y,[0,1]).all():raise ValueError('Features chứa Inf hoặc nhãn không phải 0/1.')
            meta=pd.DataFrame({dest:np.asarray(z[src]) for src,dest in aliases.items()});meta['Binary_Label']=y.astype(int)
            if len(meta)!=len(x):raise ValueError('Độ dài metadata không khớp.')
            if 'config_id' in z:config_ids.update(z['config_id'].astype(str).tolist())
            for key in ['Subject_ID','Recording_ID']:meta[key]=meta[key].astype(str)
            matrices.append(x);metas.append(meta);names=current
    x=np.concatenate(matrices);meta=pd.concat(metas,ignore_index=True)
    if meta.duplicated(KEYS).any():raise ValueError('NPZ có cửa sổ trùng; không tự ghép các dataset cùng recording.')
    if meta[['Subject_ID','Recording_ID']].isna().any().any() or any(meta[c].str.strip().eq('').any() for c in ['Subject_ID','Recording_ID']):raise ValueError('ID bệnh nhân / bản ghi rỗng.')
    numeric=meta[['Segment_ID','Window_ID','Start_Time','End_Time']].to_numpy(float)
    if not np.isfinite(numeric).all() or (meta.End_Time<=meta.Start_Time).any():raise ValueError('Metadata thời gian/ID không hợp lệ.')
    if meta.duplicated(['Subject_ID','Recording_ID','Segment_ID','Window_ID']).any():raise ValueError('Window ID trùng trong cùng recording / segment.')
    return x,meta,names,config_ids


def check_metadata(actual, expected):
    if len(actual)!=len(expected):raise ValueError('NPZ không có cùng số hàng với split manifest; dùng đúng NPZ và thứ tự input lúc train.')
    for key in KEYS:
        if key not in expected:raise ValueError('Manifest thiếu '+key)
        if key in ['Subject_ID','Recording_ID']:
            equal=np.array_equal(actual[key].astype(str).to_numpy(),expected[key].astype(str).to_numpy())
        else:equal=np.allclose(actual[key].to_numpy(float),expected[key].to_numpy(float),atol=2e-6,rtol=0)
        if not equal:raise ValueError('NPZ / manifest không khớp tại '+key+'; không tính SHAP trên dữ liệu khác lần train.')


def select_balanced_background(indices, meta, size, rng):
    if size<1:raise ValueError('Background phải >=1.')
    groups={}
    for idx in indices:
        key=(str(meta.iloc[idx].Subject_ID),int(meta.iloc[idx].Binary_Label))
        groups.setdefault(key,[]).append(int(idx))
    buckets=[rng.permutation(v).tolist() for _,v in sorted(groups.items())]
    if size<len(buckets):raise ValueError(f'Background phải >= {len(buckets)} để đại diện mỗi cặp người/lớp trong train.')
    selected=[]
    while len(selected)<min(size,len(indices)):
        active=False
        for bucket in buckets:
            if bucket and len(selected)<size:selected.append(bucket.pop());active=True
        if not active:break
    return np.asarray(selected,int)


def outcome_labels(y,p):
    return np.where(y==1,np.where(p==1,'TP','FN'),np.where(p==1,'FP','TN'))


def select_explained(indices, outcomes, size, sampling, rng):
    size=min(size,len(indices))
    if size<1:raise ValueError('Số cửa sổ giải thích phải >=1.')
    if sampling=='uniform':
        chosen=np.sort(rng.choice(np.arange(len(indices)),size=size,replace=False))
        return chosen,np.ones(size,float)
    buckets={key:np.flatnonzero(outcomes==key) for key in ['TP','FP','FN','TN'] if np.any(outcomes==key)}
    if size<len(buckets):raise ValueError('Cần đủ số mẫu để đại diện các nhóm TP/FP/FN/TN.')
    selected=[];queues={k:rng.permutation(v).tolist() for k,v in buckets.items()}
    while len(selected)<size:
        for key in queues:
            if queues[key] and len(selected)<size:selected.append(queues[key].pop())
    chosen=np.sort(selected)
    weights=np.asarray([len(buckets[outcomes[i]])/np.sum(outcomes[chosen]==outcomes[i]) for i in chosen],float)
    return chosen,weights


def model_file(directory):
    candidates=[directory/name for name in ['fog_rf.joblib','fog_svm.joblib'] if (directory/name).is_file()]
    if len(candidates)!=1:raise ValueError('Fold cần đúng một model fog_rf.joblib hoặc fog_svm.joblib.')
    return candidates[0]


def model_kind(model):
    from sklearn.ensemble import RandomForestClassifier
    return 'rf' if isinstance(model.steps[-1][1],RandomForestClassifier) else 'svm'


def prediction_scores(model, x):
    return np.asarray(model.predict_proba(x)[:,1] if model_kind(model)=='rf' else model.decision_function(x),float)


def prepare_fold(root, directory, x, meta, names, config_ids):
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    manifest=pd.read_csv(directory/'split_manifest.csv')
    check_metadata(meta,manifest)
    if not set(manifest.Split.astype(str)).issubset({'train','test'}):raise ValueError('Split manifest chỉ được có train/test.')
    train=np.flatnonzero(manifest.Split.to_numpy()=='train');test=np.flatnonzero(manifest.Split.to_numpy()=='test')
    if not len(train) or not len(test):raise ValueError('Fold thiếu train/test.')
    train_subjects=set(meta.iloc[train].Subject_ID);test_subjects=set(meta.iloc[test].Subject_ID)
    if train_subjects & test_subjects or len(test_subjects)!=1:raise ValueError('LOSO không tách biệt theo người.')
    saved_names=json.loads((root/'feature_names.json').read_text(encoding='utf-8-sig'))
    if names!=saved_names:raise ValueError('NPZ feature_names không khớp với lần train.')
    info_path=root/'dataset_info.json'
    if info_path.exists() and config_ids:
        info=json.loads(info_path.read_text(encoding='utf-8-sig'))
        known={str(c['config_id']) for c in info.get('configurations',[]) if isinstance(c,dict) and 'config_id' in c}
        if known and not config_ids.issubset(known):raise ValueError('NPZ Config_ID không khớp với lần train.')
    predictions=pd.read_csv(directory/'test_predictions.csv')
    if 'Dataset_Row' not in predictions:raise ValueError('test_predictions.csv thiếu Dataset_Row.')
    rows=predictions.Dataset_Row.to_numpy(float)
    if not np.isfinite(rows).all() or np.any(rows!=np.floor(rows)):raise ValueError('Dataset_Row không hợp lệ.')
    rows=rows.astype(int)
    if len(np.unique(rows))!=len(rows) or set(rows)!=set(test):raise ValueError('Test predictions không khớp split manifest.')
    check_metadata(meta.iloc[rows].reset_index(drop=True),predictions.reset_index(drop=True))
    with warnings.catch_warnings():
        warnings.simplefilter('error',category=InconsistentVersionWarning)
        bundle=joblib.load(model_file(directory))
    if not isinstance(bundle,dict) or 'pipeline' not in bundle:raise ValueError('Model bundle phải có pipeline.')
    if list(bundle.get('feature_names',[]))!=names:raise ValueError('Model và NPZ khác tên / thứ tự features.')
    model=bundle['pipeline']
    if not np.array_equal(np.asarray(model.classes_),np.array([0,1])):raise ValueError('SHAP yêu cầu model classes [0,1], score dương hướng FOG.')
    if set(bundle.get('test_subjects',test_subjects))!=test_subjects or set(bundle.get('train_subjects',train_subjects))!=train_subjects:raise ValueError('Model thuộc fold khác.')
    scores=prediction_scores(model,x[rows])
    column='FOG_Probability' if model_kind(model)=='rf' else 'FOG_Decision_Score'
    if column not in predictions:raise ValueError('Predictions thiếu '+column)
    if scores.shape!=(len(rows),) or not np.isfinite(scores).all():raise ValueError('Decision score model không hợp lệ.')
    if not np.allclose(scores,predictions[column].to_numpy(float),rtol=2e-4,atol=2e-5):raise ValueError('Score tính lại không khớp lần train; dùng đúng NPZ, model và môi trường sklearn.')
    thresholds=predictions.Decision_Threshold.to_numpy(float)
    if not np.isfinite(thresholds).all() or not np.allclose(thresholds,thresholds[0]):raise ValueError('Threshold không nhất quán trong fold.')
    pred=(scores>thresholds).astype(int)
    if not np.array_equal(pred,predictions.Prediction.to_numpy(int)):raise ValueError('Prediction khác kết quả đã lưu.')
    return model,train,rows,predictions,scores,pred,thresholds


def explain_fold(root,directory,x,meta,names,config_ids,args):
    import shap
    import sklearn
    destination=directory/'shap'
    if (destination/'manifest.json').exists() and not args.overwrite:raise ValueError(f'{destination} đã có kết quả; dùng --overwrite để tính lại.')
    model,train,rows,predictions,scores,pred,thresholds=prepare_fold(root,directory,x,meta,names,config_ids)
    rng=np.random.default_rng(args.seed+int(predictions.Outer_Fold.iloc[0]))
    bg=select_balanced_background(train,meta,args.background,rng)
    y=meta.iloc[rows].Binary_Label.to_numpy(int);outcomes=outcome_labels(y,pred)
    selected,weights=select_explained(rows,outcomes,args.explain,args.sampling,rng)
    chosen=rows[selected];background=x[bg];values=x[chosen]
    kind=model_kind(model)
    if kind=='rf':
        from sklearn.impute import SimpleImputer
        if len(model.steps)!=2 or not isinstance(model.steps[0][1],SimpleImputer) or model.steps[0][1].add_indicator:
            raise ValueError('Tree SHAP RF hiện hỗ trợ pipeline SimpleImputer từng cột + RandomForest, không thêm indicator.')
        transformer=model[:-1]
        transformed_background=transformer.transform(background)
        transformed_values=transformer.transform(values)
        if transformed_values.shape!=values.shape or transformed_background.shape!=background.shape:
            raise ValueError('Tree SHAP yêu cầu preprocessing một-đối-một; RF không thêm missing indicators hoặc giảm chiều.')
        masker=shap.maskers.Independent(transformed_background,max_samples=len(transformed_background))
        explainer=shap.TreeExplainer(model.steps[-1][1],data=masker,model_output='probability',feature_perturbation='interventional')
        result=explainer.shap_values(transformed_values,check_additivity=True)
        if isinstance(result,list):phi=np.asarray(result[1],float)
        else:
            array=np.asarray(result,float)
            if array.ndim!=3 or array.shape[-1]!=2:raise ValueError('TreeExplainer không trả về SHAP hai lớp.')
            phi=array[:,:,1]
        base=float(np.asarray(explainer.expected_value).reshape(-1)[1])
        l1=None
    else:
        if args.l1_reg=='0' and args.nsamples<2*len(names)+2:raise ValueError('l1-reg=0 yêu cầu nsamples >= 2*n_features+2 trong công cụ này.')
        l1=0.0 if args.l1_reg=='0' else args.l1_reg
        before=np.random.get_state();np.random.seed(args.seed)
        try:
            explainer=shap.KernelExplainer(model.decision_function,background,feature_names=names,link='identity')
            phi=np.asarray(explainer.shap_values(values,nsamples=args.nsamples,l1_reg=l1,silent=True,gc_collect=True),float)
        finally:np.random.set_state(before)
        base=float(np.asarray(explainer.expected_value).reshape(-1)[0])
    if phi.shape!=values.shape or not np.isfinite(phi).all():raise ValueError('Explainer trả về SHAP không hợp lệ.')
    reconstructed=base+phi.sum(axis=1);residual=scores[selected]-reconstructed
    if not np.allclose(reconstructed,scores[selected],rtol=1e-5,atol=1e-5):raise ValueError('SHAP không thỏa base + sum(SHAP) = decision score.')
    sample=predictions.iloc[selected].copy().reset_index(drop=True)
    sample['Outcome']=outcomes[selected];sample['Sample_Weight']=weights;sample['SHAP_Base']=base;sample['SHAP_Residual']=residual
    importance=np.average(np.abs(phi),axis=0,weights=weights)
    ranking=pd.DataFrame({'Feature':names,'Mean_Abs_SHAP':importance,'Mean_SHAP':np.average(phi,axis=0,weights=weights)})
    ranking=ranking.sort_values('Mean_Abs_SHAP',ascending=False,kind='stable').reset_index(drop=True);ranking['Rank']=np.arange(1,len(names)+1)
    selected_subjects=sorted(set(meta.iloc[bg].Subject_ID));test_subjects=sorted(set(sample.Subject_ID))
    manifest=dict(schema=SCHEMA,status='completed',scope='outer_test_explanation_only',fold=directory.name,
        model_type=kind,explainer='TreeExplainer' if kind=='rf' else 'KernelExplainer',
        explained_output='pipeline.predict_proba[:,1]' if kind=='rf' else 'pipeline.decision_function',
        positive_class=1,score_units='FOG probability' if kind=='rf' else 'SVM decision score',
        n_features=len(names),n_explained=len(chosen),n_test_windows=len(rows),n_background=len(bg),background_subjects=selected_subjects,
        test_subjects=test_subjects,background_strategy='round-robin subject × true class; training rows only',
        sampling=args.sampling,importance_weighting='sample weights; inverse sampling fraction within outcome for errors-balanced',
        nsamples=None if kind=='rf' else args.nsamples,l1_reg=l1,seed=args.seed,max_additivity_error=float(np.max(abs(residual))),
        shap_version=shap.__version__,sklearn_version=sklearn.__version__,numpy_version=np.__version__,
        approximation_note=('Tree SHAP interventional trên xác suất FOG; imputer từng cột, giữ số chiều. Không có l1 sparse selection.' if kind=='rf' else 'Kernel SHAP xấp xỉ. Với l1_reg num_features(k), mỗi lời giải có thể sparse; SHAP bằng 0 không chứng minh feature vô ích.'),
        interpretation_note='Không dùng outer-test SHAP để chọn features rồi báo cùng outer-test như một đánh giá độc lập.',
        dataset_paths=[str(Path(p).resolve()) for p in args.input],
        model_file=model_file(directory).name,model_sha256=hashlib.sha256(model_file(directory).read_bytes()).hexdigest())
    destination.mkdir(parents=True,exist_ok=True)
    atomic_json(destination/'manifest.json',dict(schema=SCHEMA,status='writing'))
    with tempfile.TemporaryDirectory(dir=destination) as td:
        td=Path(td)
        np.savez_compressed(td/'values.npz',shap_values=phi,feature_values=values,feature_names=np.asarray(names,str),
            base_value=np.asarray(base),scores=scores[selected],thresholds=thresholds[selected],dataset_rows=chosen,
            sample_weights=weights,schema=np.asarray(SCHEMA))
        sample.to_csv(td/'samples.csv',index=False)
        background_meta=meta.iloc[bg].copy();background_meta.insert(0,'Dataset_Row',bg);background_meta.to_csv(td/'background_rows.csv',index=False)
        ranking.to_csv(td/'importance.csv',index=False)
        (td/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        for name in ['values.npz','samples.csv','background_rows.csv','importance.csv','manifest.json']:os.replace(td/name,destination/name)
    return manifest


def build_parser():
    p=argparse.ArgumentParser(description='SHAP cho LOSO: SVM dùng KernelExplainer, RF dùng TreeExplainer xác suất FOG.')
    p.add_argument('--input',nargs='+',required=True,help='NPZ gốc, theo đúng thứ tự lúc train.')
    p.add_argument('--results',required=True,help='Thư mục kết quả LOSO đã giải nén, có fold models và manifest.')
    p.add_argument('--folds',nargs='+',help='Tên fold folder hoặc Subject_ID, ví dụ S06 S01.')
    p.add_argument('--background',type=int,default=24)
    p.add_argument('--explain',type=int,default=20)
    p.add_argument('--nsamples',type=int,default=256)
    p.add_argument('--l1-reg',default='num_features(20)',choices=['num_features(10)','num_features(20)','num_features(40)','aic','bic','0'])
    p.add_argument('--sampling',choices=['uniform','errors-balanced'],default='uniform')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--overwrite',action='store_true')
    return p


def run(args):
    root=Path(args.results).resolve()
    if not root.is_dir():raise ValueError('--results cần thư mục đã giải nén, không phải ZIP.')
    if min(args.background,args.explain,args.nsamples)<1:raise ValueError('Các giới hạn số mẫu phải >0.')
    folders=sorted(p for p in root.glob('fold_*') if p.is_dir())
    if args.folds:
        tokens=set(args.folds);folders=[p for p in folders if p.name in tokens or p.name.split('_')[-1] in tokens]
        matched={t for t in tokens if any(p.name==t or p.name.split('_')[-1]==t for p in folders)}
        if matched!=tokens:raise ValueError('Không tìm thấy folds: '+', '.join(sorted(tokens-matched)))
    if not folders:raise ValueError('Không tìm thấy fold folders.')
    x,meta,names,config_ids=read_dataset(args.input)
    completed=[]
    hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.input]
    for i,folder in enumerate(folders,1):
        provenance_path=root/'dataset_info.json'
        if provenance_path.exists():
            provenance=json.loads(provenance_path.read_text(encoding='utf-8-sig')).get('input_sha256')
            if provenance and hashes!=list(provenance.values()):raise ValueError('SHA256 NPZ khác dữ liệu dùng train RF; dùng đúng NPZ và thứ tự input.')
        started=time.monotonic();print(f'SHAP {i}/{len(folders)}: {folder.name}; background <= {args.background}, explain <= {args.explain}, nsamples={args.nsamples}',flush=True)
        if (folder/'shap/manifest.json').exists() and not args.overwrite:
            prior=json.loads((folder/'shap/manifest.json').read_text(encoding='utf-8'))
            if prior.get('status')!='completed':raise ValueError('SHAP lần trước chưa hoàn tất; dùng --overwrite để tính lại.')
            print('Đã có SHAP, bỏ qua. Dùng --overwrite để tính lại.',flush=True);continue
        completed.append(explain_fold(root,folder,x,meta,names,config_ids,args))
        print(f'Hoàn tất {folder.name}: {time.monotonic()-started:.1f}s',flush=True)
    return completed


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:run(args)
    except (ValueError,OSError,KeyError,ImportError,RuntimeError,Warning) as exc:
        raise SystemExit('ERROR: '+str(exc)) from None

if __name__=='__main__':main()
