"""FOG training/evaluation: original features and loader, StandardScaler + SVC.

Compatible with FOG_FIXED_1_1. Raw TXT is NOT an ML training input.
Input: one prepared CSV/NPZ configuration with Subject_ID metadata.

Modes:
- holdout (default): existing grouped holdout and validation functionality.
- loso baseline: leave one entire subject out, fixed C/gamma/threshold.
- loso --threshold-tuning: ONE inner SGKF selects threshold, fixed C/gamma.
- loso --tune --threshold-tuning: ONE inner SGKF jointly selects C/gamma/threshold.
  No GridSearch is run inside the inner fold, so this is TWO CV levels.

Examples (use a new output folder for each LOSO run):
  python train_fog.py --input fog_batch.npz --output holdout_run --test-subjects S02 S06 S09
  python train_fog.py --input fog_batch.npz --output loso_baseline --evaluation loso
  python train_fog.py --input fog_batch.npz --output loso_threshold --evaluation loso --threshold-tuning --threshold-method exact
  python train_fog.py --input fog_batch.npz --output loso_joint --evaluation loso --tune --threshold-tuning --threshold-method exact --cv-folds 3

Targets (.85/.85 by default) are constraints for inner selection, not guarantees
on outer-test patients. LOSO uses a best-balanced fallback when unmet, and reports
every subject. Null denotes undefined metrics. Outer models are evaluation models,
not a deployment model trained on all subjects. FOG iff score > saved threshold.
The legacy holdout --tune --threshold-tuning path retains its existing extra CV.
"""
RELEASE_VERSION = 'fog-unified-1.1'
import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ['Mean', 'RMS', 'Std', 'Variance', 'Min', 'Max', 'SMA',
            'Dominant_Frequency', 'Total_Power', 'Loco_Power', 'Freeze_Power',
            'Freeze_Index', 'Low_Freq_Ratio', 'PSE']
SENSORS = ['Ankle', 'Thigh', 'Trunk']
AXES = ['Forward', 'Vertical', 'Lateral']
ALIASES = {
    'subject_id': 'Subject_ID', 'Subject': 'Subject_ID',
    'recording_id': 'Recording_ID', 'Recording': 'Recording_ID',
    'segment_id': 'Segment_ID', 'window_id': 'Window_ID',
    'window_start': 'Start_Time', 'window_end': 'End_Time',
    'fog_ratio': 'FOG_Ratio', 'config_id': 'Config_ID', 'y': 'Binary_Label',
}
META = ['Subject_ID', 'Recording_ID', 'Segment_ID', 'Window_ID', 'Start_Time', 'End_Time']


def discover_inputs(inputs):
    """Prefer NPZ over its same-stem CSV when discovering directory exports."""
    result = []
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise ValueError(f'Input không tồn tại: {path}')
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in ('.csv', '.npz'))
            files = [p for p in files if not p.stem.lower().endswith(
                ('_report', '_manifest', '_predictions', '_results'))]
            npz_stems = {p.stem for p in files if p.suffix.lower() == '.npz'}
            files = [p for p in files if p.suffix.lower() != '.csv' or p.stem not in npz_stems]
            result.extend(files)
        elif path.suffix.lower() in ('.csv', '.npz'):
            result.append(path)
        else:
            raise ValueError(f'Chỉ nhận feature CSV/NPZ, không nhận raw TXT: {path.name}')
    result = list(dict.fromkeys(result))
    if not result:
        raise ValueError('Không có feature CSV/NPZ trong thư mục input.')
    return result


def _normalize_columns(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip().replace(' ', '_') for c in frame.columns]
    frame.rename(columns=ALIASES, inplace=True)
    if frame.columns.duplicated().any():
        raise ValueError('Tên cột trùng sau chuẩn hóa.')
    return frame


def _read_file(path):
    config = None
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            if not {'X', 'y', 'feature_names'}.issubset(data.files):
                raise ValueError('NPZ cần X, y, feature_names.')
            names = np.asarray(data['feature_names']).astype(str).tolist()
            X = np.asarray(data['X'], dtype=float)
            y = np.asarray(data['y'])
            if X.ndim != 2 or y.ndim != 1 or X.shape != (len(y), len(names)):
                raise ValueError('Kích thước X/y/feature_names không khớp.')
            frame = pd.DataFrame(X, columns=names)
            frame['Binary_Label'] = y
            reserved = {'X', 'y', 'feature_names', 'config_json'}
            for key in data.files:
                if key in reserved:
                    continue
                array = np.asarray(data[key])
                if array.ndim != 1 or len(array) != len(y):
                    raise ValueError(f'Độ dài metadata không hợp lệ: {key}')
                if key in frame.columns:
                    raise ValueError(f'Metadata trùng feature: {key}')
                frame[key] = array
            if 'config_json' in data.files:
                config = json.loads(str(data['config_json'].item()))
    else:
        frame = pd.read_csv(path)
    sidecar = path.with_name(path.stem + '_config.json')
    if sidecar.exists():
        external_config = json.loads(sidecar.read_text(encoding='utf-8'))
        if config is not None and config != external_config:
            raise ValueError('config_json trong NPZ khác config JSON bên cạnh file.')
        config = external_config
    return _normalize_columns(frame), config


def _metadata(frame, path):
    match = re.search(r'(S\d+R\d+)', path.stem, re.IGNORECASE)
    if 'Recording_ID' not in frame:
        if not match:
            raise ValueError('Thiếu Recording_ID. Xuất từ Batch, hoặc đặt tên file theo SxxRxx_features.')
        frame['Recording_ID'] = match.group(1).upper()
    if frame.Recording_ID.isna().any():
        raise ValueError('Recording_ID bị thiếu.')
    frame['Recording_ID'] = frame.Recording_ID.astype(str).str.strip().str.upper()
    if 'Subject_ID' not in frame:
        frame['Subject_ID'] = frame.Recording_ID.str.extract(r'^(S\d+)', expand=False)
    if frame.Subject_ID.isna().any():
        raise ValueError('Không suy ra Subject_ID; cần metadata người tham gia.')
    frame['Subject_ID'] = frame.Subject_ID.astype(str).str.strip().str.upper()
    if frame.Subject_ID.isin(['', 'UNKNOWN', 'NAN', 'NONE']).any():
        raise ValueError('Subject_ID không xác định; không thể chia test theo người tham gia.')
    inferred = frame.Recording_ID.str.extract(r'^(S\d+)', expand=False)
    if ((inferred.notna()) & (inferred != frame.Subject_ID)).any():
        raise ValueError('Subject_ID mâu thuẫn với Recording_ID.')
    for key in ['Window_ID', 'Start_Time', 'End_Time']:
        if key not in frame:
            raise ValueError(f'Thiếu metadata {key}; hãy xuất lại từ app.')
        frame[key] = pd.to_numeric(frame[key], errors='raise')
    if 'Segment_ID' not in frame:
        frame['Segment_ID'] = 0
    frame['Segment_ID'] = pd.to_numeric(frame.Segment_ID, errors='raise')
    if not np.isfinite(frame[['Window_ID', 'Segment_ID', 'Start_Time', 'End_Time']].to_numpy(float)).all():
        raise ValueError('Metadata thời gian/ID chứa NaN/Inf.')
    for key in ['Window_ID', 'Segment_ID']:
        if (frame[key] < 0).any() or (frame[key] != np.floor(frame[key])).any():
            raise ValueError(f'{key} phải là số nguyên không âm.')
        frame[key] = frame[key].astype(np.int64)
    if (frame.End_Time <= frame.Start_Time).any():
        raise ValueError('End_Time phải lớn hơn Start_Time.')
    if 'Binary_Label' not in frame:
        if 'Label' not in frame:
            raise ValueError('Không có Binary_Label hoặc Label.')
        frame['Binary_Label'] = frame.Label
    frame['Binary_Label'] = pd.to_numeric(frame.Binary_Label, errors='raise')
    if not frame.Binary_Label.isin([0, 1]).all():
        raise ValueError('Nhãn ML phải là binary 0=Non-FOG, 1=FOG; không nhận nhãn gốc 0/1/2.')
    if 'Original_Label' in frame and (frame.Original_Label == 0).any():
        raise ValueError('Có Original_Label 0 trong feature dataset; loại đoạn ngoài thí nghiệm trước windowing.')
    if 'Config_ID' in frame:
        ids = frame.Config_ID.astype(str).unique()
        if len(ids) != 1:
            raise ValueError('Một input chứa nhiều Config_ID; tách theo cấu hình trước khi train.')
    return frame


def _pivot_long(frame):
    """Combine axis rows into a single window without losing labels/metadata."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError('Long CSV/NPZ thiếu features: ' + ', '.join(sorted(missing)))
    frame = frame.copy()
    frame['Sensor'] = frame.Sensor.astype(str).str.title()
    frame['Axis'] = frame.Axis.astype(str).str.title().replace(
        {'X': 'Forward', 'Y': 'Vertical', 'Z': 'Lateral'})
    if not frame.Sensor.isin(SENSORS).all() or not frame.Axis.isin(AXES).all():
        raise ValueError('Sensor/Axis không thuộc Ankle/Thigh/Trunk và X/Y/Z (Forward/Vertical/Lateral).')
    frame['_channel'] = frame.Sensor + '_' + frame.Axis
    channels = [s + '_' + a for s in SENSORS for a in AXES
                if (s + '_' + a) in set(frame._channel)]
    identities = ['Subject_ID', 'Recording_ID', 'Window_ID']
    grouped = frame.groupby(identities, sort=False, dropna=False)
    for key in ['Segment_ID', 'Start_Time', 'End_Time', 'Binary_Label']:
        if (grouped[key].nunique(dropna=False) != 1).any():
            raise ValueError(f'Metadata {key} không nhất quán giữa các axis trong một window.')
    if frame.duplicated(identities + ['_channel']).any():
        raise ValueError('Window/axis trùng; không đưa đồng thời CSV và NPZ của cùng dataset.')
    if (grouped.size() != len(channels)).any():
        raise ValueError('Các window không có cùng bộ sensor/axis; hãy xuất cùng lựa chọn channels.')
    base = grouped[META + ['Binary_Label']].first().reset_index(drop=True)
    # Match the original window order exactly, regardless of pivot sorting.
    index = pd.MultiIndex.from_frame(base[identities])
    for feature in FEATURES:
        pivot = frame.pivot(index=identities, columns='_channel', values=feature).reindex(index)
        for channel in channels:
            base[channel + '_' + feature] = pivot[channel].to_numpy()
    names = [channel + '_' + feature for channel in channels for feature in FEATURES]
    return base, names


def _wide_features(frame):
    names = []
    for sensor in SENSORS:
        for axis in AXES:
            prefix = sensor + '_' + axis + '_'
            present = [prefix + f for f in FEATURES if prefix + f in frame]
            if present and len(present) != len(FEATURES):
                raise ValueError('Bộ feature không đầy đủ cho channel ' + prefix)
            names.extend(present)
    if not names:
        raise ValueError('Không có feature window. Chỉ nhận dữ liệu feature, không nhận raw signal.')
    return names


def _config_signature(config):
    """Ignore inactive GUI fields; retain feature algorithm family and active settings."""
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError('Configuration phải là JSON object.')
    if config.get('schema') == 'original-core-gui-v1':
        c = config['controls']
        scope = config['selection']
        value = {'family': 'original-core-gui-v1', 'source_fs': config.get('source_fs'),
                 'fs': c['target_fs_spinbox'] if c['resample_checkbox'] else config.get('source_fs', 64),
                 'filter': None, 'detrend': c['detrend_type_combo'] if c['detrend_checkbox'] else None,
                 'window': c['window_length_spinbox'], 'overlap': c['window_overlap_spinbox'],
                 'label_rule': c['window_label_mode_combo'], 'threshold': None,
                 'selection': {k: scope[k] for k in ('mode', 'exclude_zero')}}
        if c['filter_checkbox']:
            value['filter'] = [config.get('filter_implementation'), c['filter_type_combo'],
                               c['filter_order_spinbox'], c['cutoff_1_spinbox']]
            if c['filter_type_combo'] == 'Band-pass':
                value['filter'].append(c['cutoff_2_spinbox'])
        if c['window_label_mode_combo'] == 'FOG Threshold':
            value['threshold'] = c['window_fog_threshold_spinbox']
        if scope['mode'] == 1:
            value['selection'].update(start=scope['start'], end=scope['end'])
    else:
        # Modular data_prepare already generates its canonical config_id.
        if 'config_id' not in config:
            raise ValueError('Configuration không được nhận dạng (thiếu config_id).')
        value = {'family': config.get('feature_version', 'modular-data-prepare'), 'id': config['config_id']}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_datasets(paths):
    """Return the original API tuple: arrays, feature_names, processing metadata."""
    tables, names, schemas, configs, files, signatures, notes = [], None, [], [], [], set(), []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        try:
            frame, config = _read_file(path)
            if frame.empty:
                raise ValueError('Dataset rỗng.')
            frame = _metadata(frame, path)
            if config is not None and 'Config_ID' in frame:
                if str(frame.Config_ID.iloc[0]) != str(config.get('config_id')):
                    raise ValueError('Config_ID của dữ liệu khác JSON configuration.')
            is_long = 'Sensor' in frame and 'Axis' in frame
            schema = 'long-pivoted' if is_long else 'wide'
            if is_long:
                frame, current = _pivot_long(frame)
            else:
                current = _wide_features(frame)
            if frame.empty:
                raise ValueError('Dataset rỗng.')
            X = frame[current].apply(pd.to_numeric, errors='raise').to_numpy(float)
            if not np.isfinite(X).all():
                raise ValueError('Feature chứa NaN/Inf.')
            if names is not None and current != names:
                raise ValueError('Bộ feature/channels khác các input trước; không ghép cấu hình sensor khác nhau.')
            names = current
            signature = _config_signature(config)
            if signature is not None:
                signatures.add(signature)
            else:
                notes.append(f'{path.name}: không có config JSON; chưa xác minh được preprocessing/feature algorithm.')
            if len(set(schemas + [schema])) > 1:
                raise ValueError('Không trộn export long của core gốc và wide của data_prepare; công thức feature có khác biệt.')
            table = frame[META + ['Binary_Label'] + names].copy()
            table['Source_File'] = str(path)
            tables.append(table)
            files.append(str(path));schemas.append(schema);configs.append(config)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            raise ValueError(f'{path.name}: {exc}') from exc
    if not tables:
        raise ValueError('Không có dataset.')
    if len(signatures) > 1:
        raise ValueError('Các dataset dùng cấu hình preprocessing/windowing khác nhau.')
    combined = pd.concat(tables, ignore_index=True)
    # Two exports of the same windows must not be counted twice, even with changed IDs.
    identity = ['Subject_ID', 'Recording_ID', 'Start_Time', 'End_Time']
    id_keys = ['Subject_ID', 'Recording_ID', 'Segment_ID', 'Window_ID']
    if combined.duplicated(identity).any() or combined.duplicated(id_keys).any():
        raise ValueError('Có window trùng giữa các input; chỉ chọn một bản CSV hoặc NPZ cho mỗi export.')
    if combined.groupby('Recording_ID').Subject_ID.nunique().max() > 1:
        raise ValueError('Một Recording_ID thuộc nhiều Subject_ID.')
    arrays = {'X': combined[names].to_numpy(float), 'y': combined.Binary_Label.to_numpy(np.int8),
              'subject_id': combined.Subject_ID.to_numpy(str), 'recording_id': combined.Recording_ID.to_numpy(str),
              'segment_id': combined.Segment_ID.to_numpy(np.int64), 'window_id': combined.Window_ID.to_numpy(np.int64),
              'window_start': combined.Start_Time.to_numpy(float), 'window_end': combined.End_Time.to_numpy(float),
              'source_file': combined.Source_File.to_numpy(str)}
    metadata = {'input_files': files, 'input_schema': schemas[0], 'configurations': configs,
                'config_verified': not notes, 'notes': notes, 'signature': next(iter(signatures), None)}
    return arrays, names, metadata


def _split(data, test_size, seed, test_subjects):
    y = data['y']; groups = data['subject_id']
    if len(np.unique(groups)) < 2:
        raise ValueError('Cần ít nhất 2 người tham gia; S10R01 riêng lẻ không đủ để đánh giá theo subject.')
    if len(np.unique(y)) < 2:
        raise ValueError('Dataset cần cả Non-FOG=0 và FOG=1.')
    if test_subjects:
        selected = {s.upper() for s in test_subjects}
        missing = selected - set(groups)
        if missing:
            raise ValueError('Test subject không tồn tại: ' + ', '.join(sorted(missing)))
        test_idx = np.flatnonzero(np.isin(groups, list(selected)))
        train_idx = np.flatnonzero(~np.isin(groups, list(selected)))
    else:
        if not 0 < test_size < 1:
            raise ValueError('--test-size phải nằm trong (0,1).')
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(data['X'], y, groups))
    if not len(train_idx) or not len(test_idx):
        raise ValueError('Train và test đều cần ít nhất một subject.')
    for label, idx in [('train', train_idx), ('test', test_idx)]:
        if len(np.unique(y[idx])) < 2:
            raise ValueError(f'Tập {label} thiếu một lớp. Chỉ định --test-subjects theo thiết kế thí nghiệm; chương trình không tự thử seed đến khi đạt kết quả tốt.')
    return train_idx, test_idx



def diagnostic_metrics(y_true, y_pred):
    """Window-level metrics with FOG=positive(1), Non-FOG=negative(0).

    Undefined ratios are None (JSON null), rather than being reported as zero.
    Confusion matrix order is [[TN, FP], [FN, TP]].
    """
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if truth.ndim != 1 or predicted.shape != truth.shape or not truth.size:
        raise ValueError('Evaluation cần hai mảng nhãn 1D, cùng độ dài và không rỗng.')
    if not np.isin(truth, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError('Evaluation chỉ nhận nhãn 0=Non-FOG, 1=FOG.')
    tn, fp, fn, tp = map(int, confusion_matrix(truth, predicted, labels=[0, 1]).ravel())
    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None
    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ratio(tp, tp + fp),
        'npv': ratio(tn, tn + fn),
        'accuracy': ratio(tp + tn, tp + tn + fp + fn),
        'youden_index': (sensitivity + specificity - 1.0
                         if sensitivity is not None and specificity is not None else None),
        'confusion_counts': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp},
        'positive_class': 'FOG (1)',
        'evaluation_unit': 'window',
        'undefined_metric_policy': 'JSON null if denominator is zero; classification_report retains sklearn zero_division=0.'
    }


TRAINER_VERSION = 'fog-unified-1.1-loso-2level-1'


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def _group_folds(X, y, groups, n_splits, seed, require_validation_classes=False):
    if not 2 <= n_splits <= len(np.unique(groups)):
        raise ValueError('Số CV fold cần từ 2 đến số bệnh nhân của tập fit.')
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(X, y, groups))
    for fold, (fit_idx, val_idx) in enumerate(folds, 1):
        if set(groups[fit_idx]) & set(groups[val_idx]):
            raise ValueError(f'Fold {fold}: bệnh nhân trùng giữa fit và validation.')
        if len(np.unique(y[fit_idx])) < 2:
            raise ValueError(f'Fold {fold}: tập fit thiếu lớp; đổi thiết kế CV.')
        if require_validation_classes and len(np.unique(y[val_idx])) < 2:
            raise ValueError(f'Fold {fold}: validation thiếu lớp cho GridSearch F1; đổi thiết kế CV.')
    return folds


def _fit_model(pipeline, X, y, groups, *, tune, kernel, cv_folds,
               seed, n_jobs, prefix='Final'):
    if not tune:
        return clone(pipeline).fit(X, y), None, None, None, None
    folds = _group_folds(X, y, groups, cv_folds, seed,
                         require_validation_classes=True)
    grid = {'svm__C': [.1, 1., 10., 100.]}
    if kernel == 'rbf':
        grid['svm__gamma'] = ['scale', .01, .1]
    search = GridSearchCV(clone(pipeline), grid, scoring='f1', cv=folds,
                          n_jobs=n_jobs, refit=True, error_score='raise',
                          return_train_score=False)
    print(f'{prefix}: tuning C/gamma, {cv_folds} grouped folds, F1(FOG).', flush=True)
    search.fit(X, y)
    records = pd.DataFrame([
        {'Fold': i, 'Fit_Subjects': ','.join(np.unique(groups[a])),
         'Validation_Subjects': ','.join(np.unique(groups[b]))}
        for i, (a, b) in enumerate(folds, 1)])
    return (search.best_estimator_, pd.DataFrame(search.cv_results_),
            search.best_params_, float(search.best_score_), records)


def patient_validation_scores(pipeline, X, y, groups, *, cv_folds=3, seed=42,
                              tune=False, inner_cv_folds=3, kernel='rbf', n_jobs=1):
    """Out-of-fold scores; outer-validation subjects never enter scaler/SVM or inner tuning."""
    scores = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    records = []
    folds = _group_folds(X, y, groups, cv_folds, seed)
    for fold, (fit_idx, val_idx) in enumerate(folds, 1):
        print(f'Validation fold {fold}/{cv_folds}: fit '
              f'{list(np.unique(groups[fit_idx]))}; validation '
              f'{list(np.unique(groups[val_idx]))}', flush=True)
        model, _, params, cv_score, _ = _fit_model(
            pipeline, X[fit_idx], y[fit_idx], groups[fit_idx],
            tune=tune, kernel=kernel, cv_folds=inner_cv_folds,
            seed=seed, n_jobs=n_jobs, prefix=f'Outer fold {fold}')
        if model.classes_.tolist() != [0, 1]:
            raise ValueError('SVM classes phải là [0, 1] để score dương hướng về FOG.')
        scores[val_idx] = model.decision_function(X[val_idx])
        fold_ids[val_idx] = fold
        records.append({
            'Fold': fold, 'Fit_Subjects': ','.join(np.unique(groups[fit_idx])),
            'Validation_Subjects': ','.join(np.unique(groups[val_idx])),
            'Fit_Windows': len(fit_idx), 'Validation_Windows': len(val_idx),
            'Validation_FOG': int(np.sum(y[val_idx] == 1)),
            'Best_Params': json.dumps(params), 'Best_Inner_CV_F1': cv_score})
    if not np.isfinite(scores).all() or np.any(fold_ids < 1):
        raise ValueError('Có cửa sổ chưa nhận score validation hợp lệ.')
    return scores, fold_ids, pd.DataFrame(records)


def sweep_validation_thresholds(y, scores, *, min_sensitivity=.85,
                                min_specificity=.85, method='grid',
                                threshold_min=-1.5, threshold_max=.5,
                                threshold_step=.02):
    """FOG iff score > threshold. Counts include score ties consistently."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError('Scores validation không hợp lệ.')
    if len(np.unique(y)) != 2 or not np.isin(y, [0, 1]).all():
        raise ValueError('Validation gộp cần cả FOG và Non-FOG.')
    if not all(np.isfinite(v) and 0 <= v <= 1
               for v in (min_sensitivity, min_specificity)):
        raise ValueError('Mục tiêu Sensitivity/Specificity cần nằm trong [0,1].')
    if method == 'exact':
        unique = np.unique(scores)
        lower = np.nextafter(unique[0], -np.inf)
        if not np.isfinite(lower):
            raise ValueError('Score quá lớn để tạo ngưỡng exact.')
        thresholds = np.r_[lower, unique]
    elif method == 'grid':
        if (not all(np.isfinite(v) for v in (threshold_min, threshold_max, threshold_step))
                or threshold_min >= threshold_max or threshold_step <= 0):
            raise ValueError('Khoảng quét cần min < max, step >0 và hữu hạn.')
        if (threshold_max - threshold_min) / threshold_step > 20000:
            raise ValueError('Quá nhiều ngưỡng grid; tăng step hoặc dùng exact.')
        count = int(np.floor((threshold_max - threshold_min) / threshold_step))
        thresholds = np.unique(np.r_[threshold_min + np.arange(count + 1) * threshold_step,
                                     threshold_max])
    else:
        raise ValueError('Threshold method cần grid hoặc exact.')
    order = np.argsort(scores)
    sorted_scores, sorted_y = scores[order], y[order]
    prefix_positive = np.r_[0, np.cumsum(sorted_y == 1)]
    cut = np.searchsorted(sorted_scores, thresholds, side='right')
    fn = prefix_positive[cut]
    tn = cut - fn
    positives = int(np.sum(y == 1)); negatives = len(y) - positives
    tp = positives - fn; fp = negatives - tn
    sens, spec = tp / positives, tn / negatives
    ppv = np.divide(tp, tp + fp, out=np.full(len(tp), np.nan), where=(tp + fp) != 0)
    npv = np.divide(tn, tn + fn, out=np.full(len(tp), np.nan), where=(tn + fn) != 0)
    f1 = 2 * tp / (2 * tp + fp + fn)
    table = pd.DataFrame({
        'Decision_Threshold': thresholds, 'Sensitivity': sens, 'Specificity': spec,
        'Accuracy': (tp + tn) / len(y), 'PPV': ppv, 'NPV': npv,
        'F1_FOG': f1, 'Balanced_Accuracy': (sens + spec) / 2,
        'Youden_Index': sens + spec - 1,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Meets_Target': (sens >= min_sensitivity) & (spec >= min_specificity)})
    eligible = table[table.Meets_Target]
    selected = None
    if not eligible.empty:
        # Max sensitivity; tie: max specificity; remaining tie: largest threshold.
        selected = float(eligible.sort_values(
            ['Sensitivity', 'Specificity', 'Decision_Threshold'],
            ascending=[False, False, False]).iloc[0].Decision_Threshold)
    return table, selected


def _patient_metrics(y, scores, groups, threshold):
    records = []
    for subject in np.unique(groups):
        mask = groups == subject
        truth = y[mask]; prediction = (scores[mask] > threshold).astype(np.int8)
        result = diagnostic_metrics(truth, prediction)
        records.append({
            'Subject_ID': subject, 'Decision_Threshold': float(threshold),
            'N_Windows': int(np.sum(mask)), 'N_FOG': int(np.sum(truth == 1)),
            'Sensitivity': result['sensitivity'], 'Specificity': result['specificity'],
            'Accuracy': result['accuracy'], 'PPV': result['ppv'], 'NPV': result['npv'],
            'Youden_Index': result['youden_index'],
            'F1_FOG': float(f1_score(truth, prediction, zero_division=0)),
            **result['confusion_counts']})
    return pd.DataFrame(records)


def predict_with_threshold(bundle, X):
    """Inference on unscaled feature vectors in bundle feature_names order."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(bundle['feature_names']) or not np.isfinite(X).all():
        raise ValueError('X inference sai kích thước hoặc chứa NaN/Inf.')
    score = bundle['pipeline'].decision_function(X)
    threshold = float(bundle.get('decision_threshold', 0.))
    return (score > threshold).astype(np.int8), score


def _manifest(data, train_idx, test_idx):
    split = np.full(len(data['y']), 'train', dtype='<U5')
    split[test_idx] = 'test'
    return pd.DataFrame({
        'Subject_ID': data['subject_id'], 'Recording_ID': data['recording_id'],
        'Segment_ID': data['segment_id'], 'Window_ID': data['window_id'],
        'Start_Time': data['window_start'], 'End_Time': data['window_end'],
        'Binary_Label': data['y'], 'Source_File': data['source_file'], 'Split': split})


def _save_validation(output, manifest, train_idx, y, groups, scores, fold_ids,
                     folds, sweep, selected, targets, nested):
    target_met = selected is not None
    # When target unmet, report the best balanced candidate ONLY for diagnostics.
    diagnostic = float(sweep.sort_values(
        ['Balanced_Accuracy', 'F1_FOG', 'Decision_Threshold'],
        ascending=[False, False, False]).iloc[0].Decision_Threshold)
    reporting = float(selected) if target_met else diagnostic
    role = 'selected_on_validation' if target_met else 'diagnostic_only_target_unmet'
    predictions = (scores > reporting).astype(np.int8)
    sweep.to_csv(output / 'validation_threshold_sweep.csv', index=False)
    folds.to_csv(output / 'validation_folds.csv', index=False)
    windows = manifest.iloc[train_idx].copy()
    windows['Split'] = 'validation_oof'
    windows['Validation_Fold'] = fold_ids
    windows['FOG_Decision_Score'] = scores
    windows['Prediction'] = predictions
    windows['Decision_Threshold'] = reporting
    windows['Threshold_Role'] = role
    windows.to_csv(output / 'validation_scores.csv', index=False)
    patients = _patient_metrics(y, scores, groups, reporting)
    patients['Threshold_Role'] = role
    patients.to_csv(output / 'validation_patient_metrics.csv', index=False)
    fold_metrics = []
    for fold in np.unique(fold_ids):
        mask = fold_ids == fold
        result = diagnostic_metrics(y[mask], predictions[mask])
        fold_metrics.append({'Fold': int(fold), 'Decision_Threshold': reporting,
                             'Threshold_Role': role,
                             **{k: result[k] for k in ('sensitivity', 'specificity', 'accuracy')},
                             **result['confusion_counts']})
    pd.DataFrame(fold_metrics).to_csv(output / 'validation_fold_metrics.csv', index=False)
    summary = {'status': 'target_met' if target_met else 'target_not_met',
               'target_met': target_met, 'selected_threshold': selected,
               'diagnostic_candidate_threshold': diagnostic,
               'reporting_threshold': reporting, 'reporting_threshold_role': role,
               'targets': targets, 'nested_hyperparameter_tuning': bool(nested),
               'selection_rule': 'max sensitivity subject to both targets; ties: specificity, threshold',
               'evaluation_unit': 'window; pooled OOF counts, not mean patient sensitivity',
               'n_validation_windows': len(y), 'subjects': np.unique(groups).tolist(),
               'reporting_metrics': diagnostic_metrics(y, predictions),
               'notes': ['Validation scores select a threshold; they are not an independent final performance estimate.',
                         'Each outer fold fits its own scaler and SVM; refitting can change score scale.',
                         'Only thresholds in the exported sweep were considered.']}
    _write_json(output / 'validation_metrics.json', summary)
    return summary


def train(paths, output, test_size=.25, seed=42, *, kernel='rbf', C=1., gamma='scale',
          tune=False, cv_folds=3, test_subjects=None, n_jobs=1, cache_size=512.,
          decision_threshold=0., threshold_tuning=False, min_sensitivity=.85,
          min_specificity=.85, threshold_method='grid', threshold_min=-1.5,
          threshold_max=.5, threshold_step=.02, inner_cv_folds=3):
    if kernel not in ('rbf', 'linear') or not np.isfinite(C) or C <= 0 or not np.isfinite(cache_size) or cache_size <= 0:
        raise ValueError('Kernel phải là rbf/linear; C và cache-size phải >0.')
    if isinstance(gamma, (float, int)):
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError('Gamma phải >0 và hữu hạn.')
    elif gamma not in ('scale', 'auto'):
        raise ValueError('Gamma cần scale/auto hoặc số >0.')
    if not np.isfinite(decision_threshold):
        raise ValueError('Decision threshold phải hữu hạn.')
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (min_sensitivity, min_specificity)):
        raise ValueError('Mục tiêu Sensitivity/Specificity cần nằm trong [0,1].')
    if threshold_tuning and decision_threshold != 0.:
        raise ValueError('Không dùng --decision-threshold khác 0 cùng --threshold-tuning.')
    output = Path(output)
    if threshold_tuning and output.exists() and any(output.iterdir()):
        raise ValueError('Chọn thư mục output mới/rỗng cho validation để tránh trộn kết quả cũ.')
    data, names, config = load_datasets(paths)
    train_idx, test_idx = _split(data, test_size, seed, test_subjects)
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('svm', SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced',
                                     probability=False, cache_size=cache_size))])
    X_train, y_train = data['X'][train_idx], data['y'][train_idx]
    train_groups = data['subject_id'][train_idx]
    print(f'{len(data["y"]):,} windows · {len(names)} features · {len(np.unique(data["subject_id"]))} subjects', flush=True)
    print('Train:', ', '.join(np.unique(train_groups)), '| Test:', ', '.join(np.unique(data['subject_id'][test_idx])), flush=True)
    for note in config['notes']:
        print('NOTE:', note, flush=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(data, train_idx, test_idx)
    manifest.to_csv(output / 'split_manifest.csv', index=False)
    _write_json(output / 'dataset_info.json', config)
    _write_json(output / 'feature_names.json', names)
    validation_summary = None
    if threshold_tuning:
        val_scores, val_folds, fold_table = patient_validation_scores(
            pipeline, X_train, y_train, train_groups, cv_folds=cv_folds, seed=seed,
            tune=tune, inner_cv_folds=inner_cv_folds, kernel=kernel, n_jobs=n_jobs)
        sweep, selected = sweep_validation_thresholds(
            y_train, val_scores, min_sensitivity=min_sensitivity,
            min_specificity=min_specificity, method=threshold_method,
            threshold_min=threshold_min, threshold_max=threshold_max,
            threshold_step=threshold_step)
        targets = {'min_sensitivity': float(min_sensitivity),
                   'min_specificity': float(min_specificity), 'method': threshold_method,
                   'threshold_min': float(threshold_min) if threshold_method == 'grid' else None,
                   'threshold_max': float(threshold_max) if threshold_method == 'grid' else None,
                   'threshold_step': float(threshold_step) if threshold_method == 'grid' else None}
        validation_summary = _save_validation(
            output, manifest, train_idx, y_train, train_groups, val_scores, val_folds,
            fold_table, sweep, selected, targets, tune)
        if selected is None:
            stopped = {'status': 'target_not_met', 'trainer_version': TRAINER_VERSION,
                       'decision_threshold': None, 'test_evaluated': False,
                       'validation': validation_summary,
                       'train_subjects': np.unique(train_groups).tolist(),
                       'test_subjects': np.unique(data['subject_id'][test_idx]).tolist()}
            _write_json(output / 'metrics.json', stopped)
            print('Không có ngưỡng validation đạt cả hai mục tiêu trong bảng quét. '
                  'Đã lưu diagnostics; không fit model cuối hoặc đánh giá test.', flush=True)
            return stopped
        decision_threshold = selected
        print(f'Selected validation threshold: {decision_threshold:.8g}', flush=True)
    # Final fitting/tuning uses all training subjects, never test subjects.
    model, cv_results, best_params, best_score, final_cv_subjects = _fit_model(
        pipeline, X_train, y_train, train_groups, tune=tune, kernel=kernel,
        cv_folds=cv_folds, seed=seed, n_jobs=n_jobs)
    scores = model.decision_function(data['X'][test_idx])
    prediction = (scores > decision_threshold).astype(np.int8)
    y_test = data['y'][test_idx]
    report = classification_report(y_test, prediction, labels=[0, 1], target_names=['Non-FOG', 'FOG'],
                                   output_dict=True, zero_division=0)
    metrics = {'model': 'StandardScaler + SVC', 'kernel': kernel,
               'class_weight': 'balanced', 'decision_threshold': float(decision_threshold), 'seed': seed,
               'status': 'completed', 'trainer_version': TRAINER_VERSION,
               'threshold_selection': 'subject-grouped OOF validation' if threshold_tuning else 'fixed',
               'threshold_tuning': bool(threshold_tuning), 'validation': validation_summary,
               'accuracy': float(accuracy_score(y_test, prediction)),
               'balanced_accuracy': float(balanced_accuracy_score(y_test, prediction)),
               'fog_f1': float(f1_score(y_test, prediction, zero_division=0)),
               'roc_auc': float(roc_auc_score(y_test, scores)),
               'average_precision': float(average_precision_score(y_test, scores)),
               'classification_report': report,
               'confusion_matrix': confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
               'confusion_matrix_order': ['Non-FOG', 'FOG'],
               'train_subjects': np.unique(train_groups).tolist(),
               'test_subjects': np.unique(data['subject_id'][test_idx]).tolist(),
               'train_windows': len(train_idx), 'test_windows': len(test_idx), 'n_features': len(names),
               'tuning': bool(tune), 'cv_folds': cv_folds if tune else None,
               'best_cv_fog_f1': best_score, 'best_params': best_params,
               'svm_parameters': model.named_steps['svm'].get_params(),
               'support_vectors': model.named_steps['svm'].n_support_.tolist(),
               'config_verified': config['config_verified'], 'notes': config['notes'],
               'evaluation': 'One held-out subject split. Tuning/scaler fit only on training subjects. Test never used for parameter selection.',
               'versions': {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}}
    metrics.update(diagnostic_metrics(y_test, prediction))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    bundle = {'pipeline': model, 'feature_names': names, 'label_mapping': {0: 'Non-FOG', 1: 'FOG'},
              'processing_config': config, 'model_type': 'SVC', 'metrics': metrics,
              'train_subjects': metrics['train_subjects'], 'test_subjects': metrics['test_subjects']}
    bundle['decision_threshold'] = float(decision_threshold)
    bundle['trainer_version'] = TRAINER_VERSION
    joblib.dump(bundle, output / 'fog_svm.joblib')
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    evaluation_names = ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy',
                        'youden_index', 'fog_f1', 'balanced_accuracy', 'roc_auc', 'average_precision']
    pd.DataFrame([{'Metric': name, 'Value': metrics[name],
                   'Definition': {
                       'sensitivity': 'Recall FOG = TP/(TP+FN)',
                       'specificity': 'Recall Non-FOG = TN/(TN+FP)',
                       'ppv': 'Precision FOG = TP/(TP+FP)',
                       'npv': 'Precision Non-FOG = TN/(TN+FN)',
                       'accuracy': '(TP+TN)/(TP+TN+FP+FN)',
                       'youden_index': 'Sensitivity + Specificity - 1',
                       'fog_f1': 'F1 of FOG class',
                       'balanced_accuracy': '(Sensitivity + Specificity)/2',
                       'roc_auc': 'ROC AUC from SVM decision scores',
                       'average_precision': 'Average precision from SVM decision scores'
                   }[name], 'Unit': 'fraction; Youden ranges from -1 to 1'}
                  for name in evaluation_names]).to_csv(output / 'evaluation_metrics.csv', index=False)
    (output / 'dataset_info.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'feature_names.json').write_text(json.dumps(names, indent=2), encoding='utf-8')
    split = np.full(len(data['y']), 'train', dtype='<U5'); split[test_idx] = 'test'
    manifest = pd.DataFrame({'Subject_ID': data['subject_id'], 'Recording_ID': data['recording_id'],
                             'Segment_ID': data['segment_id'], 'Window_ID': data['window_id'],
                             'Start_Time': data['window_start'], 'End_Time': data['window_end'],
                             'Binary_Label': data['y'], 'Source_File': data['source_file'], 'Split': split})
    manifest.to_csv(output / 'split_manifest.csv', index=False)
    test_predictions = manifest.iloc[test_idx].copy()
    test_predictions['Prediction'] = prediction;test_predictions['FOG_Decision_Score'] = scores
    test_predictions['Decision_Threshold'] = float(decision_threshold)
    test_predictions.to_csv(output / 'test_predictions.csv', index=False)
    _patient_metrics(y_test, scores, data['subject_id'][test_idx], decision_threshold).to_csv(
        output / 'test_patient_metrics.csv', index=False)
    pd.DataFrame(metrics['confusion_matrix'], index=['Actual_NonFOG', 'Actual_FOG'],
                 columns=['Predicted_NonFOG', 'Predicted_FOG']).to_csv(output / 'confusion_matrix.csv')
    if cv_results is not None:
        cv_results.to_csv(output / 'cv_results.csv', index=False)
        final_cv_subjects.to_csv(output / 'cv_subjects.csv', index=False)
    print(f'Saved: {output / "fog_svm.joblib"}', flush=True)
    return metrics


def _loso_options(kernel, C, gamma, cache_size, decision_threshold,
                  min_sensitivity, min_specificity, n_jobs):
    if kernel not in ('rbf', 'linear') or not np.isfinite(C) or C <= 0:
        raise ValueError('Kernel cần rbf/linear và C >0.')
    if isinstance(gamma, (int, float)):
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError('Gamma phải hữu hạn và >0.')
    elif gamma not in ('scale', 'auto'):
        raise ValueError('Gamma cần scale/auto hoặc số >0.')
    if not np.isfinite(cache_size) or cache_size <= 0 or not np.isfinite(decision_threshold):
        raise ValueError('Cache-size cần >0; decision threshold phải hữu hạn.')
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (min_sensitivity, min_specificity)):
        raise ValueError('Mục tiêu metrics phải nằm trong [0,1].')
    if n_jobs == 0:
        raise ValueError('--n-jobs không được bằng 0.')


def _candidate_scores(pipeline, X, y, folds, params):
    """Fixed parameters inside each SGKF fold: no GridSearch or further split."""
    scores = np.full(len(y), np.nan)
    for fit_idx, validation_idx in folds:
        model = clone(pipeline).set_params(**params)
        model.fit(X[fit_idx], y[fit_idx])
        if model.classes_.tolist() != [0, 1]:
            raise ValueError('Classes SVM cần [0, 1].')
        scores[validation_idx] = model.decision_function(X[validation_idx])
    if not np.isfinite(scores).all():
        raise ValueError('Có cửa sổ thiếu score SGKF.')
    return scores


def select_joint_parameters(pipeline, X, y, groups, *, tune=False,
                            threshold_tuning=False, cv_folds=3, seed=42,
                            kernel='rbf', decision_threshold=0.,
                            min_sensitivity=.85, min_specificity=.85,
                            threshold_method='grid', threshold_min=-1.5,
                            threshold_max=.5, threshold_step=.02,
                            C_grid=None, gamma_grid=None, n_jobs=1,
                            save_all_sweeps=False, output=None):
    """One SGKF level selects C/gamma/threshold jointly on inner OOF predictions.

    Inner results select parameters; only the outer LOSO test estimates performance.
    When threshold targets are unmet, a predeclared best-balanced fallback is used.
    """
    folds = _group_folds(X, y, groups, cv_folds, seed)
    base = pipeline.get_params()
    c_values = list(C_grid if C_grid is not None else [.1, 1., 10., 100.]) if tune else [base['svm__C']]
    gamma_values = list(gamma_grid if gamma_grid is not None else ['scale', .01, .1]) if tune and kernel == 'rbf' else [base['svm__gamma']]
    if not c_values or not gamma_values:
        raise ValueError('Grid rỗng.')
    for c in c_values:
        _loso_options(kernel, c, gamma_values[0], 512., decision_threshold,
                      min_sensitivity, min_specificity, n_jobs)
    for g in gamma_values:
        _loso_options(kernel, c_values[0], g, 512., decision_threshold,
                      min_sensitivity, min_specificity, n_jobs)
    params_list = [{'svm__C': float(c), 'svm__gamma': g}
                   for c in dict.fromkeys(c_values) for g in dict.fromkeys(gamma_values)]
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_rows = []
    for fold, (a, b) in enumerate(folds, 1):
        fold_ids[b] = fold
        fold_rows.append({'Fold': fold,
                          'Fit_Subjects': ','.join(np.unique(groups[a])),
                          'Validation_Subjects': ','.join(np.unique(groups[b])),
                          'Fit_Windows': len(a), 'Validation_Windows': len(b)})
    print(f'Inner SGKF: {len(params_list)} candidates × {len(folds)} folds.', flush=True)
    # Parallelism is across parameter candidates, not an additional CV level.
    if n_jobs == 1:
        all_scores = [_candidate_scores(pipeline, X, y, folds, params)
                      for params in params_list]
    else:
        all_scores = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_candidate_scores)(pipeline, X, y, folds, params)
            for params in params_list)
    candidate_rows, sweeps = [], []
    for i, (params, scores) in enumerate(zip(params_list, all_scores), 1):
        if threshold_tuning:
            sweep, selected = sweep_validation_thresholds(
                y, scores, min_sensitivity=min_sensitivity,
                min_specificity=min_specificity, method=threshold_method,
                threshold_min=threshold_min, threshold_max=threshold_max,
                threshold_step=threshold_step)
            if selected is not None:
                row = sweep.loc[sweep.Decision_Threshold == selected].iloc[0]
            else:
                row = sweep.sort_values(
                    ['Balanced_Accuracy', 'F1_FOG', 'Specificity', 'Decision_Threshold'],
                    ascending=[False, False, False, False]).iloc[0]
            record = {k: row[k].item() if hasattr(row[k], 'item') else row[k]
                      for k in sweep.columns}
            record['Target_Met'] = selected is not None
            record['Threshold_Role'] = 'target_met' if selected is not None else 'fallback_best_balanced'
        else:
            pred = (scores > decision_threshold).astype(np.int8)
            result = diagnostic_metrics(y, pred)
            record = {'Decision_Threshold': float(decision_threshold),
                      'Sensitivity': result['sensitivity'], 'Specificity': result['specificity'],
                      'Accuracy': result['accuracy'], 'PPV': result['ppv'], 'NPV': result['npv'],
                      'F1_FOG': float(f1_score(y, pred, zero_division=0)),
                      'Balanced_Accuracy': (result['sensitivity'] + result['specificity']) / 2,
                      'Youden_Index': result['youden_index'],
                      **result['confusion_counts'], 'Target_Met': None,
                      'Threshold_Role': 'fixed_threshold'}
            sweep = pd.DataFrame([record])
        record.update(Candidate_ID=i, C=params['svm__C'], Gamma=params['svm__gamma'])
        candidate_rows.append(record)
        sweeps.append(sweep)
        if output is not None and save_all_sweeps:
            directory = Path(output) / 'candidate_threshold_sweeps'
            directory.mkdir(exist_ok=True)
            sweep.to_csv(directory / f'candidate_{i:02d}.csv', index=False)
    candidates = pd.DataFrame(candidate_rows)
    if threshold_tuning:
        eligible = candidates[candidates.Target_Met == True]
        if not eligible.empty:
            chosen = eligible.sort_values(
                ['Sensitivity', 'Specificity', 'Candidate_ID'],
                ascending=[False, False, True]).iloc[0]
            selection_rule = 'max sensitivity satisfying both targets; ties: specificity, candidate order'
        else:
            chosen = candidates.sort_values(
                ['Balanced_Accuracy', 'F1_FOG', 'Specificity', 'Candidate_ID'],
                ascending=[False, False, False, True]).iloc[0]
            selection_rule = 'targets unmet: best balanced accuracy; ties: F1, specificity, candidate order'
    else:
        chosen = candidates.sort_values(
            ['F1_FOG', 'Specificity', 'Candidate_ID'],
            ascending=[False, False, True]).iloc[0]
        selection_rule = 'max pooled inner OOF F1 at fixed threshold; ties: specificity, candidate order'
    chosen_id = int(chosen.Candidate_ID)
    summary = {'candidate_id': chosen_id,
               'C': float(chosen.C), 'gamma': chosen.Gamma,
               'decision_threshold': float(chosen.Decision_Threshold),
               'target_met': bool(chosen.Target_Met) if threshold_tuning else None,
               'threshold_role': chosen.Threshold_Role,
               'selection_rule': selection_rule,
               'cv_folds': cv_folds, 'n_candidates': len(candidates),
               'targets': {'min_sensitivity': min_sensitivity,
                           'min_specificity': min_specificity} if threshold_tuning else None,
               'threshold_method': threshold_method if threshold_tuning else 'fixed',
               'inner_metrics_are_for_selection_only': True,
               'score_scale_note': 'Each fold fits its own SVM; refitting may change score scale.'}
    if output is not None:
        candidates.to_csv(Path(output) / 'joint_candidates.csv', index=False)
        sweeps[chosen_id - 1].to_csv(Path(output) / 'selected_candidate_threshold_sweep.csv', index=False)
        pd.DataFrame(fold_rows).to_csv(Path(output) / 'inner_folds.csv', index=False)
        _write_json(Path(output) / 'selected_parameters.json', summary)
    return summary, all_scores[chosen_id - 1], fold_ids


def _outer_metrics(y, prediction, scores=None):
    result = diagnostic_metrics(y, prediction)
    counts = result['confusion_counts']
    n_positive = counts['TP'] + counts['FN']
    both_classes = len(np.unique(y)) == 2
    result.update(
        fog_f1=float(f1_score(y, prediction, zero_division=0)) if n_positive else None,
        balanced_accuracy=(result['sensitivity'] + result['specificity']) / 2 if both_classes else None,
        roc_auc=float(roc_auc_score(y, scores)) if both_classes and scores is not None else None,
        average_precision=float(average_precision_score(y, scores)) if both_classes and scores is not None else None,
        confusion_matrix=confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        confusion_matrix_order=['Non-FOG', 'FOG'])
    return result


def train_loso(paths, output, *, kernel='rbf', C=1., gamma='scale', seed=42,
               tune=False, threshold_tuning=False, cv_folds=3, n_jobs=1,
               cache_size=512., decision_threshold=0., min_sensitivity=.85,
               min_specificity=.85, threshold_method='grid', threshold_min=-1.5,
               threshold_max=.5, threshold_step=.02, C_grid=None, gamma_grid=None,
               save_models=True, save_all_sweeps=False):
    """LOSO outer test + ONE inner SGKF for joint parameter/threshold selection."""
    _loso_options(kernel, C, gamma, cache_size, decision_threshold,
                  min_sensitivity, min_specificity, n_jobs)
    if threshold_tuning and decision_threshold != 0.:
        raise ValueError('Không đặt threshold khác 0 cùng chế độ tự chọn ngưỡng.')
    if not tune and (C_grid is not None or gamma_grid is not None):
        raise ValueError('Grid C/gamma chỉ dùng khi --tune được bật.')
    if save_all_sweeps and not (tune or threshold_tuning):
        raise ValueError('--save-all-sweeps cần có chọn tham số/ngưỡng.')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('LOSO cần output mới/rỗng để tránh trộn các lượt đánh giá.')
    data, names, config = load_datasets(paths)
    X, y, groups = data['X'], data['y'], data['subject_id']
    if len(np.unique(groups)) < 2 or len(np.unique(y)) != 2:
        raise ValueError('LOSO cần >=2 bệnh nhân và cả hai lớp trong toàn dataset.')
    outer_folds = list(LeaveOneGroupOut().split(X, y, groups))
    needs_inner = tune or threshold_tuning
    # Preflight every fold; fail loudly instead of silently dropping difficult patients.
    for train_idx, test_idx in outer_folds:
        if len(np.unique(y[train_idx])) != 2:
            raise ValueError(f'LOSO test {groups[test_idx][0]}: nhóm train thiếu lớp.')
        if needs_inner:
            _group_folds(X[train_idx], y[train_idx], groups[train_idx], cv_folds, seed)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'dataset_info.json', config)
    _write_json(output / 'feature_names.json', names)
    settings = {'evaluation': 'loso', 'trainer_version': TRAINER_VERSION,
                'outer_folds': len(outer_folds), 'inner_cv': 'SGKF' if needs_inner else None,
                'inner_cv_folds': cv_folds if needs_inner else None,
                'cv_levels': 2 if needs_inner else 1,
                'tune': bool(tune), 'threshold_tuning': bool(threshold_tuning),
                'seed': seed, 'threshold_method': threshold_method,
                'kernel': kernel, 'fixed_C': float(C), 'fixed_gamma': gamma,
                'fixed_decision_threshold': float(decision_threshold),
                'class_weight': 'balanced', 'cache_size': cache_size, 'n_jobs': n_jobs,
                'versions': {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
                'fallback_policy': 'best inner balanced accuracy when both targets cannot be met',
                'n_features': len(names), 'input_files': config['input_files'],
                'C_grid': C_grid if C_grid is not None else [.1, 1., 10., 100.],
                'gamma_grid': gamma_grid if gamma_grid is not None else ['scale', .01, .1],
                'min_sensitivity': min_sensitivity, 'min_specificity': min_specificity,
                'threshold_grid': {'min': threshold_min, 'max': threshold_max, 'step': threshold_step}}
    _write_json(output / 'run_settings.json', settings)
    _write_json(output / 'run_status.json', {'status': 'running', 'completed_subjects': []})
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('svm', SVC(kernel=kernel, C=C, gamma=gamma,
                                     class_weight='balanced', probability=False,
                                     cache_size=cache_size))])
    subject_rows, predictions_all, fold_records, seen = [], [], [], np.zeros(len(y), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(outer_folds, 1):
        subject = str(groups[test_idx][0])
        print(f'LOSO {fold}/{len(outer_folds)}: TEST {subject}; '
              f'TRAIN {", ".join(np.unique(groups[train_idx]))}', flush=True)
        directory = output / f'fold_{fold:02d}_{subject}'
        directory.mkdir()
        manifest = _manifest(data, train_idx, test_idx)
        manifest.to_csv(directory / 'split_manifest.csv', index=False)
        if needs_inner:
            selected, inner_scores, inner_ids = select_joint_parameters(
                pipeline, X[train_idx], y[train_idx], groups[train_idx],
                tune=tune, threshold_tuning=threshold_tuning, cv_folds=cv_folds,
                seed=seed, kernel=kernel, decision_threshold=decision_threshold,
                min_sensitivity=min_sensitivity, min_specificity=min_specificity,
                threshold_method=threshold_method, threshold_min=threshold_min,
                threshold_max=threshold_max, threshold_step=threshold_step,
                C_grid=C_grid, gamma_grid=gamma_grid, n_jobs=n_jobs,
                save_all_sweeps=save_all_sweeps, output=directory)
            inner = manifest.iloc[train_idx].copy()
            inner['Split'] = 'inner_validation_oof'
            inner['Inner_Fold'] = inner_ids
            inner['FOG_Decision_Score'] = inner_scores
            inner['Decision_Threshold'] = selected['decision_threshold']
            inner['Prediction'] = (inner_scores > selected['decision_threshold']).astype(np.int8)
            inner.to_csv(directory / 'inner_validation_scores.csv', index=False)
            _patient_metrics(y[train_idx], inner_scores, groups[train_idx], selected['decision_threshold']).to_csv(
                directory / 'inner_patient_metrics.csv', index=False)
        else:
            selected = {'C': float(C), 'gamma': gamma,
                        'decision_threshold': float(decision_threshold), 'target_met': None,
                        'threshold_role': 'fixed_baseline', 'selection_rule': 'predeclared fixed parameters'}
            _write_json(directory / 'selected_parameters.json', selected)
        # One final fit on all outer-training subjects; no further tuning/CV.
        model = clone(pipeline).set_params(svm__C=selected['C'], svm__gamma=selected['gamma'])
        model.fit(X[train_idx], y[train_idx])
        scores = model.decision_function(X[test_idx])
        threshold = selected['decision_threshold']
        predicted = (scores > threshold).astype(np.int8)
        metrics = _outer_metrics(y[test_idx], predicted, scores)
        metrics.update(status='completed', evaluation='LOSO outer test',
                       trainer_version=TRAINER_VERSION, test_subject=subject,
                       train_subjects=np.unique(groups[train_idx]).tolist(),
                       test_subjects=[subject], train_windows=len(train_idx),
                       test_windows=len(test_idx), n_features=len(names),
                       decision_threshold=float(threshold), selected_parameters=selected,
                       svm_parameters=model.named_steps['svm'].get_params(),
                       support_vectors=model.named_steps['svm'].n_support_.tolist())
        _write_json(directory / 'metrics.json', metrics)
        pred_table = manifest.iloc[test_idx].copy()
        pred_table['Dataset_Row'] = test_idx
        pred_table['Outer_Fold'] = fold
        pred_table['Prediction'] = predicted
        pred_table['FOG_Decision_Score'] = scores
        pred_table['Decision_Threshold'] = threshold
        pred_table['C'] = selected['C']; pred_table['Gamma'] = selected['gamma']
        pred_table['Threshold_Role'] = selected['threshold_role']
        pred_table.to_csv(directory / 'test_predictions.csv', index=False)
        predictions_all.append(pred_table); seen[test_idx] += 1
        counts = metrics['confusion_counts']
        subject_rows.append({
            'Outer_Fold': fold, 'Subject_ID': subject,
            'N_Windows': len(test_idx), 'N_FOG': int(np.sum(y[test_idx] == 1)),
            **{k: metrics[k] for k in ['sensitivity', 'specificity', 'ppv', 'npv',
                                        'accuracy', 'fog_f1', 'balanced_accuracy',
                                        'youden_index', 'roc_auc', 'average_precision']},
            **counts, 'C': selected['C'], 'Gamma': selected['gamma'],
            'Decision_Threshold': threshold, 'Target_Met_Inner': selected['target_met'],
            'Threshold_Role': selected['threshold_role']})
        fold_records.append({'Outer_Fold': fold, 'Test_Subject': subject,
                             'Train_Subjects': ','.join(np.unique(groups[train_idx])),
                             'Train_Windows': len(train_idx), 'Test_Windows': len(test_idx)})
        if save_models:
            bundle = {'pipeline': model, 'feature_names': names,
                      'label_mapping': {0: 'Non-FOG', 1: 'FOG'}, 'processing_config': config,
                      'model_type': 'SVC', 'metrics': metrics,
                      'train_subjects': metrics['train_subjects'], 'test_subjects': [subject],
                      'decision_threshold': float(threshold), 'trainer_version': TRAINER_VERSION,
                      'purpose': 'LOSO evaluation fold; not a model fitted on the full dataset'}
            joblib.dump(bundle, directory / 'fog_svm.joblib')
        _write_json(output / 'run_status.json', {'status': 'running',
                    'completed_subjects': [row['Subject_ID'] for row in subject_rows]})
    if not np.all(seen == 1):
        raise ValueError('LOSO cần mỗi cửa sổ được test đúng một lần.')
    joined = pd.concat(predictions_all, ignore_index=True).sort_values('Dataset_Row')
    subjects = pd.DataFrame(subject_rows)
    pooled = _outer_metrics(joined.Binary_Label.to_numpy(), joined.Prediction.to_numpy())
    macro = {}
    for key in ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy', 'fog_f1',
                'balanced_accuracy', 'youden_index', 'roc_auc', 'average_precision']:
        values = subjects[key].dropna().to_numpy(float)
        macro[key] = {'mean': float(values.mean()) if len(values) else None,
                      'std_sample': float(values.std(ddof=1)) if len(values) > 1 else None,
                      'n_valid_subjects': len(values)}
    summary = {'status': 'completed', 'evaluation': 'LOSO',
               'trainer_version': TRAINER_VERSION, 'settings': settings,
               'n_subjects': len(subjects), 'n_test_windows': len(joined),
               'pooled_metrics': pooled, 'subject_macro_metrics': macro,
               'target_met_inner_folds': int(sum(row['Target_Met_Inner'] is True for row in subject_rows)) if threshold_tuning else None,
               'fallback_inner_folds': int(sum(row['Target_Met_Inner'] is False for row in subject_rows)) if threshold_tuning else None,
               'notes': ['All subjects and windows are included in outer-test aggregation.',
                         'Sensitivity/F1 for subjects with no actual FOG are null; undefined denominators are null.',
                         'Pooled ROC-AUC/AP are intentionally null: raw score scales differ across outer models.',
                         'Inner selection metrics are not independent performance estimates.',
                         'LOSO folds are correlated; subject std is descriptive, not a confidence interval.',
                         'No full-dataset deployment model is fitted by this evaluation.']}
    joined.to_csv(output / 'loso_predictions.csv', index=False)
    subjects.to_csv(output / 'loso_subject_metrics.csv', index=False)
    pd.DataFrame(fold_records).to_csv(output / 'loso_fold_manifest.csv', index=False)
    _write_json(output / 'loso_summary.json', summary)
    _write_json(output / 'run_status.json', {'status': 'completed',
                'completed_subjects': subjects.Subject_ID.tolist(), 'n_outer_folds': len(subjects)})
    print(f'LOSO completed: {len(subjects)} subjects, {len(joined):,} outer-test windows.', flush=True)
    return summary


def _gamma(value):
    if value in ('scale', 'auto'):
        return value
    try:
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError()
        return number
    except ValueError as exc:
        raise argparse.ArgumentTypeError('gamma cần scale, auto hoặc số >0.') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', nargs='+', required=True, help='Feature CSV/NPZ files or directories')
    parser.add_argument('--output', default='svm_output')
    parser.add_argument('--evaluation', choices=['holdout', 'loso'], default='holdout')
    parser.add_argument('--kernel', choices=['rbf', 'linear'], default='rbf')
    parser.add_argument('--C', '--c', dest='C', type=float, default=1.)
    parser.add_argument('--gamma', type=_gamma, default='scale')
    parser.add_argument('--test-size', type=float, default=None, help='Hold-out only; default .25 of subjects')
    parser.add_argument('--test-subjects', nargs='+', help='Hold-out only; explicit subjects')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tune', action='store_true', help='LOSO: test C/gamma candidates within ONE inner SGKF; holdout: existing GridSearch')
    parser.add_argument('--cv-folds', type=int, default=3, help='LOSO inner SGKF fold count; holdout validation/GridSearch fold count')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--cache-size', type=float, default=512.)
    parser.add_argument('--decision-threshold', type=float, default=0., help='Fixed threshold; FOG iff score > threshold')
    parser.add_argument('--threshold-tuning', action='store_true', help='LOSO: joint threshold selection with fallback; holdout: stop when targets unmet')
    parser.add_argument('--min-sensitivity', type=float, default=.85)
    parser.add_argument('--min-specificity', type=float, default=.85)
    parser.add_argument('--threshold-method', choices=['grid', 'exact'], default='grid')
    parser.add_argument('--threshold-min', type=float, default=-1.5)
    parser.add_argument('--threshold-max', type=float, default=.5)
    parser.add_argument('--threshold-step', type=float, default=.02)
    parser.add_argument('--inner-cv-folds', type=int, default=None, help='Legacy holdout only; default 3. LOSO does not use a third CV level.')
    parser.add_argument('--C-grid', nargs='+', type=float, default=None, help='LOSO --tune only; default .1 1 10 100')
    parser.add_argument('--gamma-grid', nargs='+', type=_gamma, default=None, help='LOSO RBF --tune only; default scale .01 .1')
    parser.add_argument('--no-save-models', action='store_true', help='LOSO only: omit per-fold joblib models')
    parser.add_argument('--save-all-sweeps', action='store_true', help='LOSO: save threshold sweeps for every parameter candidate (can be large)')
    args = parser.parse_args()
    try:
        if args.evaluation == 'loso':
            if args.test_subjects is not None or args.test_size is not None:
                raise ValueError('LOSO lần lượt test mọi bệnh nhân; không dùng --test-subjects/--test-size.')
            if args.inner_cv_folds is not None:
                raise ValueError('LOSO hai tầng chỉ dùng --cv-folds; bỏ --inner-cv-folds.')
            if args.kernel == 'linear' and args.gamma_grid is not None:
                raise ValueError('Gamma-grid chỉ có ý nghĩa với kernel RBF.')
        elif (args.C_grid is not None or args.gamma_grid is not None
              or args.no_save_models or args.save_all_sweeps):
            raise ValueError('Grid tùy chỉnh và tùy chọn lưu model/sweep mới chỉ dùng trong LOSO.')
        paths = discover_inputs(args.input)
        common = dict(kernel=args.kernel, C=args.C, gamma=args.gamma, seed=args.seed,
                      tune=args.tune, threshold_tuning=args.threshold_tuning,
                      cv_folds=args.cv_folds, n_jobs=args.n_jobs, cache_size=args.cache_size,
                      decision_threshold=args.decision_threshold,
                      min_sensitivity=args.min_sensitivity, min_specificity=args.min_specificity,
                      threshold_method=args.threshold_method, threshold_min=args.threshold_min,
                      threshold_max=args.threshold_max, threshold_step=args.threshold_step)
        if args.evaluation == 'loso':
            metrics = train_loso(paths, args.output, **common,
                                 C_grid=args.C_grid, gamma_grid=args.gamma_grid,
                                 save_models=not args.no_save_models,
                                 save_all_sweeps=args.save_all_sweeps)
        else:
            metrics = train(paths, args.output,
                            test_size=.25 if args.test_size is None else args.test_size,
                            test_subjects=args.test_subjects,
                            inner_cv_folds=3 if args.inner_cv_folds is None else args.inner_cv_folds,
                            **common)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')
    if args.evaluation == 'loso':
        print(json.dumps({'status': metrics['status'], 'n_subjects': metrics['n_subjects'],
                          'pooled_metrics': metrics['pooled_metrics'],
                          'subject_macro_metrics': metrics['subject_macro_metrics']},
                         ensure_ascii=False, indent=2))
        return
    if metrics.get('status') == 'target_not_met':
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        parser.exit(2, 'TARGET_NOT_MET: xem validation_threshold_sweep.csv; test chưa được đánh giá.\n')
    print(json.dumps({key: metrics[key] for key in ['sensitivity', 'specificity', 'ppv', 'npv',
                        'accuracy', 'youden_index', 'balanced_accuracy', 'fog_f1',
                        'roc_auc', 'average_precision', 'train_subjects', 'test_subjects']}, indent=2))


if __name__ == '__main__':
    main()
