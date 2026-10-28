from __future__ import annotations
import json
import math
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QComboBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QFileDialog,
    QMessageBox, QCheckBox, QScrollArea, QAbstractItemView, QSpinBox,
)

RATE_METRICS = ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy',
                'fog_f1', 'balanced_accuracy', 'youden_index']
COUNTS = ['TP', 'FP', 'FN', 'TN']
LABELS = {
    'sensitivity': 'Sensitivity / Recall', 'specificity': 'Specificity',
    'ppv': 'Precision / PPV', 'npv': 'NPV', 'accuracy': 'Accuracy',
    'fog_f1': 'FOG F1', 'balanced_accuracy': 'Balanced accuracy',
    'youden_index': 'Youden index', 'TP': 'TP', 'FP': 'FP', 'FN': 'FN', 'TN': 'TN',
}
COLORS = ['#2563eb', '#f97316', '#059669', '#9333ea',
          '#dc2626', '#0891b2', '#a16207', '#db2777']

@dataclass
class Run:
    name: str
    source: str
    root: str
    values: dict
    settings: dict
    config: dict
    warnings: list = field(default_factory=list)
    shap_data: list = field(default_factory=list)

    @property
    def n(self):
        return sum(self.values[k] for k in COUNTS)

def metrics(counts):
    c = {}
    for key in COUNTS:
        value = float(counts[key])
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError(f'Số liệu {key} không hợp lệ: {value}')
        c[key] = int(value)
    tn, fp, fn, tp = (c[k] for k in ['TN', 'FP', 'FN', 'TP'])
    def div(a, b):
        return a / b if b else float('nan')
    sensitivity = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    return dict(c, sensitivity=sensitivity, specificity=specificity,
                ppv=div(tp, tp + fp), npv=div(tn, tn + fn),
                accuracy=div(tp + tn, tp + tn + fp + fn),
                fog_f1=div(2 * tp, 2 * tp + fp + fn),
                balanced_accuracy=(sensitivity + specificity) / 2,
                youden_index=sensitivity + specificity - 1)

def load_runs(path):
    path = Path(path)
    archive = None
    try:
        if path.is_file() and path.suffix.lower() == '.zip':
            archive = zipfile.ZipFile(path)
            archive_names = {n.replace(chr(92), '/'): n for n in archive.namelist()}
            if len(archive_names) != len(archive.namelist()):
                raise ValueError('ZIP có đường dẫn trùng sau chuẩn hóa.')
            names = [n for n in archive_names if n.endswith('.json') and '__MACOSX/' not in n]
        elif path.is_dir():
            names = [p.relative_to(path).as_posix() for p in path.rglob('*.json')]
        else:
            raise ValueError('Chọn file ZIP hoặc thư mục kết quả train.')

        def read_bytes(name):
            if archive:
                if archive.getinfo(archive_names[name]).file_size > 256 * 1024 * 1024:
                    raise ValueError('File SHAP quá lớn.')
                return archive.read(archive_names[name])
            return (path / name).read_bytes()

        def read_json(name, default=None):
            if name not in names:
                return default
            if archive:
                if archive.getinfo(archive_names[name]).file_size > 16 * 1024 * 1024:
                    raise ValueError(f'File JSON quá lớn: {name}')
                raw = archive.read(archive_names[name])
            else:
                raw = (path / name).read_bytes()
            return json.loads(raw.decode('utf-8-sig'))

        summaries = [n for n in names if PurePosixPath(n).name == 'loso_summary.json']
        if not summaries:
            summaries = [n for n in names if PurePosixPath(n).name == 'metrics.json'
                         and not any(p.startswith('fold_') for p in PurePosixPath(n).parts[:-1])]
        if not summaries:
            raise ValueError('Không tìm thấy loso_summary.json hoặc metrics.json ở gốc kết quả.')
        runs = []
        for name in sorted(summaries):
            parent = str(PurePosixPath(name).parent)
            root = '' if parent == '.' else parent + '/'
            summary = read_json(name)
            saved = summary.get('pooled_metrics', summary)
            counts = saved.get('confusion_counts')
            if counts is None:
                if 'confusion_matrix' in saved:
                    matrix = np.asarray(saved['confusion_matrix'])
                    if matrix.shape != (2, 2):
                        raise ValueError('Confusion matrix phải có kích thước 2 × 2.')
                    counts = dict(zip(['TN', 'FP', 'FN', 'TP'], matrix.reshape(-1)))
                elif all(k in saved for k in COUNTS):
                    counts = saved
                else:
                    raise ValueError(f'{name}: thiếu số liệu TP / FP / FN / TN.')
            values = metrics(counts)
            warnings = []
            if summary.get('status', 'completed') != 'completed':
                warnings.append('Kết quả train chưa hoàn tất.')
            settings = read_json(root + 'run_settings.json', summary.get('settings', {})) or {}
            config = read_json(root + 'dataset_info.json', {}) or {}
            configs = [c for c in config.get('configurations', []) if isinstance(c, dict)]
            config = configs[0] if len(configs) == 1 else {'configurations': configs}
            title = PurePosixPath(parent).name if root else path.stem
            shap_data = []
            for entry in sorted(n for n in names if n.startswith(root + 'fold_') and n.endswith('/shap/manifest.json')):
                try:
                    shap_data.append(read_shap_data(entry, read_json(entry), read_bytes, read_json(root + 'feature_names.json', [])))
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    warnings.append('Không đọc SHAP ' + entry + ': ' + str(exc))
            runs.append(Run(title, str(path.resolve()), root, values, settings, config, warnings, shap_data))
        return runs
    finally:
        if archive:
            archive.close()

@dataclass
class ShapData:
    name: str
    manifest: dict
    names: list
    phi: np.ndarray
    values: np.ndarray
    scores: np.ndarray
    thresholds: np.ndarray
    base: float
    weights: np.ndarray
    samples: list

def read_shap_data(name, manifest, read_bytes, expected_names):
    import io
    import csv
    if manifest.get('schema') != 'fog-shap-v1' or manifest.get('status') != 'completed':
        raise ValueError('SHAP chưa hoàn tất hoặc schema không được hỗ trợ.')
    folder = name.rsplit('/', 1)[0] + '/'
    with np.load(io.BytesIO(read_bytes(folder + 'values.npz')), allow_pickle=False) as z:
        required = {'feature_names', 'shap_values', 'feature_values', 'scores', 'thresholds', 'base_value', 'sample_weights', 'dataset_rows', 'schema'}
        if not required.issubset(z.files) or str(z['schema'].item()) != 'fog-shap-v1':
            raise ValueError('Thiếu trường hoặc sai schema SHAP.')
        names = z['feature_names'].astype(str).tolist()
        phi = np.array(z['shap_values'], float)
        values = np.array(z['feature_values'], float)
        scores = np.array(z['scores'], float)
        thresholds = np.array(z['thresholds'], float)
        base = float(z['base_value'].item())
        weights = np.array(z['sample_weights'], float)
        rows = np.array(z['dataset_rows'], int)
    samples = list(csv.DictReader(io.StringIO(read_bytes(folder + 'samples.csv').decode('utf-8-sig'))))
    if expected_names != names or len(set(names)) != len(names):
        raise ValueError('SHAP feature names không khớp kết quả train.')
    n = len(samples)
    if len(np.unique(rows)) != len(rows):
        raise ValueError('Dataset_Row SHAP bị trùng.')
    if not n or phi.shape != (n, len(names)) or values.shape != phi.shape:
        raise ValueError('Sai kích thước SHAP.')
    if any(a.shape != (n,) for a in [scores, thresholds, weights, rows]):
        raise ValueError('Sai kích thước metadata SHAP.')
    if not all(np.isfinite(a).all() for a in [phi, scores, thresholds, weights]) or not np.isfinite(base) or np.isinf(values).any() or np.any(weights <= 0):
        raise ValueError('Giá trị SHAP không hợp lệ.')
    if not np.allclose(base + phi.sum(axis=1), scores, rtol=1e-5, atol=1e-5):
        raise ValueError('SHAP không cộng về decision score.')
    if set(manifest.get('background_subjects', [])) & set(manifest.get('test_subjects', [])):
        raise ValueError('Background SHAP lẫn người test.')
    for i, sample in enumerate(samples):
        y = int(sample['Binary_Label'])
        p = int(sample['Prediction'])
        outcome = 'TP' if y == 1 and p == 1 else 'FN' if y == 1 else 'FP' if p == 1 else 'TN'
        if y not in (0, 1) or p not in (0, 1) or sample['Outcome'] != outcome or p != int(scores[i] > thresholds[i]) or int(sample['Dataset_Row']) != rows[i]:
            raise ValueError('Metadata cửa sổ SHAP không khớp score.')
    return ShapData(manifest.get('fold', name), manifest, names, phi, values, scores, thresholds, base, weights, samples)

def shap_indices(data, outcome='All'):
    return np.asarray([i for i, s in enumerate(data.samples) if outcome == 'All' or s['Outcome'] == outcome], int)

def shap_importance(data, indices):
    return np.average(np.abs(data.phi[indices]), axis=0, weights=data.weights[indices])

def shap_units(data):
    return data.manifest.get('score_units', 'SVM decision score')

def model_label(run):
    return run.settings.get('model_name') or ('Random Forest' if run.settings.get('model_type') == 'rf' else 'SVM')

def shap_bar(data, indices, k, sort_order='High → Low'):
    importance = shap_importance(data, indices)
    ranked = np.argsort(importance, kind='stable')
    order = ranked[:k] if sort_order == 'Low → High' else ranked[::-1][:k]
    f = fig_base('SHAP Importance • ' + data.name)
    f.set_size_inches(12, max(4.2, len(order) * 0.32 + 1.2))
    ax = f.subplots()
    y = np.arange(len(order))
    bars = ax.barh(y, importance[order], color='#2563eb', height=0.68, label=data.name)
    ax.set_yticks(y, [data.names[j] for j in order], fontsize=9)
    ax.set_ylim(len(order) - 0.5, -0.5)
    for bar, v in zip(bars, importance[order]):
        ax.annotate(f'{v:.4g}', (v, bar.get_y() + bar.get_height() / 2), xytext=(5, 0), textcoords='offset points', va='center', fontsize=9)
    ax.set_xlabel('mean(|SHAP value|) • ' + shap_units(data))
    ax.set_ylabel('Feature')
    right = max(float(importance[order].max() if len(order) else 0.0) * 1.22, 1e-6)
    ax.set_xlim(0, right)
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)
    return f

def shap_beeswarm(data, indices, k, sort_order='High → Low'):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    importance = shap_importance(data, indices)
    ranked = np.argsort(importance, kind='stable')
    order = ranked[:k] if sort_order == 'Low → High' else ranked[::-1][:k]
    f = fig_base('SHAP Beeswarm • ' + data.name)
    f.set_size_inches(12, max(4.2, len(order) * 0.36 + 1.3))
    ax = f.subplots()
    for row, j in enumerate(order):
        v = data.phi[indices, j]
        raw = data.values[indices, j]
        finite = np.isfinite(raw)
        colors = np.full(len(v), 0.5)
        if finite.any():
            lo, hi = np.quantile(raw[finite], [0.05, 0.95])
            colors[finite] = np.clip((raw[finite] - lo) / (hi - lo), 0, 1) if hi > lo else 0.5
        bins = np.digitize(v, np.linspace(float(v.min()), float(v.max()) + 1e-12, 24))
        offset = np.zeros(len(v))
        for b in np.unique(bins):
            ids = np.flatnonzero(bins == b)
            count = len(ids)
            offset[ids] = (np.arange(count) - (count - 1) / 2) * min(0.11, 0.65 / max(count, 1))
        ax.scatter(v[finite], row + offset[finite], c=colors[finite], cmap='coolwarm', vmin=0, vmax=1, s=22, alpha=0.8)
        if (~finite).any():
            ax.scatter(v[~finite], row + offset[~finite], color='#94a3b8', marker='x', s=25)
    ax.set_yticks(np.arange(len(order)), [data.names[j] for j in order], fontsize=9)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.axvline(0, color='#64748b', linewidth=1)
    ax.set_xlabel('SHAP value • ' + shap_units(data) + ' • âm: Non-FOG ← → dương: FOG')
    ax.set_ylabel('Feature')
    cbar = f.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='coolwarm'), ax=ax, pad=0.02)
    cbar.set_ticks([0, 1], labels=['Low', 'High'])
    cbar.set_label('Feature value • scaled separately per feature')
    return f

def shap_waterfall(data, index, k):
    phi = data.phi[index]
    order = np.argsort(-np.abs(phi), kind='stable')[:k]
    labels = []
    changes = []
    for j in order:
        raw = data.values[index, j]
        text = f'{raw:.4g}' if np.isfinite(raw) else 'NaN → imputer'
        labels.append(data.names[j] + ' = ' + text)
        changes.append(float(phi[j]))
    if len(order) < len(phi):
        labels.append(f'Other {len(phi) - len(order)} features')
        changes.append(float(phi.sum() - phi[order].sum()))
    sample = data.samples[index]
    f = fig_base('SHAP Waterfall • ' + data.name + ' • ' + sample['Outcome'])
    f.set_size_inches(12, max(4.2, len(labels) * 0.36 + 1.4))
    ax = f.subplots()
    current = data.base
    for y, delta in enumerate(changes):
        next_value = current + delta
        left = min(current, next_value)
        ax.barh(y, abs(delta), left=left, color='#dc2626' if delta >= 0 else '#2563eb', height=0.65)
        ax.annotate(f'{delta:+.4g}', (next_value, y), xytext=(5 if delta >= 0 else -5, 0), textcoords='offset points', ha='left' if delta >= 0 else 'right', va='center', fontsize=8)
        if y < len(changes) - 1:
            ax.plot([next_value, next_value], [y + 0.33, y + 0.67], color='#94a3b8', linewidth=0.8)
        current = next_value
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.axvline(data.base, color='#64748b', linestyle='--', label=f'Base = {data.base:.4g}')
    ax.axvline(data.thresholds[index], color='#111827', linestyle=':', label=f'Threshold = {data.thresholds[index]:.4g}')
    ax.axvline(data.scores[index], color='#059669', linestyle='--', label=f'Score = {data.scores[index]:.4g}')
    ax.set_xlabel(shap_units(data) + ' • đỏ: tăng đầu ra về FOG; xanh: giảm đầu ra về Non-FOG')
    ax.set_title(f"{sample['Recording_ID']} | window {sample['Window_ID']} | {float(sample['Start_Time']):.3f}–{float(sample['End_Time']):.3f}s | True={sample['Binary_Label']} Pred={sample['Prediction']}", fontsize=10)
    ax.margins(x=0.22)
    ax.grid(axis='x', alpha=0.15)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    return f

def shap_stability(datasets, outcome, k, sort_order='High → Low'):
    valid = [(d, shap_indices(d, outcome)) for d in datasets]
    valid = [(d, i) for d, i in valid if len(i)]
    if len(valid) < 2:
        raise ValueError('Cần ít nhất 2 fold có dữ liệu để xem độ ổn định.')
    names = valid[0][0].names
    if any(d.names != names for d, _ in valid):
        raise ValueError('Không tổng hợp fold có feature schema khác nhau.')
    counts = np.zeros(len(names), int)
    for d, idx in valid:
        importance = shap_importance(d, idx)
        order = np.argsort(-importance, kind='stable')[:k]
        counts[order[importance[order] > 1e-12]] += 1
    ranked = np.argsort(counts, kind='stable')
    order = ranked[:k] if sort_order == 'Low → High' else ranked[::-1][:k]
    f = fig_base(f'Stability • top-{k} • {len(valid)} folds có dữ liệu')
    f.set_size_inches(12, max(4.2, len(order) * 0.32 + 1.2))
    ax = f.subplots()
    values = counts[order] / len(valid) * 100
    ax.barh(np.arange(len(order)), values, color='#059669', height=0.68)
    ax.set_yticks(np.arange(len(order)), [names[j] for j in order], fontsize=9)
    ax.set_ylim(len(order) - 0.5, -0.5)
    for y, j in enumerate(order):
        ax.text(values[y] + 1, y, f'{counts[j]}/{len(valid)}', va='center', fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel('Tần suất xuất hiện trong top-k (%) • không cộng SHAP thô giữa các model')
    ax.set_ylabel('Feature')
    return f

def fig_base(title):
    f = Figure(figsize=(12, 6), dpi=100, layout='constrained')
    f.suptitle(title, fontsize=13, fontweight='bold', color='#183153')
    return f

def grouped_chart(runs, keys, title, percent=False, delta=False, baseline=None):
    f = fig_base(title)
    ax = f.subplots()
    n = max(len(runs), 1)
    width = 0.8 / n
    x = np.arange(len(keys))
    for i, r in enumerate(runs):
        v = np.array([r.values.get(k, float('nan')) for k in keys], float)
        if delta and baseline:
            v -= np.array([baseline.values.get(k, float('nan')) for k in keys])
        if percent:
            v *= 100
        bars = ax.bar(x - 0.4 + width / 2 + i * width, v, width, label=r.name, color=COLORS[i % len(COLORS)])
        if n <= 6:
            for bar, value in zip(bars, v):
                if np.isfinite(value):
                    ax.annotate(f'{value:+.1f}' if delta else (f'{value:.1f}' if percent else f'{value:,.0f}'), (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 4 if value >= 0 else -12), textcoords='offset points', ha='center', fontsize=8)
    ax.set_xticks(x, [LABELS.get(k, k) for k in keys], rotation=15 if len(keys) > 5 else 0)
    ax.set_ylabel('Percentage points' if percent and delta else ('Percent (%)' if percent else 'Windows'))
    if percent and not delta:
        ax.set_ylim(min(-5, ax.get_ylim()[0]), 110)
    ax.axhline(0, color='#94a3b8', linewidth=0.7)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
    return f

def confusion_chart(runs, normalize=False):
    f = fig_base('Confusion matrix • ' + ('row normalized (%)' if normalize else 'window counts'))
    cols = min(3, len(runs))
    rows = math.ceil(len(runs) / cols)
    f.set_size_inches(12, max(5, rows * 3.8))
    axs = f.subplots(rows, cols, squeeze=False)
    vmax = 100 if normalize else max(max(r.values[k] for k in COUNTS) for r in runs)
    for ax, r in zip(axs.flat, runs):
        c = r.values
        raw = np.array([[c['TN'], c['FP']], [c['FN'], c['TP']]])
        totals = raw.sum(axis=1, keepdims=True)
        values = np.divide(raw * 100, totals, out=np.full((2, 2), np.nan), where=totals != 0) if normalize else raw
        ax.imshow(values, cmap='Blues', vmin=0, vmax=max(vmax, 1))
        for (i, j), v in np.ndenumerate(values):
            text = f'{v:.1f}%\n(n={raw[i, j]:,})' if normalize and np.isfinite(v) else ('N/A' if not np.isfinite(v) else f'{int(v):,}')
            ax.text(j, i, [['TN', 'FP'], ['FN', 'TP']][i][j] + '\n' + text, ha='center', va='center', fontsize=11, color='white' if v > vmax * 0.55 else '#183153')
        ax.set_xticks([0, 1], ['Non-FOG', 'FOG'])
        ax.set_yticks([0, 1], ['Non-FOG', 'FOG'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(r.name, fontsize=10, wrap=True)
    for ax in list(axs.flat)[len(runs):]:
        ax.set_visible(False)
    return f

class Loader(QThread):
    loaded = Signal(object, object)
    def __init__(self, paths):
        super().__init__()
        self.paths = paths
    def run(self):
        runs, errors = [], []
        for path in self.paths:
            try:
                runs.extend(load_runs(path))
            except Exception as exc:
                errors.append(f'{Path(path).name}: {exc}')
        self.loaded.emit(runs, errors)

class ViewToolbar(NavigationToolbar2QT):
    toolitems = tuple(item for item in NavigationToolbar2QT.toolitems if item[0] != 'Save')

class PlotPane(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.figure = None

    def show_figure(self, figure):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        if self.figure:
            self.figure.clear()
        self.figure = figure
        canvas = FigureCanvasQTAgg(figure)
        dpi = figure.get_dpi()
        canvas.setMinimumSize(int(figure.get_figwidth() * dpi * 0.75), int(figure.get_figheight() * dpi))
        self.layout.addWidget(ViewToolbar(canvas, self))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(canvas)
        self.layout.addWidget(scroll, 1)
        canvas.draw()
        scroll.verticalScrollBar().setValue(0)

class NumericItem(QTableWidgetItem):
    def __init__(self, text, number=None):
        super().__init__(text)
        self.number = number
    def __lt__(self, other):
        if isinstance(other, NumericItem) and self.number is not None and other.number is not None:
            return self.number < other.number
        return super().__lt__(other)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.loading = False
        self.loader = None
        self.setWindowTitle('FOG Compare')
        self.resize(1365, 820)
        self.setAcceptDrops(True)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel('FOG Compare')
        title.setStyleSheet('font-size:24px;font-weight:700;color:#183153')
        layout.addWidget(title)
        split = QSplitter()
        layout.addWidget(split, 1)
        left = QWidget()
        left.setMinimumWidth(235)
        left.setMaximumWidth(300)
        sidebar = QVBoxLayout(left)
        self.zipbutton = QPushButton('+ Thêm ZIP')
        self.zipbutton.clicked.connect(self.add_zip)
        sidebar.addWidget(self.zipbutton)
        self.folderbutton = QPushButton('+ Thêm thư mục')
        self.folderbutton.clicked.connect(self.add_folder)
        sidebar.addWidget(self.folderbutton)
        sidebar.addWidget(QLabel('Chọn kết quả cần xem:'))
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.itemChanged.connect(self.refresh)
        sidebar.addWidget(self.list, 1)
        remove = QPushButton('Bỏ mục đang chọn')
        remove.clicked.connect(self.remove)
        sidebar.addWidget(remove)
        sidebar.addWidget(QLabel('Baseline'))
        self.baseline = QComboBox()
        self.baseline.currentIndexChanged.connect(self.refresh)
        sidebar.addWidget(self.baseline)
        split.addWidget(left)
        right = QWidget()
        content = QVBoxLayout(right)
        self.summary = QLabel('Thêm ZIP hoặc thư mục kết quả train.')
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet('padding:10px;background:#e8f0fe;border-radius:5px')
        content.addWidget(self.summary)
        self.tabs = QTabWidget()
        content.addWidget(self.tabs, 1)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.table, 'Bảng số liệu')
        self.metricsplot = PlotPane()
        self.tabs.addTab(self.metricsplot, 'Chỉ số')
        matrix = QWidget()
        matrixlayout = QVBoxLayout(matrix)
        self.normalize = QCheckBox('Hiển thị tỷ lệ theo nhãn thực tế (%)')
        self.normalize.toggled.connect(self.refresh)
        matrixlayout.addWidget(self.normalize)
        self.cmplot = PlotPane()
        matrixlayout.addWidget(self.cmplot)
        self.tabs.addTab(matrix, 'Confusion matrix')
        delta = QWidget()
        deltalayout = QVBoxLayout(delta)
        self.deltakind = QComboBox()
        self.deltakind.addItems(['Chênh lệch FP / FN / TP / TN', 'Chênh lệch chỉ số (điểm %)'])
        self.deltakind.currentIndexChanged.connect(self.refresh)
        deltalayout.addWidget(self.deltakind)
        self.deltaplot = PlotPane()
        deltalayout.addWidget(self.deltaplot)
        self.tabs.addTab(delta, 'So với baseline')
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.setStyleSheet('QMainWindow{background:#f4f7fb} QPushButton{padding:8px;text-align:left} '
                           'QTableWidget{background:white;gridline-color:#e2e8f0} '
                           'QHeaderView::section{background:#e8eef7;padding:6px} QComboBox{padding:5px}')
        self.create_shap_tab()
        self.refresh()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.load([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])

    def add_zip(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Chọn ZIP kết quả', '', 'ZIP (*.zip)')
        if paths:
            self.load(paths)

    def add_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Chọn thư mục kết quả')
        if path:
            self.load([path])

    def load(self, paths):
        if self.loading or not paths:
            return
        self.loading = True
        self.zipbutton.setEnabled(False)
        self.folderbutton.setEnabled(False)
        self.summary.setText('Đang đọc kết quả…')
        self.loader = Loader(paths)
        self.loader.loaded.connect(self.loaded)
        self.loader.finished.connect(self.load_finished)
        self.loader.start()

    def load_finished(self):
        self.loading = False
        self.zipbutton.setEnabled(True)
        self.folderbutton.setEnabled(True)

    def loaded(self, runs, errors):
        self.list.blockSignals(True)
        existing = [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]
        for run in runs:
            duplicate = next((self.list.item(i) for i in range(self.list.count()) if (self.list.item(i).data(Qt.UserRole).source, self.list.item(i).data(Qt.UserRole).root) == (run.source, run.root)), None)
            if duplicate is not None:
                run.name = duplicate.text()
                duplicate.setData(Qt.UserRole, run)
                continue
            names = {r.name for r in existing}
            original, number = run.name, 2
            while run.name in names:
                run.name = f'{original} ({number})'
                number += 1
            item = QListWidgetItem(run.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, run)
            item.setToolTip(run.source)
            self.list.addItem(item)
            existing.append(run)
        self.list.blockSignals(False)
        self.refresh()
        if errors:
            QMessageBox.warning(self, 'Không đọc được kết quả', '\n\n'.join(errors))

    def selected(self):
        runs = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            run = item.data(Qt.UserRole)
            run.name = item.text().strip() or f'Run {i + 1}'
            if item.checkState() == Qt.Checked:
                runs.append(run)
        return runs

    def remove(self):
        self.list.blockSignals(True)
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self.list.blockSignals(False)
        self.refresh()

    def refresh(self, *_):
        runs = self.selected()
        self.refresh_shap_runs(runs)
        old = self.baseline.currentData()
        self.baseline.blockSignals(True)
        self.baseline.clear()
        for r in runs:
            self.baseline.addItem(r.name, r.source + '::' + r.root)
        index = self.baseline.findData(old) if old is not None else -1
        if index >= 0:
            self.baseline.setCurrentIndex(index)
        self.baseline.blockSignals(False)
        if not runs:
            self.summary.setText('Thêm ZIP hoặc thư mục kết quả train.')
            self.table.setRowCount(0)
            for pane in [self.metricsplot, self.cmplot, self.deltaplot]:
                pane.show_figure(fig_base('Chọn kết quả để xem'))
            return
        base = runs[max(0, self.baseline.currentIndex())]
        notes = [f'{len(runs)} kết quả · Baseline: {base.name} · {base.n:,} cửa sổ']
        for run in runs:
            notes.extend(f'{run.name}: {warning}' for warning in run.warnings)
        if any(r.n != base.n for r in runs):
            notes.append('Số cửa sổ khác nhau: cần so sánh thêm các chỉ số tỷ lệ.')
        different = any(
            any(r.settings.get(k) != base.settings.get(k) for k in
                ['model_type', 'evaluation', 'fixed_C', 'fixed_gamma', 'fixed_decision_threshold', 'tune', 'class_weight'])
            or any(r.config.get(k) != base.config.get(k) for k in
                   ['controls', 'selection', 'excluded_regions', 'label_mapping'])
            for r in runs
        )
        if different:
            notes.append('Các kết quả có cấu hình train hoặc tiền xử lý khác nhau.')
        self.summary.setText('\n'.join(notes))
        self.fill_table(runs, base)
        self.metricsplot.show_figure(grouped_chart(runs, RATE_METRICS, 'Chỉ số tổng hợp (%)', percent=True))
        self.cmplot.show_figure(confusion_chart(runs, self.normalize.isChecked()))
        israte = self.deltakind.currentIndex() == 1
        self.deltaplot.show_figure(grouped_chart(
            runs, RATE_METRICS if israte else ['FP', 'FN', 'TP', 'TN'],
            f'Chênh lệch so với {base.name}', percent=israte, delta=True, baseline=base))

    def fill_table(self, runs, base):
        columns = ['Run', 'Model', 'N_Windows', 'N_Features'] + COUNTS + RATE_METRICS + [
            'Delta_FP', 'Delta_FN', 'Delta_sensitivity', 'FP_reduction_percent']
        titles = {'Run': 'Kết quả', 'Model': 'Model', 'N_Windows': 'Cửa sổ', 'N_Features': 'Features',
                  'Delta_FP': 'ΔFP', 'Delta_FN': 'ΔFN', 'Delta_sensitivity': 'ΔSensitivity (điểm %)',
                  'FP_reduction_percent': 'Giảm FP (%)'}
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(runs))
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels([
            titles.get(c, LABELS.get(c, c)) + (' (%)' if c in RATE_METRICS else '') for c in columns])
        for i, run in enumerate(runs):
            row = dict(run.values, Run=run.name, Model=model_label(run), N_Windows=run.n, N_Features=run.settings.get('n_features'),
                       Delta_FP=run.values['FP'] - base.values['FP'],
                       Delta_FN=run.values['FN'] - base.values['FN'],
                       Delta_sensitivity=run.values['sensitivity'] - base.values['sensitivity'],
                       FP_reduction_percent=(base.values['FP'] - run.values['FP']) / base.values['FP'] * 100
                       if base.values['FP'] else float('nan'))
            for j, column in enumerate(columns):
                value = row[column]
                number = None
                if column in ('Run', 'Model'):
                    text = value
                elif value is None or not math.isfinite(float(value)):
                    text = 'N/A'
                else:
                    number = float(value) * (100 if column in RATE_METRICS or column == 'Delta_sensitivity' else 1)
                    if column in COUNTS + ['N_Windows', 'N_Features', 'Delta_FP', 'Delta_FN']:
                        text = f'{number:+,.0f}' if column.startswith('Delta_') else f'{number:,.0f}'
                    else:
                        text = f'{number:+.2f}' if column.startswith('Delta_') else f'{number:.2f}'
                item = NumericItem(text, number)
                item.setTextAlignment(Qt.AlignLeft if column in ('Run', 'Model') else Qt.AlignRight | Qt.AlignVCenter)
                if column in ['Delta_FP', 'Delta_FN', 'Delta_sensitivity'] and number:
                    good = number < 0 if column in ['Delta_FP', 'Delta_FN'] else number > 0
                    item.setForeground(QColor('#047857' if good else '#b91c1c'))
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def create_shap_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        first = QHBoxLayout()
        second = QHBoxLayout()
        self.shaprun = QComboBox()
        self.shapfold = QComboBox()
        self.shapmode = QComboBox()
        self.shapmode.addItems(['Importance', 'Beeswarm', 'Waterfall', 'Fold stability'])
        self.shaptop = QSpinBox()
        self.shaptop.setRange(1, 500)
        self.shaptop.setValue(20)
        self.shapsort = QComboBox()
        self.shapsort.addItems(['High → Low', 'Low → High'])
        self.shapoutcome = QComboBox()
        self.shapoutcome.addItems(['All', 'TP', 'FP', 'FN', 'TN'])
        self.shapwindow = QComboBox()
        for title, control in [('Kết quả', self.shaprun), ('Fold / người', self.shapfold), ('Biểu đồ', self.shapmode)]:
            first.addWidget(QLabel(title))
            first.addWidget(control, 1)
        for title, control in [('Số features', self.shaptop), ('Sắp xếp', self.shapsort), ('Nhóm', self.shapoutcome), ('Cửa sổ', self.shapwindow)]:
            second.addWidget(QLabel(title))
            second.addWidget(control, 2 if control is self.shapwindow else 0)
        layout.addLayout(first)
        layout.addLayout(second)
        self.shapnote = QLabel()
        self.shapnote.setWordWrap(True)
        self.shapnote.setStyleSheet('color:#475569;padding:5px')
        layout.addWidget(self.shapnote)
        self.shapplot = PlotPane()
        layout.addWidget(self.shapplot, 1)
        self.tabs.addTab(page, 'SHAP')
        rocpage = QWidget()
        roclayout = QVBoxLayout(rocpage)
        self.roc_note = QLabel()
        self.roc_note.setWordWrap(True)
        self.roc_note.setStyleSheet('color:#92400e;padding:6px;background:#fff7ed;border-radius:4px')
        roclayout.addWidget(self.roc_note)
        self.rocplot = PlotPane()
        roclayout.addWidget(self.rocplot, 1)
        self.tabs.addTab(rocpage, 'ROC curve')
        self.shaprun.currentIndexChanged.connect(self.shap_run_changed)
        self.shapfold.currentIndexChanged.connect(self.shap_fold_changed)
        self.shapoutcome.currentIndexChanged.connect(self.shap_fold_changed)
        self.shapwindow.currentIndexChanged.connect(self.draw_shap)
        self.shapmode.currentIndexChanged.connect(self.draw_shap)
        self.shaptop.valueChanged.connect(self.draw_shap)
        self.shapsort.currentIndexChanged.connect(self.draw_shap)

    def refresh_shap_runs(self, runs):
        old = self.shaprun.currentData()
        self.shaprun.blockSignals(True)
        self.shaprun.clear()
        self.shap_runs = runs
        for run in runs:
            self.shaprun.addItem(run.name, run.source + '::' + run.root)
        idx = self.shaprun.findData(old)
        if idx >= 0:
            self.shaprun.setCurrentIndex(idx)
        self.shaprun.blockSignals(False)
        self.shap_run_changed()

    def shap_run_changed(self, *_):
        old = self.shapfold.currentText()
        self.shapfold.blockSignals(True)
        self.shapfold.clear()
        idx = self.shaprun.currentIndex()
        self.current_shaps = self.shap_runs[idx].shap_data if 0 <= idx < len(self.shap_runs) else []
        for data in self.current_shaps:
            self.shapfold.addItem(data.name)
        previous = self.shapfold.findText(old)
        if previous >= 0:
            self.shapfold.setCurrentIndex(previous)
        self.shapfold.blockSignals(False)
        self.shap_fold_changed()
        self.draw_roc()

    def shap_fold_changed(self, *_):
        self.shapwindow.blockSignals(True)
        self.shapwindow.clear()
        idx = self.shapfold.currentIndex()
        data = self.current_shaps[idx] if 0 <= idx < len(self.current_shaps) else None
        if data:
            for i in shap_indices(data, self.shapoutcome.currentText()):
                sample = data.samples[i]
                self.shapwindow.addItem(f"{sample['Recording_ID']} · W{sample['Window_ID']} · {float(sample['Start_Time']):.3f}s · {sample['Outcome']}", int(i))
        self.shapwindow.blockSignals(False)
        self.draw_shap()

    def draw_shap(self, *_):
        mode = self.shapmode.currentIndex()
        self.shapwindow.setEnabled(mode == 2)
        self.shapfold.setEnabled(mode != 3)
        self.shapsort.setEnabled(mode != 2)
        idx = self.shapfold.currentIndex()
        data = self.current_shaps[idx] if 0 <= idx < len(self.current_shaps) else None
        if data is None:
            self.shapnote.setText('Chưa có kết quả SHAP hợp lệ. Chạy shap_fog.py với model và NPZ gốc, rồi nạp lại thư mục kết quả hoặc ZIP chứa SHAP.')
            self.shapplot.show_figure(fig_base('Chưa có dữ liệu SHAP'))
            return
        indices = shap_indices(data, self.shapoutcome.currentText())
        k = min(self.shaptop.value(), len(data.names))
        manifest = data.manifest
        self.shapnote.setText(
            f"{manifest.get('explainer', 'KernelExplainer')} · {shap_units(data)} · Lọc {len(indices)}/{len(data.samples)} mẫu SHAP; "
            f"outer test có {manifest.get('n_test_windows', '?')} cửa sổ · background {manifest.get('n_background', '?')} mẫu train · "
            f"sampling={manifest.get('sampling', '?')} · nsamples={manifest.get('nsamples', '?')} · l1_reg={manifest.get('l1_reg', '?')}. "
            'SHAP trên outer-test; thanh dài không có nghĩa feature tốt, SHAP=0 không chứng minh nhiễu. Không so trực tiếp độ lớn SHAP xác suất RF với decision score SVM. '
            'Beeswarm phản ánh mẫu đã chọn, không có trọng số trên vị trí điểm.'
        )
        try:
            if mode == 3:
                f = shap_stability(self.current_shaps, self.shapoutcome.currentText(), k, self.shapsort.currentText())
            elif not len(indices):
                f = fig_base('Không có mẫu thuộc nhóm đã chọn; tăng --explain hoặc dùng --sampling errors-balanced.')
            elif mode == 0:
                f = shap_bar(data, indices, k, self.shapsort.currentText())
            elif mode == 1:
                f = shap_beeswarm(data, indices, k, self.shapsort.currentText())
            else:
                f = shap_waterfall(data, int(self.shapwindow.currentData()), k)
            if f._suptitle is not None:
                f.suptitle(self.shaprun.currentText() + ' | ' + f._suptitle.get_text(), fontsize=13, fontweight='bold', color='#183153')
            self.shapplot.show_figure(f)
        except (ValueError, TypeError, IndexError) as exc:
            self.shapplot.show_figure(fig_base(str(exc)))

    def draw_roc(self):
        try:
            from sklearn.metrics import roc_curve, auc
            datasets = getattr(self, 'current_shaps', [])
            y_true = []
            scores = []
            for data in datasets:
                for sample, score in zip(data.samples, data.scores):
                    y_true.append(int(sample['Binary_Label']))
                    scores.append(float(score))
            y_true = np.asarray(y_true, dtype=int)
            scores = np.asarray(scores, dtype=float)
            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                self.roc_note.setText('Chưa đủ dữ liệu của cả hai lớp để vẽ ROC. Hãy nạp kết quả SHAP có cả FOG và Non-FOG.')
                self.rocplot.show_figure(fig_base('Không đủ dữ liệu để vẽ ROC'))
                return
            fpr, tpr, _ = roc_curve(y_true, scores)
            score_auc = auc(fpr, tpr)
            fig = fig_base('ROC Curve • ' + self.shaprun.currentText())
            ax = fig.subplots()
            ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {score_auc:.4f}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='#64748b', label='Ngẫu nhiên (AUC = 0.5)')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel('False Positive Rate (1 − Specificity)')
            ax.set_ylabel('True Positive Rate (Sensitivity)')
            ax.grid(alpha=0.25)
            ax.legend(loc='lower right')
            self.roc_note.setText(
                f'ROC tạm tính từ {len(y_true):,} mẫu được lưu trong dữ liệu SHAP của {len(datasets)} fold. '
                'Nếu SHAP được tạo bằng sampling (đặc biệt errors-balanced), phân bố mẫu không đại diện toàn bộ outer-test; '
                'AUC này chỉ để quan sát.'
            )
            self.rocplot.show_figure(fig)
        except Exception as exc:
            self.roc_note.setText('Không thể tính ROC: ' + str(exc))
            self.rocplot.show_figure(fig_base('Không thể tính ROC'))

    def closeEvent(self, event):
        if self.loading:
            event.ignore()
        else:
            event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    paths = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    if paths:
        QTimer.singleShot(100, lambda: window.load(paths))
    sys.exit(app.exec())

if __name__ == '__main__':
    main()