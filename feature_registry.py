"""Single feature schema with individual, independently selectable features.

Legacy group flags remain accepted when loading old fog_config.json / batch JSON.
New configurations use:
    feature_options.axis_features: list[str]
    feature_options.window_features: {section: list[str]}
"""
import numpy as np
from scipy import signal, stats
from scipy.integrate import trapezoid as _trapezoid

RELEASE_VERSION = 'fog-unified-1.4-cwt'

BASE = [
    'Mean', 'RMS', 'Std', 'Variance', 'Min', 'Max', 'SMA',
    'Dominant_Frequency', 'Total_Power', 'Loco_Power', 'Freeze_Power',
    'Freeze_Index', 'Low_Freq_Ratio', 'PSE'
]

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

DWT_FEATURES = [
    *[
        f'Wavelet_D{level}_{stat}'
        for level in range(1, 6)
        for stat in ('Energy', 'Relative_Energy', 'Std', 'Entropy')
    ],
    'Wavelet_D3_D4_Energy_Ratio',
]

# UI catalog: each string is one independently selectable calculated feature.
AXIS_FEATURE_SECTIONS = {
    'BASIC FEATURES (per sensor / axis)': BASE,
    'STATISTICAL FEATURES': [
        'Skewness', 'Kurtosis_Excess', 'Crest_Factor_Centered', 'Jerk_RMS'
    ],
    'SPECTRAL SHAPE FEATURES': [
        'Spectral_Centroid', 'Spectral_Spread', 'Spectral_Flatness',
        'Harmonic_Even_Odd_Ratio'
    ],
    'ENTROPY FEATURES': ['Sample_Entropy', 'Approx_Entropy'],
    'AUTOCORRELATION FEATURES': ['Autocorr_Peak', 'Autocorr_Lag_s'],
    'WAVELET FEATURES (DWT)': DWT_FEATURES,
    'RODRIGUEZ / MAZILU FEATURES (per axis)': RODRIGUEZ_AXIS_FEATURES,
}

WINDOW_FEATURE_SECTIONS = {
    'MAGNITUDE FEATURES (per sensor)': [
        'Mean', 'RMS', 'Freeze_Index', 'Jerk_RMS'
    ],
    'TEMPORAL FEATURES (per sensor / axis)': [
        'Delta_RMS', 'Delta_Freeze_Index', 'RMS_Trend_g_s'
    ],
    'WITHIN-SENSOR AXIS CORRELATION': [
        'Corr_Forward_Vertical', 'Corr_Forward_Lateral',
        'Corr_Vertical_Lateral'
    ],
    'BETWEEN-SENSOR CROSS-CORRELATION': [
        'Mag_XCorr', 'Mag_XCorr_Lag_s'
    ],
    'ANKLE–TRUNK ENERGY RATIO (ATER)': [
        'Ankle_Trunk_LogATER_Loco', 'Ankle_Trunk_LogATER_Freeze'
    ],
    'CWT FEATURES (Forward / Anterior axis)': [
        'CWT_FOG_Index_Signed', 'CWT_FOG_Index_Abs'
    ],
    'RODRIGUEZ / AHLRICHS MULTI-AXIS FEATURES': RODRIGUEZ_WINDOW_FEATURES,
}

# Legacy group catalog. Kept for migration of older saved configurations.
GROUPS = {
    'skewness': 'Skewness',
    'spectral_shape': 'Spectral centroid / spread / flatness',
    'harmonic': 'Spectral even / odd harmonic ratio',
    'wavelet': 'DWT energy / std / entropy',
    'magnitude': 'Magnitude Mean / RMS / FI / Jerk',
    'temporal': 'Temporal delta / RMS trend',
    'jerk': 'Jerk RMS (g/s)',
    'sampen': 'Sample Entropy',
    'apen': 'Approximate Entropy',
    'autocorr': 'Autocorrelation peak + lag',
    'kurtosis': 'Excess kurtosis',
    'crest': 'Centered crest factor',
    'corr': 'Within-sensor Pearson correlation (lag 0)',
    'crosscorr': 'Between-sensor magnitude cross-correlation',
    'ater': 'Log Ankle-to-Trunk energy ratio (loco / freeze)',
    'cwt': 'CWT db4 FOG index — Forward/Anterior only (signed + absolute)',
    'rodriguez_axis': 'Rodriguez/Mazilu per-axis features',
    'rodriguez_window': 'Rodriguez/Ahlrichs multi-axis features',
}

AXES = ('Forward', 'Vertical', 'Lateral')
SENSORS = ('Ankle', 'Thigh', 'Trunk')

PER_AXIS = {
    'jerk': ['Jerk_RMS'],
    'sampen': ['Sample_Entropy'],
    'apen': ['Approx_Entropy'],
    'autocorr': ['Autocorr_Peak', 'Autocorr_Lag_s'],
    'skewness': ['Skewness'],
    'spectral_shape': ['Spectral_Centroid', 'Spectral_Spread', 'Spectral_Flatness'],
    'harmonic': ['Harmonic_Even_Odd_Ratio'],
    'kurtosis': ['Kurtosis_Excess'],
    'crest': ['Crest_Factor_Centered'],
    'rodriguez_axis': RODRIGUEZ_AXIS_FEATURES,
}

DEFAULTS = dict(
    entropy_m=2, entropy_r_ratio=0.2, autocorr_min_s=0.35,
    autocorr_max_s=1.0, crosscorr_max_s=0.25, ater_epsilon_g2=1e-8,
    wavelet_name='db4', wavelet_level=4, wavelet_mode='symmetric',
    magnitude_mode='dynamic', harmonic_f0_hz=1.0, harmonic_count=8,
    temporal_history=3, cfs=False, cfs_max_features=30
)

_AXIS_OPTIONAL_ORDER = [
    'Skewness', 'Spectral_Centroid', 'Spectral_Spread',
    'Spectral_Flatness', 'Harmonic_Even_Odd_Ratio',
    *DWT_FEATURES,
    'Jerk_RMS', 'Sample_Entropy', 'Approx_Entropy',
    'Autocorr_Peak', 'Autocorr_Lag_s',
    'Kurtosis_Excess', 'Crest_Factor_Centered',
    *RODRIGUEZ_AXIS_FEATURES,
]
_ALL_AXIS_FEATURES = BASE + _AXIS_OPTIONAL_ORDER
_WINDOW_GROUP_KEYS = {
    'magnitude': 'MAGNITUDE FEATURES (per sensor)',
    'temporal': 'TEMPORAL FEATURES (per sensor / axis)',
    'corr': 'WITHIN-SENSOR AXIS CORRELATION',
    'crosscorr': 'BETWEEN-SENSOR CROSS-CORRELATION',
    'ater': 'ANKLE–TRUNK ENERGY RATIO (ATER)',
    'cwt': 'CWT FEATURES (Forward / Anterior axis)',
    'rodriguez_window': 'RODRIGUEZ / AHLRICHS MULTI-AXIS FEATURES',
}
_WINDOW_DEFAULTS = {
    'magnitude': ['Mean', 'RMS', 'Freeze_Index', 'Jerk_RMS'],
    'temporal': ['Delta_RMS', 'Delta_Freeze_Index', 'RMS_Trend_g_s'],
    'corr': ['Corr_Forward_Vertical', 'Corr_Forward_Lateral', 'Corr_Vertical_Lateral'],
    'crosscorr': ['Mag_XCorr', 'Mag_XCorr_Lag_s'],
    'ater': ['Ankle_Trunk_LogATER_Loco', 'Ankle_Trunk_LogATER_Freeze'],
    'cwt': ['CWT_FOG_Index_Signed', 'CWT_FOG_Index_Abs'],
    'rodriguez_window': list(RODRIGUEZ_WINDOW_FEATURES),
}
_WINDOW_ORDER = list(WINDOW_FEATURE_SECTIONS)
WINDOW_FEATURE_SECTION_GROUPS = {
    label: group for group, label in _WINDOW_GROUP_KEYS.items()
}


def _legacy_selections(v, params):
    """Expand old group booleans to the new feature-by-feature schema."""
    base = v.get('base_features', BASE)
    if not isinstance(base, list) or any(k not in BASE for k in base) or len(base) != len(set(base)):
        raise ValueError('Danh sách đặc trưng cơ bản không hợp lệ.')
    axis = set(base)
    for group, names in PER_AXIS.items():
        if v.get(group, False):
            axis.update(names)
    if v.get('wavelet', False):
        level = params['wavelet_level']
        axis.update(
            f'Wavelet_D{j}_{stat}'
            for j in range(1, level + 1)
            for stat in ('Energy', 'Relative_Energy', 'Std', 'Entropy')
        )
        if level >= 4:
            axis.add('Wavelet_D3_D4_Energy_Ratio')
    windows = {}
    for group in _WINDOW_GROUP_KEYS:
        if v.get(group, False):
            windows[group] = list(_WINDOW_DEFAULTS[group])
        else:
            windows[group] = []
    return axis, windows


def options(config):
    """Return normalized individual selections plus compatibility group flags."""
    v = config.get('feature_options', {})
    allowed = set(GROUPS) | set(DEFAULTS) | {
        'base_features', 'axis_features', 'window_features'
    }
    if not isinstance(v, dict) or set(v) - allowed:
        raise ValueError('Feature options không hợp lệ; kiểm tra tên feature / tham số.')

    out = dict(DEFAULTS)
    for key in DEFAULTS:
        if key in v:
            out[key] = v[key]

    if not isinstance(out['cfs'], bool):
        raise ValueError('cfs cần true/false.')
    for group in GROUPS:
        if group in v and not isinstance(v[group], bool):
            raise ValueError(group + ' cần true/false.')
    if not isinstance(out['entropy_m'], int) or isinstance(out['entropy_m'], bool) or not 1 <= out['entropy_m'] <= 3:
        raise ValueError('entropy_m phải 1..3')
    for key in DEFAULTS:
        if isinstance(DEFAULTS[key], (str, bool)):
            continue
        if not isinstance(out[key], (int, float)) or isinstance(out[key], bool) or not np.isfinite(out[key]):
            raise ValueError('Feature parameter không hữu hạn: ' + key)
    if not 0 < out['entropy_r_ratio'] <= 1:
        raise ValueError('Entropy tolerance phải trong (0,1]')
    if not 0 < out['autocorr_min_s'] < out['autocorr_max_s']:
        raise ValueError('Autocorr lag không hợp lệ')
    if not 0 < out['crosscorr_max_s'] <= 1 or out['ater_epsilon_g2'] <= 0:
        raise ValueError('Cross-correlation / ATER parameter không hợp lệ')
    if out['wavelet_name'] not in ('db4', 'db8') or out['wavelet_mode'] not in ('symmetric', 'periodization'):
        raise ValueError('Wavelet / extension mode không hợp lệ.')
    for key, lo, hi in [
        ('wavelet_level', 1, 5), ('harmonic_count', 2, 20),
        ('temporal_history', 2, 20), ('cfs_max_features', 1, 500)
    ]:
        if type(out[key]) is not int or not lo <= out[key] <= hi:
            raise ValueError(key + ' ngoài giới hạn.')
    if out['harmonic_f0_hz'] <= 0:
        raise ValueError('Harmonic f0 phải >0.')
    if out['magnitude_mode'] not in ('dynamic', 'input'):
        raise ValueError('Magnitude mode không hợp lệ.')

    # New format wins when either explicit selector is present.
    if 'axis_features' in v or 'window_features' in v:
        axis = v.get('axis_features', BASE)
        if not isinstance(axis, list) or len(axis) != len(set(axis)) or any(k not in _ALL_AXIS_FEATURES for k in axis):
            raise ValueError('axis_features chứa tên không hợp lệ hoặc bị trùng.')
        raw_windows = v.get('window_features', {})
        if not isinstance(raw_windows, dict) or set(raw_windows) - set(_WINDOW_GROUP_KEYS):
            raise ValueError('window_features phải là object gồm các nhóm feature hợp lệ.')
        windows = {}
        for group in _WINDOW_GROUP_KEYS:
            selected = raw_windows.get(group, [])
            valid = _WINDOW_DEFAULTS[group]
            if not isinstance(selected, list) or len(selected) != len(set(selected)) or any(k not in valid for k in selected):
                raise ValueError(f'window_features.{group} chứa tên không hợp lệ hoặc bị trùng.')
            windows[group] = [k for k in valid if k in selected]
        axis_set = set(axis)
    else:
        axis_set, windows = _legacy_selections(v, out)
        axis = [k for k in _ALL_AXIS_FEATURES if k in axis_set]

    # Stable ordering regardless of the order in a saved JSON file.
    out['axis_features'] = [k for k in _ALL_AXIS_FEATURES if k in set(axis)]
    out['base_features'] = [k for k in BASE if k in axis_set]
    out['window_features'] = {
        group: [k for k in _WINDOW_DEFAULTS[group] if k in set(windows[group])]
        for group in _WINDOW_GROUP_KEYS
    }

    selected = set(out['axis_features'])
    for group, names in PER_AXIS.items():
        out[group] = any(name in selected for name in names)
    out['wavelet'] = any(name in selected for name in DWT_FEATURES)
    out['magnitude'] = bool(out['window_features']['magnitude'])
    out['temporal'] = bool(out['window_features']['temporal'])
    out['corr'] = bool(out['window_features']['corr'])
    out['crosscorr'] = bool(out['window_features']['crosscorr'])
    out['ater'] = bool(out['window_features']['ater'])
    out['cwt'] = bool(out['window_features']['cwt'])
    out['rodriguez_window'] = bool(out['window_features']['rodriguez_window'])

    return out


def serialize_options(config_or_options):
    """Serialize normalized options into the new individual-selection format."""
    o = config_or_options if 'axis_features' in config_or_options else options(config_or_options)
    result = {
        'axis_features': list(o['axis_features']),
        'window_features': {
            key: list(values) for key, values in o['window_features'].items()
        },
    }
    result.update({key: o[key] for key in DEFAULTS})
    return result


def axis_names(config):
    return list(options(config)['axis_features'])


def pairs(config):
    ss = config['sensors']
    return [(a, b) for i, a in enumerate(ss) for b in ss[i + 1:]]


def window_names(config):
    o = options(config)
    selected = o['window_features']
    out = []

    if selected['magnitude']:
        if set(config['axes']) != set(AXES):
            raise ValueError('Magnitude features cần đủ 3 trục.')
        out.extend(
            f'{sensor}_Magnitude_{key}'
            for sensor in config['sensors']
            for key in selected['magnitude']
        )

    if selected['temporal']:
        out.extend(
            f'{sensor}_{axis}_{key}'
            for sensor in config['sensors']
            for axis in config['axes']
            for key in selected['temporal']
        )

    if selected['corr']:
        for sensor in config['sensors']:
            for feature in selected['corr']:
                pair = feature.removeprefix('Corr_')
                a, b = pair.split('_', 1)
                if a not in config['axes'] or b not in config['axes']:
                    raise ValueError(f'{feature} cần cả hai trục {a} và {b}.')
                out.append(f'{sensor}_{feature}')

    if selected['crosscorr']:
        if len(config['sensors']) < 2 or set(config['axes']) != set(AXES):
            raise ValueError('Cross-correlation cần >=2 vị trí và đủ 3 trục.')
        out.extend(
            f'{a}_{b}_{feature}'
            for a, b in pairs(config)
            for feature in selected['crosscorr']
        )

    if selected['ater']:
        if not {'Ankle', 'Trunk'}.issubset(config['sensors']) or set(config['axes']) != set(AXES):
            raise ValueError('ATER cần Ankle + Trunk và đủ 3 trục; dùng Batch Configuration.')
        out.extend(selected['ater'])

    if selected['cwt']:
        if 'Forward' not in config['axes']:
            raise ValueError('CWT cần chọn trục Forward (Anterior-Posterior).')
        out.extend(
            f'{sensor}_Forward_{feature}'
            for sensor in config['sensors']
            for feature in selected['cwt']
        )

    if selected['rodriguez_window']:
        if set(config['axes']) != set(AXES):
            raise ValueError('Rodriguez/Ahlrichs multi-axis cần đủ 3 trục.')
        out.extend(
            f'{sensor}_{feature}'
            for sensor in config['sensors']
            for feature in selected['rodriguez_window']
        )

    return out


def feature_names(config):
    return [
        f'{sensor}_{axis}_{feature}'
        for sensor in config['sensors']
        for axis in config['axes']
        for feature in axis_names(config)
    ] + window_names(config)


def entropy(x, m, r_ratio, approx=False):
    sd = float(np.std(x))
    n = len(x)
    if n < m + 2 or sd <= 1e-12:
        return np.nan
    x = (x - np.mean(x)) / sd
    if approx:
        def phi(k):
            v = np.lib.stride_tricks.sliding_window_view(x, k)
            counts = np.zeros(len(v))
            for start in range(0, len(v), 32):
                dist = np.max(np.abs(v[start:start + 32, None, :] - v[None, :, :]), axis=2)
                counts[start:start + 32] = np.sum(dist <= r_ratio, axis=1)
            return float(np.mean(np.log(counts / len(v))))
        return phi(m) - phi(m + 1)

    v = np.lib.stride_tricks.sliding_window_view(x, m + 1)
    B = A = 0
    for i in range(len(v) - 1):
        diff = np.abs(v[i + 1:] - v[i])
        match = np.max(diff[:, :m], axis=1) <= r_ratio
        B += int(match.sum())
        A += int((match & (diff[:, m] <= r_ratio)).sum())
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan


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


def axis_extra(x, fs, config):
    o = options(config)
    selected = set(o['axis_features']) - set(BASE)
    x = np.asarray(x, float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Raw window chứa NaN/Inf hoặc sai chiều.')
    if len(x) < 2 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Window hoặc Fs không hợp lệ.')
    xc = x - x.mean()
    energy = float(xc @ xc)
    flat = energy / len(x) <= 1e-24
    out = {}

    if 'Skewness' in selected:
        out['Skewness'] = float(stats.skew(x, bias=False)) if not flat else np.nan

    spectral_selected = selected.intersection({
        'Spectral_Centroid', 'Spectral_Spread', 'Spectral_Flatness'
    })
    if spectral_selected:
        f, p = window_psd(x, fs)
        mask = (f >= .5) & (f <= min(16., fs / 2))
        f, p = f[mask], p[mask]
        total = float(p.sum())
        if total > 1e-24:
            centroid = float(np.sum(f * p) / total)
            values = {
                'Spectral_Centroid': centroid,
                'Spectral_Spread': float(np.sqrt(np.sum((f - centroid) ** 2 * p) / total)),
                'Spectral_Flatness': float(np.exp(np.mean(np.log(np.maximum(p, 1e-300)))) / np.mean(p)),
            }
        else:
            values = {key: np.nan for key in spectral_selected}
        out.update({key: values[key] for key in spectral_selected})

    if 'Harmonic_Even_Odd_Ratio' in selected:
        f = np.fft.rfftfreq(len(x), 1 / fs)
        amp = np.abs(np.fft.rfft(xc))
        h = np.arange(1, o['harmonic_count'] + 1)
        freq = h * o['harmonic_f0_hz']
        valid = freq < fs / 2
        h, freq = h[valid], freq[valid]
        a = np.interp(freq, f, amp)
        odd = float(a[h % 2 == 1].sum())
        even = float(a[h % 2 == 0].sum())
        out['Harmonic_Even_Odd_Ratio'] = (
            even / odd if odd > 1e-12 and np.any(h % 2 == 0) and not flat else np.nan
        )

    wavelet_selected = selected.intersection(DWT_FEATURES)
    if wavelet_selected:
        pywt = wavelet_module()
        wavelet = pywt.Wavelet(o['wavelet_name'])
        level = o['wavelet_level']
        if level > pywt.dwt_max_level(len(x), wavelet.dec_len):
            raise ValueError(
                f'DWT {wavelet.name} level {level} vượt mức hữu ích cho {len(x)} mẫu. '
                'Giảm level hoặc tăng window.'
            )
        coeffs = pywt.wavedec(xc, wavelet, mode=o['wavelet_mode'], level=level)
        den = sum(float(c @ c) for c in coeffs)
        energies = {}
        for j in range(1, level + 1):
            c = coeffs[-j]
            detail_energy = float(c @ c)
            energies[j] = detail_energy
            prob = c * c / detail_energy if detail_energy > 1e-24 else np.array([])
            prob = prob[prob > 0]
            prefix = f'Wavelet_D{j}_'
            vals = {
                prefix + 'Energy': detail_energy,
                prefix + 'Relative_Energy': detail_energy / den if den > 1e-24 else np.nan,
                prefix + 'Std': float(np.std(c)),
                prefix + 'Entropy': float(-np.sum(prob * np.log2(prob))) if len(prob) else np.nan,
            }
            out.update({key: value for key, value in vals.items() if key in wavelet_selected})
        if 'Wavelet_D3_D4_Energy_Ratio' in wavelet_selected:
            out['Wavelet_D3_D4_Energy_Ratio'] = (
                energies[3] / energies[4] if level >= 4 and energies.get(4, 0) > 1e-24 else np.nan
            )

    if 'Jerk_RMS' in selected:
        out['Jerk_RMS'] = float(np.sqrt(np.mean((np.diff(x) * fs) ** 2)))
    if 'Sample_Entropy' in selected:
        out['Sample_Entropy'] = entropy(x, o['entropy_m'], o['entropy_r_ratio'])
    if 'Approx_Entropy' in selected:
        out['Approx_Entropy'] = entropy(x, o['entropy_m'], o['entropy_r_ratio'], True)
    if 'Kurtosis_Excess' in selected:
        out['Kurtosis_Excess'] = float(stats.kurtosis(x, fisher=True, bias=False)) if not flat else np.nan
    if 'Crest_Factor_Centered' in selected:
        out['Crest_Factor_Centered'] = (
            float(np.max(np.abs(xc)) / np.sqrt(energy / len(x))) if not flat else np.nan
        )

    if selected.intersection({'Autocorr_Peak', 'Autocorr_Lag_s'}):
        lo = max(1, int(np.ceil(o['autocorr_min_s'] * fs)))
        hi = min(len(x) // 2, int(np.floor(o['autocorr_max_s'] * fs)))
        peak = lag = np.nan
        if not flat and hi >= lo:
            ac = signal.correlate(xc, xc, mode='full', method='auto')[len(x) - 1:] / energy
            k = lo + int(np.argmax(ac[lo:hi + 1]))
            peak, lag = float(ac[k]), k / fs
        if 'Autocorr_Peak' in selected:
            out['Autocorr_Peak'] = peak
        if 'Autocorr_Lag_s' in selected:
            out['Autocorr_Lag_s'] = lag

    rod_selected = selected.intersection(RODRIGUEZ_AXIS_FEATURES)
    if rod_selected:
        psd_features = rod_selected.intersection({
            'Mean_Locomotor', 'Std_Locomotor', 'Entropy_Locomotor',
            'Mean_Freeze', 'Std_Freeze', 'Entropy_Freeze',
            'Peak_Amp_Locomotor', 'Peak_Freq_Locomotor',
            'Peak_Amp_Freeze', 'Peak_Freq_Freeze',
        })
        if psd_features:
            f_psd, p_psd = window_psd(x, fs)
            ml, sl, el, pal, pfl = _band_stats_and_peaks(f_psd, p_psd, .5, 3.)
            mf, sf, ef, paf, pff = _band_stats_and_peaks(f_psd, p_psd, 3., 8.)
            vals = {
                'Mean_Locomotor': ml, 'Std_Locomotor': sl, 'Entropy_Locomotor': el,
                'Mean_Freeze': mf, 'Std_Freeze': sf, 'Entropy_Freeze': ef,
                'Peak_Amp_Locomotor': pal, 'Peak_Freq_Locomotor': pfl,
                'Peak_Amp_Freeze': paf, 'Peak_Freq_Freeze': pff,
            }
            out.update({key: vals[key] for key in psd_features})

        freq_std_features = rod_selected.intersection({'Freq_Std_Low', 'Freq_Std_High'})
        if freq_std_features:
            f_fft = np.fft.rfftfreq(len(xc), 1.0 / fs)
            p_fft = (np.abs(np.fft.rfft(xc)) ** 2) / len(xc)
            vals = {
                'Freq_Std_Low': _freq_std_band(f_fft, p_fft, .1, .68),
                'Freq_Std_High': _freq_std_band(f_fft, p_fft, 8., min(20., fs / 2.)),
            }
            out.update({key: vals[key] for key in freq_std_features})

        ar_features = [f'AR_{i}' for i in range(1, 13) if f'AR_{i}' in rod_selected]
        if ar_features:
            ar_coeffs = _ar_yule_walker(xc, order=12) if not flat else [np.nan] * 12
            out.update({f'AR_{i}': ar_coeffs[i - 1] for i in range(1, 13) if f'AR_{i}' in rod_selected})

        if 'Integrals' in rod_selected:
            out['Integrals'] = float(
            _trapezoid(np.abs(xc), dx=1.0 / fs)
        )

    return out


def corr(u, v):
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    u = u - u.mean()
    v = v - v.mean()
    den = np.sqrt((u @ u) * (v @ v))
    return float(np.clip((u @ v) / den, -1, 1)) if den > 1e-24 else np.nan


def window_extra(signals, fs, config, channel_values):
    o = options(config)
    selected = o['window_features']
    out = {}

    magnitude_selected = selected['magnitude']
    if magnitude_selected:
        for sensor in config['sensors']:
            x = np.column_stack([signals[f'{sensor}_{a}_mg'] for a in AXES])
            if o['magnitude_mode'] == 'dynamic':
                x = x - x.mean(axis=0)
            vm = np.linalg.norm(x, axis=1)
            values = {}
            if 'Mean' in magnitude_selected:
                values['Mean'] = float(vm.mean())
            if 'RMS' in magnitude_selected:
                values['RMS'] = float(np.sqrt(np.mean(vm * vm)))
            if 'Freeze_Index' in magnitude_selected:
                f, p = window_psd(vm, fs)
                loco = band_power(f, p, .5, 3.)
                freeze = band_power(f, p, 3., 8.)
                values['Freeze_Index'] = freeze / loco if loco > 0 else 0.
            if 'Jerk_RMS' in magnitude_selected:
                values['Jerk_RMS'] = float(np.sqrt(np.mean((np.diff(vm) * fs) ** 2)))
            out.update({f'{sensor}_Magnitude_{key}': value for key, value in values.items()})

    corr_selected = selected['corr']
    if corr_selected:
        for sensor in config['sensors']:
            for feature in corr_selected:
                a, b = feature.removeprefix('Corr_').split('_', 1)
                if a in config['axes'] and b in config['axes']:
                    out[f'{sensor}_{feature}'] = corr(signals[f'{sensor}_{a}_mg'], signals[f'{sensor}_{b}_mg'])

    cross_selected = selected['crosscorr']
    if cross_selected:
        mags = {}
        for sensor in config['sensors']:
            x = np.column_stack([signals[f'{sensor}_{a}_mg'] for a in AXES])
            x = x - x.mean(axis=0)
            mags[sensor] = np.linalg.norm(x, axis=1)
        for sensor_a, sensor_b in pairs(config):
            u, v = mags[sensor_a], mags[sensor_b]
            lim = min(int(o['crosscorr_max_s'] * fs), len(u) // 2)
            ks = np.arange(-lim, lim + 1)
            vals = []
            for k in ks:
                uu, vv = (u[:len(u) - k], v[k:]) if k >= 0 else (u[-k:], v[:len(v) + k])
                vals.append(corr(uu, vv))
            vals = np.asarray(vals)
            peak = lag = np.nan
            if np.isfinite(vals).any():
                j = int(np.nanargmax(np.abs(vals)))
                peak, lag = float(vals[j]), float(ks[j] / fs)
            if 'Mag_XCorr' in cross_selected:
                out[f'{sensor_a}_{sensor_b}_Mag_XCorr'] = peak
            if 'Mag_XCorr_Lag_s' in cross_selected:
                out[f'{sensor_a}_{sensor_b}_Mag_XCorr_Lag_s'] = lag

    ater_selected = selected['ater']
    if ater_selected:
        for feature, band, key in [
            ('Ankle_Trunk_LogATER_Loco', 'Loco', 'Loco_Power'),
            ('Ankle_Trunk_LogATER_Freeze', 'Freeze', 'Freeze_Power'),
        ]:
            if feature in ater_selected:
                ea = sum(channel_values[('Ankle', a)][key] for a in AXES)
                et = sum(channel_values[('Trunk', a)][key] for a in AXES)
                ep = o['ater_epsilon_g2']
                out[feature] = float(np.log((ea + ep) / (et + ep)))

    cwt_selected = selected['cwt']
    if cwt_selected:
        if 'Forward' not in config['axes']:
            raise ValueError('CWT cần chọn trục Forward (Anterior-Posterior).')
        from cwt_features import features as cwt_features
        for sensor in config['sensors']:
            cwt_vals = cwt_features(signals[f'{sensor}_Forward_mg'], fs)
            for key in cwt_selected:
                out[f'{sensor}_Forward_{key}'] = cwt_vals[key]

    rod_selected = set(selected['rodriguez_window'])
    if rod_selected:
        if set(config['axes']) != set(AXES):
            raise ValueError('Rodriguez/Ahlrichs multi-axis cần đủ 3 trục.')
        for sensor in config['sensors']:
            ax_x = np.asarray(signals[f'{sensor}_Forward_mg'], float)
            ax_y = np.asarray(signals[f'{sensor}_Vertical_mg'], float)
            ax_z = np.asarray(signals[f'{sensor}_Lateral_mg'], float)
            values = {}

            if 'Spectral_CoM' in rod_selected:
                fx, px = window_psd(ax_x, fs)
                _, py = window_psd(ax_y, fs)
                _, pz = window_psd(ax_z, fs)
                p_sum = px + py + pz
                mask_com = (fx >= .1) & (fx <= 8.)
                total = float(np.sum(p_sum[mask_com]))
                values['Spectral_CoM'] = (
                    float(np.sum(fx[mask_com] * p_sum[mask_com]) / total)
                    if total > 1e-24 else np.nan
                )

            increment_selected = rod_selected.intersection({
                'Increments_AP', 'Diff_Inc_XY', 'Diff_Inc_XZ', 'Diff_Inc_YZ'
            })
            if increment_selected:
                inc_x = float(np.mean(np.abs(np.diff(ax_x))))
                inc_y = float(np.mean(np.abs(np.diff(ax_y))))
                inc_z = float(np.mean(np.abs(np.diff(ax_z))))
                all_increments = {
                    'Increments_AP': inc_x,
                    'Diff_Inc_XY': inc_x - inc_y,
                    'Diff_Inc_XZ': inc_x - inc_z,
                    'Diff_Inc_YZ': inc_y - inc_z,
                }
                values.update({key: all_increments[key] for key in increment_selected})

            pca_selected = rod_selected.intersection({'PCA_1', 'PCA_2', 'PCA_3'})
            if pca_selected:
                mat = np.column_stack([ax_x - ax_x.mean(), ax_y - ax_y.mean(), ax_z - ax_z.mean()])
                cov = (mat.T @ mat) / max(len(ax_x) - 1, 1)
                eigvals = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)
                eig_sum = float(np.sum(eigvals))
                pca = (
                    [float(eigvals[i] / eig_sum) for i in range(3)]
                    if eig_sum > 1e-24 else [np.nan, np.nan, np.nan]
                )
                values.update({f'PCA_{i + 1}': pca[i] for i in range(3) if f'PCA_{i + 1}' in pca_selected})

            out.update({f'{sensor}_{key}': value for key, value in values.items()})

    return out


def wavelet_module():
    try:
        import pywt
    except ImportError as exc:
        raise ValueError('DWT cần PyWavelets: python -m pip install PyWavelets') from exc
    return pywt


def window_psd(x, fs):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    nperseg = min(64, n)
    return signal.welch(
        x, fs=fs, window='hann', nperseg=nperseg,
        noverlap=nperseg // 2, nfft=max(256, n), detrend=False,
        scaling='density'
    )


def band_power(f, p, lo, hi):
    mask = (f >= lo) & (f <= hi)
    if mask.sum() < 2:
        return 0.0
    return float(
    _trapezoid(p[mask], x=f[mask])
    )


def temporal_extra(history, current, start, config, fs):
    o = options(config)
    selected = o['window_features']['temporal']
    out = {}
    if not selected:
        return out

    for sensor in config['sensors']:
        for axis in config['axes']:
            key = (sensor, axis)
            now = current[key]
            previous = history[-1][1][key] if history else None
            prefix = f'{sensor}_{axis}_'

            if 'Delta_RMS' in selected:
                out[prefix + 'Delta_RMS'] = (
                    now['RMS'] - previous['RMS'] if previous is not None else np.nan
                )
            if 'Delta_Freeze_Index' in selected:
                out[prefix + 'Delta_Freeze_Index'] = (
                    now['Freeze_Index'] - previous['Freeze_Index'] if previous is not None else np.nan
                )
            if 'RMS_Trend_g_s' in selected:
                recent = (history + [(start, current)])[-o['temporal_history']:]
                if len(recent) < 2:
                    slope = np.nan
                else:
                    t = np.asarray([h[0] for h in recent])
                    v = np.asarray([h[1][key]['RMS'] for h in recent])
                    t = t - t.mean()
                    slope = float(t @ (v - v.mean()) / (t @ t)) if t @ t > 0 else np.nan
                out[prefix + 'RMS_Trend_g_s'] = slope

    return out
