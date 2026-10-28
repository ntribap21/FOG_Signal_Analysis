"""Paper-inspired db4 CWT indices, with explicit numerical conventions.

Rezvanian & Lockhart (2016), doi:10.3390/s16040475, equations 1--5.
This is a specified numerical implementation, not an exact reproduction of
the authors' unpublished code. The absolute-coefficient index is an extension.
"""
from functools import lru_cache
import numpy as np

RELEASE_VERSION = 'fog-unified-1.4-cwt'
FEATURE_NAMES = ('CWT_FOG_Index_Signed', 'CWT_FOG_Index_Abs')
FREQUENCIES = tuple(float(f) for f in np.arange(.5, 8.01, .5))
PRECISION = 12
DENOM_RTOL = 1e-8


def _pywt():
    try:
        import pywt
    except ImportError as exc:
        raise ValueError('CWT db4 cần PyWavelets: python -m pip install PyWavelets') from exc
    return pywt


def _validate_fs(fs):
    if not np.isscalar(fs) or not np.isfinite(fs) or fs <= 16:
        raise ValueError('CWT 0.5–8 Hz cần Fs > 16 Hz (khuyến nghị 64 Hz).')
    return float(fs)


@lru_cache(maxsize=16)
def _filter_bank(fs):
    """Bin-integrated, centered db4 correlation kernels; zero outside window.

    Implements the unnormalized integral in paper Eq. 1, including dt=1/fs.
    Integrating psi over each sample bin avoids point-sampling at coarse scales.
    Mother-wavelet support is centered at its midpoint; this alignment is an
    implementation choice absent from the paper's specification.
    """
    fs = _validate_fs(fs)
    pywt = _pywt()
    wavelet = pywt.Wavelet('db4')
    integral, grid = pywt.integrate_wavelet(wavelet, precision=PRECISION)
    center = .5 * (grid[0] + grid[-1])
    fc = float(pywt.central_frequency(wavelet, precision=PRECISION))
    scales = fc * fs / np.asarray(FREQUENCIES)
    filters = []
    for scale in scales:
        radius = int(np.ceil(scale * .5 * (grid[-1] - grid[0]) + .5))
        offsets = np.arange(-radius, radius + 1, dtype=float)
        upper = center + (offsets + .5) / scale
        lower = center + (offsets - .5) / scale
        # Constant primitive beyond the support gives zero wavelet there.
        kernel = (scale / fs) * (np.interp(upper, grid, integral)
                                - np.interp(lower, grid, integral))
        kernel.setflags(write=False)
        filters.append((kernel, radius))
    scales.setflags(write=False)
    return tuple(filters), scales, fc


def settings(fs):
    _, scales, fc = _filter_bank(float(fs))
    return dict(algorithm='db4-bin-integral-centered-zero-v1',
                pywavelets_version=_pywt().__version__,
                reference_doi='10.3390/s16040475', wavelet='db4',
                wavelet_precision=PRECISION, fs_hz=float(fs),
                central_frequency=fc, frequencies_hz=list(FREQUENCIES),
                scales=scales.tolist(), loco_frequencies_hz=list(FREQUENCIES[:6]),
                freeze_frequencies_hz=list(FREQUENCIES[5:]),
                shared_boundary_hz=3., extension='zero within each independent window',
                alignment='mother support centered at midpoint',
                normalization='none; paper Eq.1 integral, includes dt=1/fs',
                input='processed window; no additional centering or filtering',
                flat_window_policy='both indices NaN if peak-to-peak <= 1e-12 input units',
                denominator_relative_tolerance=DENOM_RTOL,
                undefined_policy='NaN if any time point has near-zero denominator; no nanmean or clipping',
                features=list(FEATURE_NAMES),
                signed_definition='mean(100 * sum(C_loco) / (sum(C_loco) + sum(C_freeze)))',
                abs_definition='mean(100 * sum(abs(C_loco)) / (sum(abs(C_loco)) + sum(abs(C_freeze))))',
                abs_is_paper_extension=True,
                exact_author_code_reproduction=False)


def coefficients(x, fs):
    """Return shape (16 frequencies, n samples); no data outside this window."""
    x = np.asarray(x, dtype=float)
    fs = _validate_fs(fs)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError('CWT cần cửa sổ 1D hữu hạn, ít nhất 2 mẫu.')
    if not np.isfinite(np.ptp(x)):
        raise ValueError('Biên độ CWT vượt giới hạn số thực.')
    # A flat window contains no gait oscillation. Keep both indices undefined.
    if np.ptp(x) <= 1e-12:
        return np.zeros((len(FREQUENCIES), x.size), dtype=float)
    filters, _, _ = _filter_bank(fs)
    rows = []
    for kernel, radius in filters:
        full = np.convolve(x, kernel[::-1], mode='full')
        rows.append(full[radius:radius + x.size])
    result = np.asarray(rows)
    if not np.isfinite(result).all():
        raise ValueError('CWT coefficients chứa NaN/Inf.')
    return result


def _ratio_mean(loco, freeze, scale_reference):
    denominator = loco + freeze
    reference = float(np.max(scale_reference))
    if reference <= np.finfo(float).tiny:
        return np.nan
    # Do not silently omit troublesome timestamps or clamp signed ratios.
    if np.any(np.abs(denominator) <= DENOM_RTOL * reference):
        return np.nan
    value = float(np.mean(100. * loco / denominator))
    return value if np.isfinite(value) else np.nan


def features(x, fs):
    c = coefficients(x, fs)
    absolute = np.abs(c)
    loco, freeze = c[:6].sum(axis=0), c[5:].sum(axis=0)
    loco_abs, freeze_abs = absolute[:6].sum(axis=0), absolute[5:].sum(axis=0)
    reference = loco_abs + freeze_abs
    return dict(zip(FEATURE_NAMES,
                    (_ratio_mean(loco, freeze, reference),
                     _ratio_mean(loco_abs, freeze_abs, reference))))
