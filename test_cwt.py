import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pywt
import cwt_features as cw
import feature_registry as fr
import signal_processing as sp
from test_features import configuration


class CWTTests(unittest.TestCase):
    def test_frequency_scales_and_boundary(self):
        s=cw.settings(64)
        self.assertEqual(s['loco_frequencies_hz'],[.5,1,1.5,2,2.5,3])
        self.assertEqual(s['freeze_frequencies_hz'],list(np.arange(3,8.01,.5)))
        np.testing.assert_allclose(np.asarray(s['scales'])*s['frequencies_hz'],64*s['central_frequency'])
        self.assertEqual(len(s['frequencies_hz']),16)
        json.dumps(s,allow_nan=False)

    def test_independent_dense_quadrature(self):
        # Independently integrate a piecewise-constant input against a dense db4
        # wavefun, rather than reusing the bin primitive/filter implementation.
        x=np.random.default_rng(2026).normal(size=128)
        actual=cw.coefficients(x,64)
        _,psi,u=pywt.Wavelet('db4').wavefun(level=16)
        center=.5*(u[0]+u[-1]);fc=pywt.central_frequency('db4',precision=16)
        integrate=getattr(np,'trapezoid',np.trapz)
        for row in [0,5,15]:
            scale=fc*64/cw.FREQUENCIES[row]
            for tau in [0,64,127]:
                sample=np.floor(tau+scale*(u-center)+.5).astype(int)
                inside=(sample>=0)&(sample<len(x));amplitude=np.zeros_like(u)
                amplitude[inside]=x[sample[inside]]
                expected=scale/64*integrate(amplitude*psi,u)
                self.assertAlmostEqual(actual[row,tau],expected,delta=3e-4)

    def test_mean_of_ratios_and_signed_cancellation(self):
        # Shared 3 Hz row belongs to LC and FC. Mean of pointwise ratios is
        # different from ratio of total coefficients for these timestamps.
        c=np.zeros((16,2));c[0]=[1,10];c[5]=[1,1];c[6]=[6,1]
        with patch.object(cw,'coefficients',return_value=c):
            values=cw.features(np.ones(2),64)
        expected=np.mean([100*2/9,100*11/13])
        self.assertAlmostEqual(values[cw.FEATURE_NAMES[0]],expected)
        self.assertAlmostEqual(values[cw.FEATURE_NAMES[1]],expected)
        self.assertNotAlmostEqual(expected,100*13/22)
        c[0]=[1,1];c[5]=0;c[6]=[-1,-1]
        with patch.object(cw,'coefficients',return_value=c):
            values=cw.features(np.ones(2),64)
        self.assertTrue(np.isnan(values[cw.FEATURE_NAMES[0]]))
        self.assertEqual(values[cw.FEATURE_NAMES[1]],50)

    def test_sines_polarity_amplitude_and_flat(self):
        t=np.arange(128)/64;x=np.sin(2*np.pi*2*t)
        normal=cw.features(x,64);high=cw.features(np.sin(2*np.pi*6*t),64)
        self.assertGreater(normal[cw.FEATURE_NAMES[1]],high[cw.FEATURE_NAMES[1]])
        for scale in [-1,5,1e-4]:
            v=cw.features(scale*x,64)
            for key in cw.FEATURE_NAMES:
                np.testing.assert_allclose(v[key],normal[key],rtol=1e-9)
        for x in [np.zeros(128),np.ones(128)]:
            self.assertTrue(all(np.isnan(v) for v in cw.features(x,64).values()))

    def test_invalid_inputs_and_nyquist(self):
        for x,fs in [([1,np.nan],64),([1,np.inf],64),([1],64),([[1,2]],64),([1,2],16),([1,2],np.nan)]:
            with self.assertRaises(ValueError):cw.features(x,fs)
        c=configuration({'cwt':True});c['controls']['target_fs_spinbox']=16
        with self.assertRaises(ValueError):sp.canonical_config(c)

    def test_existing_configs_unchanged_and_cwt_metadata(self):
        root=Path(__file__).parent/'presets'
        for name in ['baseline_126','legacy_206','all_features_443','rms_fi_only_18']:
            old=json.loads((root/(name+'.json')).read_text())
            self.assertEqual(sp.canonical_config(old)['config_id'],old['config_id'])
        old=configuration();new=copy.deepcopy(old)
        new['feature_options']={'cwt':True};new=sp.canonical_config(new)
        self.assertEqual(new['controls'],old['controls'])
        self.assertNotEqual(new['config_id'],old['config_id'])
        self.assertIn('cwt_settings',new)
        self.assertEqual(len(fr.feature_names(new)),132)
        self.assertEqual(sp.canonical_config(new)['config_id'],new['config_id'])
        new['feature_options']['cwt']=False;new=sp.canonical_config(new)
        self.assertNotIn('cwt_settings',new)
        self.assertEqual(new['config_id'],old['config_id'])


if __name__=='__main__':unittest.main(verbosity=2)
