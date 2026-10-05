import unittest
import numpy as np

from mallorn.physics_features import object_features,thermal_fit,luminosity_distance_cm
from mallorn.features import WAVELENGTHS
from mallorn.data import BANDS


class PhysicsTests(unittest.TestCase):
    def task(self):
        t=np.tile(np.linspace(0,180,18),6)
        band=np.repeat(np.arange(6),18)
        flux=(1+.1*band)*20*np.exp(-.5*((t-70)/25)**2)
        error=np.ones(len(t))*.8
        return ('fixture',.2,.02,t,flux,error,band)

    def test_gp_features_are_calendar_translation_invariant(self):
        task=self.task();_,a=object_features(task)
        shifted=(*task[:3],task[3]+40000,*task[4:])
        _,b=object_features(shifted)
        self.assertEqual(set(a),set(b))
        np.testing.assert_allclose(list(a.values()),list(b.values()),rtol=1e-6,atol=1e-6,equal_nan=True)
        self.assertFalse(np.isinf(list(a.values())).any())

    def test_thermal_fit_recovers_synthetic_temperature_and_radius(self):
        z=.15;distance=luminosity_distance_cm(z);temperature=20000.;radius=1e15
        nu=2.99792458e18/np.array([WAVELENGTHS[b] for b in BANDS])*(1+z)
        bb=2*6.62607015e-27*nu**3/2.99792458e10**2/np.expm1(6.62607015e-27*nu/(1.380649e-16*temperature))
        flux=(1+z)*np.pi*bb*(radius/distance)**2/1e-29
        fitted=thermal_fit(flux,.03*flux,z,distance)
        self.assertLess(abs(fitted[0]-np.log10(temperature)),.02)
        self.assertLess(abs(fitted[1]-np.log10(radius)),.04)

    def test_thermal_fit_rejects_unsupported_measurements(self):
        self.assertTrue(np.isnan(thermal_fit(np.ones(6),np.ones(6),.2,luminosity_distance_cm(.2))).all())

    def test_distance_small_redshift_limit(self):
        z=.0001
        self.assertAlmostEqual(luminosity_distance_cm(z)/(299792.458/70*z*3.085677581491367e24),1.,places=3)


if __name__=='__main__':unittest.main()
