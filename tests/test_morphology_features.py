import unittest

import numpy as np

from mallorn.morphology_features import object_features


class MorphologyTests(unittest.TestCase):
    def task(self):
        phase = np.tile(np.linspace(-80, 220, 101), 3)
        band = np.repeat([1, 2, 3], 101)
        shape = np.exp(-.5 * (np.minimum(phase, 0) / 20) ** 2)
        shape *= (1 + np.maximum(phase, 0) / 50) ** (-5 / 3)
        flux = (20 + 5 * band) * shape
        return ('synthetic', .2, 0., 50000 + 1.2 * phase,
                flux, np.ones(len(flux)) * .3, band)

    def test_powerlaw_fit_recovers_rest_frame_scales(self):
        _, values = object_features(self.task())
        self.assertAlmostEqual(values['flare_powerlaw_rise'], 20., places=3)
        self.assertAlmostEqual(values['flare_powerlaw_fall'], 50., places=3)
        self.assertLess(values['flare_powerlaw_chi2'], 1e-8)
        self.assertLess(values['flare_powerlaw_exp_chi2_ratio'], 1e-6)

    def test_calendar_translation_does_not_change_features(self):
        task = self.task()
        _, original = object_features(task)
        _, shifted = object_features((*task[:3], task[3] + 20000, *task[4:]))
        self.assertEqual(original.keys(), shifted.keys())
        np.testing.assert_allclose(list(original.values()), list(shifted.values()),
                                   rtol=1e-5, atol=1e-5, equal_nan=True)
        self.assertFalse(np.isinf(list(original.values())).any())


if __name__ == '__main__':
    unittest.main()
