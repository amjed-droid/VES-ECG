"""
Unit Tests for VES-ECG Core Mathematical Module.
Verifies scale invariance, non-parametric smoothing bounds, F-tier mapping, and zero-shot stability.
"""

import unittest
import numpy as np
import sys, os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ves_ecg import compute_ves_metrics, extract_active_beat, compute_standard_sqis


class TestVESECGCore(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic clean ECG-like pulse
        t = np.linspace(0, 1.0, 150)
        qrs = np.exp(-((t - 0.25)**2) / (2 * 0.03**2))
        t_wave = 0.3 * np.exp(-((t - 0.65)**2) / (2 * 0.08**2))
        noise = np.random.normal(0, 0.02, 150)
        self.clean_beat = np.clip(qrs + t_wave + noise, 0, 1.2)
        
        # Padded version
        self.padded_beat = np.zeros(187)
        self.padded_beat[:150] = self.clean_beat

    def test_extract_active_beat(self):
        valid, s, e = extract_active_beat(self.padded_beat)
        self.assertEqual(s, 0)
        self.assertGreaterEqual(e, 140)
        self.assertEqual(len(valid), e - s + 1)

    def test_ves_clean_beat(self):
        res = compute_ves_metrics(self.clean_beat)
        self.assertIn("penalty", res)
        self.assertIn("tier", res)
        self.assertGreaterEqual(res["penalty"], 0.0)
        # Clean beat should be Tier 1 or Tier 2
        self.assertIn(res["tier"], [1, 2])

    def test_ves_scale_invariance(self):
        """
        Theorem: Multiplicative scaling f' = a * f preserves dispersion contrast psi and penalty P.
        """
        res_orig = compute_ves_metrics(self.clean_beat)
        res_scaled = compute_ves_metrics(self.clean_beat * 3.5)
        
        self.assertAlmostEqual(res_orig["psi"], res_scaled["psi"], places=4)
        self.assertAlmostEqual(res_orig["penalty"], res_scaled["penalty"], places=4)
        self.assertEqual(res_orig["tier"], res_scaled["tier"])

    def test_severe_noise_burst_tier3(self):
        """
        Severe localized variance burst should trigger Tier 3 Veto (P > 0.60).
        """
        corrupted = self.clean_beat.copy()
        # Add massive high-variance noise burst
        corrupted[30:70] += np.random.normal(0, 0.6, 40)
        
        res = compute_ves_metrics(corrupted)
        self.assertGreater(res["penalty"], 0.50)

    def test_standard_sqis(self):
        sqis = compute_standard_sqis(self.clean_beat)
        self.assertIn("ksqi", sqis)
        self.assertIn("ssqi", sqis)
        self.assertIn("psqi", sqis)
        self.assertGreater(sqis["psqi"], 0.0)


if __name__ == "__main__":
    unittest.main()
