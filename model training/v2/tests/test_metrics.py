import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.metrics import quick_score, summarize, paired_patient_change, patient_bootstrap


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([[100., 60.], [110., 70.], [120., 80.]])
        self.p = self.y + np.array([[-5., 10.], [5., -10.], [20., 0.]])
        self.s = np.array(["a", "a", "b"])

    def test_hand_calculated(self):
        r = summarize(self.y, self.p, self.s)
        np.testing.assert_allclose(r["pooled"]["mae"], [10, 20 / 3])
        np.testing.assert_allclose(r["macro"]["mae"], [12.5, 5])
        np.testing.assert_allclose(r["pooled"]["bias"], [20 / 3, 0])
        np.testing.assert_allclose(r["pooled"]["error_sd"], (self.p-self.y).std(0, ddof=1))
        np.testing.assert_allclose(r["pooled"]["within_5"], [2 / 3, 1 / 3])
        np.testing.assert_allclose(r["pooled"]["within_10"], [2 / 3, 1])
        np.testing.assert_allclose(quick_score(self.y, self.p, self.s)["macro_mae"], [12.5, 5])

    def test_constant_and_singleton(self):
        r = summarize(np.ones((2, 2)), np.ones((2, 2)), ["a", "b"])
        self.assertEqual(r["pooled"]["pearson_r"], [None, None])
        self.assertEqual(r["pooled"]["r2"], [None, None])
        self.assertEqual(r["per_patient"]["a"]["error_sd"], [None, None])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            summarize(self.y, self.p[:2], self.s)
        with self.assertRaises(ValueError):
            summarize(self.y, self.p * np.nan, self.s)

    def test_paired_bootstrap(self):
        base = summarize(self.y, self.p, self.s)
        perfect = summarize(self.y, self.y, self.s)
        paired = paired_patient_change([perfect] * 3, [base] * 3)
        np.testing.assert_allclose(paired["estimate"], [-12.5, -5])
        fixed = patient_bootstrap(np.array([[2, -1]] * 4))
        self.assertEqual(fixed["ci_lower"], [2, -1])
        self.assertEqual(fixed["ci_upper"], [2, -1])


if __name__ == "__main__":
    unittest.main()
