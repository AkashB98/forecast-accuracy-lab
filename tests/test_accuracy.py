import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy import compute_metrics


class TestHandComputed(unittest.TestCase):
    def test_all_metrics_on_small_case(self):
        m = compute_metrics([100.0, 200.0], [110.0, 180.0])
        self.assertEqual(m["n"], 2)
        self.assertEqual(m["mae"], 15.0)
        self.assertEqual(m["rmse"], round(math.sqrt(250.0), 4))
        self.assertEqual(m["mape"], 10.0)
        self.assertEqual(m["smape"], 10.0251)
        self.assertEqual(m["bias"], -5.0)  # under-forecast on average
        self.assertEqual(m["bias_pct"], -3.3333)
        self.assertEqual(m["mape_skipped"], 0)

    def test_perfect_forecast_is_zero_everywhere(self):
        m = compute_metrics([10.0, 20.0], [10.0, 20.0])
        self.assertEqual((m["mae"], m["rmse"], m["mape"], m["smape"],
                          m["bias"]), (0.0, 0.0, 0.0, 0.0, 0.0))

    def test_length_mismatch_and_empty_raise(self):
        with self.assertRaises(ValueError):
            compute_metrics([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            compute_metrics([], [])


class TestZeroActuals(unittest.TestCase):
    def test_zero_actuals_skipped_for_mape_never_crash(self):
        m = compute_metrics([0.0, 0.0, 100.0], [10.0, 20.0, 110.0])
        self.assertEqual(m["mape"], 10.0)  # only the 100 point counts
        self.assertEqual(m["mape_skipped"], 2)
        self.assertTrue(math.isfinite(m["smape"]))

    def test_all_zero_actuals_mape_and_bias_pct_are_none(self):
        m = compute_metrics([0.0, 0.0], [5.0, 5.0])
        self.assertIsNone(m["mape"])
        self.assertIsNone(m["bias_pct"])
        self.assertEqual(m["mape_skipped"], 2)

    def test_both_zero_point_contributes_zero_smape(self):
        m = compute_metrics([0.0, 100.0], [0.0, 100.0])
        self.assertEqual(m["smape"], 0.0)

    def test_zero_actual_forecast_nonzero_smape_bounded(self):
        m = compute_metrics([0.0], [10.0])
        self.assertEqual(m["smape"], 200.0)  # sMAPE's documented ceiling


class TestBiasSign(unittest.TestCase):
    def test_over_forecast_bias_positive(self):
        m = compute_metrics([100.0, 100.0], [130.0, 120.0])
        self.assertGreater(m["bias"], 0)
        self.assertGreater(m["bias_pct"], 0)

    def test_under_forecast_bias_negative(self):
        m = compute_metrics([100.0, 100.0], [70.0, 80.0])
        self.assertLess(m["bias"], 0)
        self.assertLess(m["bias_pct"], 0)


if __name__ == "__main__":
    unittest.main()
