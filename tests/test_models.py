import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (ALPHA_GRID, BETA_GRID, forecast_all, forecast_holt,
                    forecast_model, forecast_naive, forecast_seasonal_naive,
                    forecast_ses)


class TestNaive(unittest.TestCase):
    def test_repeats_last_value(self):
        out = forecast_naive([10.0, 20.0, 30.0], 3)
        self.assertEqual(out["forecasts"], [30.0, 30.0, 30.0])
        self.assertEqual(out["params"], {})

    def test_empty_train_raises(self):
        with self.assertRaises(ValueError):
            forecast_naive([], 3)

    def test_bad_horizon_raises(self):
        with self.assertRaises(ValueError):
            forecast_naive([1.0], 0)


class TestSeasonalNaive(unittest.TestCase):
    def test_cycles_last_season(self):
        train = [float(i) for i in range(104)]
        out = forecast_seasonal_naive(train, 4)
        self.assertEqual(out["forecasts"], [52.0, 53.0, 54.0, 55.0])

    def test_wraps_beyond_one_season(self):
        train = [float(i) for i in range(52)]
        out = forecast_seasonal_naive(train, 54)
        self.assertEqual(out["forecasts"][52], 0.0)
        self.assertEqual(out["forecasts"][53], 1.0)

    def test_short_history_falls_back_to_naive(self):
        out = forecast_seasonal_naive([5.0, 7.0], 2)
        self.assertEqual(out["forecasts"], [7.0, 7.0])
        self.assertEqual(out["params"]["fallback"], "naive")


class TestSES(unittest.TestCase):
    def test_constant_series_stays_constant(self):
        out = forecast_ses([50.0] * 20, 4)
        self.assertEqual(out["forecasts"], [50.0] * 4)

    def test_forecast_is_flat(self):
        out = forecast_ses([10.0, 14.0, 12.0, 18.0, 20.0], 5)
        self.assertEqual(len(set(out["forecasts"])), 1)

    def test_alpha_comes_from_grid(self):
        out = forecast_ses([10.0, 14.0, 12.0, 18.0, 20.0, 22.0], 2)
        self.assertIn(out["params"]["alpha"], ALPHA_GRID)


class TestHolt(unittest.TestCase):
    def test_exact_on_linear_series(self):
        train = [10.0 + 2.0 * i for i in range(30)]
        out = forecast_holt(train, 3)
        self.assertEqual(out["forecasts"], [70.0, 72.0, 74.0])

    def test_params_come_from_grids(self):
        out = forecast_holt([10.0, 12.0, 15.0, 19.0, 24.0], 2)
        self.assertIn(out["params"]["alpha"], ALPHA_GRID)
        self.assertIn(out["params"]["beta"], BETA_GRID)

    def test_constant_series_stays_constant(self):
        out = forecast_holt([50.0] * 20, 3)
        self.assertEqual(out["forecasts"], [50.0] * 3)


class TestDispatch(unittest.TestCase):
    def test_forecast_all_order_and_shape(self):
        out = forecast_all([float(i % 52) + 100.0 for i in range(120)], 4)
        self.assertEqual([m["model"] for m in out],
                         ["naive", "seasonal_naive", "ses", "holt"])
        self.assertTrue(all(len(m["forecasts"]) == 4 for m in out))

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            forecast_model("holt_winters", [1.0, 2.0], 2)


if __name__ == "__main__":
    unittest.main()
