import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import (assert_no_leakage, default_origins,
                      rolling_origin_backtest)
from generator import generate_weeks, values_of
from helpers import pure_seasonal_values


class TestBacktestStructure(unittest.TestCase):
    def setUp(self):
        self.values = values_of(generate_weeks())
        self.bt = rolling_origin_backtest(self.values, horizon=8)

    def test_default_origins_for_156_weeks(self):
        self.assertEqual(default_origins(156, 8),
                         [104, 112, 120, 128, 136, 144])
        self.assertEqual(self.bt["origins"], [104, 112, 120, 128, 136, 144])

    def test_run_count_and_windows(self):
        self.assertEqual(len(self.bt["runs"]), 24)  # 6 origins x 4 models
        for run in self.bt["runs"]:
            self.assertEqual(run["train_end"], run["origin"] - 1)
            self.assertEqual(run["train_size"], run["origin"])
            self.assertEqual(len(run["forecasts"]), 8)
            self.assertEqual(run["actuals"],
                             [round(v, 4) for v in
                              self.values[run["origin"]:run["origin"] + 8]])

    def test_aggregates_cover_all_models(self):
        self.assertEqual(set(self.bt["aggregates"]),
                         {"naive", "seasonal_naive", "ses", "holt"})
        self.assertEqual(self.bt["best_model"], "holt")  # pinned, seed 42
        self.assertEqual(self.bt["aggregates"]["holt"]["smape"], 11.4011)

    def test_deterministic(self):
        again = rolling_origin_backtest(self.values, horizon=8)
        self.assertEqual(self.bt, again)

    def test_invalid_origin_raises(self):
        with self.assertRaises(ValueError):
            rolling_origin_backtest(self.values, origins=[150], horizon=8)
        with self.assertRaises(ValueError):
            rolling_origin_backtest(self.values, origins=[0], horizon=8)

    def test_horizon_four(self):
        bt = rolling_origin_backtest(self.values, origins=[120], horizon=4)
        self.assertTrue(all(len(r["forecasts"]) == 4 for r in bt["runs"]))


class TestLeakageGuard(unittest.TestCase):
    def test_poisoning_the_future_changes_no_forecast(self):
        values = values_of(generate_weeks())
        clean = rolling_origin_backtest(values, origins=[104], horizon=8)
        poisoned = list(values)
        for i in range(104, len(poisoned)):
            poisoned[i] = poisoned[i] * 100.0 + 12345.0
        dirty = rolling_origin_backtest(poisoned, origins=[104], horizon=8)
        for c, d in zip(clean["runs"], dirty["runs"]):
            self.assertEqual(c["model"], d["model"])
            self.assertEqual(c["forecasts"], d["forecasts"])
            self.assertNotEqual(c["actuals"], d["actuals"])  # poison landed

    def test_assert_no_leakage_rejects_tampered_result(self):
        bt = rolling_origin_backtest(values_of(generate_weeks()),
                                     origins=[104], horizon=8)
        tampered = dict(bt["runs"][0])
        tampered["train_end"] = tampered["origin"]  # used the origin week
        with self.assertRaises(AssertionError):
            assert_no_leakage(tampered)
        tampered2 = dict(bt["runs"][0])
        tampered2["train_size"] = tampered2["origin"] + 52
        with self.assertRaises(AssertionError):
            assert_no_leakage(tampered2)


class TestSeasonalitySignal(unittest.TestCase):
    def test_seasonal_naive_beats_naive_when_seasonality_dominates(self):
        bt = rolling_origin_backtest(pure_seasonal_values(), horizon=8)
        agg = bt["aggregates"]
        self.assertLess(agg["seasonal_naive"]["smape"],
                        agg["naive"]["smape"])
        self.assertEqual(agg["seasonal_naive"]["smape"], 1.9431)  # pinned

    def test_bias_sign_on_step_series(self):
        # Level drops at the origin: naive keeps forecasting the old
        # level -> systematic OVER-forecast -> positive bias.
        down = [200.0] * 104 + [100.0] * 52
        bt = rolling_origin_backtest(down, origins=[104], horizon=8,
                                     models=["naive"])
        self.assertGreater(bt["aggregates"]["naive"]["bias"], 0)
        # And the mirror image under-forecasts.
        up = [100.0] * 104 + [200.0] * 52
        bt = rolling_origin_backtest(up, origins=[104], horizon=8,
                                     models=["naive"])
        self.assertLess(bt["aggregates"]["naive"]["bias"], 0)


if __name__ == "__main__":
    unittest.main()
