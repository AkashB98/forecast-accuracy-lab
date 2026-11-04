import hashlib
import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import (LEVEL_SHIFT_WEEK, generate_weeks, seasonal_factor,
                       values_of, weeks_to_csv, weeks_to_jsonl)


class TestGenerator(unittest.TestCase):
    def test_default_shape(self):
        rows = generate_weeks()
        self.assertEqual(len(rows), 156)
        self.assertEqual(rows[0]["week"], 0)
        self.assertEqual(rows[0]["week_start"], "2023-01-02")
        self.assertEqual(date(2023, 1, 2).weekday(), 0)  # Monday
        self.assertEqual(rows[-1]["week"], 155)
        self.assertEqual(rows[-1]["week_start"], "2025-12-22")

    def test_deterministic_same_seed(self):
        self.assertEqual(generate_weeks(), generate_weeks())
        self.assertEqual(weeks_to_jsonl(generate_weeks()),
                         weeks_to_jsonl(generate_weeks()))

    def test_pinned_md5(self):
        md5 = hashlib.md5(weeks_to_jsonl(generate_weeks()).encode()).hexdigest()
        self.assertEqual(md5, "c193f8df1b8c242337dd897658bc68f8")

    def test_different_seed_differs(self):
        self.assertNotEqual(generate_weeks(seed=1), generate_weeks(seed=2))

    def test_week_steps_are_seven_days(self):
        rows = generate_weeks(10)
        dates = [date.fromisoformat(r["week_start"]) for r in rows]
        for a, b in zip(dates, dates[1:]):
            self.assertEqual((b - a).days, 7)

    def test_seasonal_factor_peak_and_trough(self):
        self.assertAlmostEqual(seasonal_factor(10), 1.30, places=9)
        self.assertAlmostEqual(seasonal_factor(36), 0.70, places=9)
        self.assertAlmostEqual(seasonal_factor(10), seasonal_factor(62))

    def test_level_shift_lifts_year_three(self):
        rows = generate_weeks()
        vals = values_of(rows)
        year2 = sum(vals[52:104]) / 52
        year3 = sum(vals[LEVEL_SHIFT_WEEK:]) / 52
        # Trend alone would give ~1.23x; the 1.18 shift stacks on top.
        self.assertGreater(year3, year2 * 1.35)

    def test_revenues_positive(self):
        self.assertTrue(all(v > 0 for v in values_of(generate_weeks())))

    def test_zero_and_negative_weeks(self):
        self.assertEqual(generate_weeks(0), [])
        with self.assertRaises(ValueError):
            generate_weeks(-1)

    def test_csv_round_trip_shape(self):
        text = weeks_to_csv(generate_weeks(3))
        lines = text.strip().splitlines()
        self.assertEqual(lines[0], "week,week_start,revenue")
        self.assertEqual(len(lines), 4)


if __name__ == "__main__":
    unittest.main()
