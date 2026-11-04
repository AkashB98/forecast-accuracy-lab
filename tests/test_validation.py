import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_weeks
from helpers import week_row, weeks_block
from validation import validate_row, validate_series


class TestValidation(unittest.TestCase):
    def test_generated_series_is_clean(self):
        self.assertEqual(validate_series(generate_weeks()), [])

    def test_empty_series_flagged(self):
        self.assertEqual(validate_series([]), ["series is empty"])

    def test_missing_field(self):
        row = week_row(0)
        del row["revenue"]
        self.assertTrue(any("missing field" in i
                            for i in validate_row(row)))

    def test_negative_revenue(self):
        self.assertTrue(any(">= 0" in i
                            for i in validate_row(week_row(0, -1.0))))

    def test_bool_revenue_rejected(self):
        row = week_row(0)
        row["revenue"] = True
        self.assertTrue(any("must be a number" in i
                            for i in validate_row(row)))

    def test_bad_date(self):
        row = week_row(0)
        row["week_start"] = "02/01/2023"
        self.assertTrue(any("ISO date" in i for i in validate_row(row)))

    def test_gap_detected(self):
        rows = weeks_block(10)
        del rows[5]  # a missing week breaks both sequence and cadence
        issues = validate_series(rows)
        self.assertTrue(any("gap" in i or "sequence" in i for i in issues))

    def test_duplicate_week_detected(self):
        rows = weeks_block(5)
        rows.append(dict(rows[-1]))
        issues = validate_series(rows)
        self.assertTrue(any("duplicate" in i for i in issues))

    def test_never_raises_on_garbage(self):
        issues = validate_series([{"week": "x"}, {}])
        self.assertTrue(len(issues) >= 2)


if __name__ == "__main__":
    unittest.main()
