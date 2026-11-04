"""Shared builders for the test suite (not a test module)."""

import os
import random
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import seasonal_factor  # noqa: E402

START = date(2023, 1, 2)


def week_row(i, revenue=50000.0):
    return {"week": i,
            "week_start": (START + timedelta(weeks=i)).isoformat(),
            "revenue": revenue}


def weeks_block(n, revenue=50000.0):
    return [week_row(i, revenue) for i in range(n)]


def pure_seasonal_values(n=156, seed=7, base=50000.0, noise=0.02):
    """Series whose ONLY signal is the planted yearly seasonality."""
    rng = random.Random(seed)
    return [round(base * seasonal_factor(i) * rng.gauss(1.0, noise), 2)
            for i in range(n)]
