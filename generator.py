"""Deterministic synthetic weekly revenue generator — Helios Home (fictional).

SYNTHETIC DATA ONLY. No real company, customer, or revenue data is ever
used. Every week is fabricated from a seeded PRNG so the same seed always
yields the same series, byte-for-byte.

One row per week, 156 weeks (3 years) by default starting Monday
2023-01-02:

  week, week_start, revenue

Planted patterns (pinned by tests and golden evals, seed 42):
  * trend — revenue grows ~+0.4% per week, compounding;
  * yearly seasonality — a 52-week sine wave, amplitude ~30% of base,
    peaking around week 10 of each year;
  * noise — multiplicative Gaussian noise, sigma ~4%;
  * level shift — from week 104 (the start of year 3) revenue steps
    up by a permanent factor of 1.18 (a planted distribution gain).
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Dict, List

START_DATE = date(2023, 1, 2)  # a Monday
DEFAULT_SEED = 42
DEFAULT_WEEKS = 156
SEASON_LENGTH = 52

WEEK_FIELDS = ["week", "week_start", "revenue"]

BASE_REVENUE = 50000.0
WEEKLY_TREND = 0.004
SEASON_AMPLITUDE = 0.30
SEASON_PHASE_WEEKS = 10  # seasonal peak lands near this week of the year
NOISE_SIGMA = 0.04
LEVEL_SHIFT_WEEK = 104
LEVEL_SHIFT_FACTOR = 1.18


def seasonal_factor(week_index: int) -> float:
    """Yearly seasonality multiplier for a 0-based week index."""
    angle = 2.0 * math.pi * ((week_index - SEASON_PHASE_WEEKS) % SEASON_LENGTH) / SEASON_LENGTH
    return 1.0 + SEASON_AMPLITUDE * math.cos(angle)


def generate_weeks(weeks: int = DEFAULT_WEEKS, seed: int = DEFAULT_SEED,
                   start: date = START_DATE) -> List[Dict]:
    """Return ``weeks`` synthetic weekly rows, deterministic for ``seed``."""
    if weeks < 0:
        raise ValueError("weeks must be >= 0")
    rng = random.Random(seed)
    rows: List[Dict] = []
    for i in range(weeks):
        d = start + timedelta(weeks=i)
        trend = (1.0 + WEEKLY_TREND) ** i
        season = seasonal_factor(i)
        noise = rng.gauss(1.0, NOISE_SIGMA)
        shift = LEVEL_SHIFT_FACTOR if i >= LEVEL_SHIFT_WEEK else 1.0
        revenue = round(BASE_REVENUE * trend * season * shift * noise, 2)
        rows.append({
            "week": i,
            "week_start": d.isoformat(),
            "revenue": revenue,
        })
    return rows


def values_of(rows: List[Dict]) -> List[float]:
    """Extract the revenue series (floats) from generated/loaded rows."""
    return [float(r["revenue"]) for r in rows]


def weeks_to_csv(rows: List[Dict]) -> str:
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=WEEK_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in WEEK_FIELDS})
    return buf.getvalue()


def weeks_to_jsonl(rows: List[Dict]) -> str:
    import json
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
