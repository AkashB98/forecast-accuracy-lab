"""Data-quality gate for weekly series rows.

Reports issues as strings and NEVER raises on bad data, so the CLI
``check`` gate can map data problems to exit code 2, distinctly from
an accuracy breach (exit 1).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

REQUIRED_FIELDS = ("week", "week_start", "revenue")


def validate_row(row: Dict) -> List[str]:
    issues: List[str] = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            issues.append(f"missing field: {field}")
    if issues:
        return issues
    if isinstance(row["week"], bool) or not isinstance(row["week"], int):
        issues.append(f"week must be an int, got {type(row['week']).__name__}")
    if isinstance(row["revenue"], bool) or \
            not isinstance(row["revenue"], (int, float)):
        issues.append("revenue must be a number")
    elif row["revenue"] < 0:
        issues.append(f"revenue must be >= 0, got {row['revenue']}")
    try:
        date.fromisoformat(str(row["week_start"]))
    except ValueError:
        issues.append(f"week_start is not an ISO date: {row['week_start']}")
    return issues


def validate_series(rows: List[Dict]) -> List[str]:
    issues: List[str] = []
    if not rows:
        return ["series is empty"]
    seen_weeks = set()
    prev_date = None
    for i, row in enumerate(rows):
        for issue in validate_row(row):
            issues.append(f"row {i}: {issue}")
        if "week" in row:
            if row["week"] in seen_weeks:
                issues.append(f"row {i}: duplicate week {row['week']}")
            seen_weeks.add(row["week"])
            if row["week"] != i:
                issues.append(
                    f"row {i}: week index {row['week']} out of sequence")
        try:
            d = date.fromisoformat(str(row.get("week_start")))
        except ValueError:
            continue
        if prev_date is not None and d - prev_date != timedelta(weeks=1):
            issues.append(
                f"row {i}: week_start gap — {prev_date} -> {d} "
                f"is not exactly 7 days")
        prev_date = d
    return issues
