"""Forecast accuracy metrics.

Error convention: error = forecast - actual (so a POSITIVE bias means
the model over-forecasts).

  MAE   — mean absolute error.
  RMSE  — root mean squared error.
  MAPE  — mean absolute percentage error over NON-ZERO actuals only.
          Zero actuals are skipped and counted in ``mape_skipped``;
          if every actual is zero, ``mape`` is None. Never crashes.
  sMAPE — symmetric MAPE: mean of 2*|err| / (|actual| + |forecast|),
          in percent. A point where both actual and forecast are 0
          contributes 0 (no division by zero).
  bias  — mean error (signed).
  bias_pct — 100 * sum(errors) / sum(actuals); None when the actuals
          sum to zero.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def compute_metrics(actuals: Sequence[float],
                    forecasts: Sequence[float]) -> Dict:
    if len(actuals) != len(forecasts):
        raise ValueError("actuals and forecasts must have equal length")
    n = len(actuals)
    if n == 0:
        raise ValueError("no points to score")
    errors = [float(f) - float(a) for a, f in zip(actuals, forecasts)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    bias = sum(errors) / n

    ape_terms: List[float] = []
    skipped = 0
    for a, e in zip(actuals, errors):
        if float(a) == 0.0:
            skipped += 1
        else:
            ape_terms.append(abs(e) / abs(float(a)))
    mape: Optional[float] = (100.0 * sum(ape_terms) / len(ape_terms)
                             if ape_terms else None)

    smape_terms = []
    for a, f in zip(actuals, forecasts):
        denom = abs(float(a)) + abs(float(f))
        smape_terms.append(0.0 if denom == 0.0
                           else 2.0 * abs(float(f) - float(a)) / denom)
    smape = 100.0 * sum(smape_terms) / n

    total_actual = sum(float(a) for a in actuals)
    bias_pct: Optional[float] = (100.0 * sum(errors) / total_actual
                                 if total_actual != 0.0 else None)

    return {
        "n": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4) if mape is not None else None,
        "mape_skipped": skipped,
        "smape": round(smape, 4),
        "bias": round(bias, 4),
        "bias_pct": round(bias_pct, 4) if bias_pct is not None else None,
    }
