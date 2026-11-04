"""Rolling-origin backtest with a hard leakage guard.

A rolling-origin backtest replays history honestly: at each origin
(index ``o``), every model is fit on ``values[:o]`` ONLY and scored
against ``values[o:o+horizon]``. Nothing after the origin may reach a
model — that is the single easiest way to fabricate a good forecast,
so this module guards it twice:

  1. structurally — models only ever receive the sliced prefix;
  2. by assertion — ``assert_no_leakage`` verifies, for every result,
     that the recorded ``train_end`` (the last training index used)
     is strictly before the origin, and the scoring window starts
     exactly at the origin.

The poison test in the test suite proves the guard is real: mutating
every value after an origin leaves that origin's forecasts
byte-identical.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from accuracy import compute_metrics
from models import MODEL_NAMES, forecast_model

DEFAULT_HORIZON = 8
# Origins for the default 156-week series: every 8 weeks through
# year 3, so the planted level shift (week 104) sits inside the
# training window for the later origins.
DEFAULT_ORIGINS = [104, 112, 120, 128, 136, 144]


def assert_no_leakage(result: Dict) -> None:
    """Raise AssertionError if a backtest result used future data."""
    origin = result["origin"]
    assert result["train_end"] == origin - 1, (
        f"leakage: train_end {result['train_end']} is not origin-1 "
        f"({origin - 1})")
    assert result["train_size"] == origin, (
        f"leakage: train_size {result['train_size']} != origin {origin}")
    assert len(result["actuals"]) == result["horizon"], (
        "scoring window size mismatch")


def default_origins(n: int, horizon: int = DEFAULT_HORIZON) -> List[int]:
    origins = [o for o in DEFAULT_ORIGINS if o + horizon <= n]
    if not origins:
        # Short series: fall back to origins in the back half.
        origins = [o for o in range(max(2, n // 2), n - horizon + 1, 4)]
    return origins


def rolling_origin_backtest(values: Sequence[float],
                            origins: Sequence[int] | None = None,
                            horizon: int = DEFAULT_HORIZON,
                            models: Sequence[str] | None = None,
                            season: int = 52) -> Dict:
    """Run the backtest. Returns per-model aggregates + per-origin detail."""
    vals = [float(v) for v in values]
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if origins is None:
        origins = default_origins(len(vals), horizon)
    if not origins:
        raise ValueError("no valid origins for this series length/horizon")
    names = list(models) if models else list(MODEL_NAMES)

    runs: List[Dict] = []
    for origin in origins:
        if origin < 1 or origin + horizon > len(vals):
            raise ValueError(
                f"origin {origin} invalid for n={len(vals)}, "
                f"horizon={horizon}")
        train = vals[:origin]  # the ONLY data a model may see
        actuals = vals[origin:origin + horizon]
        for name in names:
            fitted = forecast_model(name, train, horizon, season=season)
            metrics = compute_metrics(actuals, fitted["forecasts"])
            result = {
                "model": name,
                "origin": origin,
                "train_end": origin - 1,
                "train_size": len(train),
                "horizon": horizon,
                "params": fitted["params"],
                "forecasts": [round(f, 4) for f in fitted["forecasts"]],
                "actuals": [round(a, 4) for a in actuals],
                "metrics": metrics,
            }
            assert_no_leakage(result)
            runs.append(result)

    aggregates: Dict[str, Dict] = {}
    for name in names:
        mine = [r for r in runs if r["model"] == name]
        k = len(mine)
        mapes = [r["metrics"]["mape"] for r in mine
                 if r["metrics"]["mape"] is not None]
        aggregates[name] = {
            "model": name,
            "origins": k,
            "mae": round(sum(r["metrics"]["mae"] for r in mine) / k, 4),
            "rmse": round(sum(r["metrics"]["rmse"] for r in mine) / k, 4),
            "mape": round(sum(mapes) / len(mapes), 4) if mapes else None,
            "smape": round(sum(r["metrics"]["smape"] for r in mine) / k, 4),
            "bias": round(sum(r["metrics"]["bias"] for r in mine) / k, 4),
            "bias_pct": round(
                sum(r["metrics"]["bias_pct"] for r in mine
                    if r["metrics"]["bias_pct"] is not None)
                / max(1, sum(1 for r in mine
                             if r["metrics"]["bias_pct"] is not None)), 4),
        }

    ranking = sorted(aggregates.values(), key=lambda a: a["smape"])
    return {
        "n": len(vals),
        "horizon": horizon,
        "origins": list(origins),
        "models": names,
        "aggregates": aggregates,
        "ranking": [a["model"] for a in ranking],
        "best_model": ranking[0]["model"] if ranking else None,
        "runs": runs,
    }
