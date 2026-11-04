"""Forecast-vs-actual retrospective report.

Assembles the backtest into a deterministic retrospective: per-model
aggregate accuracy, the ranking, and a plain-English narrative copied
from the computed numbers (never recomputed in the text layer).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from backtest import rolling_origin_backtest


def build_narrative(backtest: Dict) -> str:
    agg = backtest["aggregates"]
    best_name = backtest["best_model"]
    best = agg[best_name]
    lines = [
        f"Retrospective over {len(backtest['origins'])} rolling origins "
        f"(horizon {backtest['horizon']} weeks, "
        f"{len(backtest['models'])} models).",
        f"Best model by sMAPE: {best_name} at {best['smape']}% "
        f"(MAPE {best['mape']}%, bias {best['bias_pct']:+}%).",
    ]
    naive = agg.get("naive")
    seasonal = agg.get("seasonal_naive")
    if naive and seasonal:
        if seasonal["smape"] < naive["smape"]:
            lines.append(
                f"Seasonal-naive sMAPE {seasonal['smape']}% vs naive "
                f"{naive['smape']}% — the gap is the value of respecting "
                f"the yearly cycle instead of repeating last week.")
        else:
            lines.append(
                f"Seasonal-naive sMAPE {seasonal['smape']}% vs naive "
                f"{naive['smape']}% — here the year-ago values are "
                f"stale (trend plus a level shift), so repeating last "
                f"year loses to repeating last week.")
    biased = max(agg.values(), key=lambda a: abs(a["bias_pct"]))
    if abs(biased["bias_pct"]) >= 2.0:
        direction = "over-forecasts" if biased["bias_pct"] > 0 \
            else "under-forecasts"
        lines.append(
            f"Watch the bias: {biased['model']} systematically "
            f"{direction} by {biased['bias_pct']:+}% on average.")
    return "\n".join(lines)


def build_report(rows: List[Dict], horizon: int = 8) -> Dict:
    from generator import values_of
    values = values_of(rows)
    backtest = rolling_origin_backtest(values, horizon=horizon)
    return {
        "weeks": len(values),
        "horizon": horizon,
        "origins": backtest["origins"],
        "best_model": backtest["best_model"],
        "ranking": backtest["ranking"],
        "aggregates": backtest["aggregates"],
        "runs": backtest["runs"],
        "narrative": build_narrative(backtest),
    }


def forecast_payload(values: Sequence[float], horizon: int = 8,
                     model: str | None = None) -> Dict:
    """Forecast the future (past the end of ``values``) with one or all
    models. Used by the CLI ``forecast`` command and the serve API."""
    from models import MODEL_NAMES, forecast_all, forecast_model
    vals = [float(v) for v in values]
    if model:
        fitted = [forecast_model(model, vals, horizon)]
    else:
        fitted = forecast_all(vals, horizon)
    return {
        "train_size": len(vals),
        "horizon": horizon,
        "models": [
            {"model": f["model"], "params": f["params"],
             "forecasts": [round(x, 4) for x in f["forecasts"]]}
            for f in fitted
        ],
        "model_names": MODEL_NAMES if not model else [model],
    }
