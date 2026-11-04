"""Forecast models — deliberately small, classical, inspectable.

Four models, stdlib only:

  * naive           — tomorrow looks like today: repeat the last value.
  * seasonal_naive  — this week looks like the same week last year
                      (52 weeks back), cycling through the last season.
  * ses             — simple exponential smoothing: a weighted level,
                      flat forecast. ``alpha`` chosen by a small grid
                      search minimising one-step-ahead SSE on the
                      TRAINING data only.
  * holt            — Holt's additive-trend method: level + trend,
                      linear forecast. ``alpha``/``beta`` chosen by the
                      same training-only grid search.

There is intentionally NO Holt-Winters, NO prediction intervals, and
NO parameter optimisation beyond the grids below — see the README's
scope section.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

SEASON_LENGTH = 52

ALPHA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
BETA_GRID = [0.05, 0.1, 0.2, 0.3]

MODEL_NAMES = ["naive", "seasonal_naive", "ses", "holt"]


def _check_train(train: Sequence[float], horizon: int) -> None:
    if len(train) == 0:
        raise ValueError("training series is empty")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")


def forecast_naive(train: Sequence[float], horizon: int) -> Dict:
    _check_train(train, horizon)
    last = float(train[-1])
    return {"model": "naive", "params": {},
            "forecasts": [last] * horizon}


def forecast_seasonal_naive(train: Sequence[float], horizon: int,
                            season: int = SEASON_LENGTH) -> Dict:
    _check_train(train, horizon)
    if len(train) < season:
        # Not enough history for a full season: fall back to naive.
        return {"model": "seasonal_naive",
                "params": {"season": season, "fallback": "naive"},
                "forecasts": [float(train[-1])] * horizon}
    base = len(train) - season
    forecasts = [float(train[base + (h % season)]) for h in range(horizon)]
    return {"model": "seasonal_naive", "params": {"season": season},
            "forecasts": forecasts}


def _ses_level(train: Sequence[float], alpha: float) -> float:
    level = float(train[0])
    for value in train[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level
    return level


def _ses_one_step_sse(train: Sequence[float], alpha: float) -> float:
    level = float(train[0])
    sse = 0.0
    for value in train[1:]:
        err = float(value) - level
        sse += err * err
        level = alpha * float(value) + (1.0 - alpha) * level
    return sse


def forecast_ses(train: Sequence[float], horizon: int) -> Dict:
    _check_train(train, horizon)
    best_alpha, best_sse = ALPHA_GRID[0], None
    for alpha in ALPHA_GRID:
        sse = _ses_one_step_sse(train, alpha)
        if best_sse is None or sse < best_sse:
            best_alpha, best_sse = alpha, sse
    level = _ses_level(train, best_alpha)
    return {"model": "ses", "params": {"alpha": best_alpha},
            "forecasts": [level] * horizon}


def _holt_state(train: Sequence[float], alpha: float, beta: float):
    level = float(train[0])
    trend = float(train[1]) - float(train[0]) if len(train) > 1 else 0.0
    sse = 0.0
    for value in train[1:]:
        predicted = level + trend
        err = float(value) - predicted
        sse += err * err
        new_level = alpha * float(value) + (1.0 - alpha) * predicted
        trend = beta * (new_level - level) + (1.0 - beta) * trend
        level = new_level
    return level, trend, sse


def forecast_holt(train: Sequence[float], horizon: int) -> Dict:
    _check_train(train, horizon)
    best = None
    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            level, trend, sse = _holt_state(train, alpha, beta)
            if best is None or sse < best[0]:
                best = (sse, alpha, beta, level, trend)
    _, alpha, beta, level, trend = best
    forecasts = [level + (h + 1) * trend for h in range(horizon)]
    return {"model": "holt", "params": {"alpha": alpha, "beta": beta},
            "forecasts": forecasts}


_FORECASTERS = {
    "naive": forecast_naive,
    "seasonal_naive": forecast_seasonal_naive,
    "ses": forecast_ses,
    "holt": forecast_holt,
}


def forecast_model(name: str, train: Sequence[float], horizon: int,
                   season: int = SEASON_LENGTH) -> Dict:
    """Forecast with one named model. ``train`` must contain ONLY data
    available at the forecast origin — callers enforce that."""
    if name not in _FORECASTERS:
        raise ValueError(f"unknown model: {name}")
    if name == "seasonal_naive":
        return forecast_seasonal_naive(train, horizon, season=season)
    return _FORECASTERS[name](train, horizon)


def forecast_all(train: Sequence[float], horizon: int,
                 season: int = SEASON_LENGTH) -> List[Dict]:
    return [forecast_model(name, train, horizon, season=season)
            for name in MODEL_NAMES]
