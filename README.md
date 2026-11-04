# forecast-accuracy-lab

**Production-style forecasting, graded honestly: four classical forecast models, a rolling-origin backtest that cannot peek at the future, and per-model accuracy tracking (MAPE, sMAPE, RMSE, MAE, bias) with a forecast-vs-actual retrospective — generated, tested, and reproducible end to end.**

Forecasting is easy to demo and hard to do honestly. A model fit on all the data, scored on the same data, always looks brilliant — until it meets next quarter. The work that matters in production forecasting teams is the discipline around the model: replay history the way it actually unfolded (**rolling-origin backtesting** — refit at successive points in the past using only what was known then), guard against **leakage** (any use of future data in training, the quiet bug behind most too-good forecast results), track each model's **bias** (its signed average error — a model can be accurate on average and still systematically over-promise revenue), and keep a retrospective that says which model earned trust and why. This project builds that whole loop around a small, inspectable model set, with tests and golden evals pinning every claim.

> **Synthetic data only.** Every figure is fabricated by a seeded generator for a fictional smart-home store (**Helios Home**). No real company, customer, or revenue data is used anywhere in this project.

Stdlib-only Python. Zero network, zero API keys, zero external dependencies.

## How this differs from its sibling projects

> **MMM explains past drivers, exec-kpi-kit reports what happened, this forecasts and grades itself.**

The marketing-mix model looks backwards and attributes; the KPI kit reports the present against targets. This one makes a claim about the future, then grades that claim against what actually happened — the retrospective is the product.

## Quickstart

```bash
# 1. Generate the synthetic Helios Home series (156 weeks, seed 42)
python3 cli.py generate --out series.jsonl

# 2. Forecast 8 weeks past the end of the series (all four models)
python3 cli.py forecast --input series.jsonl --horizon 8

# 3. Rolling-origin backtest: every model, every origin, leakage-guarded
python3 cli.py backtest --input series.jsonl

# 4. Retrospective report: aggregates + ranking + narrative (JSON)
python3 cli.py report --input series.jsonl

# 5. CI gate: exit 0 = pass, 1 = accuracy breach (best-model sMAPE
#    above --max-smape), 2 = data-quality / IO error
python3 cli.py check --input series.jsonl

# 6. One-command end-to-end demo (generate -> forecast -> backtest
#    -> report -> check -> serve)
python3 demo.py

# 7. Tiny localhost API: POST /forecast, GET /health on 127.0.0.1
python3 cli.py serve --port 8765
```

## The seeded series at a glance (seed 42, 156 weeks, 2023-01-02 → 2025-12-22)

**$11,526,365.04** of revenue over 3 years of weeks. The generator plants four patterns: a **+0.4%/week trend** (compounding to ~+23%/year), **yearly seasonality** (a 52-week sine, ±30%, peaking near week 10), **~4% multiplicative noise**, and one **level shift** — from week 104, revenue steps up permanently ×1.18 (a planted distribution gain).

Backtest results — 6 rolling origins (104…144), 8-week horizon, mean across origins:

| Model | sMAPE | MAPE | RMSE | Bias % |
|---|---|---|---|---|
| **Holt (additive trend)** | **11.40%** | **10.98%** | **12,429** | −2.47% |
| Naive | 13.07% | 12.45% | 13,705 | −2.20% |
| SES | 13.28% | 12.68% | 13,980 | −2.03% |
| Seasonal-naive | 36.73% | 30.92% | 30,711 | **−31.02%** |

Read the last row carefully, because it is the most instructive result in the project: seasonal-naive — the model that "respects seasonality" — is by far the *worst* here. Its year-ago reference values predate both a year of +23% trend growth and the ×1.18 level shift, so it systematically under-forecasts by 31%. On a series where seasonality is the *only* signal, the same model wins decisively (sMAPE 1.94% vs naive's 10.14%, pinned by eval E2). Both facts are pinned, not cherry-picked: a model's reputation is a property of the series, not the model. These numbers are pinned by golden evals, so any change to the generator, models, or backtest shows up as an eval failure, not a surprise.

## How it works

**The models.** Four classical baselines, each small enough to audit line by line: **naive** (repeat the last value), **seasonal-naive** (repeat the same week last year, cycling the last season; falls back to naive with under a year of history), **SES** (**simple exponential smoothing** — a weighted level, flat forecast), and **Holt** (**Holt's additive-trend method** — level plus trend, linear forecast). SES's `alpha` and Holt's `alpha`/`beta` are chosen by a small grid search (5 × 4 points) minimising one-step-ahead error **on the training window only**. There is deliberately no Holt-Winters and no optimiser — see Scope.

**The backtest.** At each origin, every model is fit on the prefix `values[:origin]` and scored on the next 8 weeks. Leakage is guarded twice: structurally (models only ever receive the sliced prefix) and by assertion (`train_end == origin − 1`, `train_size == origin` on every result, checked inside the backtest itself). A poison test proves the guard: multiply every post-origin value by 100 and the origin's forecasts are byte-identical (eval E5).

**The metrics.** MAE and RMSE for scale, **MAPE** (**mean absolute percentage error**) and **sMAPE** (**symmetric MAPE** — error relative to the average of actual and forecast, so it stays bounded) for comparability, and **bias** (mean signed error, plus bias % of total actuals) for direction. Zero actuals are handled, never crashed on: MAPE skips them and reports how many it skipped (`mape_skipped`), and is `None` when *every* actual is zero; an sMAPE point with actual 0 and forecast 10 scores the documented 200% ceiling rather than dividing by zero.

**The retrospective.** `report` assembles per-model aggregates, the sMAPE ranking, per-origin forecast-vs-actual detail, and a template narrative copied from the computed numbers — including which model is systematically biased and in which direction. Same series, same narrative, word for word (eval E12).

**Data quality first.** `validation.py` runs before any forecast: required fields, week-sequence and 7-day cadence continuity, duplicates, non-negative revenue, int/bool traps. It reports issues and never raises on bad data, so the `check` gate maps them to exit code 2, distinctly from an accuracy breach (exit 1).

## Architecture

| Module | Job |
|---|---|
| `generator.py` | Deterministic synthetic Helios Home weekly revenue (156 weeks): trend, yearly seasonality, noise, one planted level shift; CSV / JSONL serialisers |
| `models.py` | The four forecasters (naive, seasonal-naive, SES, Holt) + training-only grid search over declared alpha/beta grids |
| `accuracy.py` | MAE, RMSE, MAPE (zero-safe, skip-counted), sMAPE, bias and bias % — error = forecast − actual throughout |
| `backtest.py` | Rolling-origin engine: prefix-only fitting, per-origin results, leakage assertion on every run, per-model aggregates + ranking |
| `report.py` | Forecast-vs-actual retrospective (aggregates, ranking, per-origin detail) + deterministic narrative; also the shared forecast payload used by CLI and API |
| `validation.py` | Data-quality gate: fields, week sequence/cadence, duplicates, ranges — reports issues, never raises on bad data |
| `cli.py` | `generate` / `forecast` / `backtest` / `report` / `check` (CI exit 0/1/2) / `demo` / `serve` (stdlib `http.server` on 127.0.0.1: `POST /forecast`, `GET /health`); CSV, JSONL, and JSON-array inputs |
| `demo.py` | End-to-end narrative with asserts: determinism, planted-pattern behaviour, backtest ranking, check exit codes, live localhost serve round-trip |
| `fixtures/` | The full seeded 156-week series (`sample_156w.jsonl` + `.csv`) with pinned totals |

## Evals

`python3 evals/run_evals.py` → **12/12 golden evals**, report committed at `evals/eval_report.json` (md5 `14e5b9fa3f172a395d2a6dc084219137`, byte-identical across 3 verified runs):

- **E1** generator determinism — seed 42 JSONL bytes pinned by md5
- **E2** on a planted-seasonality series, seasonal-naive beats naive on sMAPE (1.94% vs 10.14%)
- **E3** bias sign correct on series built to over-forecast (+100.0) and under-forecast (−100.0)
- **E4** MAPE/sMAPE handle zero actuals (skipped + counted; all-zero → `None`, never a crash)
- **E5** leakage guard — poison test changes no forecast; a tampered `train_end` trips the assertion
- **E6** accuracy metrics hand-computed on a 2-point case (MAE 15, MAPE 10, sMAPE 10.0251, bias −5)
- **E7** grid-only tuning — chosen parameters stay inside the declared grids; Holt exact on a linear series
- **E8** headline backtest pinned (Holt best at sMAPE 11.4011%; seasonal-naive bias −31.0173% pinned, not hidden)
- **E9** CLI `check` exit codes 0 / 1 / 2
- **E10** serve round-trip matches local forecast exactly
- **E11** bundled fixture pinned (156 weeks, total revenue $11,526,365.04, validates clean)
- **E12** report determinism — canonical JSON identical across independent builds

Unit tests: `python3 -m unittest discover -s tests` → **66 hermetic tests**, including a no-network enforcement test (sockets stubbed), the leakage poison test, zero-actual edge cases, bias-sign constructions, hand-computed metrics, grid-membership checks, validation defects, CLI exit codes, JSON/CSV round-trips, and serve round-trips.

## Development loop — what the checks actually caught

1. **Real find: the "seasonal" model was the worst model, and the first draft assumed otherwise.** The demo originally asserted seasonal-naive would beat naive on the headline series — the textbook expectation. It failed immediately: seasonal-naive scored sMAPE 36.7% with a −31% bias. The model wasn't buggy (it wins 1.94% vs 10.14% on the pure-seasonality series, eval E2); the *assumption* was. With +23%/year trend and a ×1.18 level shift at week 104, last year's values are simply stale. Fix: the assertion was replaced by the pinned, honest result (Holt best; seasonal bias pinned in E8), and the README table leads with it. A backtest that only confirms your prior is marketing, not evaluation.
2. **Real bug: the narrative template asserted the wrong lesson.** The first retrospective narrative printed "the gap is the value of respecting the yearly cycle" whenever it compared seasonal-naive to naive — including when seasonal-naive had *lost* by 23 points, so the report confidently explained the opposite of its own numbers. Fixed: the narrative branches on the actual comparison — credit when the seasonal model wins, a staleness explanation when it loses. Pinned indirectly by E12 determinism and covered by the report tests.
3. **The check threshold is a judgement call, and it is set where the data argues.** `check` fails when even the *best* model breaches `--max-smape` (default 20%): Holt's 11.40% passes with headroom, while a series so noisy or shifted that no model can track it fails the gate instead of shipping a confident number. The default is documented here rather than buried in the CLI help.

## Scope and honest limits

- **No Holt-Winters, no prediction intervals, no parameter optimisation beyond the declared grids.** Those are the obvious next model upgrades; the project's claim is the evaluation discipline around the models, not model sophistication.
- **No scenario or driver analysis** — the retrospective says *which* model was wrong and by how much, not *why* the business moved. Adding driver/scenario analysis on top of the backtest (e.g. "what would accuracy have been without the level shift?") is the named next step.
- **Seasonal-naive assumes a stationary level.** As the headline table shows, trend and level shifts break it; that failure is documented and pinned rather than patched over with an unprincipled fix.
- **Origins are fixed 8-week steps in year 3** of the seeded series, so the level shift sits inside later training windows — representative of monitoring a live forecast, not of cold-start performance.
- **The API is a demo surface.** It binds 127.0.0.1 only, holds its series in memory, and resets on restart.

## Why this project

It takes the author's day-job craft — marketing and portfolio analytics: revenue reporting, KPI definitions, weekly business reviews — and extends it into the forward-looking half of that work: making a forecast, replaying it honestly against history, and keeping score of your own accuracy over time. The same discipline a forward-deployed engineer brings to a customer problem: make the logic inspectable, make the outputs reproducible, and make the failure modes exit codes.
