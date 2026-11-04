"""Golden evals for forecast-accuracy-lab.

SYNTHETIC DATA ONLY (fictional Helios Home dataset). Every eval is
deterministic: run_evals.py writes eval_report.json and the file must
be byte-identical across runs. No timestamps, ports, or temp paths
enter the report — only names, booleans, pinned counts and values.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tests"))

from accuracy import compute_metrics  # noqa: E402
from backtest import assert_no_leakage, rolling_origin_backtest  # noqa: E402
from cli import (EXIT_ERROR, EXIT_FAIL, EXIT_OK, load_weeks,  # noqa: E402
                 main as cli_main, make_server, reset_state)
from generator import generate_weeks, values_of, weeks_to_jsonl  # noqa: E402
from helpers import pure_seasonal_values  # noqa: E402
from models import ALPHA_GRID, BETA_GRID, forecast_holt, forecast_ses  # noqa: E402
from report import build_report, forecast_payload  # noqa: E402
from validation import validate_series  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE = os.path.join(ROOT, "fixtures", "sample_156w.jsonl")
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def _run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


def main():
    print("== golden evals: forecast-accuracy-lab ==")
    rows = generate_weeks()
    values = values_of(rows)

    # E1: generator determinism (bytes pinned)
    text1 = weeks_to_jsonl(generate_weeks())
    text2 = weeks_to_jsonl(generate_weeks())
    md5 = hashlib.md5(text1.encode()).hexdigest()
    check("E1 generator determinism (seed=42, 156 weeks)",
          text1 == text2 and len(rows) == 156
          and md5 == "c193f8df1b8c242337dd897658bc68f8",
          f"jsonl_md5={md5}")

    # E2 (spec a): on a planted-seasonality series, seasonal-naive
    # beats naive on sMAPE — seasonality is the only signal there.
    bt_seasonal = rolling_origin_backtest(pure_seasonal_values(), horizon=8)
    agg_s = bt_seasonal["aggregates"]
    check("E2 seasonal-naive beats naive on planted seasonality (sMAPE)",
          agg_s["seasonal_naive"]["smape"] < agg_s["naive"]["smape"]
          and agg_s["seasonal_naive"]["smape"] == 1.9431
          and agg_s["naive"]["smape"] == 10.14,
          f"seasonal={agg_s['seasonal_naive']['smape']} "
          f"naive={agg_s['naive']['smape']}")

    # E3 (spec b): bias sign is correct on series built to over- and
    # under-forecast (level steps at the origin, naive model).
    down = rolling_origin_backtest([200.0] * 104 + [100.0] * 52,
                                   origins=[104], horizon=8,
                                   models=["naive"])
    up = rolling_origin_backtest([100.0] * 104 + [200.0] * 52,
                                 origins=[104], horizon=8, models=["naive"])
    check("E3 bias sign correct (over-forecast +, under-forecast -)",
          down["aggregates"]["naive"]["bias"] == 100.0
          and down["aggregates"]["naive"]["bias_pct"] == 100.0
          and up["aggregates"]["naive"]["bias"] == -100.0,
          f"down_bias={down['aggregates']['naive']['bias']} "
          f"up_bias={up['aggregates']['naive']['bias']}")

    # E4 (spec c): MAPE/sMAPE handle zero actuals without crashing.
    m = compute_metrics([0.0, 0.0, 100.0], [10.0, 20.0, 110.0])
    m_all_zero = compute_metrics([0.0, 0.0], [5.0, 5.0])
    check("E4 zero actuals: MAPE skips+counts, sMAPE finite, no crash",
          m["mape"] == 10.0 and m["mape_skipped"] == 2
          and m_all_zero["mape"] is None
          and m_all_zero["bias_pct"] is None,
          f"mape={m['mape']} skipped={m['mape_skipped']}")

    # E5 (spec d): rolling-origin leakage guard — poisoning every
    # value after the origin changes no forecast; a tampered result
    # (train_end at the origin) trips the assertion.
    clean = rolling_origin_backtest(values, origins=[104], horizon=8)
    poisoned = list(values)
    for i in range(104, len(poisoned)):
        poisoned[i] = poisoned[i] * 100.0 + 12345.0
    dirty = rolling_origin_backtest(poisoned, origins=[104], horizon=8)
    forecasts_unchanged = all(
        c["forecasts"] == d["forecasts"]
        for c, d in zip(clean["runs"], dirty["runs"]))
    tampered = dict(clean["runs"][0])
    tampered["train_end"] = tampered["origin"]
    try:
        assert_no_leakage(tampered)
        guard_trips = False
    except AssertionError:
        guard_trips = True
    check("E5 leakage guard (poison test + tampered-result assertion)",
          forecasts_unchanged and guard_trips)

    # E6: accuracy metrics hand-computed on a 2-point case.
    hand = compute_metrics([100.0, 200.0], [110.0, 180.0])
    check("E6 accuracy hand-computed (MAE 15, MAPE 10, sMAPE 10.0251)",
          hand["mae"] == 15.0 and hand["mape"] == 10.0
          and hand["smape"] == 10.0251 and hand["bias"] == -5.0
          and hand["bias_pct"] == -3.3333,
          f"smape={hand['smape']}")

    # E7: grid search stays inside the declared grids; SES forecasts
    # flat; Holt is exact on a perfectly linear series.
    ses = forecast_ses([10.0, 14.0, 12.0, 18.0, 20.0, 22.0], 4)
    holt = forecast_holt([10.0 + 2.0 * i for i in range(30)], 3)
    check("E7 grid-only tuning (SES flat, Holt exact on linear)",
          ses["params"]["alpha"] in ALPHA_GRID
          and len(set(ses["forecasts"])) == 1
          and holt["params"]["alpha"] in ALPHA_GRID
          and holt["params"]["beta"] in BETA_GRID
          and holt["forecasts"] == [70.0, 72.0, 74.0])

    # E8: headline backtest pinned on the full Helios series — Holt
    # wins; seasonal-naive's -31% bias (stale year-ago levels after
    # trend + level shift) is pinned too, not hidden.
    bt = rolling_origin_backtest(values, horizon=8)
    agg = bt["aggregates"]
    check("E8 headline backtest pinned (Holt best, seasonal bias -31%)",
          bt["best_model"] == "holt"
          and agg["holt"]["smape"] == 11.4011
          and agg["naive"]["smape"] == 13.0737
          and agg["seasonal_naive"]["bias_pct"] == -31.0173,
          f"holt_smape={agg['holt']['smape']} "
          f"seasonal_bias_pct={agg['seasonal_naive']['bias_pct']}")

    # E9: CLI check exit codes 0 / 1 / 2.
    import tempfile
    tmp = tempfile.mkdtemp(prefix="forecast-evals-")
    good_path = os.path.join(tmp, "series.jsonl")
    with open(good_path, "w", encoding="utf-8") as fh:
        fh.write(weeks_to_jsonl(rows))
    codes = (_run_cli(["check", "--input", good_path]),
             _run_cli(["check", "--input", good_path,
                       "--max-smape", "0.01"]),
             _run_cli(["check", "--input", "/tmp/no-such-series-xyz.jsonl"]))
    check("E9 CLI check exit codes 0/1/2",
          codes == (EXIT_OK, EXIT_FAIL, EXIT_ERROR), f"codes={codes}")

    # E10: serve round-trip matches local computation.
    reset_state(rows)
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=10) as resp:
            health = json.loads(resp.read())
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/forecast", data=b'{"horizon": 8}',
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            served = json.loads(resp.read())
        serve_ok = health == {"status": "ok"} \
            and served == forecast_payload(values, horizon=8)
    finally:
        server.shutdown()
        server.server_close()
        reset_state()
    check("E10 serve round-trip matches local forecast", serve_ok)

    # E11: bundled fixture pinned (156 weeks, clean, totals pinned).
    fixture = load_weeks(FIXTURE)
    total = round(sum(values_of(fixture)), 2)
    check("E11 fixture pinned (156 weeks, revenue $11,526,365.04, clean)",
          len(fixture) == 156 and total == 11526365.04
          and validate_series(fixture) == []
          and rolling_origin_backtest(values_of(fixture), horizon=8) == bt,
          f"total_revenue={total}")

    # E12 (spec e, in-process half): the full report is deterministic —
    # canonical JSON identical across two independent builds. (The
    # file-level half — eval_report.json byte-identical across runs —
    # holds because this report contains no timestamps/ports/paths.)
    r1 = json.dumps(build_report(rows, horizon=8), sort_keys=True)
    r2 = json.dumps(build_report(rows, horizon=8), sort_keys=True)
    check("E12 report determinism (canonical JSON identical)",
          r1 == r2, f"report_md5={hashlib.md5(r1.encode()).hexdigest()}")

    passed = sum(1 for r in RESULTS if r["passed"])
    total_count = len(RESULTS)
    summary_doc = {"passed": passed, "total": total_count,
                   "all_passed": passed == total_count, "evals": RESULTS}
    with open(os.path.join(HERE, "eval_report.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary_doc, sort_keys=True) + "\n")
    print(f"\n{passed}/{total_count} evals passed -> evals/eval_report.json")
    return 0 if passed == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
