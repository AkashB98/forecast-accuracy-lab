"""End-to-end demo: generate -> validate -> forecast -> backtest ->
report -> check -> serve.

SYNTHETIC DATA ONLY (fictional Helios Home dataset). Runs fully
offline; the serve round-trip is loopback-only on an ephemeral port.
Every stage asserts.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import EXIT_FAIL, EXIT_OK, main as cli_main, make_server, reset_state  # noqa: E402
from backtest import rolling_origin_backtest  # noqa: E402
from generator import generate_weeks, values_of, weeks_to_jsonl  # noqa: E402
from report import build_report, forecast_payload  # noqa: E402
from validation import validate_series  # noqa: E402


def _request(port: int, path: str, payload=None, method: str = "GET"):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def main() -> None:
    print("== 1. generate synthetic Helios Home weekly series (seed=42, 156 weeks) ==")
    rows = generate_weeks()
    again = generate_weeks()
    assert rows == again, "generator must be deterministic"
    assert len(rows) == 156 and rows[0]["week_start"] == "2023-01-02"
    assert validate_series(rows) == [], validate_series(rows)[:3]
    values = values_of(rows)
    total = round(sum(values), 2)
    print(f"   156 weeks, deterministic re-generation verified, "
          f"total revenue ${total:,.2f}")

    print("\n== 2. forecast: all four models, 8 weeks ahead ==")
    payload = forecast_payload(values, horizon=8)
    assert [m["model"] for m in payload["models"]] == [
        "naive", "seasonal_naive", "ses", "holt"]
    assert all(len(m["forecasts"]) == 8 for m in payload["models"])
    for m in payload["models"]:
        print(f"   {m['model']:<15} params={m['params']} "
              f"first={m['forecasts'][0]:,.2f}")

    print("\n== 3. backtest: rolling origins, leakage-guarded ==")
    bt = rolling_origin_backtest(values, horizon=8)
    assert bt["origins"] == [104, 112, 120, 128, 136, 144]
    assert len(bt["runs"]) == 24  # 6 origins x 4 models
    agg = bt["aggregates"]
    assert bt["best_model"] == "holt"
    # Honest find (see README dev loop): on THIS series seasonal-naive
    # is the worst model — trend (+23%/yr) plus the week-104 level
    # shift leave year-ago values ~31% stale. It only wins where
    # seasonality is the dominant signal (pinned in eval E2).
    assert agg["seasonal_naive"]["bias_pct"] < -20.0
    for name in ("naive", "seasonal_naive", "ses", "holt"):
        a = agg[name]
        print(f"   {name:<15} sMAPE={a['smape']}% MAPE={a['mape']}% "
              f"bias={a['bias_pct']:+}%")
    print(f"   best model: {bt['best_model']}")

    print("\n== 4. retrospective report: deterministic narrative ==")
    report = build_report(rows, horizon=8)
    again_report = build_report(rows, horizon=8)
    assert report == again_report, "report must be deterministic"
    print("   " + report["narrative"].replace("\n", "\n   "))

    print("\n== 5. CLI check gate exit codes ==")
    import tempfile
    tmp = tempfile.mkdtemp(prefix="forecast-demo-")
    good_path = os.path.join(tmp, "series.jsonl")
    with open(good_path, "w", encoding="utf-8") as fh:
        fh.write(weeks_to_jsonl(rows))
    code_ok = cli_main(["check", "--input", good_path])
    code_breach = cli_main(["check", "--input", good_path,
                            "--max-smape", "0.01"])
    code_err = cli_main(["check", "--input", "/tmp/no-such-series-xyz.jsonl"])
    print(f"   normal={code_ok} impossible-threshold={code_breach} "
          f"missing-file={code_err}")
    assert code_ok == EXIT_OK and code_breach == EXIT_FAIL and code_err == 2

    print("\n== 6. live localhost serve round-trip ==")
    reset_state(rows)
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, health = _request(port, "/health")
        assert status == 200 and health == {"status": "ok"}
        status, served = _request(port, "/forecast",
                                  payload={"horizon": 8}, method="POST")
        assert status == 200 and served == payload
        status, served_one = _request(
            port, "/forecast", payload={"horizon": 4, "model": "ses"},
            method="POST")
        local_one = forecast_payload(values, horizon=4, model="ses")
        assert status == 200 and served_one == local_one
        print("   POST /forecast (all models + single model) matches "
              "local computation")
    finally:
        server.shutdown()
        server.server_close()
        reset_state()

    print("\nDemo OK: generate, validate, forecast, backtest, report, "
          "check and serve verified.")


if __name__ == "__main__":
    main()
