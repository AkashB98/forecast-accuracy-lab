"""CLI: generate | forecast | backtest | report | check | demo | serve.

SYNTHETIC DATA ONLY — Helios Home is a fictional company and every
dataset handled here is fabricated sample data.

Exit codes:
  check : 0 = pass, 1 = accuracy breach (best-model sMAPE above
          --max-smape), 2 = usage / IO / data-quality error
  others: 0 = success, 2 = usage / IO error
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import DEFAULT_HORIZON, rolling_origin_backtest  # noqa: E402
from generator import generate_weeks, values_of, weeks_to_csv, weeks_to_jsonl  # noqa: E402
from report import build_report, forecast_payload  # noqa: E402
from validation import validate_series  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2
DEFAULT_MAX_SMAPE = 20.0


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True)


def _coerce_numbers(row: Dict) -> Dict:
    out = dict(row)
    if isinstance(out.get("week"), str):
        try:
            out["week"] = int(float(out["week"]))
        except ValueError:
            pass
    if isinstance(out.get("revenue"), str):
        try:
            out["revenue"] = float(out["revenue"])
        except ValueError:
            pass
    if isinstance(out.get("week"), float) and out["week"].is_integer():
        out["week"] = int(out["week"])
    return out


def load_weeks(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    stripped = text.strip()
    if not stripped:
        return []
    if path.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        return [_coerce_numbers(dict(row)) for row in reader]
    if stripped.startswith("["):
        return [_coerce_numbers(dict(x)) for x in json.loads(stripped)]
    rows = []
    for line in stripped.splitlines():
        if line.strip():
            rows.append(_coerce_numbers(json.loads(line)))
    return rows


def _load_or_error(path: str) -> Optional[List[Dict]]:
    try:
        return load_weeks(path)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
    except (ValueError, OSError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
    return None


def _write_out(args, text: str, summary: str) -> int:
    if getattr(args, "out", None):
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(summary)
    else:
        print(text, end="")
    return EXIT_OK


def _validated_rows(args):
    rows = _load_or_error(args.input)
    if rows is None:
        return None, EXIT_ERROR
    issues = validate_series(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}", file=sys.stderr)
        return None, EXIT_ERROR
    return rows, EXIT_OK


def cmd_generate(args) -> int:
    rows = generate_weeks(weeks=args.weeks, seed=args.seed)
    fmt = args.format
    if fmt is None:
        fmt = "csv" if (args.out and args.out.endswith(".csv")) else "jsonl"
    text = weeks_to_csv(rows) if fmt == "csv" else weeks_to_jsonl(rows)
    return _write_out(args, text,
                      f"generated {len(rows)} synthetic weeks "
                      f"(seed={args.seed}) -> {args.out}")


def cmd_forecast(args) -> int:
    rows, code = _validated_rows(args)
    if rows is None:
        return code
    try:
        payload = forecast_payload(values_of(rows), horizon=args.horizon,
                                   model=args.model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(payload) + "\n",
                      f"forecast -> {args.out}")


def cmd_backtest(args) -> int:
    rows, code = _validated_rows(args)
    if rows is None:
        return code
    try:
        result = rolling_origin_backtest(values_of(rows),
                                         horizon=args.horizon)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(result) + "\n",
                      f"backtest -> {args.out}")


def cmd_report(args) -> int:
    rows, code = _validated_rows(args)
    if rows is None:
        return code
    try:
        report = build_report(rows, horizon=args.horizon)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(report) + "\n",
                      f"report -> {args.out}")


def cmd_check(args) -> int:
    rows = _load_or_error(args.input)
    if rows is None:
        return EXIT_ERROR
    issues = validate_series(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}")
        print(f"check FAIL (data quality): {len(issues)} issue(s)")
        return EXIT_ERROR
    try:
        result = rolling_origin_backtest(values_of(rows),
                                         horizon=args.horizon)
    except ValueError as exc:
        print(f"error: {exc}")
        return EXIT_ERROR
    best = result["aggregates"][result["best_model"]]
    print(f"weeks={len(rows)} origins={len(result['origins'])} "
          f"horizon={args.horizon} best={result['best_model']} "
          f"best_smape={best['smape']} max_smape={args.max_smape}")
    if best["smape"] > args.max_smape:
        print(f"check FAIL: best-model sMAPE {best['smape']}% above "
              f"limit {args.max_smape}%")
        return EXIT_FAIL
    print("check PASS")
    return EXIT_OK


def cmd_demo(args) -> int:
    import demo as demo_mod
    try:
        demo_mod.main()
    except AssertionError as exc:
        print(f"demo FAILED: {exc}", file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK


# ------------------------------------------------------------------- serve

class _State:
    """Mutable series held by the serve process."""
    def __init__(self):
        self.rows: List[Dict] = generate_weeks()


_STATE = _State()


def get_state() -> _State:
    return _STATE


def reset_state(rows: Optional[List[Dict]] = None) -> None:
    _STATE.rows = list(rows) if rows is not None else generate_weeks()


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: Any) -> None:
        self._send(code, (canonical(payload) + "\n").encode(),
                   "application/json")

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        return self._send_json(404, {"error": "unknown endpoint"})

    def do_POST(self):  # noqa: N802
        if self.path != "/forecast":
            return self._send_json(404, {"error": "unknown endpoint"})
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw) if raw.strip() else {}
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            if "rows" in body:
                rows = [_coerce_numbers(dict(x)) for x in body["rows"]]
                issues = validate_series(rows)
                if issues:
                    return self._send_json(
                        400, {"error": f"data quality: {issues[0]}"})
                _STATE.rows = rows
                values = values_of(rows)
            elif "values" in body:
                values = [float(v) for v in body["values"]]
            else:
                values = values_of(_STATE.rows)
            horizon = int(body.get("horizon", DEFAULT_HORIZON))
            model = body.get("model")
            payload = forecast_payload(values, horizon=horizon,
                                       model=model)
            return self._send_json(200, payload)
        except (ValueError, TypeError, KeyError) as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})

    def log_message(self, *a: Any) -> None:
        pass


def make_server(port: int) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), _Handler)


def cmd_serve(args) -> int:
    if args.input:
        rows = _load_or_error(args.input)
        if rows is None:
            return EXIT_ERROR
        reset_state(rows)
    server = make_server(args.port)
    print(f"serving on http://127.0.0.1:{args.port}  "
          f"(POST /forecast, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forecast-accuracy-lab",
        description="Forecasting + rolling-origin backtest + accuracy "
                    "tracking (Helios Home sample data).")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a synthetic weekly series")
    g.add_argument("--weeks", type=int, default=156)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default=None)
    g.add_argument("--format", choices=["csv", "jsonl"], default=None)
    g.set_defaults(func=cmd_generate)

    f = sub.add_parser("forecast", help="forecast past the end of a series")
    f.add_argument("--input", required=True)
    f.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    f.add_argument("--model", default=None,
                   choices=["naive", "seasonal_naive", "ses", "holt"])
    f.add_argument("--out", default=None)
    f.set_defaults(func=cmd_forecast)

    b = sub.add_parser("backtest", help="rolling-origin backtest")
    b.add_argument("--input", required=True)
    b.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    b.add_argument("--out", default=None)
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("report", help="forecast-vs-actual retrospective")
    r.add_argument("--input", required=True)
    r.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    r.add_argument("--out", default=None)
    r.set_defaults(func=cmd_report)

    c = sub.add_parser("check", help="CI gate: exit 0 pass / 1 breach / 2 error")
    c.add_argument("--input", required=True)
    c.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    c.add_argument("--max-smape", type=float, default=DEFAULT_MAX_SMAPE)
    c.set_defaults(func=cmd_check)

    sub.add_parser("demo", help="run the end-to-end demo").set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="localhost JSON API")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--input", default=None)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
