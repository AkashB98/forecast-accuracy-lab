import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import (EXIT_ERROR, EXIT_FAIL, EXIT_OK, load_weeks, main as cli_main,
                 make_server, reset_state)
from generator import generate_weeks, values_of, weeks_to_jsonl
from report import forecast_payload


def _run(argv):
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        code = cli_main(argv)
    return code, out.getvalue()


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forecast-cli-")
        self.path = os.path.join(self.tmp, "series.jsonl")
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(weeks_to_jsonl(generate_weeks()))

    def test_generate_stdout_is_loadable(self):
        code, text = _run(["generate", "--weeks", "10"])
        self.assertEqual(code, EXIT_OK)
        rows = [json.loads(line) for line in text.strip().splitlines()]
        self.assertEqual(len(rows), 10)

    def test_load_weeks_jsonl_and_csv(self):
        rows = load_weeks(self.path)
        self.assertEqual(len(rows), 156)
        csv_path = os.path.join(self.tmp, "series.csv")
        code, _ = _run(["generate", "--out", csv_path, "--format", "csv"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(load_weeks(csv_path), rows)

    def test_forecast_command(self):
        code, text = _run(["forecast", "--input", self.path, "--horizon", "4"])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(text)
        self.assertEqual(payload["horizon"], 4)
        self.assertEqual(len(payload["models"]), 4)

    def test_forecast_single_model(self):
        code, text = _run(["forecast", "--input", self.path,
                           "--model", "ses"])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(text)
        self.assertEqual([m["model"] for m in payload["models"]], ["ses"])

    def test_backtest_and_report_commands(self):
        code, text = _run(["backtest", "--input", self.path])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(text)["best_model"], "holt")
        code, text = _run(["report", "--input", self.path])
        self.assertEqual(code, EXIT_OK)
        report = json.loads(text)
        self.assertIn("Best model by sMAPE: holt", report["narrative"])

    def test_check_exit_codes(self):
        code, _ = _run(["check", "--input", self.path])
        self.assertEqual(code, EXIT_OK)
        code, _ = _run(["check", "--input", self.path,
                        "--max-smape", "0.01"])
        self.assertEqual(code, EXIT_FAIL)
        code, _ = _run(["check", "--input", "/tmp/no-such-series-xyz.jsonl"])
        self.assertEqual(code, EXIT_ERROR)

    def test_check_data_quality_is_exit_2(self):
        bad_path = os.path.join(self.tmp, "bad.jsonl")
        rows = generate_weeks()
        rows[3]["revenue"] = -5.0
        with open(bad_path, "w", encoding="utf-8") as fh:
            fh.write(weeks_to_jsonl(rows))
        code, _ = _run(["check", "--input", bad_path])
        self.assertEqual(code, EXIT_ERROR)


class TestServe(unittest.TestCase):
    def setUp(self):
        self.rows = generate_weeks()
        reset_state(self.rows)
        self.server = make_server(0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        reset_state()

    def _req(self, path, payload=None, method="GET"):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health(self):
        status, body = self._req("/health")
        self.assertEqual((status, body), (200, {"status": "ok"}))

    def test_post_forecast_matches_local(self):
        status, body = self._req("/forecast", {"horizon": 8}, method="POST")
        self.assertEqual(status, 200)
        self.assertEqual(body, forecast_payload(values_of(self.rows),
                                                horizon=8))

    def test_post_forecast_with_values_and_model(self):
        values = values_of(self.rows)[:120]
        status, body = self._req(
            "/forecast", {"values": values, "horizon": 4, "model": "holt"},
            method="POST")
        self.assertEqual(status, 200)
        self.assertEqual(body, forecast_payload(values, horizon=4,
                                                model="holt"))

    def test_unknown_endpoint_404(self):
        status, _ = self._req("/nope")
        self.assertEqual(status, 404)
        status, _ = self._req("/forecast")  # GET not supported
        self.assertEqual(status, 404)

    def test_unknown_model_400(self):
        status, body = self._req(
            "/forecast", {"values": [1.0, 2.0], "model": "prophet"},
            method="POST")
        self.assertEqual(status, 400)
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
