"""No-network enforcement: with sockets blocked, generating,
forecasting, backtesting, reporting and validating must all still
work. (``serve`` is the only intentional exception — it binds loopback
on demand, and is not used here.)"""

import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Blocker:
    def __init__(self, *a, **k):
        raise RuntimeError("network access blocked in hermetic test")


class TestNoNetwork(unittest.TestCase):
    def setUp(self):
        self._real_socket = socket.socket
        socket.socket = _Blocker
        import urllib.request as urlreq
        self._real_urlopen = urlreq.urlopen
        urlreq.urlopen = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("network access blocked in hermetic test"))

    def tearDown(self):
        socket.socket = self._real_socket
        import urllib.request as urlreq
        urlreq.urlopen = self._real_urlopen

    def test_full_pipeline_offline(self):
        from backtest import rolling_origin_backtest
        from generator import generate_weeks, values_of
        from report import build_report
        from validation import validate_series
        rows = generate_weeks()
        self.assertEqual(validate_series(rows), [])
        bt = rolling_origin_backtest(values_of(rows), horizon=8)
        self.assertEqual(bt["best_model"], "holt")
        report = build_report(rows, horizon=8)
        self.assertEqual(report["best_model"], "holt")

    def test_no_module_opens_sockets_at_import(self):
        for mod in ("generator", "models", "accuracy", "backtest",
                    "report", "validation", "cli", "demo"):
            sys.modules.pop(mod, None)
        import accuracy, backtest, cli, demo, generator, models  # noqa: F401
        import report, validation  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
