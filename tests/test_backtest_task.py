from datetime import datetime, timedelta, timezone
import time

import pytest

pytest.importorskip("vectorbt")

from quant.tasks import build_task
from quant.webapp import JobManager, find_run


def _fixture(path, symbol, days=120):
    rows = ["symbol,timestamp,open,high,low,close,volume"]
    price = 10.0
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(days):
        price += 0.05
        day = (start + timedelta(days=index)).date().isoformat()
        rows.append(f"{symbol},{day},{price:.2f},{price + 0.2:.2f},{price - 0.2:.2f},{price:.2f},{1000 + index}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _wait(runs_dir, run_id, timeout=120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = find_run(run_id, runs_dir)
        if record is not None and record.status not in {"QUEUED", "RUNNING"}:
            return record
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_backtest_task_produces_metrics_and_equity_curve(tmp_path):
    snapshot = _fixture(tmp_path / "bars.csv", "600000")
    runs_dir = tmp_path / "runs"
    data_dir = tmp_path / "data"
    manager = JobManager(runs_dir)
    try:
        download = build_task(
            "download",
            {"provider": "local", "file": str(snapshot), "symbols": ["600000"], "start": "2026-01-01", "end": "2026-12-31"},
            data_dir=data_dir,
            configs_dir=tmp_path / "configs",
        )
        assert _wait(runs_dir, manager.submit("download", download).run_id).status == "PASS"
        backtest = build_task(
            "backtest",
            {"fast_window": 10, "slow_window": 30},
            data_dir=data_dir,
            configs_dir=tmp_path / "configs",
        )
        record = _wait(runs_dir, manager.submit("backtest", backtest).run_id)
    finally:
        manager.shutdown()

    assert record.status == "PASS", record.payload
    assert (record.run_dir / "backtest_metrics.json").is_file()
    assert (record.run_dir / "equity_curve.csv").is_file()
