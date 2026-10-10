from datetime import datetime, timedelta, timezone
import time

import pytest

pd = pytest.importorskip("pandas")

from quant.domain import ExecutedTrade
from quant.portfolio import PortfolioStore
from quant.tasks import build_task
from quant.webapp import JobManager, find_run


def _fixture(path, symbol="600000", days=90):
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


def test_held_symbol_without_price_blocks_instead_of_erroring(tmp_path):
    snapshot = _fixture(tmp_path / "bars.csv")
    runs_dir = tmp_path / "runs"
    data_dir = tmp_path / "data"
    db_path = tmp_path / "portfolio.sqlite"
    store = PortfolioStore(db_path)
    store.set_initial_cash(100_000)
    # 999999 is held but absent from the snapshot, so the portfolio cannot be valued.
    store.append_trade(
        ExecutedTrade("t-1", "999999", "BUY", 100, 10.0, 1.0, datetime(2026, 1, 5, tzinfo=timezone.utc))
    )
    manager = JobManager(runs_dir)
    try:
        download = build_task(
            "download",
            {"provider": "local", "file": str(snapshot), "symbols": ["600000"], "start": "2026-01-01", "end": "2026-12-31"},
            data_dir=data_dir,
            configs_dir=tmp_path / "configs",
        )
        assert _wait(runs_dir, manager.submit("download", download).run_id).status == "PASS"
        signals = build_task(
            "generate-signals", {}, data_dir=data_dir, configs_dir=tmp_path / "configs", db_path=db_path
        )
        record = _wait(runs_dir, manager.submit("generate-signals", signals).run_id)
    finally:
        manager.shutdown()

    assert record.status == "BLOCKED"
    assert any("cannot be valued" in reason for reason in record.payload["risk_reasons"])
    assert (record.run_dir / "orders.csv").is_file()
    assert (record.run_dir / "risk_report.json").is_file()
    assert (record.run_dir / "report.html").is_file()


def test_stale_data_blocks_suggestions(tmp_path):
    snapshot = _fixture(tmp_path / "bars.csv", days=60)
    runs_dir = tmp_path / "runs"
    data_dir = tmp_path / "data"
    db_path = tmp_path / "portfolio.sqlite"
    PortfolioStore(db_path).set_initial_cash(100_000)
    manager = JobManager(runs_dir)
    try:
        download = build_task(
            "download",
            {"provider": "local", "file": str(snapshot), "symbols": ["600000"], "start": "2026-01-01", "end": "2026-12-31"},
            data_dir=data_dir,
            configs_dir=tmp_path / "configs",
        )
        assert _wait(runs_dir, manager.submit("download", download).run_id).status == "PASS"
        signals = build_task(
            "generate-signals", {}, data_dir=data_dir, configs_dir=tmp_path / "configs", db_path=db_path
        )
        record = _wait(runs_dir, manager.submit("generate-signals", signals).run_id)
    finally:
        manager.shutdown()

    assert record.status == "BLOCKED"
    assert any("day(s) old" in reason for reason in record.payload["risk_reasons"])
    rows = (record.run_dir / "orders.csv").read_text(encoding="utf-8-sig").splitlines()
    assert rows[1].startswith("600000")
    assert "BLOCKED" in rows[1]
