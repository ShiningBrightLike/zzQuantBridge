from datetime import datetime, timedelta, timezone
from threading import Event
import time

from quant.domain import ExecutedTrade
from quant.webapp import JobConflictError, JobManager, find_run, parse_fill_fields, parse_fill_rows, preview_trades
from quant.portfolio import PortfolioStore
from quant.tasks import build_task


def _wait_for(runs_dir, run_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = find_run(run_id, runs_dir)
        if record is not None and record.status not in {"QUEUED", "RUNNING"}:
            return record
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


def _write_local_snapshot(path, symbol="600000", days=120):
    rows = ["symbol,timestamp,open,high,low,close,volume"]
    price = 10.0
    for index in range(days):
        price += 0.05
        stamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
        rows.append(
            f"{symbol},{stamp.date().isoformat()},{price:.2f},{price + 0.2:.2f},"
            f"{price - 0.2:.2f},{price:.2f},{1000 + index}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_download_task_finishes_with_pass_and_archives_snapshot(tmp_path):
    snapshot = _write_local_snapshot(tmp_path / "bars.csv")
    runs_dir = tmp_path / "runs"
    manager = JobManager(runs_dir)
    try:
        task = build_task(
            "download",
            {"provider": "local", "file": str(snapshot), "symbols": ["600000"], "start": "2026-01-01", "end": "2026-12-31"},
            data_dir=tmp_path / "data",
            configs_dir=tmp_path / "configs",
        )
        context = manager.submit("download", task)
        record = _wait_for(runs_dir, context.run_id)
    finally:
        manager.shutdown()

    assert record.status == "PASS"
    assert record.payload["rows"] > 0
    assert (tmp_path / "data" / "metadata").is_dir()


def test_generate_signals_task_writes_orders(tmp_path):
    snapshot = _write_local_snapshot(tmp_path / "bars.csv")
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
        first = manager.submit("download", download)
        assert _wait_for(runs_dir, first.run_id).status == "PASS"
        signals = build_task(
            "generate-signals",
            {"initial_cash": 100000},
            data_dir=data_dir,
            configs_dir=tmp_path / "configs",
            db_path=tmp_path / "portfolio.sqlite",
        )
        context = manager.submit("generate-signals", signals)
        record = _wait_for(runs_dir, context.run_id)
    finally:
        manager.shutdown()

    assert record.status in {"PASS", "BLOCKED"}
    assert (record.run_dir / "orders.csv").is_file()
    assert (record.run_dir / "signals.csv").is_file()


def test_fill_preview_does_not_write_and_duplicate_is_reported(tmp_path):
    db = tmp_path / "portfolio.sqlite"
    store = PortfolioStore(db)
    existing = ExecutedTrade("fill-1", "600000", "BUY", 100, 10, 1, datetime.now(timezone.utc))
    store.append_trade(existing)

    rows = parse_fill_rows(
        "trade_id,symbol,side,quantity,price,fee,executed_at\n"
        "fill-1,600000,BUY,100,10,1,2026-10-10T09:30:00+08:00\n"
        "fill-2,600000,SELL,100,11,1,2026-10-10T10:30:00+08:00\n"
    )
    preview = preview_trades(rows, db)

    assert preview["count"] == 2
    assert preview["duplicates"] == ["fill-1"]
    assert len(store.trades()) == 1


def test_job_manager_allows_only_one_active_job(tmp_path):
    manager = JobManager(tmp_path / "runs")
    gate = Event()
    try:
        started = manager.submit("backtest", lambda context: gate.wait(timeout=2))
        assert started.run_id
        try:
            manager.submit("download", lambda context: None)
        except JobConflictError:
            pass
        else:
            raise AssertionError("second active job should be rejected")
    finally:
        gate.set()
        manager.shutdown()


def test_single_fill_form_uses_same_domain_parser():
    trades = parse_fill_fields(
        {
            "trade_id": "fill-1",
            "symbol": "600000",
            "side": "BUY",
            "quantity": "100",
            "price": "10",
            "fee": "1",
            "executed_at": "2026-10-10T09:30:00+08:00",
        }
    )

    assert trades[0].quantity == 100
