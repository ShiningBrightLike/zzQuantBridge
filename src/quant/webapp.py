"""Application services shared by the local Web UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import io
import json
from pathlib import Path
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Iterable

from .domain import ExecutedTrade
from .portfolio import PortfolioStore
from .runs import RunContext, configure_run_logging, create_run, write_status


RUN_KINDS = {"download", "backtest", "generate-signals"}


class JobConflictError(RuntimeError):
    """Raised when a second long-running job is submitted."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    command: str
    run_dir: Path
    status: str
    payload: dict[str, Any]


class JobManager:
    """Run one auditable background job at a time in the local process."""

    def __init__(self, runs_dir: str | Path = "runs") -> None:
        self.runs_dir = Path(runs_dir)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quant-job")
        self._lock = threading.Lock()
        self._active: Future[None] | None = None
        self._active_run_id: str | None = None

    @property
    def active_run_id(self) -> str | None:
        with self._lock:
            if self._active is None or self._active.done():
                return None
            return self._active_run_id

    def submit(
        self,
        kind: str,
        task: Callable[[RunContext], dict[str, Any] | None],
        *,
        config_paths: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> RunContext:
        if kind not in RUN_KINDS:
            raise ValueError(f"unsupported run kind: {kind}")
        with self._lock:
            if self._active is not None and not self._active.done():
                raise JobConflictError(f"a job is already running: {self._active_run_id}")
            context = create_run(self.runs_dir, command=kind, config_paths=config_paths, params=params)
            self._write_status(context, "QUEUED", {"command": kind})
            future = self._executor.submit(self._run, context, kind, task)
            self._active = future
            self._active_run_id = context.run_id
            return context

    def _run(self, context: RunContext, kind: str, task: Callable[[RunContext], dict[str, Any] | None]) -> None:
        logger = configure_run_logging(context)
        self._write_status(context, "RUNNING", {"command": kind})
        try:
            payload = dict(task(context) or {})
            status = str(payload.pop("status", "PASS"))
            self._write_status(context, status, payload)
            logger.info("job completed status=%s", status)
        except Exception as exc:  # pragma: no cover - exercised by integration failures
            self._write_status(context, "ERROR", {"error": str(exc), "error_type": type(exc).__name__})
            logger.exception("job failed")

    @staticmethod
    def _write_status(context: RunContext, status: str, payload: dict[str, Any]) -> None:
        write_status(context.run_dir, status, payload)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def list_runs(runs_dir: str | Path = "runs", *, limit: int = 20) -> list[RunRecord]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    records: list[RunRecord] = []
    for manifest_path in root.glob("*/*/input_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status_path = manifest_path.with_name("status.json")
            payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            records.append(
                RunRecord(
                    run_id=str(manifest["run_id"]),
                    command=str(manifest["command"]),
                    run_dir=manifest_path.parent,
                    status=str(payload.get("status", "UNKNOWN")),
                    payload=payload,
                )
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    def modified(record: RunRecord) -> float:
        try:
            return record.run_dir.stat().st_mtime
        except OSError:
            # The directory can disappear while a cleanup runs; keep listing.
            return 0.0

    records.sort(key=modified, reverse=True)
    return records[:limit]


def find_run(run_id: str, runs_dir: str | Path = "runs") -> RunRecord | None:
    return next((item for item in list_runs(runs_dir, limit=1000) if item.run_id == run_id), None)


def parse_fill_rows(content: str) -> list[ExecutedTrade]:
    """Parse uploaded fill CSV into domain objects without writing anything."""

    trades: list[ExecutedTrade] = []
    for row in csv.DictReader(io.StringIO(content)):
        try:
            executed_at = datetime.fromisoformat(row["executed_at"].replace("Z", "+00:00"))
            trades.append(
                ExecutedTrade(
                    trade_id=row["trade_id"],
                    symbol=row["symbol"],
                    side=row["side"].upper(),
                    quantity=int(row["quantity"]),
                    price=float(row["price"]),
                    fee=float(row.get("fee") or 0),
                    executed_at=executed_at,
                    broker_reference=row.get("broker_reference") or None,
                    note=row.get("note") or None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid fill row: {row}") from exc
    return trades


def parse_fill_fields(fields: dict[str, str | None]) -> list[ExecutedTrade]:
    """Parse one form fill using the same domain validation as CSV imports."""

    try:
        executed_at = datetime.fromisoformat((fields["executed_at"] or "").replace("Z", "+00:00"))
        return [
            ExecutedTrade(
                trade_id=fields["trade_id"] or "",
                symbol=fields["symbol"] or "",
                side=(fields["side"] or "").upper(),
                quantity=int(fields["quantity"] or "0"),
                price=float(fields["price"] or "0"),
                fee=float(fields.get("fee") or 0),
                executed_at=executed_at,
                broker_reference=fields.get("broker_reference") or None,
                note=fields.get("note") or None,
            )
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid single fill form") from exc


def preview_trades(trades: Iterable[ExecutedTrade], db_path: str | Path) -> dict[str, Any]:
    store = PortfolioStore(db_path)
    rows = list(trades)
    existing = {trade.trade_id for trade in store.trades()}
    seen: set[str] = set()
    duplicates = []
    for trade in rows:
        if trade.trade_id in existing or trade.trade_id in seen:
            duplicates.append(trade.trade_id)
        seen.add(trade.trade_id)
    return {
        "count": len(rows),
        "trade_ids": [trade.trade_id for trade in rows],
        "duplicates": duplicates,
        "will_insert": len(rows) - len(duplicates),
        "initial_cash": store.get_initial_cash(),
    }


def commit_trades(trades: Iterable[ExecutedTrade], db_path: str | Path) -> dict[str, Any]:
    store = PortfolioStore(db_path)
    inserted = store.append_trades(list(trades))
    snapshot = store.snapshot(store.get_initial_cash() or 0.0)
    return {
        "inserted": inserted,
        "cash": snapshot.cash,
        "positions": snapshot.positions,
        "realized_pnl": snapshot.realized_pnl,
        "fees": snapshot.fees,
    }


def empty_plot(title: str) -> dict[str, Any]:
    return {
        "data": [],
        "layout": {
            "title": {"text": title},
            "template": "plotly_dark",
            "annotations": [{"text": "暂无数据", "showarrow": False}],
        },
    }
