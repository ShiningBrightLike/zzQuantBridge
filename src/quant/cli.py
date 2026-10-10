"""Unified command line entry point for the quant workflow."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime
import json
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .config import ConfigError, load_yaml_config
from .data.archive import ArchiveError
from .data.providers import DataFetchError, LocalDataProvider
from .data.validation import DataValidationError
from .domain import ExecutedTrade
from .maintenance import MaintenanceError, clear_history, format_bytes, history_summary
from .portfolio import PortfolioStore
from .reports import write_run_report
from .runs import RunContext, configure_run_logging, create_run, write_status
from .tasks import build_task, resolve_initial_cash
from .webapp import find_run, list_runs


COMMANDS = (
    "download",
    "validate-data",
    "backtest",
    "generate-signals",
    "reconcile",
    "report",
    "clear-history",
)
TASK_COMMANDS = ("download", "backtest", "generate-signals")
STRATEGY_KEYS = (
    "fast_window",
    "slow_window",
    "max_positions",
    "max_single_weight",
    "model_version",
    "init_cash",
    "initial_cash",
    "signal_delay",
    "fees",
    "slippage",
    "max_gross_weight",
    "cash_buffer",
    "max_turnover",
    "max_data_age_days",
    "lot_size",
    "min_order_value",
    "industries",
    "industry_limits",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant", description="Manual-execution quant workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--date", dest="run_date", help="Run date in YYYY-MM-DD format")
        subparser.add_argument("--runs-dir", default="runs", help="Root directory for run artifacts")
        subparser.add_argument("--data-dir", default="data", help="Market data archive directory")
        subparser.add_argument("--configs-dir", default="configs", help="Directory holding YAML configs")
        subparser.add_argument("--db-path", default="db/portfolio.sqlite", help="Portfolio SQLite file")
        subparser.add_argument(
            "--config",
            required=command == "backtest",
            action="append",
            default=[],
            help="YAML config path (repeatable)",
        )
        if command == "download":
            subparser.add_argument("--provider", choices=("akshare", "local"), default="akshare")
            subparser.add_argument("--file", help="Local CSV or Parquet input when provider=local")
            subparser.add_argument("--symbols", nargs="+", default=[], help="Symbols to download")
            subparser.add_argument("--start", help="First date to download")
            subparser.add_argument("--end", help="Last date to download")
            subparser.add_argument("--adjustment", choices=("", "qfq", "hfq"), default="")
        if command == "validate-data":
            subparser.add_argument("--file", required=True, help="CSV or Parquet OHLCV snapshot")
            subparser.add_argument("--symbols", nargs="+", default=[], help="Symbols to validate")
            subparser.add_argument("--start", default="1900-01-01", help="First date to include")
            subparser.add_argument("--end", default="2100-01-01", help="Last date to include")
        if command == "generate-signals":
            subparser.add_argument("--initial-cash", type=float, default=None)
        if command == "reconcile":
            subparser.add_argument("--file", required=True, help="CSV file containing executed fills")
            subparser.add_argument("--initial-cash", type=float, default=None)
            subparser.add_argument("--db", default=None, help="Override the portfolio database path")
        if command == "report":
            subparser.add_argument("--run-id", help="Run to render; defaults to the latest run")
        if command == "clear-history":
            subparser.add_argument("--runs", action="store_true", help="Remove run directories")
            subparser.add_argument("--data", action="store_true", help="Remove market data snapshots")
            subparser.add_argument("--db", action="store_true", help="Remove the portfolio SQLite database")
            subparser.add_argument("--keep-days", type=int, default=None, help="Keep history newer than N days")
            subparser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    return parser


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"Invalid --date, expected YYYY-MM-DD: {value}") from exc


def _selection_flags(selection: dict[str, bool]) -> list[str]:
    flags = []
    if selection["runs"]:
        flags.append("--runs")
    if selection["data"]:
        flags.append("--data")
    if selection["database"]:
        flags.append("--db")
    return flags


def _db_path(args: argparse.Namespace) -> str:
    override = getattr(args, "db", None)
    return override or args.db_path


def _task_params(args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI flags with any strategy/risk config files into one parameter set."""

    params: dict[str, Any] = {}
    for key in ("symbols", "start", "end", "provider", "file", "adjustment", "initial_cash"):
        value = getattr(args, key, None)
        if value not in (None, "", []):
            params[key] = value
    for config_path in getattr(args, "config", []) or []:
        payload = load_yaml_config(config_path)
        for key in STRATEGY_KEYS:
            if key in payload:
                params[key] = payload[key]
    return params


def _run_task_command(
    args: argparse.Namespace, context: RunContext, logger: Any
) -> int:
    params = _task_params(args)
    task = build_task(
        args.command,
        params,
        data_dir=args.data_dir,
        configs_dir=args.configs_dir,
        db_path=_db_path(args),
    )
    payload = dict(task(context) or {})
    status = str(payload.pop("status", "PASS"))
    write_status(context.run_dir, status, payload)
    logger.info("%s finished with status=%s", args.command, status)
    print(json.dumps({"run_id": context.run_id, "status": status, **payload}, indent=2, ensure_ascii=False))
    return 0 if status != "ERROR" else 2


def _run_validate_data(args: argparse.Namespace, context: RunContext, logger: Any) -> int:
    from .data.quality import validate_market_data

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    frame = LocalDataProvider(args.file).fetch(args.symbols, start, end)
    # The expected calendar is the set of trading days observed in the snapshot;
    # a symbol missing one of those days is a gap, not a holiday.
    expected = sorted({item.date() for item in frame["timestamp"]}) if len(frame) else None
    quality = validate_market_data(frame, symbols=args.symbols or None, expected_dates=expected)
    payload = {
        "row_count": quality.row_count,
        "errors": list(quality.errors),
        "warnings": list(quality.warnings),
        "expected_days": len(expected) if expected else 0,
    }
    write_status(context.run_dir, quality.status, payload)
    (context.run_dir / "data_quality.json").write_text(
        json.dumps({"status": quality.status, **payload}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if quality.blocked:
        logger.error("data quality blocked: %s", "; ".join(quality.errors))
        print(context.run_dir)
        return 2
    print(context.run_dir)
    return 0


def _run_reconcile(args: argparse.Namespace, context: RunContext, logger: Any) -> int:
    store = PortfolioStore(_db_path(args))
    imported = 0
    with open(args.file, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                executed_at = datetime.fromisoformat(row["executed_at"].replace("Z", "+00:00"))
                trade = ExecutedTrade(
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
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError(f"invalid fill row: {row}") from exc
            imported += int(store.append_trade(trade))
    initial_cash = resolve_initial_cash(
        store,
        {} if args.initial_cash is None else {"initial_cash": args.initial_cash},
        Path(args.configs_dir),
    )
    snapshot = store.snapshot(initial_cash)
    payload = {
        "cash": snapshot.cash,
        "positions": snapshot.positions,
        "average_cost": snapshot.average_cost,
        "realized_pnl": snapshot.realized_pnl,
        "fees": snapshot.fees,
        "imported_trades": imported,
        "initial_cash": initial_cash,
    }
    write_status(context.run_dir, "PASS", payload)
    (context.run_dir / "portfolio_snapshot.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_run_report(context.run_dir)
    logger.info("reconciled %s new fill(s)", imported)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_report(args: argparse.Namespace, context: RunContext, logger: Any) -> int:
    target = find_run(args.run_id, args.runs_dir) if args.run_id else None
    if target is None:
        candidates = [item for item in list_runs(args.runs_dir, limit=50) if item.run_id != context.run_id]
        target = candidates[0] if candidates else None
    if target is None:
        raise ConfigError("no run available to report on; run download/backtest first")
    path = write_run_report(target.run_dir)
    write_status(context.run_dir, "PASS", {"report": str(path), "source_run": target.run_id})
    logger.info("report written for %s", target.run_id)
    print(path)
    return 0


def _run_clear_history(args: argparse.Namespace) -> int:
    selection = {
        "runs": args.runs or not (args.runs or args.data or args.db),
        "data": args.data or not (args.runs or args.data or args.db),
        "database": args.db,
    }
    summary = history_summary(runs_dir=args.runs_dir, data_dir=args.data_dir, db_path=_db_path(args))
    total_bytes = sum(
        section.get("bytes", 0)
        for key, section in summary.items()
        if key != "paths" and isinstance(section, dict)
    )
    print(
        f"history: runs={summary['runs']['count']}"
        f" snapshots={summary['snapshots']['count']}"
        f" metadata={summary['metadata']['count']}"
        f" db_files={summary['database']['count']}"
        f" size={format_bytes(total_bytes)}"
    )
    report = clear_history(
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
        db_path=_db_path(args),
        runs=selection["runs"],
        data=selection["data"],
        database=selection["database"],
        keep_days=args.keep_days,
        confirm=args.yes,
        dry_run=not args.yes,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not args.yes:
        flags = " ".join(_selection_flags(selection)) or "--runs --data"
        print(
            f"\npreview only: {report['total']} item(s), {format_bytes(report['bytes_freed'])} would be removed."
            f" Re-run with --yes to delete: quant clear-history {flags} --yes"
        )
        return 0
    print(f"\nremoved {report['total']} item(s), freed {format_bytes(report['bytes_freed'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context: RunContext | None = None
    try:
        run_date = _parse_date(args.run_date)
        context = create_run(args.runs_dir, run_date, command=args.command)
        logger = configure_run_logging(context)
        logger.info("command=%s started", args.command)

        if args.command == "clear-history":
            return _run_clear_history(args)
        if args.command in TASK_COMMANDS:
            return _run_task_command(args, context, logger)
        if args.command == "validate-data":
            return _run_validate_data(args, context, logger)
        if args.command == "reconcile":
            return _run_reconcile(args, context, logger)
        if args.command == "report":
            return _run_report(args, context, logger)
        raise ConfigError(f"unhandled command: {args.command}")
    except (
        ArchiveError,
        ConfigError,
        DataFetchError,
        DataValidationError,
        MaintenanceError,
        ValueError,
    ) as exc:
        if context is not None:
            write_status(context.run_dir, "ERROR", {"error": str(exc), "error_type": type(exc).__name__})
            print(context.run_dir)
        print(f"quant: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
