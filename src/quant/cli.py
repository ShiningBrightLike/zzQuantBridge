"""Unified command line entry point for the quant workflow."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime
import json
import sys
from zoneinfo import ZoneInfo

from .config import ConfigError, load_yaml_config
from .data.archive import ArchiveError
from .data.providers import DataFetchError
from .data.validation import DataValidationError
from .runs import configure_run_logging, create_run


COMMANDS = (
    "download",
    "validate-data",
    "backtest",
    "generate-signals",
    "reconcile",
    "report",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant", description="Manual-execution quant workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--date", dest="run_date", help="Run date in YYYY-MM-DD format")
        subparser.add_argument("--runs-dir", default="runs", help="Root directory for run artifacts")
        subparser.add_argument(
            "--config",
            required=command == "backtest",
            action="append",
            default=[],
            help="YAML config path (repeatable)",
        )
        if command == "reconcile":
            subparser.add_argument("--file", required=True, help="CSV file containing executed fills")
            subparser.add_argument("--db", default="db/portfolio.sqlite", help="Portfolio SQLite database")
            subparser.add_argument("--initial-cash", type=float, default=0.0)
        if command == "validate-data":
            subparser.add_argument("--file", required=True, help="CSV or Parquet OHLCV snapshot")
            subparser.add_argument("--symbols", nargs="+", default=[], help="Symbols to validate")
            subparser.add_argument("--start", default="1900-01-01", help="First date to include")
            subparser.add_argument("--end", default="2100-01-01", help="Last date to include")
        if command == "download":
            subparser.add_argument("--provider", choices=("akshare", "local"), default="akshare")
            subparser.add_argument("--file", help="Local CSV or Parquet input when provider=local")
            subparser.add_argument("--symbols", nargs="+", default=[], help="Symbols to download")
            subparser.add_argument("--start", help="First date to download")
            subparser.add_argument("--end", help="Last date to download")
            subparser.add_argument("--data-dir", default="data", help="Market data archive directory")
            subparser.add_argument("--adjustment", choices=("", "qfq", "hfq"), default="")
    return parser


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"Invalid --date, expected YYYY-MM-DD: {value}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = None
    try:
        run_date = _parse_date(args.run_date)
        loaded_configs = {}
        for config_path in args.config:
            loaded_configs[config_path] = load_yaml_config(config_path)
        context = create_run(
            args.runs_dir,
            run_date,
            command=args.command,
            config_paths=args.config,
        )
        logger = configure_run_logging(context)
        logger.info("initialized command=%s", args.command)
        if args.command == "download":
            from .data.archive import archive_snapshot
            from .data.providers import AkShareDataProvider, LocalDataProvider

            universe = next(
                (value for value in loaded_configs.values() if isinstance(value.get("symbols"), list)),
                {},
            )
            symbols = args.symbols or universe.get("symbols", [])
            if not symbols:
                raise ConfigError("download requires --symbols or a config containing symbols")
            end = _parse_date(args.end) or run_date or datetime.now(ZoneInfo("Asia/Shanghai")).date()
            start = _parse_date(args.start) or end
            if start > end:
                raise ConfigError("download start date must not be after end date")
            if args.provider == "local":
                if not args.file:
                    raise ConfigError("download --provider local requires --file")
                provider = LocalDataProvider(args.file)
            else:
                provider = AkShareDataProvider(adjustment=args.adjustment)
            frame = provider.fetch(symbols, start, end)
            manifest_path = archive_snapshot(
                frame,
                args.data_dir,
                source=args.provider,
                request={"symbols": symbols, "start": start.isoformat(), "end": end.isoformat()},
                adjustment=args.adjustment,
            )
            (context.run_dir / "data_manifest.json").write_text(
                json.dumps({"status": "PASS", "manifest": str(manifest_path)}, indent=2) + "\n",
                encoding="utf-8",
            )
            (context.run_dir / "status.json").write_text(
                json.dumps({"status": "PASS", "command": args.command}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(context.run_dir)
            return 0
        if args.command == "validate-data":
            from .data.providers import LocalDataProvider
            from .data.quality import validate_market_data

            start = _parse_date(args.start)
            end = _parse_date(args.end)
            frame = LocalDataProvider(args.file).fetch(args.symbols, start, end)
            quality = validate_market_data(frame, symbols=args.symbols or None)
            (context.run_dir / "data_quality.json").write_text(
                json.dumps(
                    {
                        "status": quality.status,
                        "row_count": quality.row_count,
                        "errors": list(quality.errors),
                        "warnings": list(quality.warnings),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (context.run_dir / "status.json").write_text(
                json.dumps({"status": quality.status, "command": args.command}, indent=2) + "\n",
                encoding="utf-8",
            )
            if quality.blocked:
                logger.error("data quality blocked: %s", "; ".join(quality.errors))
                print(context.run_dir)
                return 2
            print(context.run_dir)
            return 0
        if args.command == "reconcile":
            from .domain import ExecutedTrade
            from .portfolio import PortfolioStore

            store = PortfolioStore(args.db)
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
            snapshot = store.snapshot(args.initial_cash)
            (context.run_dir / "portfolio_snapshot.json").write_text(
                json.dumps(
                    {
                        "cash": snapshot.cash,
                        "positions": snapshot.positions,
                        "average_cost": snapshot.average_cost,
                        "realized_pnl": snapshot.realized_pnl,
                        "fees": snapshot.fees,
                        "imported_trades": imported,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (context.run_dir / "status.json").write_text(
                json.dumps({"status": "PASS", "command": args.command}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(context.run_dir)
            return 0
        (context.run_dir / "status.json").write_text(
            json.dumps({"status": "INITIALIZED", "command": args.command}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(context.run_dir)
        return 0
    except (ArchiveError, ConfigError, DataFetchError, DataValidationError) as exc:
        if context is not None:
            (context.run_dir / "status.json").write_text(
                json.dumps({"status": "ERROR", "error": str(exc)}, indent=2) + "\n", encoding="utf-8"
            )
            print(context.run_dir)
        print(f"quant: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
