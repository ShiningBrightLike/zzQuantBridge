"""Real domain work dispatched by the local Web UI job runner."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from .config import load_yaml_config
from .data.archive import archive_snapshot
from .data.providers import AkShareDataProvider, LocalDataProvider
from .data.validation import normalize_ohlcv
from .portfolio import PortfolioStore
from .runs import RunContext
from .signals import generate_suggestions
from .strategies import MovingAverageStrategy, MovingAverageStrategyConfig


Task = Callable[[RunContext], dict[str, Any]]


def _today() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def _as_date(value: Any, default: date) -> date:
    if value in (None, ""):
        return default
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _config_symbols(configs_dir: Path) -> list[str]:
    path = configs_dir / "universe.yaml"
    if not path.is_file():
        return []
    payload = load_yaml_config(path)
    return [str(item).strip() for item in payload.get("symbols", []) if str(item).strip()]


def _strategy_settings(configs_dir: Path, params: dict[str, Any]) -> MovingAverageStrategyConfig:
    payload: dict[str, Any] = {}
    path = configs_dir / "strategy.yaml"
    if path.is_file():
        payload = load_yaml_config(path)
    for key in ("fast_window", "slow_window", "max_positions", "max_single_weight", "model_version"):
        if params.get(key) not in (None, ""):
            payload[key] = params[key]
    allowed = {"fast_window", "slow_window", "max_positions", "max_single_weight", "model_version"}
    return MovingAverageStrategyConfig(**{key: value for key, value in payload.items() if key in allowed})


def _risk_settings(configs_dir: Path, params: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, Any] = {}
    path = configs_dir / "risk.yaml"
    if path.is_file():
        payload = load_yaml_config(path)
    lot_size = int(params.get("lot_size") or payload.get("lot_size") or 100)
    min_order_value = float(params.get("min_order_value") or payload.get("min_order_value") or 0.0)
    commission_rate = float(params.get("commission_rate") or payload.get("commission_rate") or 0.0)
    slippage_bps = float(params.get("slippage_bps") or payload.get("slippage_bps") or 0.0)
    return {
        "lot_size": lot_size,
        "min_order_value": min_order_value,
        "commission_rate": commission_rate,
        "slippage": slippage_bps / 10_000.0,
    }


def _mean_scalar(value: Any) -> float:
    """Collapse vectorbt's per-column results into one portfolio-level number."""

    if hasattr(value, "mean"):
        return float(value.mean())
    return float(value)


def latest_snapshot(data_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Return the newest archived raw snapshot path and its manifest."""

    metadata_dir = data_dir / "metadata"
    manifests = (
        sorted(metadata_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        if metadata_dir.is_dir()
        else []
    )
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for entry in payload.get("files", []):
            if entry.get("kind") == "raw":
                frame_path = data_dir / str(entry.get("path", ""))
                if frame_path.is_file():
                    return frame_path, payload
    raise FileNotFoundError("no archived market snapshot found; run download first")


def load_snapshot(data_dir: Path) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    frame_path, manifest = latest_snapshot(data_dir)
    frame = normalize_ohlcv(pd.read_parquet(frame_path))
    return frame, manifest


def _download_task(params: dict[str, Any], *, data_dir: Path, configs_dir: Path) -> Task:
    def run(context: RunContext) -> dict[str, Any]:
        symbols = [str(item).strip() for item in params.get("symbols") or []] or _config_symbols(configs_dir)
        if not symbols:
            raise ValueError("no symbols configured; set symbols in configs/universe.yaml or run parameters")
        end = _as_date(params.get("end"), _today())
        start = _as_date(params.get("start"), end - timedelta(days=365))
        if start > end:
            raise ValueError("start date must not be after end date")
        provider_name = str(params.get("provider") or "akshare").strip().lower()
        adjustment = str(params.get("adjustment") or "")
        if provider_name == "local":
            file_path = params.get("file")
            if not file_path:
                raise ValueError("local provider requires a file path")
            provider: Any = LocalDataProvider(file_path)
        elif provider_name == "akshare":
            provider = AkShareDataProvider(adjustment=adjustment)
        else:
            raise ValueError(f"unsupported provider: {provider_name}")
        frame = provider.fetch(symbols, start, end)
        manifest_path = archive_snapshot(
            frame,
            data_dir,
            source=provider_name,
            request={"symbols": symbols, "start": start.isoformat(), "end": end.isoformat()},
            adjustment=adjustment,
        )
        summary = {
            "rows": int(len(frame)),
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "provider": provider_name,
            "manifest": str(manifest_path),
        }
        endpoints = getattr(provider, "last_sources", None)
        if endpoints:
            summary["endpoints"] = dict(endpoints)
        (context.run_dir / "data_manifest.json").write_text(
            json.dumps({"status": "PASS", **summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary

    return run


def _backtest_task(params: dict[str, Any], *, data_dir: Path, configs_dir: Path) -> Task:
    def run(context: RunContext) -> dict[str, Any]:
        from .backtest import BacktestConfig, run_signal_backtest

        frame, manifest = load_snapshot(data_dir)
        settings = _strategy_settings(configs_dir, params)
        risk = _risk_settings(configs_dir, params)
        close = frame.pivot_table(index="timestamp", columns="symbol", values="close").sort_index()
        fast = close.rolling(settings.fast_window, min_periods=settings.fast_window).mean()
        slow = close.rolling(settings.slow_window, min_periods=settings.slow_window).mean()
        ready = fast.notna() & slow.notna()
        entries = (fast > slow) & ready
        exits = (fast < slow) & ready
        entries = entries & ~entries.shift(1).fillna(False)
        exits = exits & ~exits.shift(1).fillna(False)
        config = BacktestConfig(
            init_cash=float(params.get("init_cash") or 100_000.0),
            fees=risk["commission_rate"],
            slippage=risk["slippage"],
            signal_delay=int(params.get("signal_delay") or 1),
        )
        result = run_signal_backtest(close, entries, exits, config)
        metrics = {str(key): value for key, value in result.metrics.items()}
        for key, value in list(metrics.items()):
            try:
                json.dumps(value)
            except TypeError:
                metrics[key] = str(value)
        (context.run_dir / "backtest_metrics.json").write_text(
            json.dumps({"status": "PASS", "metrics": metrics}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        equity = result.portfolio.value()
        if hasattr(equity, "to_frame"):
            equity = equity.to_frame("value")
        equity.to_csv(context.run_dir / "equity_curve.csv")
        trades = result.portfolio.trades.records_readable
        if trades is not None and len(trades):
            trades.to_csv(context.run_dir / "trades.csv", index=False)
        return {
            "snapshot": manifest.get("snapshot_id"),
            "symbols": [str(item) for item in close.columns],
            "rows": int(len(close)),
            "signal_delay": config.signal_delay,
            "fees": config.fees,
            "slippage": config.slippage,
            "total_return": _mean_scalar(result.portfolio.total_return()),
            "max_drawdown": _mean_scalar(result.portfolio.max_drawdown()),
            "per_symbol_return": {
                str(key): float(item) for key, item in result.portfolio.total_return().items()
            }
            if hasattr(result.portfolio.total_return(), "items")
            else {},
            "trade_count": int(len(trades)) if trades is not None else 0,
            "artifacts": ["backtest_metrics.json", "equity_curve.csv", "trades.csv"],
        }

    return run


def _signals_task(params: dict[str, Any], *, data_dir: Path, configs_dir: Path, db_path: Path) -> Task:
    def run(context: RunContext) -> dict[str, Any]:
        frame, manifest = load_snapshot(data_dir)
        settings = _strategy_settings(configs_dir, params)
        risk = _risk_settings(configs_dir, params)
        strategy = MovingAverageStrategy(settings)
        targets = strategy.generate_targets(frame)
        prices = {str(symbol): float(value) for symbol, value in frame.groupby("symbol")["close"].last().items()}
        initial_cash = float(params.get("initial_cash") or 100_000.0)
        snapshot = PortfolioStore(db_path).snapshot(initial_cash)
        suggestions = generate_suggestions(
            targets,
            snapshot,
            prices,
            lot_size=risk["lot_size"],
            min_order_value=risk["min_order_value"],
        )
        (context.run_dir / "signals.csv").write_text(
            "symbol,timestamp,target_weight,reason,model_version\n"
            + "".join(
                f"{item.symbol},{item.timestamp.isoformat()},{item.target_weight},{item.reason},{item.model_version}\n"
                for item in targets
            ),
            encoding="utf-8",
        )
        (context.run_dir / "orders.csv").write_text(
            "symbol,action,quantity,target_weight,reference_price,status,risk_flags,reason\n"
            + "".join(
                f"{item.symbol},{item.action},{item.quantity},{item.target_weight},{item.reference_price},"
                f"{item.status},{'|'.join(item.risk_flags)},{item.reason}\n"
                for item in suggestions
            ),
            encoding="utf-8",
        )
        blocked = [item.symbol for item in suggestions if item.status == "BLOCKED"]
        return {
            "status": "BLOCKED" if blocked else "PASS",
            "snapshot": manifest.get("snapshot_id"),
            "cash": snapshot.cash,
            "positions": snapshot.positions,
            "targets": len(targets),
            "suggestions": len(suggestions),
            "blocked": blocked,
            "artifacts": ["signals.csv", "orders.csv"],
        }

    return run


def build_task(
    kind: str,
    params: dict[str, Any] | None = None,
    *,
    data_dir: str | Path = "data",
    configs_dir: str | Path = "configs",
    db_path: str | Path = "db/portfolio.sqlite",
) -> Task:
    """Build the real task for a Web UI run kind."""

    settings = params or {}
    data_root = Path(data_dir)
    configs_root = Path(configs_dir)
    if kind == "download":
        return _download_task(settings, data_dir=data_root, configs_dir=configs_root)
    if kind == "backtest":
        return _backtest_task(settings, data_dir=data_root, configs_dir=configs_root)
    if kind == "generate-signals":
        return _signals_task(settings, data_dir=data_root, configs_dir=configs_root, db_path=Path(db_path))
    raise ValueError(f"unsupported run kind: {kind}")


def market_figure(frame: Any) -> tuple[dict[str, Any], str]:
    """Build a candlestick + volume figure for the first available symbol."""

    symbols = sorted(str(item) for item in frame["symbol"].unique())
    symbol = symbols[0]
    data = frame.loc[frame["symbol"] == symbol].sort_values("timestamp")
    figure = {
        "data": [
            {
                "type": "candlestick",
                "name": symbol,
                "x": [item.isoformat() for item in data["timestamp"]],
                "open": [float(item) for item in data["open"]],
                "high": [float(item) for item in data["high"]],
                "low": [float(item) for item in data["low"]],
                "close": [float(item) for item in data["close"]],
            },
            {
                "type": "bar",
                "name": "volume",
                "x": [item.isoformat() for item in data["timestamp"]],
                "y": [float(item) for item in data["volume"]],
                "yaxis": "y2",
                "marker": {"color": "rgba(115,167,255,.35)"},
            },
        ],
        "layout": {
            "template": "plotly_dark",
            "title": {"text": f"{symbol} 日线"},
            "xaxis": {"rangeslider": {"visible": False}},
            "yaxis": {"title": {"text": "price"}, "domain": [0.28, 1.0]},
            "yaxis2": {"title": {"text": "volume"}, "domain": [0.0, 0.2], "anchor": "x"},
            "legend": {"orientation": "h"},
        },
    }
    message = f"共 {len(data)} 根 bar；快照标的：{', '.join(symbols)}"
    return figure, message


def _latest_artifact(runs_dir: Path, filename: str) -> Path | None:
    candidates = sorted(runs_dir.glob(f"*/*/{filename}"), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def equity_figure(runs_dir: str | Path) -> tuple[dict[str, Any], str]:
    path = _latest_artifact(Path(runs_dir), "equity_curve.csv")
    if path is None:
        return {"data": [], "layout": {"template": "plotly_dark", "annotations": [{"text": "暂无回测结果", "showarrow": False}]}}, "尚未运行回测"
    import pandas as pd

    frame = pd.read_csv(path)
    x_column = frame.columns[0]
    value_column = "value" if "value" in frame.columns else frame.columns[-1]
    figure = {
        "data": [{"type": "scatter", "mode": "lines", "name": "portfolio value", "x": list(frame[x_column]), "y": list(frame[value_column])}],
        "layout": {"template": "plotly_dark", "title": {"text": "组合净值"}, "yaxis": {"title": {"text": "value"}}},
    }
    return figure, f"来源：{path.parent.name}"


def latest_orders(runs_dir: str | Path) -> tuple[list[dict[str, str]], str, list[str]]:
    path = _latest_artifact(Path(runs_dir), "orders.csv")
    if path is None:
        return [], "ERROR", ["尚未生成建议；请先运行下载和生成建议"]
    import csv as csv_module

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv_module.DictReader(handle))
    blocked = [f"{row['symbol']}: {row['risk_flags']}" for row in rows if row.get("status") == "BLOCKED"]
    status = "BLOCKED" if blocked else "PASS"
    return rows, status, blocked or [f"来源：{path.parent.name}"]
