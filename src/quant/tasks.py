"""Real domain work shared by the CLI and the local Web UI job runner."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .backtest import BacktestConfig, crossovers, run_signal_backtest
from .config import load_yaml_config
from .data.archive import archive_snapshot
from .data.providers import AkShareDataProvider, LocalDataProvider
from .data.validation import normalize_ohlcv
from .domain import TargetPosition, TradeSuggestion
from .portfolio import PortfolioStore
from .reports import write_run_report
from .risk import RiskPolicy, evaluate_risk
from .runs import RunContext, write_status
from .signals import generate_suggestions
from .strategies import MovingAverageStrategy, MovingAverageStrategyConfig


Task = Callable[[RunContext], dict[str, Any]]
DEFAULT_INITIAL_CASH = 100_000.0


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


def _optional_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    return _as_date(value, _today())


def _config_payload(configs_dir: Path, name: str) -> dict[str, Any]:
    path = configs_dir / name
    return load_yaml_config(path) if path.is_file() else {}


def _config_symbols(configs_dir: Path) -> list[str]:
    payload = _config_payload(configs_dir, "universe.yaml")
    return [str(item).strip() for item in payload.get("symbols", []) if str(item).strip()]


def _strategy_settings(configs_dir: Path, params: Mapping[str, Any]) -> MovingAverageStrategyConfig:
    payload: dict[str, Any] = dict(_config_payload(configs_dir, "strategy.yaml"))
    for key in ("fast_window", "slow_window", "max_positions", "max_single_weight", "model_version"):
        if params.get(key) not in (None, ""):
            payload[key] = params[key]
    allowed = {"fast_window", "slow_window", "max_positions", "max_single_weight", "model_version"}
    return MovingAverageStrategyConfig(**{key: value for key, value in payload.items() if key in allowed})


def _risk_payload(configs_dir: Path, params: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = dict(_config_payload(configs_dir, "risk.yaml"))
    for key in (
        "lot_size",
        "min_order_value",
        "max_single_weight",
        "max_gross_weight",
        "cash_buffer",
        "max_turnover",
        "max_data_age_days",
        "commission_rate",
        "slippage_bps",
        "initial_cash",
        "industries",
        "industry_limits",
    ):
        if params.get(key) not in (None, ""):
            payload[key] = params[key]
    return payload


def _risk_policy(configs_dir: Path, params: Mapping[str, Any]) -> RiskPolicy:
    return RiskPolicy.from_config(_risk_payload(configs_dir, params))


def _mean_scalar(value: Any) -> float:
    """Collapse vectorbt's per-column results into one portfolio-level number."""

    if hasattr(value, "mean"):
        return float(value.mean())
    return float(value)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write CSV through the stdlib writer so commas or quotes cannot shift columns."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


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


def data_as_of(frame: Any) -> date | None:
    if frame is None or len(frame) == 0:
        return None
    return max(item.date() for item in frame["timestamp"])


def resolve_initial_cash(store: PortfolioStore, params: Mapping[str, Any], configs_dir: Path) -> float:
    """Keep one cash basis for every entry point.

    An explicit value wins and is persisted; otherwise the stored value is
    reused; otherwise the configured default is applied and persisted.
    """

    explicit = params.get("initial_cash")
    if explicit not in (None, ""):
        value = float(explicit)
        store.set_initial_cash(value)
        return value
    stored = store.get_initial_cash()
    if stored is not None:
        return stored
    default = float(_risk_payload(configs_dir, {}).get("initial_cash", DEFAULT_INITIAL_CASH))
    store.set_initial_cash(default)
    return default


def _download_task(params: Mapping[str, Any], *, data_dir: Path, configs_dir: Path) -> Task:
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
        summary: dict[str, Any] = {
            "rows": int(len(frame)),
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "data_as_of": data_as_of(frame).isoformat() if data_as_of(frame) else None,
            "provider": provider_name,
            "manifest": str(manifest_path),
        }
        endpoints = getattr(provider, "last_sources", None)
        if endpoints:
            summary["endpoints"] = dict(endpoints)
        (context.run_dir / "data_manifest.json").write_text(
            json.dumps({"status": "PASS", **summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_run_report(context.run_dir)
        # Status is written last: a poller that sees a terminal status must be
        # able to trust that every artifact already exists.
        write_status(context.run_dir, "PASS", summary)
        return summary

    return run


def _window_pairs(settings: MovingAverageStrategyConfig) -> list[tuple[int, int]]:
    fast, slow = settings.fast_window, settings.slow_window
    pairs = {(fast, slow)}
    pairs.add((max(2, fast // 2), slow))
    if fast * 2 < slow:
        pairs.add((fast * 2, slow))
    else:
        pairs.add((max(2, slow - 1), slow))
    return sorted(pairs)


def _backtest_task(params: Mapping[str, Any], *, data_dir: Path, configs_dir: Path) -> Task:
    def run(context: RunContext) -> dict[str, Any]:
        settings = _strategy_settings(configs_dir, params)
        risk = _risk_payload(configs_dir, params)
        frame, manifest = load_snapshot(data_dir)
        close = frame.pivot_table(index="timestamp", columns="symbol", values="close").sort_index()
        init_cash = float(params.get("init_cash") or risk.get("initial_cash") or DEFAULT_INITIAL_CASH)
        base_fees = float(params.get("fees") or risk.get("commission_rate") or 0.0)
        base_slippage = float(params.get("slippage") or 0.0)
        if "slippage" not in params and risk.get("slippage_bps") is not None:
            base_slippage = float(risk["slippage_bps"]) / 10_000.0
        base_delay = int(params.get("signal_delay") or 1)

        def simulate(fast_window: int, slow_window: int, fees: float, slippage: float, delay: int) -> Any:
            fast = close.rolling(fast_window, min_periods=fast_window).mean()
            slow = close.rolling(slow_window, min_periods=slow_window).mean()
            ready = fast.notna() & slow.notna()
            entries = crossovers((fast > slow) & ready)
            exits = crossovers((fast < slow) & ready)
            config = BacktestConfig(
                init_cash=init_cash, fees=fees, slippage=slippage, signal_delay=delay
            )
            return run_signal_backtest(close, entries, exits, config)

        def summarize(result: Any) -> dict[str, Any]:
            portfolio = result.portfolio
            trades = portfolio.trades.records_readable
            return {
                "total_return": round(_mean_scalar(portfolio.total_return()), 6),
                "max_drawdown": round(_mean_scalar(portfolio.max_drawdown()), 6),
                "trade_count": int(len(trades)) if trades is not None else 0,
            }

        result = simulate(settings.fast_window, settings.slow_window, base_fees, base_slippage, base_delay)
        metrics = {str(key): value for key, value in result.metrics.items()}
        for key, value in list(metrics.items()):
            try:
                json.dumps(value)
            except TypeError:
                metrics[key] = str(value)
        (context.run_dir / "backtest_metrics.json").write_text(
            json.dumps({"status": "PASS", "metrics": metrics}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        equity = result.portfolio.value()
        if hasattr(equity, "to_frame"):
            equity = equity.to_frame("value")
        equity.to_csv(context.run_dir / "equity_curve.csv")
        trades = result.portfolio.trades.records_readable
        if trades is not None and len(trades):
            trades.to_csv(context.run_dir / "trades.csv", index=False)

        # Three sensitivity analyses required by the plan (section 7.3). The
        # baseline scenario is reused instead of being simulated twice.
        base_summary = summarize(result)
        cost: dict[str, Any] = {}
        for multiplier in (1.0, 2.0, 3.0):
            key = f"x{multiplier:g}"
            cost[key] = (
                base_summary
                if multiplier == 1.0
                else summarize(
                    simulate(
                        settings.fast_window,
                        settings.slow_window,
                        base_fees * multiplier,
                        base_slippage * multiplier,
                        base_delay,
                    )
                )
            )
        delay: dict[str, Any] = {}
        for value in sorted({base_delay, base_delay + 1, max(0, base_delay - 1)}):
            delay[f"delay_{value}"] = (
                base_summary
                if value == base_delay
                else summarize(simulate(settings.fast_window, settings.slow_window, base_fees, base_slippage, value))
            )
        window: dict[str, Any] = {}
        for fast, slow in _window_pairs(settings):
            key = f"fast_{fast}_slow_{slow}"
            window[key] = (
                base_summary
                if (fast, slow) == (settings.fast_window, settings.slow_window)
                else summarize(simulate(fast, slow, base_fees, base_slippage, base_delay))
            )
        sensitivity: dict[str, Any] = {
            "cost_multiplier": cost,
            "signal_delay": delay,
            "window_stability": window,
        }
        (context.run_dir / "sensitivity.json").write_text(
            json.dumps(sensitivity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summary: dict[str, Any] = {
            "snapshot": manifest.get("snapshot_id"),
            "symbols": [str(item) for item in close.columns],
            "rows": int(len(close)),
            "data_as_of": data_as_of(frame).isoformat() if data_as_of(frame) else None,
            "signal_delay": base_delay,
            "fees": base_fees,
            "slippage": base_slippage,
            **base_summary,
            "artifacts": [
                "backtest_metrics.json",
                "equity_curve.csv",
                "trades.csv",
                "sensitivity.json",
                "report.html",
            ],
        }
        write_run_report(context.run_dir)
        write_status(context.run_dir, "PASS", summary)
        return summary

    return run


def _signals_task(params: Mapping[str, Any], *, data_dir: Path, configs_dir: Path, db_path: Path) -> Task:
    def run(context: RunContext) -> dict[str, Any]:
        frame, manifest = load_snapshot(data_dir)
        settings = _strategy_settings(configs_dir, params)
        policy = _risk_policy(configs_dir, params)
        strategy = MovingAverageStrategy(settings)
        targets = list(strategy.generate_targets(frame))
        prices = {str(symbol): float(value) for symbol, value in frame.groupby("symbol")["close"].last().items()}
        store = PortfolioStore(db_path)
        initial_cash = resolve_initial_cash(store, params, configs_dir)
        snapshot = store.snapshot(initial_cash)
        as_of = _today()
        bar_date = data_as_of(frame)

        valuation_error: str | None = None
        try:
            suggestions = list(
                generate_suggestions(
                    targets,
                    snapshot,
                    prices,
                    lot_size=policy.lot_size,
                    min_order_value=policy.min_order_value,
                )
            )
        except ValueError as exc:
            # A portfolio that cannot be valued is a risk failure, not a crash.
            valuation_error = str(exc)
            suggestions = [
                TradeSuggestion(
                    symbol=target.symbol,
                    action="HOLD",
                    quantity=0,
                    target_weight=target.target_weight,
                    reference_price=None,
                    max_price=None,
                    min_price=None,
                    risk_flags=[f"cannot value portfolio: {exc}"],
                    reason=target.reason,
                    status="BLOCKED",
                )
                for target in targets
            ]

        risk = evaluate_risk(
            targets=targets,
            suggestions=suggestions,
            portfolio=snapshot,
            prices=prices,
            policy=policy,
            as_of=as_of,
            data_as_of=bar_date,
        )
        reasons = list(risk.reasons)
        if valuation_error is not None:
            reasons.append(f"portfolio cannot be valued: {valuation_error}")

        finalized: list[TradeSuggestion] = []
        for item in suggestions:
            # Run-level failures block every suggestion, per the agreed rule
            # that a global risk failure cannot leave executable rows behind.
            extra = [
                flag
                for flag in (*risk.symbol_flags.get(item.symbol, ()), *reasons)
                if flag not in item.risk_flags
            ]
            flags = [*item.risk_flags, *extra]
            finalized.append(
                replace(item, risk_flags=flags, status="BLOCKED" if flags else "PASS")
            )

        _write_csv(
            context.run_dir / "signals.csv",
            ("symbol", "timestamp", "target_weight", "reason", "model_version"),
            [
                {
                    "symbol": target.symbol,
                    "timestamp": target.timestamp.isoformat(),
                    "target_weight": target.target_weight,
                    "reason": target.reason,
                    "model_version": target.model_version,
                }
                for target in targets
            ],
        )
        _write_csv(
            context.run_dir / "orders.csv",
            ("symbol", "action", "quantity", "target_weight", "reference_price", "status", "risk_flags", "reason"),
            [
                {
                    "symbol": item.symbol,
                    "action": item.action,
                    "quantity": item.quantity,
                    "target_weight": item.target_weight,
                    "reference_price": item.reference_price if item.reference_price is not None else "",
                    "status": item.status,
                    "risk_flags": " | ".join(item.risk_flags),
                    "reason": item.reason,
                }
                for item in finalized
            ],
        )
        (context.run_dir / "risk_report.json").write_text(
            json.dumps(
                {"status": risk.status, "reasons": reasons, "symbol_flags": risk.as_dict()["symbol_flags"]},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (context.run_dir / "portfolio_snapshot.json").write_text(
            json.dumps(
                {
                    "initial_cash": initial_cash,
                    "cash": snapshot.cash,
                    "positions": snapshot.positions,
                    "average_cost": snapshot.average_cost,
                    "realized_pnl": snapshot.realized_pnl,
                    "fees": snapshot.fees,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        blocked_symbols = sorted({item.symbol for item in finalized if item.status == "BLOCKED"})
        status = "BLOCKED" if reasons or blocked_symbols else "PASS"
        summary: dict[str, Any] = {
            "snapshot": manifest.get("snapshot_id"),
            "data_as_of": bar_date.isoformat() if bar_date else None,
            "initial_cash": initial_cash,
            "cash": snapshot.cash,
            "positions": snapshot.positions,
            "targets": len(targets),
            "suggestions": len(finalized),
            "blocked": blocked_symbols,
            "risk_reasons": reasons,
            "artifacts": ["signals.csv", "orders.csv", "risk_report.json", "portfolio_snapshot.json", "report.html"],
        }
        write_run_report(context.run_dir)
        write_status(context.run_dir, status, summary)
        return {"status": status, **summary}

    return run


def build_task(
    kind: str,
    params: Mapping[str, Any] | None = None,
    *,
    data_dir: str | Path = "data",
    configs_dir: str | Path = "configs",
    db_path: str | Path = "db/portfolio.sqlite",
) -> Task:
    """Build the real task for a run kind; shared by the CLI and the Web UI."""

    settings = dict(params or {})
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
        empty = {
            "data": [],
            "layout": {
                "template": "plotly_dark",
                "annotations": [{"text": "暂无回测结果", "showarrow": False}],
            },
        }
        return empty, "尚未运行回测"
    import pandas as pd

    frame = pd.read_csv(path)
    x_column = frame.columns[0]
    value_column = "value" if "value" in frame.columns else frame.columns[-1]
    figure = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "portfolio value",
                "x": list(frame[x_column]),
                "y": list(frame[value_column]),
            }
        ],
        "layout": {"template": "plotly_dark", "title": {"text": "组合净值"}, "yaxis": {"title": {"text": "value"}}},
    }
    return figure, f"来源：{path.parent.name}"


def latest_orders(runs_dir: str | Path) -> tuple[list[dict[str, str]], str, list[str]]:
    path = _latest_artifact(Path(runs_dir), "orders.csv")
    if path is None:
        return [], "ERROR", ["尚未生成建议；请先运行下载和生成建议"]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    blocked = [f"{row['symbol']}: {row['risk_flags']}" for row in rows if row.get("status") == "BLOCKED"]
    status = "BLOCKED" if blocked else "PASS"
    return rows, status, blocked or [f"来源：{path.parent.name}"]
