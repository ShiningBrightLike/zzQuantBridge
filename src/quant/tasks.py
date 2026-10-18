"""Real domain work shared by the CLI and the local Web UI job runner."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .backtest import (
    BacktestConfig,
    CrossSectionalBacktestConfig,
    crossovers,
    run_cross_sectional_backtest,
    run_signal_backtest,
)
from .artifacts import (
    write_csv,
    write_multifactor_backtest_artifacts,
    write_selection_artifacts,
    write_target_signals,
)
from .config import load_yaml_config
from .data.archive import archive_snapshot
from .data.providers import AkShareDataProvider, DataFetchError, LocalDataProvider
from .data.quality import QualityReport, validate_research_snapshot
from .data.universe import (
    config_names,
    config_symbols,
    fetch_all_a_candidates,
    load_universe_config,
)
from .data.validation import normalize_ohlcv
from .domain import TargetPosition, TradeSuggestion
from .portfolio import PortfolioStore
from .reports import write_run_report
from .risk import RiskPolicy, evaluate_risk
from .runs import RunContext, write_status
from .signals import generate_suggestions
from .strategies import (
    MovingAverageStrategy,
    MovingAverageStrategyConfig,
    MultifactorStrategy,
    MultifactorStrategyConfig,
    SelectionResult,
)


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


def _strategy_settings(configs_dir: Path, params: Mapping[str, Any]) -> MovingAverageStrategyConfig:
    payload: dict[str, Any] = dict(_config_payload(configs_dir, "strategy.yaml"))
    for key in ("fast_window", "slow_window", "max_positions", "max_single_weight", "model_version"):
        if params.get(key) not in (None, ""):
            payload[key] = params[key]
    allowed = {"fast_window", "slow_window", "max_positions", "max_single_weight", "model_version"}
    return MovingAverageStrategyConfig(**{key: value for key, value in payload.items() if key in allowed})


def _strategy_name(configs_dir: Path, params: Mapping[str, Any], default: str = "moving_average") -> str:
    explicit = str(params.get("strategy") or "").strip().lower()
    config_payload = params.get("config_payload")
    if not explicit and isinstance(config_payload, Mapping):
        explicit = str(config_payload.get("strategy") or config_payload.get("name") or "").strip().lower()
    if not explicit:
        payload = _config_payload(configs_dir, "strategy.yaml")
        explicit = str(payload.get("strategy") or payload.get("name") or default).strip().lower()
    if "multi" in explicit:
        return "multifactor"
    if "moving" in explicit or explicit in {"ma", "ma-baseline"}:
        return "moving_average"
    return explicit or default


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _multifactor_payload(configs_dir: Path, params: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = dict(_config_payload(configs_dir, "multifactor.yaml"))
    explicit = params.get("config_payload")
    if isinstance(explicit, Mapping):
        name = str(explicit.get("strategy") or explicit.get("name") or "").strip().lower()
        if "multi" in name:
            payload = _deep_merge(payload, explicit)
    for section in ("universe", "factors", "training", "selection", "backtest"):
        override = params.get(section)
        if isinstance(override, Mapping):
            payload[section] = _deep_merge(payload.get(section) or {}, override)
    for key in ("model_version", "top_n", "hold_buffer", "max_single_weight", "cash_buffer", "rebalance_days"):
        if params.get(key) not in (None, ""):
            if key in {"top_n", "hold_buffer", "max_single_weight", "cash_buffer", "rebalance_days"}:
                payload.setdefault("selection", {})
                payload["selection"][key] = params[key]
            else:
                payload[key] = params[key]
    return payload


def _multifactor_settings(configs_dir: Path, params: Mapping[str, Any]) -> MultifactorStrategyConfig:
    return MultifactorStrategyConfig.from_mapping(_multifactor_payload(configs_dir, params))


def _cross_sectional_backtest_config(
    configs_dir: Path, params: Mapping[str, Any]
) -> CrossSectionalBacktestConfig:
    multifactor = _multifactor_payload(configs_dir, params)
    payload = dict(multifactor.get("backtest") or {})
    selection = multifactor.get("selection") or {}
    if isinstance(selection, Mapping) and selection.get("lot_size") not in (None, ""):
        # One lot size governs both the affordability screen and the
        # target-weight simulation.
        payload["lot_size"] = selection["lot_size"]
    for key in (
        "init_cash",
        "fees",
        "slippage",
        "lot_size",
        "signal_delay",
        "min_order_value",
        "min_oos_days",
        "backtest_start",
    ):
        if params.get(key) not in (None, ""):
            payload[key] = params[key]
    if payload.get("backtest_start") in ("", None):
        payload.pop("backtest_start", None)
    return CrossSectionalBacktestConfig(**payload)


def _selection_backtest_config(
    configs_dir: Path,
    params: Mapping[str, Any],
    initial_cash: float,
) -> CrossSectionalBacktestConfig:
    """Use the portfolio's resolved cash basis for the selection backtest."""

    return replace(
        _cross_sectional_backtest_config(configs_dir, params),
        init_cash=float(initial_cash),
    )


def _quality_gate(
    context: RunContext,
    frame: Any,
    configs_dir: Path,
    params: Mapping[str, Any],
) -> QualityReport:
    payload = _multifactor_payload(configs_dir, params).get("quality") or {}
    report = validate_research_snapshot(
        frame,
        max_missing_days=int(payload.get("max_missing_days", 20)),
        max_missing_ratio=float(payload.get("max_missing_ratio", 0.02)),
    )
    (context.run_dir / "data_quality.json").write_text(
        json.dumps(
            {
                "status": report.status,
                "row_count": report.row_count,
                "errors": list(report.errors),
                "warnings": list(report.warnings),
                "max_missing_days": int(payload.get("max_missing_days", 20)),
                "max_missing_ratio": float(payload.get("max_missing_ratio", 0.02)),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _quality_blocked_summary(context: RunContext, report: QualityReport) -> dict[str, Any]:
    payload = {
        "status": "BLOCKED",
        "quality_errors": list(report.errors),
        "quality_warnings": list(report.warnings),
        "artifacts": ["data_quality.json", "report.html"],
    }
    write_status(context.run_dir, "BLOCKED", payload)
    write_run_report(context.run_dir)
    return payload


def _fetch_akshare_frames(
    provider: AkShareDataProvider,
    symbols: Sequence[str],
    start: date,
    end: date,
) -> tuple[Any, dict[str, str]]:
    """Fetch symbols independently so one failed name does not abort a broad run."""

    import pandas as pd

    failures: dict[str, str] = {}
    frames = []
    if len(symbols) > 1:
        for symbol in symbols:
            try:
                frames.append(provider.fetch([symbol], start, end))
            except DataFetchError as exc:
                failures[symbol] = str(exc)
    else:
        frames.append(provider.fetch(symbols, start, end))
    if not frames:
        raise DataFetchError("all requested symbols failed to download")
    return pd.concat(frames, ignore_index=True), failures


def _merge_execution_close(frame: Any, execution_frame: Any) -> Any:
    """Attach the latest unadjusted close for execution without replacing research prices."""

    if execution_frame is None or len(execution_frame) == 0:
        return frame
    execution = execution_frame[["symbol", "timestamp", "close"]].rename(
        columns={"close": "execution_close"}
    )
    result = frame.drop(columns=["execution_close"], errors="ignore")
    return result.merge(execution, on=["symbol", "timestamp"], how="left")


def _latest_prices(frame: Any, manifest: Mapping[str, Any]) -> tuple[dict[str, float], str]:
    """Return execution prices and whether they are unadjusted."""

    import pandas as pd

    if "execution_close" in frame.columns and frame["execution_close"].notna().any():
        values = frame.groupby("symbol")["execution_close"].last()
        source = "unadjusted"
    else:
        values = frame.groupby("symbol")["close"].last()
        source = "unadjusted" if not manifest.get("adjustment") else "adjusted"
    prices = {
        str(symbol): float(value)
        for symbol, value in values.items()
        if pd.notna(value) and math.isfinite(float(value)) and float(value) > 0
    }
    return prices, source


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


def _snapshot_names(configs_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    names = config_names(configs_dir)
    request = manifest.get("request")
    if isinstance(request, Mapping):
        raw = request.get("names")
        if isinstance(raw, Mapping):
            names.update(
                {
                    str(key): re.sub(r"^(XD|XR|DR)", "", str(value))
                    for key, value in raw.items()
                    if value
                }
            )
    return names


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
        provider_name = str(params.get("provider") or "akshare").strip().lower()
        adjustment = str(params.get("adjustment") or "")
        universe = load_universe_config(
            configs_dir,
            params.get("universe") if isinstance(params.get("universe"), Mapping) else None,
        )
        explicit_symbols = [str(item).strip() for item in params.get("symbols") or []]
        symbols = explicit_symbols or (
            [] if universe.mode == "all_a" else config_symbols(configs_dir)
        )
        universe_rows: list[dict[str, Any]] = []
        if not symbols and provider_name == "akshare" and universe.mode == "all_a":
            candidates = fetch_all_a_candidates(universe)
            symbols = [str(item) for item in candidates["symbol"].tolist()]
            universe_rows = candidates.to_dict(orient="records")
        if not symbols and not (provider_name == "local" and params.get("file")):
            raise ValueError(
                "no symbols configured; set symbols in configs/universe.yaml, "
                "set universe.mode=all_a, or pass --symbols"
            )
        end = _as_date(params.get("end"), _today())
        default_days = 365 * 5 if universe.mode == "all_a" else 365
        start = _as_date(params.get("start"), end - timedelta(days=default_days))
        if start > end:
            raise ValueError("start date must not be after end date")
        if provider_name == "local":
            file_path = params.get("file")
            if not file_path:
                raise ValueError("local provider requires a file path")
            provider: Any = LocalDataProvider(file_path)
        elif provider_name == "akshare":
            provider = AkShareDataProvider(adjustment=adjustment)
        else:
            raise ValueError(f"unsupported provider: {provider_name}")
        failures: dict[str, str] = {}
        execution_failures: dict[str, str] = {}
        if provider_name == "akshare":
            frame, failures = _fetch_akshare_frames(provider, symbols, start, end)
        else:
            frame = provider.fetch(symbols, start, end)
        configured_execution = params.get("execution_prices")
        if configured_execution is None:
            config_payload = params.get("config_payload")
            if isinstance(config_payload, Mapping):
                configured_execution = config_payload.get("execution_prices")
        include_execution = True if configured_execution is None else bool(configured_execution)
        if provider_name == "akshare" and include_execution and adjustment:
            # Only the recent unadjusted close is needed for manual execution;
            # avoid duplicating the full five-year history download.
            execution_provider = AkShareDataProvider(adjustment="")
            execution_start = max(start, end - timedelta(days=10))
            try:
                execution_frame, execution_failures = _fetch_akshare_frames(
                    execution_provider, symbols, execution_start, end
                )
                frame = _merge_execution_close(frame, execution_frame)
            except DataFetchError as exc:
                execution_failures = {"all": str(exc)}
        elif provider_name == "local" and not adjustment and "execution_close" not in frame.columns:
            frame = frame.copy()
            frame["execution_close"] = frame["close"]
        actual_symbols = sorted(str(item) for item in frame["symbol"].unique())
        actual_set = set(actual_symbols)
        if universe_rows:
            universe_rows = [
                row for row in universe_rows if str(row.get("symbol")) in actual_set
            ]
        if not universe_rows:
            universe_rows = [
                {"symbol": symbol, "name": "", "tradable": 1}
                for symbol in actual_symbols
            ]
        write_csv(
            context.run_dir / "universe.csv",
            ("symbol", "name", "board", "amount", "source"),
            universe_rows,
        )
        names = {
            str(row.get("symbol")): str(row.get("name"))
            for row in universe_rows
            if row.get("symbol") and row.get("name")
        }
        if not names:
            names = config_names(configs_dir)
        manifest_path = archive_snapshot(
            frame,
            data_dir,
            source=provider_name,
            request={
                "symbols": symbols,
                "names": names,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "execution_prices": include_execution,
            },
            adjustment=adjustment,
        )
        summary: dict[str, Any] = {
            "rows": int(len(frame)),
            "symbols": actual_symbols,
            "universe_mode": universe.mode,
            "candidate_pool_size": universe.candidate_pool_size,
            "universe_size": len(actual_symbols),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "data_as_of": data_as_of(frame).isoformat() if data_as_of(frame) else None,
            "provider": provider_name,
            "execution_price_source": (
                "unadjusted"
                if "execution_close" in frame.columns and frame["execution_close"].notna().any()
                else ("adjusted" if adjustment else "unadjusted")
            ),
            "manifest": str(manifest_path),
        }
        sources = {
            str(row.get("source")) for row in universe_rows if row.get("source")
        }
        if sources:
            summary["universe_source"] = sorted(sources)
        if failures:
            summary["failed_symbols"] = failures
            (context.run_dir / "download_failures.json").write_text(
                json.dumps(failures, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if execution_failures:
            summary["execution_price_failures"] = execution_failures
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
        frame, manifest = load_snapshot(data_dir)
        if _strategy_name(configs_dir, params) == "multifactor":
            quality = _quality_gate(context, frame, configs_dir, params)
            if quality.blocked:
                return _quality_blocked_summary(context, quality)
            settings = _multifactor_settings(configs_dir, params)
            backtest_config = _cross_sectional_backtest_config(configs_dir, params)
            result = run_cross_sectional_backtest(
                frame,
                settings,
                config=backtest_config,
                backtest_start=backtest_config.backtest_start,
            )
            summary = write_multifactor_backtest_artifacts(
                context, frame, result, settings, backtest_config
            )
            summary["snapshot"] = manifest.get("snapshot_id")
            return summary
        settings = _strategy_settings(configs_dir, params)
        risk = _risk_payload(configs_dir, params)
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
        policy = _risk_policy(configs_dir, params)
        store = PortfolioStore(db_path)
        initial_cash = resolve_initial_cash(store, params, configs_dir)
        snapshot = store.snapshot(initial_cash)
        strategy_name = _strategy_name(configs_dir, params)
        selection: SelectionResult | None = None
        if strategy_name == "multifactor":
            quality = _quality_gate(context, frame, configs_dir, params)
            if quality.blocked:
                return _quality_blocked_summary(context, quality)
            multifactor_settings = _multifactor_settings(configs_dir, params)
            selection = MultifactorStrategy(multifactor_settings).run_selection(
                frame, snapshot, multifactor_settings
            )
            targets = list(selection.targets)
            write_selection_artifacts(
                context,
                selection,
                names=_snapshot_names(configs_dir, manifest),
                top_n=multifactor_settings.top_n,
                affordability_filter=multifactor_settings.affordability_filter,
            )
        else:
            settings = _strategy_settings(configs_dir, params)
            strategy = MovingAverageStrategy(settings)
            targets = list(strategy.generate_targets(frame))
        prices, execution_price_source = _latest_prices(frame, manifest)
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
        if execution_price_source == "adjusted":
            reasons.append(
                "execution prices are adjusted; an unadjusted latest close is required"
            )

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

        blocked_symbols = sorted({item.symbol for item in finalized if item.status == "BLOCKED"})
        status = "BLOCKED" if reasons or blocked_symbols else "PASS"

        write_target_signals(context, targets)
        write_csv(
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
                {"status": status, "reasons": reasons, "symbol_flags": risk.as_dict()["symbol_flags"]},
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
        summary: dict[str, Any] = {
            "snapshot": manifest.get("snapshot_id"),
            "data_as_of": bar_date.isoformat() if bar_date else None,
            "strategy": strategy_name,
            "execution_price_source": execution_price_source,
            "model_version": (
                selection.model.model_version if selection is not None else targets[0].model_version if targets else ""
            ),
            "selected": len(targets),
            "initial_cash": initial_cash,
            "cash": snapshot.cash,
            "positions": snapshot.positions,
            "targets": len(targets),
            "suggestions": len(finalized),
            "blocked": blocked_symbols,
            "risk_reasons": reasons,
            "artifacts": [
                "signals.csv",
                "orders.csv",
                "risk_report.json",
                "portfolio_snapshot.json",
                *(
                    ["factor_snapshot.csv", "factor_weights.csv", "selections.csv", "universe.csv", "model_metadata.json"]
                    if selection is not None
                    else []
                ),
                "report.html",
            ],
        }
        write_run_report(context.run_dir)
        write_status(context.run_dir, status, summary)
        return {"status": status, **summary}

    return run


def _selection_task(params: Mapping[str, Any], *, data_dir: Path, configs_dir: Path, db_path: Path) -> Task:
    def run(context: RunContext) -> dict[str, Any]:
        frame, manifest = load_snapshot(data_dir)
        quality = _quality_gate(context, frame, configs_dir, params)
        if quality.blocked:
            return _quality_blocked_summary(context, quality)
        settings = _multifactor_settings(configs_dir, params)
        store = PortfolioStore(db_path)
        initial_cash = resolve_initial_cash(store, params, configs_dir)
        backtest_config = _selection_backtest_config(configs_dir, params, initial_cash)
        snapshot = store.snapshot(initial_cash)
        selection = MultifactorStrategy(settings).run_selection(frame, snapshot, settings)
        write_selection_artifacts(
            context,
            selection,
            names=_snapshot_names(configs_dir, manifest),
            top_n=settings.top_n,
            affordability_filter=settings.affordability_filter,
        )
        write_target_signals(context, selection.targets)
        backtest_error: str | None = None
        backtest_summary: dict[str, Any] = {}
        try:
            result = run_cross_sectional_backtest(
                frame,
                settings,
                config=backtest_config,
                backtest_start=backtest_config.backtest_start,
            )
            backtest_summary = write_multifactor_backtest_artifacts(
                context, frame, result, settings, backtest_config
            )
            if backtest_summary.get("status") == "BLOCKED":
                backtest_error = (
                    f"insufficient out-of-sample history: active_days "
                    f"{backtest_summary.get('active_days', 0)} < "
                    f"min_oos_days {backtest_config.min_oos_days}"
                )
        except (ImportError, ValueError, RuntimeError) as exc:
            backtest_error = str(exc)
            (context.run_dir / "backtest_metrics.json").write_text(
                json.dumps(
                    {"status": "BLOCKED", "error": backtest_error},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        status = "PASS" if selection.targets and backtest_error is None else "BLOCKED"
        artifact_names = [
            "factor_snapshot.csv",
            "factor_weights.csv",
            "selections.csv",
            "signals.csv",
            "universe.csv",
            "model_metadata.json",
            "backtest_metrics.json",
            "equity_curve.csv",
            "trades.csv",
            "backtest_model_metrics.csv",
            "backtest_factor_weights.csv",
            "backtest_selections.csv",
            "sensitivity.json",
            "report.html",
        ]
        summary: dict[str, Any] = {
            "snapshot": manifest.get("snapshot_id"),
            "data_as_of": str(frame["timestamp"].max().date()),
            "strategy": "multifactor",
            "model_version": selection.model.model_version,
            "universe_size": len(selection.tradable_symbols),
            "selected": len(selection.targets),
            "selected_symbols": [target.symbol for target in selection.targets],
            "initial_cash": initial_cash,
            "backtest": backtest_summary if backtest_summary else {},
            "backtest_error": backtest_error,
            "artifacts": [
                name for name in artifact_names if (context.run_dir / name).is_file()
            ],
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
    if kind == "select-stocks":
        return _selection_task(settings, data_dir=data_root, configs_dir=configs_root, db_path=Path(db_path))
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
    if "equity" in frame.columns:
        value_column = "equity"
    elif "value" in frame.columns:
        value_column = "value"
    else:
        value_column = frame.columns[-1]
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


def _latest_artifact_for_command(runs_dir: Path, command: str, filename: str) -> Path | None:
    candidates: list[Path] = []
    for path in runs_dir.glob(f"*/*/{filename}"):
        manifest_path = path.parent / "input_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(manifest.get("command")) == command:
            candidates.append(path)
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def latest_selections(
    runs_dir: str | Path,
) -> tuple[list[dict[str, str]], str, list[str], dict[str, Any]]:
    """Load the latest ``select-stocks`` cross-section for the Web UI."""

    path = _latest_artifact_for_command(Path(runs_dir), "select-stocks", "selections.csv")
    if path is None:
        return [], "ERROR", ["尚未生成多因子选股结果；请先下载行情并运行选股"], {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    status_path = path.parent / "status.json"
    status = "UNKNOWN"
    if status_path.is_file():
        try:
            status = str(json.loads(status_path.read_text(encoding="utf-8")).get("status", "UNKNOWN"))
        except (OSError, ValueError):
            pass
    metadata_path = path.parent / "model_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = {}
    reasons = [
        f"模型 {metadata.get('model_version', 'unknown')}",
        f"因子 {len(metadata.get('factor_names', []))} 个",
        f"股票池 {metadata.get('universe_size', 0)} 只",
        f"来源 {path.parent.name}",
    ]
    if metadata.get("selection_limited"):
        reasons.append(
            f"最低得分和整手预算过滤后入选 {metadata.get('selected_count', 0)} / "
            f"{metadata.get('top_n', '?')} 只；可提高初始资金或减少 top_n"
        )
    if metadata.get("fallback_used"):
        reasons.append("严格 IC/ICIR 阈值未满足，本次使用了宽松因子回退")
    return rows, status, reasons, metadata
