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
from .data.calendar import latest_completed_trading_date
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
) -> tuple[Any, QualityReport]:
    payload = _multifactor_payload(configs_dir, params).get("quality") or {}
    max_missing_days = int(payload.get("max_missing_days", 20))
    max_missing_ratio = float(payload.get("max_missing_ratio", 0.02))
    excess_gap_policy = str(payload.get("excess_gap_policy", "block"))
    report = validate_research_snapshot(
        frame,
        max_missing_days=max_missing_days,
        max_missing_ratio=max_missing_ratio,
        excess_gap_policy=excess_gap_policy,
    )
    excluded = set(report.excluded_symbols)
    filtered = frame
    if excluded:
        filtered = frame.loc[
            ~frame["symbol"].astype(str).isin(excluded)
        ].reset_index(drop=True)
        write_csv(
            context.run_dir / "quality_exclusions.csv",
            ("symbol", "reason"),
            [
                {"symbol": symbol, "reason": reason}
                for symbol, reason in zip(
                    report.excluded_symbols,
                    report.exclusion_reasons,
                )
            ],
        )
    (context.run_dir / "data_quality.json").write_text(
        json.dumps(
            {
                "status": report.status,
                "row_count": report.row_count,
                "errors": list(report.errors),
                "warnings": list(report.warnings),
                "excluded_symbols": list(report.excluded_symbols),
                "exclusion_reasons": list(report.exclusion_reasons),
                "max_missing_days": max_missing_days,
                "max_missing_ratio": max_missing_ratio,
                "excess_gap_policy": excess_gap_policy,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return filtered, report


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


def _manifest_data_as_of(data_dir: Path, payload: Mapping[str, Any]) -> date | None:
    value = payload.get("data_as_of")
    if value:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            pass
    try:
        import pandas as pd

        for entry in payload.get("files", []):
            if entry.get("kind") != "raw":
                continue
            path = data_dir / str(entry.get("path", ""))
            if not path.is_file():
                continue
            timestamps = pd.read_parquet(path, columns=["timestamp"])["timestamp"]
            return max(item.date() for item in timestamps)
    except Exception:
        pass
    request = payload.get("request")
    if isinstance(request, Mapping):
        try:
            return date.fromisoformat(str(request.get("end"))[:10])
        except ValueError:
            return None
    return None


def _manifest_created_at(payload: Mapping[str, Any]) -> datetime:
    try:
        value = datetime.fromisoformat(str(payload.get("created_at", "")))
        return value if value.tzinfo else value.replace(tzinfo=ZoneInfo("UTC"))
    except ValueError:
        return datetime.min.replace(tzinfo=ZoneInfo("UTC"))


def latest_snapshot(data_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Return the raw snapshot with the newest actual data coverage."""

    metadata_dir = data_dir / "metadata"
    manifests = metadata_dir.glob("*.json") if metadata_dir.is_dir() else ()
    candidates: list[tuple[date, datetime, float, Path, dict[str, Any]]] = []
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        raw_path: Path | None = None
        for entry in payload.get("files", []):
            if entry.get("kind") == "raw":
                candidate = data_dir / str(entry.get("path", ""))
                if candidate.is_file():
                    raw_path = candidate
                    break
        if raw_path is None:
            continue
        candidates.append(
            (
                _manifest_data_as_of(data_dir, payload) or date.min,
                _manifest_created_at(payload),
                manifest_path.stat().st_mtime,
                raw_path,
                payload,
            )
        )
    if candidates:
        _, _, _, frame_path, payload = max(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
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


def _download_task(
    params: Mapping[str, Any],
    *,
    data_dir: Path,
    configs_dir: Path,
    finalize: bool = True,
) -> Task:
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
        if finalize:
            write_run_report(context.run_dir)
            # Status is written last: a poller that sees a terminal status must
            # be able to trust that every artifact already exists.
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
            frame, quality = _quality_gate(context, frame, configs_dir, params)
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
            frame, quality = _quality_gate(context, frame, configs_dir, params)
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


def _multifactor_snapshot_ready(
    frame: Any | None,
    settings: MultifactorStrategyConfig,
    *,
    required_data_as_of: date | None = None,
) -> bool:
    """Check whether a local snapshot is broad, long, and fresh enough to train."""

    return _multifactor_snapshot_issue(
        frame,
        settings,
        required_data_as_of=required_data_as_of,
    ) is None


def _multifactor_snapshot_issue(
    frame: Any | None,
    settings: MultifactorStrategyConfig,
    *,
    required_data_as_of: date | None = None,
) -> str | None:
    """Return a human-readable reason why a snapshot needs replacement."""

    if frame is None or len(frame) == 0:
        return "no market snapshot is available"
    symbol_count = int(frame["symbol"].nunique())
    date_count = int(frame["timestamp"].nunique())
    if symbol_count < settings.training.min_cross_section:
        return (
            f"snapshot has {symbol_count} symbols, below the required "
            f"{settings.training.min_cross_section}"
        )
    required_dates = (
        settings.universe.min_history_days
        + settings.training.label_horizon
        + max(settings.training.lookback_days, settings.training.min_ic_days)
    )
    if date_count < required_dates:
        return (
            f"snapshot has {date_count} trading dates, below the required "
            f"{required_dates}"
        )
    actual_data_as_of = data_as_of(frame)
    if (
        required_data_as_of is not None
        and actual_data_as_of is not None
        and actual_data_as_of < required_data_as_of
    ):
        return (
            f"snapshot data ends {actual_data_as_of.isoformat()}, before the latest "
            f"completed trading day {required_data_as_of.isoformat()}"
        )
    return None


def _multifactor_download_params(
    params: Mapping[str, Any],
    settings: MultifactorStrategyConfig,
) -> dict[str, Any]:
    """Build an all-A download request from the multifactor universe settings."""

    universe = settings.universe
    return {
        "provider": "akshare",
        "adjustment": "qfq",
        "start": params.get("start"),
        "end": params.get("end"),
        "execution_prices": params.get("execution_prices", True),
        "universe": {
            "mode": "all_a",
            "max_symbols": universe.max_symbols,
            "candidate_pool_size": universe.candidate_pool_size,
            "min_history_days": universe.min_history_days,
            "min_avg_dollar_volume": universe.min_avg_dollar_volume,
            "exclude_st": universe.exclude_st,
            "exclude_bj": universe.exclude_bj,
            "include_chi_next": universe.include_chi_next,
            "include_star": universe.include_star,
        },
    }


def _freshness_blocked_summary(
    context: RunContext,
    *,
    reason: str,
    expected_data_as_of: date,
    actual_data_as_of: date | None,
    freshness_source: str,
    auto_download: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "status": "BLOCKED",
        "data_freshness_error": reason,
        "expected_data_as_of": expected_data_as_of.isoformat(),
        "snapshot_data_as_of": (
            actual_data_as_of.isoformat() if actual_data_as_of is not None else None
        ),
        "freshness_source": freshness_source,
        "auto_download": auto_download,
        "artifacts": ["report.html"],
    }
    write_status(context.run_dir, "BLOCKED", payload)
    write_run_report(context.run_dir)
    return payload


def _selection_task(params: Mapping[str, Any], *, data_dir: Path, configs_dir: Path, db_path: Path) -> Task:
    def run(context: RunContext) -> dict[str, Any]:
        settings = _multifactor_settings(configs_dir, params)
        snapshot_error: Exception | None = None
        try:
            frame, manifest = load_snapshot(data_dir)
        except (FileNotFoundError, ValueError) as exc:
            frame, manifest, snapshot_error = None, None, exc

        auto_download: dict[str, Any] | None = None
        expected_data_as_of: date | None = None
        freshness_source: str | None = None
        refresh_reason: str | None = None
        provider = str(params.get("provider") or "akshare").strip().lower()
        should_auto_download = (
            settings.universe.auto_download
            and settings.universe.mode == "all_a"
            and provider != "local"
        )
        if should_auto_download:
            refresh_reason = _multifactor_snapshot_issue(frame, settings)
            if refresh_reason is None:
                expected_data_as_of, freshness_source = latest_completed_trading_date()
                refresh_reason = _multifactor_snapshot_issue(
                    frame,
                    settings,
                    required_data_as_of=expected_data_as_of,
                )
        if should_auto_download and refresh_reason is not None:
            write_status(
                context.run_dir,
                "RUNNING",
                {
                    "phase": "DOWNLOADING_MARKET_DATA",
                    "message": f"正在刷新全市场行情：{refresh_reason}",
                },
            )
            download_params = _multifactor_download_params(params, settings)
            download_summary = _download_task(
                download_params,
                data_dir=data_dir,
                configs_dir=configs_dir,
                finalize=False,
            )(context)
            frame, manifest = load_snapshot(data_dir)
            auto_download = {
                "trigger": refresh_reason,
                "snapshot": manifest.get("snapshot_id"),
                "rows": download_summary.get("rows"),
                "universe_size": download_summary.get("universe_size"),
                "start": download_summary.get("start"),
                "end": download_summary.get("end"),
                "data_as_of": download_summary.get("data_as_of"),
                "failed_symbol_count": len(download_summary.get("failed_symbols", {})),
            }
            if (context.run_dir / "download_failures.json").is_file():
                auto_download["failure_report"] = "download_failures.json"
            expected_data_as_of, freshness_source = latest_completed_trading_date()
            post_download_issue = _multifactor_snapshot_issue(
                frame,
                settings,
                required_data_as_of=expected_data_as_of,
            )
            if post_download_issue is not None:
                return _freshness_blocked_summary(
                    context,
                    reason=post_download_issue,
                    expected_data_as_of=expected_data_as_of,
                    actual_data_as_of=data_as_of(frame),
                    freshness_source=freshness_source,
                    auto_download=auto_download,
                )

        if frame is None or manifest is None:
            if snapshot_error is not None:
                raise snapshot_error
            raise FileNotFoundError("no archived market snapshot found; run download first")
        write_status(
            context.run_dir,
            "RUNNING",
            {"phase": "COMPUTING_FACTORS", "message": "正在训练因子并生成横截面选股结果"},
        )
        frame, quality = _quality_gate(context, frame, configs_dir, params)
        if quality.blocked:
            return _quality_blocked_summary(context, quality)
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
            "data_quality.json",
            "quality_exclusions.csv",
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
            "auto_download": auto_download,
            "snapshot_data_as_of": data_as_of(frame).isoformat() if data_as_of(frame) else None,
            "expected_data_as_of": (
                expected_data_as_of.isoformat() if expected_data_as_of is not None else None
            ),
            "freshness_source": freshness_source,
            "quality_excluded": len(quality.excluded_symbols),
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


def market_symbols(frame: Any) -> list[str]:
    """Return the symbols available in the current market snapshot."""

    return sorted(str(item) for item in frame["symbol"].unique())


def market_figure(
    frame: Any, symbol: str | None = None
) -> tuple[dict[str, Any], str, str]:
    """Build a candlestick + volume figure for the requested snapshot symbol."""

    symbols = market_symbols(frame)
    if not symbols:
        raise ValueError("market snapshot contains no symbols")
    requested = str(symbol).strip() if symbol else None
    selected = requested if requested in symbols else symbols[0]
    data = frame.loc[frame["symbol"].astype(str) == selected].sort_values("timestamp")
    figure = {
        "data": [
            {
                "type": "candlestick",
                "name": selected,
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
            "title": {"text": f"{selected} 日线"},
            "xaxis": {"rangeslider": {"visible": False}},
            "yaxis": {"title": {"text": "price"}, "domain": [0.28, 1.0]},
            "yaxis2": {"title": {"text": "volume"}, "domain": [0.0, 0.2], "anchor": "x"},
            "legend": {"orientation": "h"},
        },
    }
    message = f"共 {len(data)} 根 bar；快照标的：{', '.join(symbols)}"
    if requested and requested not in symbols:
        message = f"标的不在当前快照中：{requested}；已显示 {selected}。{message}"
    return figure, message, selected


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
