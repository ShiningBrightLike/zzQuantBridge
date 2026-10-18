"""Walk-forward selection and deterministic target-weight backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
from typing import Any, Mapping, Sequence

from ..features.multifactor import (
    FACTOR_NAMES,
    compute_factor_panel,
    fit_factor_model,
    score_cross_section,
    select_from_scores,
)
from ..strategies.multifactor import MultifactorStrategyConfig, SelectionResult


@dataclass(frozen=True)
class CrossSectionalBacktestConfig:
    init_cash: float = 100_000.0
    fees: float = 0.0003
    slippage: float = 0.0005
    lot_size: int = 100
    signal_delay: int = 1
    min_order_value: float = 0.0
    min_oos_days: int = 504
    backtest_start: date | None = None

    def __post_init__(self) -> None:
        if self.init_cash <= 0:
            raise ValueError("init_cash must be positive")
        if self.fees < 0 or self.slippage < 0:
            raise ValueError("fees and slippage must not be negative")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.signal_delay < 0:
            raise ValueError("signal_delay must not be negative")
        if self.min_order_value < 0:
            raise ValueError("min_order_value must not be negative")
        if self.min_oos_days < 0:
            raise ValueError("min_oos_days must not be negative")


@dataclass
class CrossSectionalBacktestResult:
    equity: Any
    daily_returns: Any
    trades: Any
    selections: Any
    target_weights: Any
    raw_target_weights: Any
    model_metrics: Any
    skipped_rebalances: Any
    metrics: dict[str, Any] = field(default_factory=dict)


def _as_timestamp(value: Any) -> Any:
    import pandas as pd

    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def _price_matrix(frame: Any, column: str) -> Any:
    import pandas as pd

    return (
        frame.pivot_table(index="timestamp", columns="symbol", values=column, aggfunc="last")
        .sort_index()
        .astype(float)
    )


def walk_forward_targets(
    frame: Any,
    settings: MultifactorStrategyConfig,
    *,
    backtest_start: date | None = None,
    portfolio_value: float | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Build daily target weights by refitting the model at every rebalance."""

    import pandas as pd

    from ..data.universe import build_universe_membership

    membership = build_universe_membership(frame, settings.universe)
    panel = compute_factor_panel(
        frame,
        settings.factors,
        label_horizon=settings.training.label_horizon,
    ).merge(
        membership[["symbol", "timestamp", "tradable"]],
        on=["symbol", "timestamp"],
        how="left",
    )
    dates = list(pd.to_datetime(panel["timestamp"], utc=True).drop_duplicates().sort_values())
    if len(dates) <= 2:
        raise ValueError("not enough dates to run a walk-forward multifactor backtest")
    first_index = min(len(dates) - 1, max(1, settings.training.lookback_days))
    if backtest_start is not None:
        cutoff = _as_timestamp(backtest_start)
        eligible = [index for index, value in enumerate(dates) if value >= cutoff]
        if eligible:
            first_index = min(first_index, eligible[0])
    rebalance_indices = list(range(first_index, len(dates), settings.rebalance_days))
    if not rebalance_indices:
        raise ValueError("backtest_start leaves no rebalance dates")

    weights = pd.DataFrame(0.0, index=pd.DatetimeIndex(dates), columns=sorted(frame["symbol"].astype(str).unique()))
    selection_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    metric_frames: list[Any] = []
    previous: list[str] = []
    current_weights: dict[str, float] = {}
    successful_rebalances = 0
    for index in rebalance_indices:
        stamp = dates[index]
        tradable_panel = panel.loc[panel["tradable"].fillna(False)]
        try:
            model = fit_factor_model(tradable_panel, stamp, settings.training)
            scores = score_cross_section(tradable_panel, model, stamp)
            if settings.affordability_filter and "close" in scores.columns:
                budget = (float(portfolio_value or 100_000.0) * (1.0 - settings.cash_buffer) / settings.top_n)
                scores = scores.loc[scores["close"] * settings.lot_size <= budget].copy()
                if scores.empty:
                    continue
            selected = select_from_scores(
                scores,
                top_n=settings.top_n,
                hold_buffer=settings.hold_buffer,
                max_single_weight=settings.max_single_weight,
                cash_buffer=settings.cash_buffer,
                min_score=settings.min_score,
                previous_symbols=previous,
            )
        except ValueError as exc:
            skipped_rows.append({"timestamp": stamp, "reason": str(exc)})
            continue
        successful_rebalances += 1
        previous = list(selected.loc[selected["selected"], "symbol"].astype(str))
        current_weights = {
            str(row.symbol): float(row.target_weight)
            for row in selected.loc[selected["selected"]].itertuples(index=False)
        }
        for row in selected.loc[selected["selected"]].itertuples(index=False):
            selection_rows.append(
                {
                    "timestamp": stamp,
                    "symbol": str(row.symbol),
                    "rank": int(row.rank),
                    "score": float(row.score),
                    "valid_factor_count": int(row.valid_factor_count),
                    "target_weight": float(row.target_weight),
                    "model_version": model.model_version,
                }
            )
        metrics = model.metrics.copy()
        metrics["weight"] = metrics["factor"].map(model.weights).fillna(0.0)
        metrics["used"] = metrics["factor"].isin(model.factor_names)
        metrics["rebalance_date"] = stamp
        metrics["model_version"] = model.model_version
        metric_frames.append(metrics)
        # The selected weights become effective after this close. The
        # execution simulator shifts them by signal_delay.
        next_index = index + settings.rebalance_days
        weights.loc[stamp:dates[min(next_index - 1, len(dates) - 1)]] = 0.0
        for symbol, weight in current_weights.items():
            if symbol in weights.columns:
                weights.loc[stamp:dates[min(next_index - 1, len(dates) - 1)], symbol] = weight

    if successful_rebalances == 0:
        raise ValueError("no rebalance date had enough point-in-time history to fit the model")
    selections = pd.DataFrame(selection_rows)
    model_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows)
    return weights, selections, model_metrics, panel, skipped


def simulate_target_weights(
    close: Any,
    open_prices: Any,
    target_weights: Any,
    config: CrossSectionalBacktestConfig | None = None,
) -> tuple[Any, Any, Any]:
    """Deterministic long-only target-weight simulation with A-share lots."""

    import numpy as np
    import pandas as pd

    settings = config or CrossSectionalBacktestConfig()
    close = close.sort_index().astype(float)
    open_prices = open_prices.reindex(index=close.index, columns=close.columns).astype(float)
    open_prices = open_prices.fillna(close)
    weights = target_weights.reindex(index=close.index, columns=close.columns).fillna(0.0).astype(float)
    if settings.signal_delay:
        weights = weights.shift(settings.signal_delay).fillna(0.0)
    active = weights.abs().sum(axis=1) > 1e-12
    if active.any():
        first_active = active.idxmax()
        close = close.loc[first_active:]
        open_prices = open_prices.loc[first_active:]
        weights = weights.loc[first_active:]

    cash = float(settings.init_cash)
    shares = pd.Series(0, index=close.columns, dtype="int64")
    equity_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for stamp in close.index:
        prices = open_prices.loc[stamp]
        close_prices = close.loc[stamp]
        valid = prices.notna() & (prices > 0)
        gross_before = float((shares * prices.where(valid, 0.0)).sum())
        equity_open = cash + gross_before
        traded_value = 0.0
        for symbol in close.columns:
            target = float(weights.at[stamp, symbol])
            price = prices.get(symbol)
            if not valid.get(symbol, False):
                # A suspended or delisted symbol keeps its last known shares;
                # the portfolio is marked with the available close below.
                continue
            desired = equity_open * target
            desired_qty = int(math.floor(desired / float(price) / settings.lot_size) * settings.lot_size)
            current_qty = int(shares.at[symbol])
            delta = desired_qty - current_qty
            if delta > 0:
                # Buys are constrained by cash after fees.
                affordable = int(math.floor(cash / (float(price) * (1 + settings.slippage) * (1 + settings.fees)) / settings.lot_size) * settings.lot_size)
                delta = min(delta, max(0, affordable))
            if delta == 0:
                continue
            execution_price = float(price) * (1 + settings.slippage if delta > 0 else 1 - settings.slippage)
            value = abs(delta) * execution_price
            if value < settings.min_order_value:
                continue
            fee = value * settings.fees
            if delta > 0 and cash < value + fee:
                continue
            cash += -value - fee if delta > 0 else value - fee
            shares.at[symbol] = current_qty + delta
            traded_value += value
            trades.append(
                {
                    "timestamp": stamp,
                    "symbol": symbol,
                    "side": "BUY" if delta > 0 else "SELL",
                    "quantity": abs(int(delta)),
                    "price": execution_price,
                    "value": value,
                    "fee": fee,
                    "target_weight": target,
                }
            )
        marked = (shares * close_prices.where(close_prices.notna(), 0.0)).sum()
        equity_rows.append(
            {
                "timestamp": stamp,
                "cash": cash,
                "market_value": float(marked),
                "equity": float(cash + marked),
                "turnover": traded_value / equity_open if equity_open > 0 else 0.0,
            }
        )
    equity = pd.DataFrame(equity_rows).set_index("timestamp").sort_index()
    equity["returns"] = equity["equity"].pct_change().fillna(0.0)
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1.0
    trade_frame = pd.DataFrame(trades)
    return equity, trade_frame, weights


def performance_metrics(equity: Any, trades: Any, benchmark_returns: Any | None = None) -> dict[str, Any]:
    import numpy as np

    values = equity["equity"].astype(float)
    returns = equity["returns"].astype(float).iloc[1:]
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0) if len(values) > 1 else 0.0
    periods = max(1, len(returns))
    annual_return = float((1.0 + total_return) ** (252.0 / periods) - 1.0) if total_return > -1 else -1.0
    volatility = float(returns.std(ddof=0) * math.sqrt(252)) if len(returns) else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=0) * math.sqrt(252)) if len(returns) and returns.std(ddof=0) else 0.0
    max_drawdown = float(equity["drawdown"].min()) if len(equity) else 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown else 0.0
    result: dict[str, Any] = {
        "total_return": round(total_return, 6),
        "annual_return": round(annual_return, 6),
        "annual_volatility": round(volatility, 6),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(max_drawdown, 6),
        "calmar": round(calmar, 6),
        "win_rate": round(float((returns > 0).mean()), 6) if len(returns) else 0.0,
        "trade_count": int(len(trades)),
        "turnover": round(float(equity["turnover"].mean()), 6) if len(equity) else 0.0,
        "fees_paid": round(float(trades["fee"].sum()), 2) if len(trades) else 0.0,
        "average_invested_weight": round(
            float((equity["market_value"] / equity["equity"]).mean()), 6
        ) if len(equity) else 0.0,
        "active_days": int(len(equity)),
        "first_trade_date": str(equity.index[0].date()) if len(equity) else "",
    }
    if benchmark_returns is not None and len(benchmark_returns):
        benchmark_total = float((1.0 + benchmark_returns.fillna(0.0)).prod() - 1.0)
        result["benchmark_total_return"] = round(benchmark_total, 6)
        result["excess_total_return"] = round(total_return - benchmark_total, 6)
    return result


def run_cross_sectional_backtest(
    frame: Any,
    settings: MultifactorStrategyConfig,
    *,
    config: CrossSectionalBacktestConfig | None = None,
    backtest_start: date | None = None,
) -> CrossSectionalBacktestResult:
    """Run walk-forward selection and target-weight execution on one snapshot."""

    close = _price_matrix(frame, "close")
    open_prices = _price_matrix(frame, "open")
    weights, selections, model_metrics, panel, skipped = walk_forward_targets(
        frame,
        settings,
        backtest_start=backtest_start,
        portfolio_value=(config or CrossSectionalBacktestConfig()).init_cash,
    )
    equity, trades, shifted_weights = simulate_target_weights(
        close, open_prices, weights, config
    )
    benchmark = close.pct_change().mean(axis=1).reindex(equity.index)
    return CrossSectionalBacktestResult(
        equity=equity,
        daily_returns=equity["returns"],
        trades=trades,
        selections=selections,
        target_weights=shifted_weights,
        model_metrics=model_metrics,
        skipped_rebalances=skipped,
        raw_target_weights=weights,
        metrics=performance_metrics(equity, trades, benchmark),
    )
