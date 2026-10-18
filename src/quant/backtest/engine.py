"""Small vectorbt wrapper with explicit execution assumptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BacktestConfig:
    init_cash: float = 100_000.0
    fees: float = 0.0003
    slippage: float = 0.0005
    signal_delay: int = 1
    freq: str = "1D"

    def __post_init__(self) -> None:
        if self.init_cash <= 0:
            raise ValueError("init_cash must be positive")
        if self.fees < 0 or self.slippage < 0:
            raise ValueError("fees and slippage must not be negative")
        if self.signal_delay < 0:
            raise ValueError("signal_delay must not be negative")


@dataclass
class BacktestResult:
    portfolio: Any
    metrics: dict[str, Any] = field(default_factory=dict)


def run_signal_backtest(close: Any, entries: Any, exits: Any, config: BacktestConfig | None = None) -> BacktestResult:
    """Run signal-based vectorbt simulation with an explicit signal delay.

    Signals are shifted forward before execution, so a signal observed at the
    close of bar *t* executes on bar *t+1* by default. No strategy code is
    allowed to inspect data after the signal timestamp.
    """

    settings = config or BacktestConfig()
    try:
        import vectorbt as vbt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("vectorbt is required for backtests; install the backtest extra") from exc
    # numba rejects tz-aware DatetimeIndex and object dtypes. pandas 3 turns
    # boolean signal frames into object columns after ``shift``/``fillna``, so
    # rebuild all frames as plain numpy dtypes before they reach vectorbt.
    close = _plain_frame(close)
    entries = _plain_frame(entries)
    exits = _plain_frame(exits)
    if settings.signal_delay:
        entries = entries.shift(settings.signal_delay)
        exits = exits.shift(settings.signal_delay)
    close = _float_frame(close)
    entries = _bool_frame(entries)
    exits = _bool_frame(exits)
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=settings.init_cash,
        fees=settings.fees,
        slippage=settings.slippage,
        freq=settings.freq,
    )
    # vectorbt 0.27's aggregate ``stats()`` mutates internal NumPy arrays that
    # newer NumPy versions expose as read-only. Compute the small set of
    # portfolio-level metrics we actually persist instead of relying on that
    # fragile aggregation path.
    def scalar(value: Any) -> float:
        if hasattr(value, "mean"):
            return float(value.mean())
        return float(value)

    total_return = scalar(portfolio.total_return())
    drawdown = scalar(portfolio.max_drawdown())
    import numpy as np

    returns = np.asarray(portfolio.returns(), dtype="float64").ravel()
    returns = returns[np.isfinite(returns)]
    periods = max(1, int(returns.size))
    mean_return = float(returns.mean()) if returns.size else 0.0
    dispersion = float(returns.std(ddof=0)) if returns.size else 0.0
    volatility = dispersion * (252**0.5)
    sharpe = mean_return / dispersion * (252**0.5) if dispersion > 0 else 0.0
    annual_return = (
        float((1.0 + total_return) ** (252.0 / periods) - 1.0)
        if total_return > -1.0
        else -1.0
    )
    trades = portfolio.trades.records_readable
    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown,
        "trade_count": int(len(trades)) if trades is not None else 0,
    }
    return BacktestResult(portfolio=portfolio, metrics=metrics)


def _plain_frame(frame: Any) -> Any:
    """Drop timezone information so vectorbt's numba kernels accept the frame."""

    if hasattr(frame, "to_frame") and not hasattr(frame, "columns"):
        frame = frame.to_frame()
    index = getattr(frame, "index", None)
    if index is not None and getattr(index, "tz", None) is not None:
        frame = frame.copy()
        frame.index = index.tz_localize(None)
    return frame


def _float_frame(frame: Any) -> Any:
    import pandas as pd

    values = frame.to_numpy(dtype="float64", copy=True)
    return pd.DataFrame(values.copy(), index=frame.index, columns=frame.columns)


def _bool_frame(frame: Any) -> Any:
    import pandas as pd

    values = frame.to_numpy(dtype=bool, na_value=False, copy=True)
    return pd.DataFrame(
        values.copy(),
        index=frame.index,
        columns=frame.columns,
    )


def crossovers(signals: Any) -> Any:
    """Keep only the bars where a boolean signal turns on.

    pandas turns a shifted boolean frame into ``object`` dtype, and ``~`` on
    those values yields Python ints (``True`` -> -2, ``False`` -> -1) which are
    both truthy. That silently disables the filter, so compare plain numpy
    arrays instead.
    """

    import pandas as pd

    current = signals.to_numpy(dtype=bool, na_value=False)
    previous = signals.shift(1).to_numpy(dtype=bool, na_value=False)
    return pd.DataFrame(current & ~previous, index=signals.index, columns=signals.columns)
