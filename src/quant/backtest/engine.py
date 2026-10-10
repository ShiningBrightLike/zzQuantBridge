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
    stats = portfolio.stats(silence_warnings=True)
    metrics = {str(key): value for key, value in stats.items()}
    metrics["total_return"] = portfolio.total_return()
    metrics["max_drawdown"] = portfolio.max_drawdown()
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

    return pd.DataFrame(frame.to_numpy(dtype="float64"), index=frame.index, columns=frame.columns)


def _bool_frame(frame: Any) -> Any:
    import pandas as pd

    return pd.DataFrame(
        frame.to_numpy(dtype=bool, na_value=False),
        index=frame.index,
        columns=frame.columns,
    )
