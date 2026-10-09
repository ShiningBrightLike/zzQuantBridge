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
    if settings.signal_delay:
        entries = entries.shift(settings.signal_delay).fillna(False)
        exits = exits.shift(settings.signal_delay).fillna(False)
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=settings.init_cash,
        fees=settings.fees,
        slippage=settings.slippage,
        freq=settings.freq,
    )
    stats = portfolio.stats()
    metrics = {str(key): value for key, value in stats.items()}
    metrics["total_return"] = portfolio.total_return()
    metrics["max_drawdown"] = portfolio.max_drawdown()
    return BacktestResult(portfolio=portfolio, metrics=metrics)
