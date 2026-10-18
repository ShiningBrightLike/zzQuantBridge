"""vectorbt backtest boundary."""

from .engine import BacktestConfig, BacktestResult, crossovers, run_signal_backtest
from .multifactor import (
    CrossSectionalBacktestConfig,
    CrossSectionalBacktestResult,
    performance_metrics,
    run_cross_sectional_backtest,
    simulate_target_weights,
    walk_forward_targets,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "CrossSectionalBacktestConfig",
    "CrossSectionalBacktestResult",
    "performance_metrics",
    "crossovers",
    "run_cross_sectional_backtest",
    "run_signal_backtest",
    "simulate_target_weights",
    "walk_forward_targets",
]
