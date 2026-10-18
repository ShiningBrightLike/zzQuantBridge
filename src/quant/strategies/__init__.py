"""Pluggable target-position strategies."""

from .baseline import MovingAverageStrategy, MovingAverageStrategyConfig
from .multifactor import MultifactorStrategy, MultifactorStrategyConfig, SelectionResult
from .protocol import TargetStrategy

__all__ = [
    "MovingAverageStrategy",
    "MovingAverageStrategyConfig",
    "MultifactorStrategy",
    "MultifactorStrategyConfig",
    "SelectionResult",
    "TargetStrategy",
]
