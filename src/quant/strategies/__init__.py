"""Pluggable target-position strategies."""

from .baseline import MovingAverageStrategy, MovingAverageStrategyConfig
from .protocol import TargetStrategy

__all__ = ["MovingAverageStrategy", "MovingAverageStrategyConfig", "TargetStrategy"]
