"""Strategy protocol shared by rule-based and future model strategies."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from ..domain import TargetPosition


class TargetStrategy(Protocol):
    model_version: str

    def generate_targets(self, features: Any, portfolio: Any, config: Any) -> Sequence[TargetPosition]:
        """Generate target positions without placing orders or changing state."""
