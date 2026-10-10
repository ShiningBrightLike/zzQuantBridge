"""The first explainable, long-only strategy implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..domain import TargetPosition


@dataclass(frozen=True)
class MovingAverageStrategyConfig:
    fast_window: int = 20
    slow_window: int = 60
    max_positions: int = 5
    max_single_weight: float = 0.2
    model_version: str = "ma-baseline-v1"

    def __post_init__(self) -> None:
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("moving-average windows must be positive")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if not 0 < self.max_single_weight <= 1:
            raise ValueError("max_single_weight must be between 0 and 1")


class MovingAverageStrategy:
    """Select symbols whose fast moving average is above their slow average."""

    def __init__(self, settings: MovingAverageStrategyConfig | None = None) -> None:
        self.settings = settings or MovingAverageStrategyConfig()
        self.model_version = self.settings.model_version

    def generate_targets(self, features: Any, portfolio: Any = None, config: Any = None) -> list[TargetPosition]:
        del portfolio
        settings = self._settings_for(config)
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for MovingAverageStrategy") from exc
        required = {"symbol", "timestamp", "close"}
        missing = required.difference(features.columns)
        if missing:
            raise ValueError(f"features missing columns: {', '.join(sorted(missing))}")
        frame = features.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.sort_values(["symbol", "timestamp"], kind="stable")
        grouped = frame.groupby("symbol", sort=True, group_keys=False)
        frame["fast_ma"] = grouped["close"].transform(
            lambda values: values.rolling(settings.fast_window, min_periods=settings.fast_window).mean()
        )
        frame["slow_ma"] = grouped["close"].transform(
            lambda values: values.rolling(settings.slow_window, min_periods=settings.slow_window).mean()
        )
        latest = frame.groupby("symbol", sort=True, as_index=False).tail(1).copy()
        eligible = latest.dropna(subset=["fast_ma", "slow_ma"])
        eligible = eligible[eligible["fast_ma"] > eligible["slow_ma"]].sort_values(
            ["fast_ma", "symbol"], ascending=[False, True], kind="stable"
        )
        selected = eligible.head(settings.max_positions)
        weight = min(1.0 / len(selected), settings.max_single_weight) if len(selected) else 0.0
        selected_symbols = set(selected["symbol"])
        targets: list[TargetPosition] = []
        for row in latest.itertuples(index=False):
            is_selected = row.symbol in selected_symbols
            reason = (
                f"fast MA ({settings.fast_window}) above slow MA ({settings.slow_window})"
                if is_selected
                else "trend filter not satisfied or warm-up incomplete"
            )
            targets.append(
                TargetPosition(
                    symbol=str(row.symbol),
                    timestamp=row.timestamp.to_pydatetime(),
                    target_weight=weight if is_selected else 0.0,
                    reason=reason,
                    model_version=settings.model_version,
                )
            )
        return targets

    def _settings_for(self, config: Any) -> MovingAverageStrategyConfig:
        """Honour the per-call config when the caller passes one."""

        if config is None:
            return self.settings
        if isinstance(config, MovingAverageStrategyConfig):
            return config
        if isinstance(config, Mapping):
            payload = asdict(self.settings)
            for key in tuple(payload):
                if config.get(key) is not None:
                    payload[key] = config[key]
            return MovingAverageStrategyConfig(**payload)
        raise TypeError("config must be a MovingAverageStrategyConfig or a mapping")
