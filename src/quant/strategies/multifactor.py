"""Cross-sectional A-share multifactor selection strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Mapping, Sequence

from ..data.universe import (
    UniverseConfig,
    build_universe_membership,
    current_tradable_symbols,
)
from ..domain import TargetPosition
from ..features.multifactor import (
    FactorConfig,
    FactorModel,
    FactorTrainingConfig,
    compute_factor_panel,
    fit_factor_model,
    score_cross_section,
    select_from_scores,
)


@dataclass(frozen=True)
class MultifactorStrategyConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    training: FactorTrainingConfig = field(default_factory=FactorTrainingConfig)
    top_n: int = 20
    hold_buffer: int = 30
    max_single_weight: float = 0.05
    cash_buffer: float = 0.10
    min_score: float = 0.0
    rebalance_days: int = 20
    lot_size: int = 100
    affordability_filter: bool = True
    model_version: str = "multifactor-v1"

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.hold_buffer < self.top_n:
            raise ValueError("hold_buffer must be at least top_n")
        if not 0 < self.max_single_weight <= 1:
            raise ValueError("max_single_weight must be between zero and one")
        if not 0 <= self.cash_buffer < 1:
            raise ValueError("cash_buffer must be in [0, 1)")
        if not math.isfinite(self.min_score):
            raise ValueError("min_score must be finite")
        if self.rebalance_days <= 0:
            raise ValueError("rebalance_days must be positive")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "MultifactorStrategyConfig":
        data = dict(payload or {})
        universe = data.get("universe") or {}
        factors = data.get("factors") or {}
        training = data.get("training") or {}
        selection = data.get("selection") or {}
        if not isinstance(universe, Mapping):
            raise ValueError("multifactor universe config must be a mapping")
        if not isinstance(factors, Mapping):
            raise ValueError("multifactor factors config must be a mapping")
        if not isinstance(training, Mapping):
            raise ValueError("multifactor training config must be a mapping")
        if not isinstance(selection, Mapping):
            raise ValueError("multifactor selection config must be a mapping")
        universe_payload = {
            key: universe[key]
            for key in (
                "mode",
                "auto_download",
                "max_symbols",
                "candidate_pool_size",
                "min_history_days",
                "min_avg_dollar_volume",
                "exclude_st",
                "exclude_bj",
                "include_chi_next",
                "include_star",
            )
            if key in universe
        }
        factor_payload = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in factors.items()
            if key in {
                "momentum_windows",
                "volatility_windows",
                "reversal_window",
                "liquidity_window",
                "long_liquidity_window",
                "rsi_window",
                "high_window",
            }
        }
        training_payload = {
            key: training[key]
            for key in (
                "lookback_days",
                "label_horizon",
                "min_cross_section",
                "min_ic_days",
                "min_abs_ic",
                "min_abs_icir",
                "weight_alpha",
                "max_factors",
            )
            if key in training
        }
        selection_payload = {
            key: selection[key]
            for key in (
                "top_n",
                "hold_buffer",
                "max_single_weight",
                "cash_buffer",
                "min_score",
                "rebalance_days",
                "lot_size",
                "affordability_filter",
            )
            if key in selection
        }
        return cls(
            universe=UniverseConfig(**universe_payload),
            factors=FactorConfig(**factor_payload),
            training=FactorTrainingConfig(**training_payload),
            model_version=str(data.get("model_version") or "multifactor-v1"),
            **selection_payload,
        )


@dataclass
class SelectionResult:
    """Everything needed to audit one cross-sectional selection run."""

    as_of: datetime
    model: FactorModel
    panel: Any
    membership: Any
    scores: Any
    selected: Any
    targets: list[TargetPosition]
    tradable_symbols: list[str]


class MultifactorStrategy:
    """Rank a point-in-time A-share universe with rolling IC/ICIR weights."""

    def __init__(self, settings: MultifactorStrategyConfig | None = None) -> None:
        self.settings = settings or MultifactorStrategyConfig()
        self.model_version = self.settings.model_version

    def _settings_for(self, config: Any) -> MultifactorStrategyConfig:
        if config is None:
            return self.settings
        if isinstance(config, MultifactorStrategyConfig):
            return config
        if isinstance(config, Mapping):
            return MultifactorStrategyConfig.from_mapping(config)
        raise TypeError("config must be a MultifactorStrategyConfig or a mapping")

    def run_selection(
        self,
        features: Any,
        portfolio: Any = None,
        config: Any = None,
    ) -> SelectionResult:
        settings = self._settings_for(config)
        membership = build_universe_membership(features, settings.universe)
        panel = compute_factor_panel(
            features,
            settings.factors,
            label_horizon=settings.training.label_horizon,
        )
        panel = panel.merge(
            membership[["symbol", "timestamp", "tradable"]],
            on=["symbol", "timestamp"],
            how="left",
        )
        as_of = panel["timestamp"].max()
        tradable = current_tradable_symbols(membership, as_of)
        if not tradable:
            raise ValueError("no stocks satisfy the point-in-time universe filters")
        membership = membership.loc[membership["symbol"].isin(tradable)].copy()
        model = fit_factor_model(panel, as_of, settings.training)
        full_scores = score_cross_section(panel.loc[panel["tradable"].fillna(False)], model, as_of)
        full_scores = full_scores.loc[full_scores["symbol"].isin(tradable)].copy()
        scores = full_scores.copy()
        if "rank" not in scores.columns:
            scores["rank"] = scores["score"].rank(ascending=False, method="first").astype(int)
        if settings.affordability_filter and "close" in scores.columns:
            price_by_symbol = {
                str(row.symbol): float(row.close)
                for row in scores.itertuples(index=False)
                if float(row.close) > 0
            }
            cash = float(getattr(portfolio, "cash", 0.0)) if portfolio is not None else 0.0
            position_map = getattr(portfolio, "positions", {}) if portfolio is not None else {}
            market_value = cash + sum(
                int(quantity) * price_by_symbol.get(str(symbol), 0.0)
                for symbol, quantity in (position_map or {}).items()
            )
            if market_value > 0:
                position_budget = (
                    market_value * (1.0 - settings.cash_buffer) / settings.top_n
                )
                affordable = scores["close"] * settings.lot_size <= position_budget
                if not affordable.any():
                    raise ValueError(
                        "no tradable stock fits one A-share lot within the per-position budget; "
                        "increase initial cash, reduce top_n, or raise selection.lot_size budget"
                    )
                scores = scores.loc[affordable].copy()
        previous_symbols: Sequence[str] = ()
        positions = getattr(portfolio, "positions", None) if portfolio is not None else None
        if isinstance(positions, Mapping):
            previous_symbols = [str(symbol) for symbol, quantity in positions.items() if int(quantity) > 0]
        selected = select_from_scores(
            scores,
            top_n=settings.top_n,
            hold_buffer=settings.hold_buffer,
            max_single_weight=settings.max_single_weight,
            cash_buffer=settings.cash_buffer,
            min_score=settings.min_score,
            previous_symbols=previous_symbols,
        )
        targets: list[TargetPosition] = []
        stamp = as_of.to_pydatetime() if hasattr(as_of, "to_pydatetime") else as_of
        for row in selected.loc[selected["selected"]].itertuples(index=False):
            reason = (
                f"multifactor rank {int(row.rank)} score {float(row.score):.4f}; "
                f"factors {int(row.valid_factor_count)}; model {model.model_version}"
            )
            targets.append(
                TargetPosition(
                    symbol=str(row.symbol),
                    timestamp=stamp,
                    target_weight=float(row.target_weight),
                    reason=reason,
                    model_version=model.model_version,
                )
            )
        return SelectionResult(
            as_of=stamp,
            model=model,
            panel=panel,
            membership=membership,
            scores=full_scores,
            selected=selected,
            targets=targets,
            tradable_symbols=tradable,
        )

    def generate_targets(self, features: Any, portfolio: Any = None, config: Any = None) -> list[TargetPosition]:
        """Protocol-compatible entry point used by the shared suggestion layer."""

        return self.run_selection(features, portfolio, config).targets
