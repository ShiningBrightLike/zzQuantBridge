"""Mandatory pre-trade risk checks from the project plan (section 7.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
from typing import Any, Mapping, Sequence

from ..domain import TargetPosition, TradeSuggestion
from ..portfolio.store import PortfolioSnapshot


def _number(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


@dataclass(frozen=True)
class RiskPolicy:
    """Configurable limits; each one maps to a check in :func:`evaluate_risk`."""

    max_single_weight: float = 0.2
    max_gross_weight: float = 1.0
    cash_buffer: float = 0.1
    max_turnover: float = 0.5
    min_order_value: float = 0.0
    lot_size: int = 100
    max_data_age_days: int | None = 5
    industries: Mapping[str, str] = field(default_factory=dict)
    industry_limits: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.max_single_weight <= 1:
            raise ValueError("max_single_weight must be between 0 and 1")
        if self.max_gross_weight <= 0:
            raise ValueError("max_gross_weight must be positive")
        if not 0 <= self.cash_buffer < 1:
            raise ValueError("cash_buffer must be between 0 and 1")
        if self.max_turnover < 0:
            raise ValueError("max_turnover must not be negative")
        if self.min_order_value < 0:
            raise ValueError("min_order_value must not be negative")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.max_data_age_days is not None and self.max_data_age_days < 0:
            raise ValueError("max_data_age_days must not be negative")

    @classmethod
    def from_config(cls, payload: Mapping[str, Any] | None) -> "RiskPolicy":
        data = dict(payload or {})
        return cls(
            max_single_weight=_number(data.get("max_single_weight"), 0.2),
            max_gross_weight=_number(data.get("max_gross_weight"), 1.0),
            cash_buffer=_number(data.get("cash_buffer"), 0.1),
            max_turnover=_number(data.get("max_turnover"), 0.5),
            min_order_value=_number(data.get("min_order_value"), 0.0),
            lot_size=int(_number(data.get("lot_size"), 100)),
            max_data_age_days=_optional_int(data.get("max_data_age_days", 5)),
            industries={str(key): str(value) for key, value in (data.get("industries") or {}).items()},
            industry_limits={
                str(key): float(value) for key, value in (data.get("industry_limits") or {}).items()
            },
        )


@dataclass(frozen=True)
class RiskReport:
    status: str
    reasons: tuple[str, ...] = ()
    symbol_flags: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCKED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "symbol_flags": {key: list(value) for key, value in self.symbol_flags.items()},
        }


def evaluate_risk(
    *,
    targets: Sequence[TargetPosition],
    suggestions: Sequence[TradeSuggestion],
    portfolio: PortfolioSnapshot,
    prices: Mapping[str, float],
    policy: RiskPolicy,
    as_of: date,
    data_as_of: date | None = None,
) -> RiskReport:
    """Check every limit in the plan; any failure blocks instead of passing."""

    reasons: list[str] = []
    flags: dict[str, list[str]] = {}

    def flag(symbol: str, message: str) -> None:
        flags.setdefault(symbol, [])
        if message not in flags[symbol]:
            flags[symbol].append(message)

    gross_weight = sum(target.target_weight for target in targets)
    if gross_weight > policy.max_gross_weight + 1e-9:
        reasons.append(f"target gross weight {gross_weight:.4f} exceeds limit {policy.max_gross_weight:.4f}")

    for target in targets:
        if target.target_weight > policy.max_single_weight + 1e-9:
            flag(
                target.symbol,
                f"target weight {target.target_weight:.4f} exceeds single-name limit "
                f"{policy.max_single_weight:.4f}",
            )

    if policy.industries and policy.industry_limits:
        by_industry: dict[str, float] = {}
        members: dict[str, list[str]] = {}
        for target in targets:
            industry = policy.industries.get(target.symbol)
            if not industry:
                continue
            by_industry[industry] = by_industry.get(industry, 0.0) + target.target_weight
            members.setdefault(industry, []).append(target.symbol)
        for industry, weight in by_industry.items():
            limit = policy.industry_limits.get(industry)
            if limit is not None and weight > limit + 1e-9:
                for symbol in members[industry]:
                    flag(symbol, f"industry {industry} weight {weight:.4f} exceeds limit {limit:.4f}")

    for symbol in {item.symbol for item in suggestions}:
        price = prices.get(symbol)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            flag(symbol, f"missing or invalid price for {symbol}")

    if data_as_of is not None and policy.max_data_age_days is not None:
        age = (as_of - data_as_of).days
        if age > policy.max_data_age_days:
            reasons.append(
                f"market data is {age} day(s) old (last bar {data_as_of.isoformat()}); "
                f"limit is {policy.max_data_age_days}"
            )

    market_value = portfolio.cash + sum(
        portfolio.positions.get(symbol, 0) * float(prices[symbol])
        for symbol in portfolio.positions
        if symbol in prices
    )
    if market_value > 0:
        traded_value = 0.0
        cash_delta = 0.0
        for item in suggestions:
            if item.quantity <= 0 or item.reference_price is None:
                continue
            value = item.quantity * float(item.reference_price)
            traded_value += abs(value)
            cash_delta += -value if item.action == "BUY" else value
        turnover = traded_value / market_value
        if turnover > policy.max_turnover + 1e-9:
            reasons.append(f"turnover {turnover:.4f} exceeds limit {policy.max_turnover:.4f}")
        projected_cash_ratio = (portfolio.cash + cash_delta) / market_value
        if projected_cash_ratio < policy.cash_buffer - 1e-9:
            reasons.append(
                f"projected cash {projected_cash_ratio:.4f} of portfolio falls below buffer "
                f"{policy.cash_buffer:.4f}"
            )

    return RiskReport(
        status="BLOCKED" if reasons or flags else "PASS",
        reasons=tuple(reasons),
        symbol_flags={key: tuple(value) for key, value in flags.items()},
    )
