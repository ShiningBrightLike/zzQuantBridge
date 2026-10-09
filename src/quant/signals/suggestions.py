"""Deterministic target-position to manual-suggestion translation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

from ..domain import TargetPosition, TradeSuggestion
from ..portfolio.store import PortfolioSnapshot


def generate_suggestions(
    targets: Iterable[TargetPosition],
    portfolio: PortfolioSnapshot,
    prices: Mapping[str, float],
    *,
    lot_size: int = 100,
    min_order_value: float = 0.0,
) -> list[TradeSuggestion]:
    """Create suggestions without mutating the portfolio or treating them as fills."""

    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    if min_order_value < 0:
        raise ValueError("min_order_value must not be negative")
    target_list = list(targets)
    market_value = portfolio.cash + sum(
        portfolio.positions.get(symbol, 0) * _price(prices, symbol) for symbol in portfolio.positions
    )
    if market_value <= 0:
        raise ValueError("portfolio market value must be positive")

    by_symbol = {target.symbol: target for target in target_list}
    symbols = sorted(set(by_symbol) | set(portfolio.positions))
    suggestions: list[TradeSuggestion] = []
    for symbol in symbols:
        target = by_symbol.get(symbol)
        target_weight = target.target_weight if target else 0.0
        price = _price(prices, symbol)
        current_quantity = portfolio.positions.get(symbol, 0)
        desired_quantity = math.floor((market_value * target_weight) / price)
        delta = desired_quantity - current_quantity
        if delta > 0:
            quantity = (delta // lot_size) * lot_size
            action = "BUY"
        elif delta < 0:
            # Selling may use an odd-lot remainder when reducing or closing an
            # existing position.
            quantity = -delta
            action = "SELL" if target_weight == 0 else "REDUCE"
        else:
            quantity = 0
            action = "HOLD"
        reason = target.reason if target else "position is not present in current targets"
        risk_flags: list[str] = []
        if quantity and quantity * price < min_order_value:
            risk_flags.append(f"order value below minimum {min_order_value:g}")
            quantity = 0
            action = "HOLD"
        if action == "BUY" and quantity * price > portfolio.cash:
            risk_flags.append("insufficient cash")
            quantity = 0
            action = "HOLD"
        suggestions.append(
            TradeSuggestion(
                symbol=symbol,
                action=action,
                quantity=quantity,
                target_weight=target_weight,
                reference_price=price,
                max_price=None,
                min_price=None,
                risk_flags=risk_flags,
                reason=reason,
                status="BLOCKED" if risk_flags else "PASS",
            )
        )
    return suggestions


def _price(prices: Mapping[str, float], symbol: str) -> float:
    try:
        price = float(prices[symbol])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid price for {symbol}") from exc
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"price for {symbol} must be positive")
    return price
