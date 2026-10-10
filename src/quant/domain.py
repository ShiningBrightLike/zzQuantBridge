"""Small, dependency-free domain objects shared by research and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Literal

TradeAction = Literal["BUY", "SELL", "REDUCE", "HOLD"]
TradeSide = Literal["BUY", "SELL"]


def _require_symbol(symbol: str) -> None:
    if not symbol or not symbol.strip():
        raise ValueError("symbol must not be empty")


@dataclass(frozen=True)
class TargetPosition:
    symbol: str
    timestamp: datetime
    target_weight: float
    reason: str
    model_version: str

    def __post_init__(self) -> None:
        _require_symbol(self.symbol)
        if not math.isfinite(self.target_weight) or not 0.0 <= self.target_weight <= 1.0:
            raise ValueError("target_weight must be between 0 and 1 for long-only portfolios")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")


@dataclass(frozen=True)
class TradeSuggestion:
    symbol: str
    action: TradeAction
    quantity: int
    target_weight: float
    reference_price: float | None
    max_price: float | None
    min_price: float | None
    risk_flags: list[str] = field(default_factory=list)
    reason: str = ""
    status: Literal["PASS", "BLOCKED"] = "PASS"

    def __post_init__(self) -> None:
        _require_symbol(self.symbol)
        if self.action not in {"BUY", "SELL", "REDUCE", "HOLD"}:
            raise ValueError("invalid trade action")
        if self.quantity < 0:
            raise ValueError("quantity must not be negative")
        if not math.isfinite(self.target_weight) or not 0.0 <= self.target_weight <= 1.0:
            raise ValueError("target_weight must be between 0 and 1")
        if self.reference_price is not None and (
            not math.isfinite(self.reference_price) or self.reference_price <= 0
        ):
            raise ValueError("reference_price must be positive when provided")
        if self.status == "PASS" and self.reference_price is None:
            raise ValueError("a passing suggestion requires a reference_price")
        if self.max_price is not None and (not math.isfinite(self.max_price) or self.max_price <= 0):
            raise ValueError("max_price must be positive")
        if self.min_price is not None and (not math.isfinite(self.min_price) or self.min_price <= 0):
            raise ValueError("min_price must be positive")
        if self.status == "PASS" and self.risk_flags:
            raise ValueError("a passing suggestion cannot contain risk flags")
        if self.status == "BLOCKED" and not self.risk_flags:
            raise ValueError("a blocked suggestion must contain risk flags")


@dataclass(frozen=True)
class ExecutedTrade:
    trade_id: str
    symbol: str
    side: TradeSide
    quantity: int
    price: float
    fee: float
    executed_at: datetime
    broker_reference: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.trade_id.strip():
            raise ValueError("trade_id must not be empty")
        _require_symbol(self.symbol)
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("invalid trade side")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("price must be positive")
        if not math.isfinite(self.fee) or self.fee < 0:
            raise ValueError("fee must not be negative")
