"""Quality gates applied before research or recommendation generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

from .validation import DataValidationError, normalize_ohlcv


@dataclass(frozen=True)
class QualityReport:
    status: str
    row_count: int
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCKED"


def validate_market_data(
    frame: Any,
    *,
    symbols: Iterable[str] | None = None,
    expected_dates: Iterable[date] | None = None,
    halted: Iterable[tuple[str, date]] = (),
) -> QualityReport:
    """Validate canonical OHLCV and return diagnostics instead of raising."""

    try:
        normalized = normalize_ohlcv(frame)
    except DataValidationError as exc:
        return QualityReport(status="BLOCKED", row_count=0, errors=(str(exc),))

    errors: list[str] = []
    halted_set = set(halted)
    if symbols is not None and expected_dates is not None:
        expected = set(expected_dates)
        for symbol in symbols:
            observed = set(normalized.loc[normalized["symbol"] == symbol, "timestamp"].dt.date)
            missing = sorted(expected - observed - {day for item, day in halted_set if item == symbol})
            if missing:
                errors.append(f"{symbol} missing bars: {', '.join(day.isoformat() for day in missing)}")
    return QualityReport(
        status="BLOCKED" if errors else "PASS",
        row_count=len(normalized),
        errors=tuple(errors),
    )
