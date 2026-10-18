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


def validate_research_snapshot(
    frame: Any,
    *,
    max_missing_days: int = 20,
    max_missing_ratio: float = 0.02,
    halted: Iterable[tuple[str, date]] = (),
) -> QualityReport:
    """Validate a multi-symbol research snapshot without rejecting normal halts.

    Gaps are measured from each symbol's first observed bar to its last
    observed bar, so pre-listing dates are not treated as missing history.
    Small gaps are warnings; gaps above either threshold block the run. Callers
    can declare known halts through ``halted`` when stricter accounting is
    required.
    """

    try:
        normalized = normalize_ohlcv(frame)
    except DataValidationError as exc:
        return QualityReport(status="BLOCKED", row_count=0, errors=(str(exc),))
    if max_missing_days < 0:
        raise ValueError("max_missing_days must not be negative")
    if not 0 <= max_missing_ratio <= 1:
        raise ValueError("max_missing_ratio must be between zero and one")

    calendar = sorted(set(normalized["timestamp"].dt.date))
    halted_set = set(halted)
    errors: list[str] = []
    warnings: list[str] = []
    for symbol, group in normalized.groupby("symbol", sort=True):
        observed = set(group["timestamp"].dt.date)
        first, last = min(observed), max(observed)
        expected = {day for day in calendar if first <= day <= last}
        missing = sorted(
            expected - observed - {day for item, day in halted_set if item == symbol}
        )
        if not missing:
            continue
        ratio = len(missing) / max(1, len(expected))
        message = (
            f"{symbol} missing {len(missing)} bar(s) between "
            f"{first.isoformat()} and {last.isoformat()} ({ratio:.2%})"
        )
        if len(missing) > max_missing_days or ratio > max_missing_ratio:
            errors.append(message)
        else:
            warnings.append(message)
    return QualityReport(
        status="BLOCKED" if errors else "PASS",
        row_count=len(normalized),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
