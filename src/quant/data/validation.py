"""Validation and canonicalization for OHLCV data frames."""

from __future__ import annotations

from typing import Any

REQUIRED_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume")


class DataValidationError(ValueError):
    """Raised when a market data frame violates the canonical schema."""


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise DataValidationError("pandas is required for market data operations; install the data extra") from exc
    return pd


def normalize_ohlcv(frame: Any) -> Any:
    """Return a sorted, timezone-aware copy in the canonical OHLCV schema."""

    pd = _pandas()
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"missing OHLCV columns: {', '.join(missing)}")
    if frame.empty:
        raise DataValidationError("OHLCV data is empty")
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.strip()
    if result["symbol"].eq("").any():
        raise DataValidationError("symbol must not be empty")
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")
    if result["timestamp"].isna().any():
        raise DataValidationError("timestamp contains invalid values")
    numeric = ["open", "high", "low", "close", "volume"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise DataValidationError(f"{column} contains non-numeric values")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("OHLC prices must be positive")
    if (result["volume"] < 0).any():
        raise DataValidationError("volume must not be negative")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any():
        raise DataValidationError("high must be at least open, close, and low")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise DataValidationError("low must be at most open, close, and high")
    if result.duplicated(["symbol", "timestamp"]).any():
        raise DataValidationError("duplicate symbol/timestamp bars")
    return result.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
