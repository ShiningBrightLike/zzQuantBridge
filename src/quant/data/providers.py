"""Market data provider implementations."""

from __future__ import annotations

from datetime import date
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

from .validation import REQUIRED_COLUMNS, normalize_ohlcv


class MarketDataProvider(Protocol):
    def fetch(self, symbols: Sequence[str], start: date, end: date, timeframe: str = "1d") -> Any:
        """Fetch canonical OHLCV data for the requested symbols and range."""


class DataFetchError(RuntimeError):
    """Raised when a provider cannot return verified data."""


class LocalDataProvider:
    """Read a previously archived CSV or Parquet snapshot."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self, symbols: Sequence[str], start: date, end: date, timeframe: str = "1d") -> Any:
        if timeframe != "1d":
            raise DataFetchError("LocalDataProvider currently supports only 1d data")
        if not self.path.is_file():
            raise DataFetchError(f"local data file does not exist: {self.path}")
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise DataFetchError("pandas is required for local market data") from exc
        try:
            frame = pd.read_parquet(self.path) if self.path.suffix.lower() == ".parquet" else pd.read_csv(self.path)
        except Exception as exc:  # pragma: no cover - backend-specific errors
            raise DataFetchError(f"failed to read local data: {self.path}") from exc
        result = normalize_ohlcv(frame)
        wanted = set(symbols)
        if wanted:
            result = result[result["symbol"].isin(wanted)]
        result = result[(result["timestamp"].dt.date >= start) & (result["timestamp"].dt.date <= end)]
        return result.reset_index(drop=True)


class AkShareDataProvider:
    """Fetch daily A-share history through the optional AkShare dependency."""

    def __init__(
        self,
        *,
        adjustment: str = "",
        retries: int = 2,
        backoff_seconds: float = 1.0,
        sleep: Any = time.sleep,
    ) -> None:
        if adjustment not in {"", "qfq", "hfq"}:
            raise ValueError("adjustment must be '', 'qfq', or 'hfq'")
        self.adjustment = adjustment
        self.retries = max(0, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self._sleep = sleep

    def fetch(self, symbols: Sequence[str], start: date, end: date, timeframe: str = "1d") -> Any:
        if timeframe != "1d":
            raise DataFetchError("AkShareDataProvider currently supports only 1d data")
        try:
            import akshare as ak
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise DataFetchError("AkShare and pandas are required; install the data extra") from exc

        frames = []
        for symbol in symbols:
            provider_symbol = symbol.split(".", 1)[0]
            last_error: Exception | None = None
            for attempt in range(self.retries + 1):
                try:
                    raw = ak.stock_zh_a_hist(
                        symbol=provider_symbol,
                        period="daily",
                        start_date=start.strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"),
                        adjust=self.adjustment,
                    )
                    mapped = raw.rename(
                        columns={
                            "\u65e5\u671f": "timestamp",
                            "\u5f00\u76d8": "open",
                            "\u6700\u9ad8": "high",
                            "\u6700\u4f4e": "low",
                            "\u6536\u76d8": "close",
                            "\u6210\u4ea4\u91cf": "volume",
                        }
                    )
                    mapped["symbol"] = symbol
                    frames.append(mapped[["symbol", *REQUIRED_COLUMNS[1:]]])
                    last_error = None
                    break
                except Exception as exc:  # provider errors vary by endpoint
                    last_error = exc
                    if attempt < self.retries:
                        self._sleep(self.backoff_seconds * (2**attempt))
            if last_error is not None:
                raise DataFetchError(f"AkShare request failed for {symbol}") from last_error
        if not frames:
            raise DataFetchError("AkShare returned no data")
        return normalize_ohlcv(pd.concat(frames, ignore_index=True))
