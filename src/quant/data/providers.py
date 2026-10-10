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
            if self.path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(self.path)
            else:
                # A-share codes start with zeros, so the symbol column must stay
                # a string instead of being inferred as an integer.
                frame = pd.read_csv(self.path, dtype={"symbol": str})
        except Exception as exc:  # pragma: no cover - backend-specific errors
            raise DataFetchError(f"failed to read local data: {self.path}") from exc
        result = normalize_ohlcv(frame)
        wanted = set(symbols)
        if wanted:
            result = result[result["symbol"].isin(wanted)]
        result = result[(result["timestamp"].dt.date >= start) & (result["timestamp"].dt.date <= end)]
        return result.reset_index(drop=True)


class AkShareDataProvider:
    """Fetch daily A-share history through AkShare with endpoint fallbacks.

    AkShare aggregates several public endpoints. Any single one can be blocked,
    rate limited, or changed by its upstream provider, so requests fall through
    Eastmoney, Sina, and Tencent before the symbol is reported as failed.
    """

    ENDPOINTS = ("eastmoney", "sina", "tencent")
    _COLUMN_MAP = {
        "eastmoney": {
            "\u65e5\u671f": "timestamp",
            "\u5f00\u76d8": "open",
            "\u6700\u9ad8": "high",
            "\u6700\u4f4e": "low",
            "\u6536\u76d8": "close",
            "\u6210\u4ea4\u91cf": "volume",
        },
        "sina": {
            "date": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        "tencent": {
            "date": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
    }

    def __init__(
        self,
        *,
        adjustment: str = "",
        retries: int = 2,
        backoff_seconds: float = 1.0,
        endpoints: Sequence[str] | None = None,
        timeout: float | None = 15.0,
        sleep: Any = time.sleep,
    ) -> None:
        if adjustment not in {"", "qfq", "hfq"}:
            raise ValueError("adjustment must be '', 'qfq', or 'hfq'")
        self.adjustment = adjustment
        self.retries = max(0, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.endpoints = tuple(endpoints or self.ENDPOINTS)
        unknown = set(self.endpoints).difference(self.ENDPOINTS)
        if unknown:
            raise ValueError(f"unsupported AkShare endpoints: {', '.join(sorted(unknown))}")
        self.timeout = timeout
        self._sleep = sleep
        self.last_sources: dict[str, str] = {}

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
            last_error: Exception | None = None
            attempt_errors: list[str] = []
            for attempt in range(self.retries + 1):
                attempt_errors = []
                for endpoint in self.endpoints:
                    try:
                        frames.append(self._request(ak, endpoint, symbol, start, end))
                    except Exception as exc:  # provider errors vary by endpoint
                        last_error = exc
                        attempt_errors.append(f"{endpoint}: {exc}")
                        continue
                    self.last_sources[symbol] = endpoint
                    last_error = None
                    break
                if last_error is None:
                    break
                if attempt < self.retries:
                    self._sleep(self.backoff_seconds * (2**attempt))
            if last_error is not None:
                raise DataFetchError(
                    f"AkShare request failed for {symbol}: " + "; ".join(attempt_errors)
                ) from last_error
        if not frames:
            raise DataFetchError("AkShare returned no data")
        return normalize_ohlcv(pd.concat(frames, ignore_index=True))

    def _request(self, ak: Any, endpoint: str, symbol: str, start: date, end: date) -> Any:
        code = _plain_code(symbol)
        start_text = start.strftime("%Y%m%d")
        end_text = end.strftime("%Y%m%d")
        if endpoint == "eastmoney":
            raw = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_text,
                end_date=end_text,
                adjust=self.adjustment,
                timeout=self.timeout,
            )
        elif endpoint == "sina":
            raw = ak.stock_zh_a_daily(
                symbol=_prefixed_code(code, symbol),
                start_date=start_text,
                end_date=end_text,
                adjust=self.adjustment,
            )
        else:
            raw = ak.stock_zh_a_hist_tx(
                symbol=_prefixed_code(code, symbol),
                start_date=start_text,
                end_date=end_text,
                adjust=self.adjustment,
                timeout=self.timeout,
            )
        if raw is None or len(raw) == 0:
            raise ValueError(f"{endpoint} returned an empty frame")
        mapped = raw.rename(columns=self._COLUMN_MAP[endpoint])
        wanted = [column for column in REQUIRED_COLUMNS if column != "symbol"]
        missing = [column for column in wanted if column not in mapped.columns]
        if missing:
            raise ValueError(f"{endpoint} response missing columns: {', '.join(missing)}")
        mapped = mapped.copy()
        mapped["symbol"] = symbol
        return mapped[["symbol", *wanted]]


def _plain_code(symbol: str) -> str:
    return symbol.strip().split(".", 1)[0]


def _prefixed_code(code: str, original: str) -> str:
    upper = original.strip().upper()
    if upper.endswith((".SH", ".SZ", ".BJ")):
        return upper.rsplit(".", 1)[1].lower() + code
    if code.startswith(("5", "6", "9")):
        return "sh" + code
    if code.startswith(("0", "1", "2", "3")):
        return "sz" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sh" + code
