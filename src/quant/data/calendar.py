"""A-share trading calendar helpers for snapshot freshness checks."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
MARKET_CLOSE = time(15, 0)
_TRADE_DATE_CACHE: tuple[date, ...] | None = None


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    return date.fromisoformat(text[:10])


def _load_ashare_trade_dates() -> tuple[date, ...]:
    global _TRADE_DATE_CACHE
    if _TRADE_DATE_CACHE is not None:
        return _TRADE_DATE_CACHE
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - optional data dependency
        raise RuntimeError("AkShare is required to load the A-share trading calendar") from exc
    frame = ak.tool_trade_date_hist_sina()
    if frame is None or len(frame) == 0 or "trade_date" not in frame.columns:
        raise RuntimeError("A-share trading calendar response is empty or malformed")
    dates = tuple(sorted({_as_date(value) for value in frame["trade_date"]}))
    if not dates:
        raise RuntimeError("A-share trading calendar contains no dates")
    _TRADE_DATE_CACHE = dates
    return dates


def _latest_weekday(now: datetime) -> date:
    candidate = now.date()
    if now.time() < MARKET_CLOSE:
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def latest_completed_trading_date(
    *,
    now: datetime | None = None,
    trade_dates: Iterable[object] | None = None,
) -> tuple[date, str]:
    """Return the latest completed A-share session and the calendar source."""

    current = now or datetime.now(SHANGHAI_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=SHANGHAI_TZ)
    else:
        current = current.astimezone(SHANGHAI_TZ)

    source = "akshare"
    if trade_dates is None:
        try:
            dates = _load_ashare_trade_dates()
        except Exception:
            return _latest_weekday(current), "weekday_fallback"
    else:
        dates = tuple(sorted({_as_date(value) for value in trade_dates}))
        source = "provided"

    cutoff = current.date()
    if cutoff in dates and current.time() < MARKET_CLOSE:
        cutoff -= timedelta(days=1)
    eligible = [item for item in dates if item <= cutoff]
    if not eligible:
        return _latest_weekday(current), "weekday_fallback"
    return eligible[-1], source
