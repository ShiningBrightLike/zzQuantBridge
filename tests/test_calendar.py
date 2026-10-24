from datetime import datetime

from quant.data.calendar import SHANGHAI_TZ, latest_completed_trading_date


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=SHANGHAI_TZ)


def test_latest_completed_trading_date_waits_for_market_close():
    dates = ("2026-10-22", "2026-10-23", "2026-10-26")

    before_close, source = latest_completed_trading_date(
        now=_at("2026-10-23T14:59:00"),
        trade_dates=dates,
    )
    after_close, _ = latest_completed_trading_date(
        now=_at("2026-10-23T15:00:00"),
        trade_dates=dates,
    )

    assert before_close.isoformat() == "2026-10-22"
    assert after_close.isoformat() == "2026-10-23"
    assert source == "provided"


def test_latest_completed_trading_date_handles_weekends_and_holidays():
    dates = ("2026-09-30", "2026-10-09", "2026-10-23", "2026-10-26")

    weekend, _ = latest_completed_trading_date(
        now=_at("2026-10-24T12:00:00"),
        trade_dates=dates,
    )
    holiday, _ = latest_completed_trading_date(
        now=_at("2026-10-03T12:00:00"),
        trade_dates=dates,
    )

    assert weekend.isoformat() == "2026-10-23"
    assert holiday.isoformat() == "2026-09-30"


def test_latest_completed_trading_date_falls_back_to_weekdays():
    latest, source = latest_completed_trading_date(
        now=_at("2026-10-24T12:00:00"),
        trade_dates=(),
    )

    assert latest.isoformat() == "2026-10-23"
    assert source == "weekday_fallback"
