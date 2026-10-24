from datetime import date
import sys
import types

import pytest

pd = pytest.importorskip("pandas")

from quant.data.providers import AkShareDataProvider, DataFetchError, LocalDataProvider
from quant.tasks import _fetch_akshare_frames


def test_local_csv_keeps_leading_zero_symbol_as_string(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "symbol,timestamp,open,high,low,close,volume\n"
        "600000,2026-10-09,10,11,9,10.5,1000\n"
        "000001,2026-10-09,20,21,19,20.5,2000\n",
        encoding="utf-8",
    )

    frame = LocalDataProvider(path).fetch(["600000", "000001"], date(2026, 1, 1), date(2026, 12, 31))

    assert sorted(frame["symbol"]) == ["000001", "600000"]
    assert frame["symbol"].map(type).eq(str).all()


class _FakeAkShare(types.ModuleType):
    def __init__(self):
        super().__init__("akshare")
        self.calls: list[str] = []
        self.sina_frame = pd.DataFrame(
            {
                "date": ["2026-10-08", "2026-10-09"],
                "open": [10.0, 10.5],
                "high": [11.0, 11.5],
                "low": [9.5, 10.0],
                "close": [10.6, 11.0],
                "volume": [1000, 1200],
            }
        )

    def stock_zh_a_hist(self, **kwargs):
        self.calls.append("eastmoney")
        raise ConnectionError("Remote end closed connection without response")

    def stock_zh_a_daily(self, **kwargs):
        self.calls.append("sina")
        assert kwargs["symbol"] == "sh600000"
        return self.sina_frame

    def stock_zh_a_hist_tx(self, **kwargs):  # pragma: no cover - not reached
        self.calls.append("tencent")
        return self.sina_frame


def test_akshare_provider_falls_back_to_next_endpoint(monkeypatch):
    fake = _FakeAkShare()
    monkeypatch.setitem(sys.modules, "akshare", fake)

    frame = AkShareDataProvider(retries=0, timeout=1.0).fetch(
        ["600000"], date(2026, 1, 1), date(2026, 12, 31)
    )

    assert fake.calls == ["eastmoney", "sina"]
    assert len(frame) == 2
    assert set(frame["symbol"]) == {"600000"}


def test_akshare_provider_falls_back_when_endpoint_returns_invalid_ohlc(monkeypatch):
    fake = _FakeAkShare()
    invalid = fake.sina_frame.rename(
        columns={
            "date": "日期",
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
            "volume": "成交量",
        }
    )
    invalid.loc[0, "开盘"] = 0.0

    def invalid_eastmoney(**kwargs):
        fake.calls.append("eastmoney")
        return invalid

    fake.stock_zh_a_hist = invalid_eastmoney
    monkeypatch.setitem(sys.modules, "akshare", fake)

    frame = AkShareDataProvider(retries=0, timeout=1.0).fetch(
        ["600000"], date(2026, 1, 1), date(2026, 12, 31)
    )

    assert fake.calls == ["eastmoney", "sina"]
    assert len(frame) == 2
    assert (frame[["open", "high", "low", "close"]] > 0).all().all()


def test_akshare_provider_converts_invalid_ohlc_to_fetch_error(monkeypatch):
    fake = _FakeAkShare()
    invalid = fake.sina_frame.rename(
        columns={
            "date": "日期",
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
            "volume": "成交量",
        }
    )
    invalid.loc[0, "收盘"] = 0.0

    def invalid_endpoint(**kwargs):
        return invalid

    fake.stock_zh_a_hist = invalid_endpoint
    fake.stock_zh_a_daily = invalid_endpoint
    fake.stock_zh_a_hist_tx = invalid_endpoint
    monkeypatch.setitem(sys.modules, "akshare", fake)

    with pytest.raises(DataFetchError, match="OHLC prices must be positive"):
        AkShareDataProvider(retries=0, timeout=1.0).fetch(
            ["600000"], date(2026, 1, 1), date(2026, 12, 31)
        )


def test_batch_download_continues_when_one_symbol_is_invalid():
    class PartialProvider:
        def fetch(self, symbols, start, end):
            if symbols == ["BAD"]:
                raise DataFetchError("BAD: OHLC prices must be positive")
            return pd.DataFrame(
                [
                    {
                        "symbol": "GOOD",
                        "timestamp": "2026-10-08",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.5,
                        "volume": 1000,
                    }
                ]
            )

    frame, failures = _fetch_akshare_frames(
        PartialProvider(), ["GOOD", "BAD"], date(2026, 1, 1), date(2026, 12, 31)
    )

    assert set(frame["symbol"]) == {"GOOD"}
    assert "OHLC prices must be positive" in failures["BAD"]


def test_akshare_provider_reports_underlying_error(monkeypatch):
    fake = _FakeAkShare()
    fake.stock_zh_a_daily = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("sina blocked"))
    fake.stock_zh_a_hist_tx = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("tencent blocked"))
    monkeypatch.setitem(sys.modules, "akshare", fake)

    with pytest.raises(DataFetchError, match="sina blocked"):
        AkShareDataProvider(retries=0, timeout=1.0).fetch(["600000"], date(2026, 1, 1), date(2026, 12, 31))


def test_prefixed_code_maps_exchange_and_suffix():
    from quant.data.providers import _prefixed_code

    assert _prefixed_code("600000", "600000") == "sh600000"
    assert _prefixed_code("000001", "000001") == "sz000001"
    assert _prefixed_code("300750", "300750.SZ") == "sz300750"
    assert _prefixed_code("830799", "830799") == "bj830799"
