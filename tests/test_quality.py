import pytest
from datetime import date

pd = pytest.importorskip("pandas")

from quant.data.quality import validate_market_data


def test_missing_bar_blocks_unless_declared_halted():
    frame = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2026-10-08", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
        ]
    )
    report = validate_market_data(
        frame,
        symbols=["AAA"],
        expected_dates=[date(2026, 10, 8), date(2026, 10, 9)],
        halted=[("AAA", date(2026, 10, 9))],
    )

    assert report.status == "PASS"
