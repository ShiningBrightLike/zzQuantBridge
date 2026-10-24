import pytest
from datetime import date

pd = pytest.importorskip("pandas")

from quant.data.quality import validate_market_data, validate_research_snapshot


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


def test_research_snapshot_warns_on_small_gap_and_blocks_large_gap():
    frame = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2026-10-08", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "AAA", "timestamp": "2026-10-09", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "AAA", "timestamp": "2026-10-13", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "BBB", "timestamp": "2026-10-08", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "BBB", "timestamp": "2026-10-09", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "BBB", "timestamp": "2026-10-12", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "BBB", "timestamp": "2026-10-13", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
        ]
    )

    warned = validate_research_snapshot(frame, max_missing_days=1, max_missing_ratio=0.5)
    blocked = validate_research_snapshot(frame, max_missing_days=0, max_missing_ratio=0.0)

    assert warned.status == "PASS"
    assert warned.warnings
    assert blocked.status == "BLOCKED"
    assert blocked.errors


def test_research_snapshot_can_exclude_only_symbols_with_large_gaps():
    frame = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2026-10-08", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "AAA", "timestamp": "2026-10-09", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "AAA", "timestamp": "2026-10-13", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "BBB", "timestamp": "2026-10-08", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "BBB", "timestamp": "2026-10-09", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "BBB", "timestamp": "2026-10-12", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"symbol": "BBB", "timestamp": "2026-10-13", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
        ]
    )

    report = validate_research_snapshot(
        frame,
        max_missing_days=0,
        max_missing_ratio=0,
        excess_gap_policy="exclude",
    )

    assert report.status == "PASS"
    assert report.excluded_symbols == ("AAA",)
    assert "AAA" in report.exclusion_reasons[0]
    assert not report.errors
