from datetime import datetime, timezone

import pytest

from quant.domain import ExecutedTrade, TargetPosition, TradeSuggestion


def test_domain_objects_keep_suggestions_separate_from_fills():
    timestamp = datetime(2026, 10, 8, tzinfo=timezone.utc)
    target = TargetPosition("600000", timestamp, 0.2, "trend", "ma-v1")
    suggestion = TradeSuggestion("600000", "BUY", 100, target.target_weight, 10.0, None, None)
    fill = ExecutedTrade("fill-1", "600000", "BUY", 100, 10.0, 1.0, timestamp)

    assert target.target_weight == suggestion.target_weight
    assert fill.trade_id != suggestion.symbol


def test_blocked_suggestion_requires_risk_flag():
    with pytest.raises(ValueError, match="risk flags"):
        TradeSuggestion("600000", "BUY", 100, 0.2, 10.0, None, None, status="BLOCKED")
