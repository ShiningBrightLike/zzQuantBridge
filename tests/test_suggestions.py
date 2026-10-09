from datetime import datetime, timezone

from quant.domain import ExecutedTrade, TargetPosition
from quant.portfolio import PortfolioSnapshot
from quant.signals import generate_suggestions


def test_buy_uses_lots_and_sell_can_close_odd_lot():
    timestamp = datetime(2026, 10, 8, tzinfo=timezone.utc)
    portfolio = PortfolioSnapshot(cash=10000, positions={"AAA": 150}, average_cost={"AAA": 10})
    targets = [TargetPosition("AAA", timestamp, 0.0, "trend off", "ma-v1")]

    suggestions = generate_suggestions(targets, portfolio, {"AAA": 10})

    assert suggestions[0].action == "SELL"
    assert suggestions[0].quantity == 150


def test_cash_shortfall_blocks_buy():
    timestamp = datetime(2026, 10, 8, tzinfo=timezone.utc)
    portfolio = PortfolioSnapshot(cash=50, positions={"AAA": 100}, average_cost={"AAA": 10})
    targets = [TargetPosition("BBB", timestamp, 1.0, "trend on", "ma-v1")]

    suggestion = next(item for item in generate_suggestions(targets, portfolio, {"AAA": 10, "BBB": 10}) if item.symbol == "BBB")

    assert suggestion.status == "BLOCKED"
    assert "insufficient cash" in suggestion.risk_flags
