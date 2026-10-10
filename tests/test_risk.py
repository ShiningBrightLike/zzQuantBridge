from datetime import date, datetime, timezone

import pytest

from quant.domain import TargetPosition, TradeSuggestion
from quant.portfolio import PortfolioSnapshot
from quant.risk import RiskPolicy, evaluate_risk


def _target(symbol: str, weight: float) -> TargetPosition:
    return TargetPosition(symbol, datetime(2026, 10, 9, tzinfo=timezone.utc), weight, "trend", "ma-v1")


def _suggestion(
    symbol: str,
    quantity: int = 0,
    price: float | None = 10.0,
    action: str = "HOLD",
    status: str = "PASS",
    risk_flags: list[str] | None = None,
) -> TradeSuggestion:
    return TradeSuggestion(symbol, action, quantity, 0.2, price, None, None, risk_flags or [], "test", status)


def test_policy_from_config_reads_limits():
    policy = RiskPolicy.from_config(
        {
            "max_single_weight": 0.1,
            "cash_buffer": 0.2,
            "max_turnover": 0.3,
            "lot_size": 200,
            "max_data_age_days": 2,
        }
    )

    assert policy.max_single_weight == 0.1
    assert policy.cash_buffer == 0.2
    assert policy.lot_size == 200
    assert policy.max_data_age_days == 2


def test_single_name_cap_blocks_only_that_symbol():
    report = evaluate_risk(
        targets=[_target("AAA", 0.5), _target("BBB", 0.2)],
        suggestions=[_suggestion("AAA"), _suggestion("BBB")],
        portfolio=PortfolioSnapshot(cash=100_000),
        prices={"AAA": 10.0, "BBB": 10.0},
        policy=RiskPolicy(max_single_weight=0.2),
        as_of=date(2026, 10, 9),
        data_as_of=date(2026, 10, 9),
    )

    assert report.status == "BLOCKED"
    assert "AAA" in report.symbol_flags
    assert "BBB" not in report.symbol_flags


def test_gross_weight_and_stale_data_block_the_run():
    report = evaluate_risk(
        targets=[_target("AAA", 0.7), _target("BBB", 0.7)],
        suggestions=[_suggestion("AAA"), _suggestion("BBB")],
        portfolio=PortfolioSnapshot(cash=100_000),
        prices={"AAA": 10.0, "BBB": 10.0},
        policy=RiskPolicy(max_gross_weight=1.0, max_data_age_days=3),
        as_of=date(2026, 10, 9),
        data_as_of=date(2026, 9, 1),
    )

    assert report.status == "BLOCKED"
    assert any("gross weight" in reason for reason in report.reasons)
    assert any("day(s) old" in reason for reason in report.reasons)


def test_turnover_and_cash_buffer_block_the_run():
    report = evaluate_risk(
        targets=[_target("AAA", 0.2)],
        suggestions=[_suggestion("AAA", quantity=8000, action="BUY")],
        portfolio=PortfolioSnapshot(cash=100_000),
        prices={"AAA": 10.0},
        policy=RiskPolicy(max_turnover=0.1, cash_buffer=0.5),
        as_of=date(2026, 10, 9),
    )

    assert report.status == "BLOCKED"
    assert any("turnover" in reason for reason in report.reasons)
    assert any("cash" in reason for reason in report.reasons)


def test_missing_price_blocks_that_symbol():
    report = evaluate_risk(
        targets=[_target("AAA", 0.2)],
        suggestions=[_suggestion("AAA", price=None, status="BLOCKED", risk_flags=["no price"])],
        portfolio=PortfolioSnapshot(cash=100_000),
        prices={},
        policy=RiskPolicy(),
        as_of=date(2026, 10, 9),
    )

    assert report.status == "BLOCKED"
    assert any("price" in flag for flag in report.symbol_flags["AAA"])


def test_industry_limit_flags_every_member():
    report = evaluate_risk(
        targets=[_target("AAA", 0.2), _target("BBB", 0.2)],
        suggestions=[_suggestion("AAA"), _suggestion("BBB")],
        portfolio=PortfolioSnapshot(cash=100_000),
        prices={"AAA": 10.0, "BBB": 10.0},
        policy=RiskPolicy(
            max_single_weight=1.0,
            industries={"AAA": "banks", "BBB": "banks"},
            industry_limits={"banks": 0.3},
        ),
        as_of=date(2026, 10, 9),
    )

    assert set(report.symbol_flags) == {"AAA", "BBB"}
    assert all("industry" in flag for flags in report.symbol_flags.values() for flag in flags)


def test_policy_rejects_invalid_limits():
    with pytest.raises(ValueError, match="cash_buffer"):
        RiskPolicy(cash_buffer=1.0)
