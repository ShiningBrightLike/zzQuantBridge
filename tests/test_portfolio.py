from datetime import datetime, timezone

import pytest

from quant.domain import ExecutedTrade
from quant.portfolio import PortfolioStore, TradeConflictError


def test_trade_import_is_idempotent_and_rebuilds_state(tmp_path):
    timestamp = datetime(2026, 10, 8, tzinfo=timezone.utc)
    trade = ExecutedTrade("t-1", "600000", "BUY", 100, 10.0, 1.0, timestamp)
    store = PortfolioStore(tmp_path / "portfolio.sqlite")

    assert store.append_trade(trade) is True
    assert store.append_trade(trade) is False
    snapshot = store.snapshot(2000.0)

    assert snapshot.cash == 999.0
    assert snapshot.positions == {"600000": 100}


def test_reusing_trade_id_with_different_values_is_rejected(tmp_path):
    timestamp = datetime(2026, 10, 8, tzinfo=timezone.utc)
    store = PortfolioStore(tmp_path / "portfolio.sqlite")
    store.append_trade(ExecutedTrade("t-1", "600000", "BUY", 100, 10.0, 1.0, timestamp))

    with pytest.raises(TradeConflictError):
        store.append_trade(ExecutedTrade("t-1", "600000", "BUY", 200, 10.0, 1.0, timestamp))
