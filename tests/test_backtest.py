import pytest

pytest.importorskip("vectorbt")

from quant.backtest import BacktestConfig


def test_backtest_config_requires_non_negative_costs():
    with pytest.raises(ValueError, match="fees"):
        BacktestConfig(fees=-0.01)
