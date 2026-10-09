import pytest

pd = pytest.importorskip("pandas")

from quant.strategies import MovingAverageStrategy, MovingAverageStrategyConfig


def test_moving_average_strategy_returns_targets_without_future_rows():
    rows = []
    for day in range(1, 7):
        rows.append({"symbol": "AAA", "timestamp": f"2026-10-0{day}", "close": float(day)})
        rows.append({"symbol": "BBB", "timestamp": f"2026-10-0{day}", "close": 10.0})
    frame = pd.DataFrame(rows)
    strategy = MovingAverageStrategy(
        MovingAverageStrategyConfig(fast_window=2, slow_window=3, max_positions=1, max_single_weight=1.0)
    )

    targets = strategy.generate_targets(frame)
    weights = {target.symbol: target.target_weight for target in targets}

    assert weights["AAA"] == 1.0
    assert weights["BBB"] == 0.0
    assert all(target.model_version == "ma-baseline-v1" for target in targets)
