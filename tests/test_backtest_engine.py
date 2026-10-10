import pytest

pd = pytest.importorskip("pandas")

from quant.backtest import crossovers


def test_crossovers_keeps_only_the_first_true_bar():
    signals = pd.DataFrame({"AAA": [False, False, True, True, True, False, True]})

    result = crossovers(signals)

    assert result["AAA"].tolist() == [False, False, True, False, False, False, True]
    assert result["AAA"].dtype == bool


def test_crossovers_handles_object_dtype_signals():
    # pandas 3 turns a shifted boolean column into object dtype; the naive
    # ``~signals.shift(1)`` trick silently disabled the filter.
    signals = pd.DataFrame({"AAA": [True, True, True, False, True]}, dtype=object)

    result = crossovers(signals)

    assert result["AAA"].tolist() == [True, False, False, False, True]
