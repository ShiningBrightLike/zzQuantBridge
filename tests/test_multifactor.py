from datetime import datetime, timedelta, timezone
import time

import pytest

pd = pytest.importorskip("pandas")

from quant.data.archive import archive_snapshot
from quant.data.universe import (
    UniverseConfig,
    build_universe_membership,
    current_tradable_symbols,
)
from quant.features.multifactor import (
    FactorTrainingConfig,
    compute_factor_panel,
    fit_factor_model,
    score_cross_section,
    select_from_scores,
)
from quant.tasks import build_task, latest_selections
from quant.tasks import (
    _cross_sectional_backtest_config,
    _latest_prices,
    _merge_execution_close,
    _multifactor_snapshot_issue,
    _multifactor_snapshot_ready,
    _selection_backtest_config,
)
from quant.strategies.multifactor import MultifactorStrategyConfig
from quant.webapp import JobManager, find_run


def _frame(symbols=2, days=80, start="2025-01-01"):
    dates = pd.bdate_range(start, periods=days, tz="UTC")
    rows = []
    for index in range(symbols):
        price = 10.0 + index
        for offset, stamp in enumerate(dates):
            price *= 1.0 + 0.002 * ((index + 1) % 3 - 1) + 0.001 * (offset % 4)
            rows.append(
                {
                    "symbol": f"{index:06d}",
                    "timestamp": stamp,
                    "open": price * 0.999,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000 + index * 1000,
                }
            )
    return pd.DataFrame(rows)


def test_factor_panel_does_not_change_when_future_bars_change():
    frame = _frame(symbols=3, days=80)
    panel = compute_factor_panel(frame, label_horizon=5)
    cut = panel["timestamp"].sort_values().unique()[50]
    before = panel.loc[
        (panel["symbol"] == "000000") & (panel["timestamp"] == cut), "momentum_20"
    ].iloc[0]
    changed = frame.copy()
    changed.loc[changed["timestamp"] > cut, "close"] *= 10.0
    changed.loc[changed["timestamp"] > cut, "open"] *= 10.0
    after_panel = compute_factor_panel(changed, label_horizon=5)
    after = after_panel.loc[
        (after_panel["symbol"] == "000000") & (after_panel["timestamp"] == cut), "momentum_20"
    ].iloc[0]

    assert before == pytest.approx(after)
    assert panel.loc[
        (panel["symbol"] == "000000") & (panel["timestamp"] == cut), "forward_return"
    ].iloc[0] != pytest.approx(
        after_panel.loc[
            (after_panel["symbol"] == "000000") & (after_panel["timestamp"] == cut),
            "forward_return",
        ].iloc[0]
    )


def test_training_ignores_labels_not_realized_at_cutoff():
    dates = pd.bdate_range("2025-01-01", periods=60, tz="UTC")
    rows = []
    for stamp in dates:
        for rank in range(20):
            rows.append(
                {
                    "timestamp": stamp,
                    "symbol": f"{rank:06d}",
                    "past_factor": float(rank),
                    "future_factor": float(rank) if stamp >= dates[-6] else 0.0,
                    "forward_return": rank / 1000.0,
                    "label_end": stamp + pd.Timedelta(days=5),
                }
            )
    panel = pd.DataFrame(rows)
    cutoff = dates[40]
    model = fit_factor_model(
        panel,
        cutoff,
        FactorTrainingConfig(
            lookback_days=100,
            label_horizon=5,
            min_cross_section=10,
            min_ic_days=20,
            min_abs_ic=0.01,
            min_abs_icir=0.0,
            max_factors=2,
        ),
        factor_names=("past_factor", "future_factor"),
    )

    assert "past_factor" in model.factor_names
    assert "future_factor" not in model.factor_names


def test_selection_uses_hold_buffer_and_caps_weights():
    scores = pd.DataFrame(
        {
            "symbol": [f"{index:06d}" for index in range(20)],
            "score": list(range(20, 0, -1)),
            "rank": list(range(1, 21)),
        }
    )
    selected = select_from_scores(
        scores,
        top_n=5,
        hold_buffer=10,
        max_single_weight=0.2,
        cash_buffer=0.1,
        previous_symbols=("000007",),
    )

    chosen = selected.loc[selected["selected"]]
    assert len(chosen) == 5
    assert "000007" in set(chosen["symbol"])
    assert chosen["target_weight"].sum() == pytest.approx(0.9)


def test_selection_does_not_fill_top_n_with_negative_scores():
    scores = pd.DataFrame(
        {
            "symbol": ["000000", "000001", "000002", "000003"],
            "score": [0.8, 0.2, -0.1, -0.5],
            "rank": [1, 2, 3, 4],
        }
    )

    selected = select_from_scores(
        scores,
        top_n=4,
        hold_buffer=4,
        max_single_weight=0.25,
        cash_buffer=0.1,
        min_score=0.0,
    )

    assert set(selected.loc[selected["selected"], "symbol"]) == {"000000", "000001"}


def test_universe_membership_limits_by_history_and_liquidity():
    frame = _frame(symbols=8, days=40)
    membership = build_universe_membership(
        frame,
        UniverseConfig(max_symbols=4, min_history_days=20, min_avg_dollar_volume=0),
    )
    latest = membership["timestamp"].max()
    tradable = membership.loc[(membership["timestamp"] == latest) & membership["tradable"]]

    assert 1 <= len(tradable) <= 4
    assert current_tradable_symbols(membership, latest) == sorted(tradable["symbol"].tolist())


def test_multifactor_configs_share_lot_size_and_portfolio_cash(tmp_path):
    (tmp_path / "multifactor.yaml").write_text(
        "\n".join(
            [
                "name: multifactor_selection",
                "strategy: multifactor",
                "selection:",
                "  lot_size: 200",
                "backtest:",
                "  init_cash: 100000",
                "  min_oos_days: 30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = _cross_sectional_backtest_config(tmp_path, {})
    selection_config = _selection_backtest_config(tmp_path, {}, 250000)

    assert config.lot_size == 200
    assert config.min_oos_days == 30
    assert selection_config.init_cash == 250000


def test_latest_prices_prefers_unadjusted_execution_close():
    frame = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2026-10-08", "close": 10.0, "execution_close": 9.5},
            {"symbol": "AAA", "timestamp": "2026-10-09", "close": 11.0, "execution_close": 10.0},
            {"symbol": "BBB", "timestamp": "2026-10-09", "close": 20.0},
        ]
    )

    prices, source = _latest_prices(frame, {"adjustment": "qfq"})
    assert source == "unadjusted"
    assert prices["AAA"] == 10.0
    assert "BBB" not in prices

    adjusted, adjusted_source = _latest_prices(frame[["symbol", "timestamp", "close"]], {"adjustment": "qfq"})
    assert adjusted_source == "adjusted"
    assert adjusted["AAA"] == 11.0


def test_execution_close_merge_does_not_replace_research_close():
    research = pd.DataFrame(
        [{"symbol": "AAA", "timestamp": "2026-10-09", "close": 11.0}]
    )
    execution = pd.DataFrame(
        [{"symbol": "AAA", "timestamp": "2026-10-09", "close": 10.0}]
    )

    merged = _merge_execution_close(research, execution)

    assert merged.loc[0, "close"] == 11.0
    assert merged.loc[0, "execution_close"] == 10.0


def _snapshot(path, symbols=20, days=320):
    frame = _frame(symbols=symbols, days=days)
    frame.to_csv(path, index=False)
    return path


def _wait(runs_dir, run_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = find_run(run_id, runs_dir)
        if record is not None and record.status not in {"QUEUED", "RUNNING"}:
            # Pollers can observe the task's terminal status just before the
            # manager publishes the same completed payload. Re-read once that
            # hand-off has settled so assertions see the final summary.
            time.sleep(0.1)
            return find_run(run_id, runs_dir) or record
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_select_stocks_task_writes_factor_and_backtest_artifacts(tmp_path):
    snapshot = _snapshot(tmp_path / "bars.csv")
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    configs_dir = tmp_path / "configs"
    manager = JobManager(runs_dir)
    params = {
        "universe": {
            "mode": "config",
            "max_symbols": 20,
            "min_history_days": 40,
            "min_avg_dollar_volume": 0,
        },
        "training": {
            "lookback_days": 100,
            "label_horizon": 20,
            "min_cross_section": 10,
            "min_ic_days": 30,
            "max_factors": 4,
        },
        "selection": {
            "top_n": 5,
            "hold_buffer": 8,
            "max_single_weight": 0.15,
            "cash_buffer": 0.1,
            "rebalance_days": 20,
        },
        "backtest": {"init_cash": 100000, "fees": 0.0003, "slippage": 0.0005},
    }
    try:
        download = build_task(
            "download",
            {
                "provider": "local",
                "file": str(snapshot),
                "symbols": [f"{index:06d}" for index in range(20)],
                "start": "2025-01-01",
                "end": "2026-12-31",
            },
            data_dir=data_dir,
            configs_dir=configs_dir,
        )
        assert _wait(runs_dir, manager.submit("download", download).run_id).status == "PASS"
        selection = build_task(
            "select-stocks",
            params,
            data_dir=data_dir,
            configs_dir=configs_dir,
            db_path=tmp_path / "portfolio.sqlite",
        )
        record = _wait(runs_dir, manager.submit("select-stocks", selection).run_id)
    finally:
        manager.shutdown()

    assert record.status in {"PASS", "BLOCKED"}, record.payload
    assert (record.run_dir / "factor_snapshot.csv").is_file()
    assert (record.run_dir / "factor_weights.csv").is_file()
    assert (record.run_dir / "selections.csv").is_file()
    assert (record.run_dir / "selections.csv").read_text(encoding="utf-8-sig").count("\n") >= 2
    assert (record.run_dir / "report.html").is_file()
    assert record.payload["backtest"]["min_oos_days"] == 504
    assert record.payload["backtest"]["status"] == "BLOCKED"
    rows, status, reasons, metadata = latest_selections(runs_dir)
    assert rows
    assert status == record.status
    assert reasons
    assert metadata["model_version"]


def test_select_stocks_auto_downloads_all_a_snapshot(tmp_path, monkeypatch):
    frame = _frame(symbols=20, days=320)
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    calls = []

    def fake_download_task(params, *, data_dir, configs_dir, finalize=True):
        calls.append((params, finalize))

        def run(context):
            manifest = archive_snapshot(
                frame,
                data_dir,
                source="test",
                request={"symbols": sorted(frame["symbol"].unique().tolist())},
                adjustment="qfq",
            )
            return {
                "rows": len(frame),
                "universe_size": int(frame["symbol"].nunique()),
                "start": str(frame["timestamp"].min().date()),
                "end": str(frame["timestamp"].max().date()),
                "data_as_of": str(frame["timestamp"].max().date()),
                "manifest": str(manifest),
            }

        return run

    monkeypatch.setattr("quant.tasks._download_task", fake_download_task)
    latest_date = frame["timestamp"].max().date()
    monkeypatch.setattr(
        "quant.tasks.latest_completed_trading_date",
        lambda: (latest_date, "provided"),
    )
    manager = JobManager(runs_dir)
    params = {
        "provider": "akshare",
        "universe": {
            "mode": "all_a",
            "auto_download": True,
            "max_symbols": 20,
            "min_history_days": 40,
            "min_avg_dollar_volume": 0,
        },
        "training": {
            "lookback_days": 100,
            "label_horizon": 20,
            "min_cross_section": 10,
            "min_ic_days": 30,
            "max_factors": 4,
        },
        "selection": {
            "top_n": 5,
            "hold_buffer": 8,
            "max_single_weight": 0.15,
            "cash_buffer": 0.1,
            "rebalance_days": 20,
        },
        "backtest": {"init_cash": 100000, "fees": 0.0003, "slippage": 0.0005},
    }
    try:
        task = build_task(
            "select-stocks",
            params,
            data_dir=data_dir,
            configs_dir=tmp_path / "configs",
            db_path=tmp_path / "portfolio.sqlite",
        )
        record = _wait(runs_dir, manager.submit("select-stocks", task).run_id)
    finally:
        manager.shutdown()

    assert record.status in {"PASS", "BLOCKED"}, record.payload
    assert calls
    assert calls[0][1] is False
    assert calls[0][0]["universe"]["mode"] == "all_a"
    assert calls[0][0]["adjustment"] == "qfq"
    assert "auto_download" in record.payload, record.payload
    assert record.payload["auto_download"]["universe_size"] == 20


def test_multifactor_snapshot_readiness_requires_breadth_and_history():
    settings = MultifactorStrategyConfig.from_mapping(
        {
            "universe": {
                "mode": "all_a",
                "min_history_days": 40,
                "max_symbols": 20,
            },
            "training": {
                "lookback_days": 100,
                "label_horizon": 20,
                "min_cross_section": 10,
                "min_ic_days": 30,
            },
        }
    )

    frame = _frame(symbols=20, days=320)
    latest_date = frame["timestamp"].max().date()

    assert _multifactor_snapshot_ready(
        frame,
        settings,
        required_data_as_of=latest_date,
    )
    assert not _multifactor_snapshot_ready(_frame(symbols=2, days=320), settings)
    assert not _multifactor_snapshot_ready(_frame(symbols=20, days=100), settings)
    assert not _multifactor_snapshot_ready(
        frame,
        settings,
        required_data_as_of=latest_date + timedelta(days=1),
    )
    assert "before the latest completed trading day" in (
        _multifactor_snapshot_issue(
            frame,
            settings,
            required_data_as_of=latest_date + timedelta(days=1),
        )
        or ""
    )
