from datetime import datetime, timezone

import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from quant.data.archive import archive_snapshot
from quant.server import create_app
from quant.tasks import market_figure


def _market_frame():
    rows = []
    for symbol, base in (("000001", 10.0), ("600000", 20.0)):
        for offset in range(3):
            close = base + offset
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": datetime(2026, 1, 1 + offset, tzinfo=timezone.utc),
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000 + offset,
                }
            )
    return pd.DataFrame(rows)


def test_home_treats_request_as_request_object(tmp_path):
    app = create_app(runs_dir=tmp_path / "runs", db_path=tmp_path / "portfolio.sqlite")

    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Field required" not in response.text


def test_run_trigger_requires_csrf_token(tmp_path):
    app = create_app(runs_dir=tmp_path / "runs", db_path=tmp_path / "portfolio.sqlite")

    with TestClient(app) as client:
        response = client.post("/api/runs/backtest")
        assert response.status_code == 403


def test_selection_page_renders_empty_state(tmp_path):
    app = create_app(runs_dir=tmp_path / "runs", db_path=tmp_path / "portfolio.sqlite")

    with TestClient(app) as client:
        response = client.get("/selection")
        assert response.status_code == 200
        assert "多因子选股" in response.text
        assert "还没有多因子选股结果" in response.text


def test_market_figure_uses_requested_symbol():
    figure, message, selected = market_figure(_market_frame(), "600000")

    assert selected == "600000"
    assert figure["data"][0]["name"] == "600000"
    assert "600000" in figure["layout"]["title"]["text"]
    assert "000001" in message


def test_market_figure_falls_back_for_unknown_symbol():
    figure, message, selected = market_figure(_market_frame(), "999999")

    assert selected == "000001"
    assert figure["data"][0]["name"] == "000001"
    assert "标的不在当前快照中" in message


def test_market_page_switches_selected_symbol(tmp_path):
    archive_snapshot(
        _market_frame(),
        tmp_path / "data",
        source="test",
        request={"names": {"600000": "浦发银行"}},
        adjustment="",
    )
    app = create_app(
        runs_dir=tmp_path / "runs",
        db_path=tmp_path / "portfolio.sqlite",
        data_dir=tmp_path / "data",
        configs_dir=tmp_path / "configs",
    )

    with TestClient(app) as client:
        response = client.get("/market", params={"symbol": "600000"})

    assert response.status_code == 200
    assert 'value="600000" selected' in response.text
    assert "浦发银行" in response.text


def test_history_clear_needs_csrf_and_confirm_token(tmp_path):
    app = create_app(
        runs_dir=tmp_path / "runs",
        db_path=tmp_path / "portfolio.sqlite",
        data_dir=tmp_path / "data",
    )

    with TestClient(app) as client:
        client.get("/")
        token = client.cookies.get("quant_csrf")
        assert token

        without_csrf = client.post("/api/history/clear", json={"scopes": ["runs"], "confirm": "CLEAR"})
        assert without_csrf.status_code == 403

        bad_token = client.post(
            "/api/history/clear",
            json={"scopes": ["runs"], "confirm": "nope"},
            headers={"X-CSRF-Token": token},
        )
        assert bad_token.status_code == 422
