import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
from fastapi.testclient import TestClient

from quant.server import create_app


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
