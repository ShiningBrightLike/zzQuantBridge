"""Local FastAPI server for the manual-execution quant dashboard."""

import argparse
from contextlib import asynccontextmanager
from datetime import date
import json
from pathlib import Path
import secrets
from typing import Any

from .webapp import (
    JobConflictError,
    JobManager,
    commit_trades,
    empty_plot,
    find_run,
    list_runs,
    parse_fill_fields,
    parse_fill_rows,
    preview_trades,
)
from .tasks import build_task, equity_figure, latest_orders, load_snapshot, market_figure
from .maintenance import CONFIRM_TOKEN, MaintenanceError, clear_history, format_mb, history_summary


def create_app(
    *,
    runs_dir: str | Path = "runs",
    db_path: str | Path = "db/portfolio.sqlite",
    data_dir: str | Path = "data",
    configs_dir: str | Path = "configs",
    templates_dir: str | Path | None = None,
) -> Any:
    try:
        from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
        from fastapi.templating import Jinja2Templates
    except ImportError as exc:  # pragma: no cover - optional web extra
        raise RuntimeError("Web UI dependencies are required; install the web extra") from exc

    base_dir = Path(__file__).resolve().parent
    template_root = Path(templates_dir) if templates_dir else base_dir / "templates"
    templates = Jinja2Templates(directory=str(template_root))
    static_root = base_dir / "static"
    manager = JobManager(runs_dir)

    @asynccontextmanager
    async def lifespan(_app: Any):
        yield
        manager.shutdown()

    app = FastAPI(title="zzQuantBridge", version="0.1.0", lifespan=lifespan)
    app.state.job_manager = manager
    app.state.runs_dir = Path(runs_dir)
    app.state.db_path = Path(db_path)
    app.state.data_dir = Path(data_dir)
    app.state.configs_dir = Path(configs_dir)

    def ensure_csrf(request: Request, x_csrf_token: str | None = Header(default=None)) -> str:
        cookie = request.cookies.get("quant_csrf")
        if not cookie or not x_csrf_token or not secrets.compare_digest(cookie, x_csrf_token):
            raise HTTPException(status_code=403, detail="CSRF token is missing or invalid")
        return cookie

    def page_context(request: Request, **values: Any) -> dict[str, Any]:
        token = request.cookies.get("quant_csrf") or secrets.token_urlsafe(24)
        return {"request": request, "csrf_token": token, **values}

    def page_response(template: str, request: Request, **values: Any) -> HTMLResponse:
        context = page_context(request, **values)
        response = templates.TemplateResponse(request, template, context)
        # This is a local single-user tool: never let the browser serve a stale
        # page or asset after the code changes.
        response.headers["Cache-Control"] = "no-store"
        if "quant_csrf" not in request.cookies:
            response.set_cookie("quant_csrf", context["csrf_token"], httponly=True, samesite="lax")
        return response

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return page_response("home.html", request, runs=list_runs(runs_dir, limit=10), active_run_id=manager.active_run_id)

    @app.get("/market", response_class=HTMLResponse)
    def market(request: Request) -> HTMLResponse:
        try:
            frame, manifest = load_snapshot(Path(data_dir))
        except (FileNotFoundError, ImportError, ValueError) as exc:
            return page_response("market.html", request, figure=empty_plot("行情"), message=str(exc))
        figure, message = market_figure(frame)
        return page_response("market.html", request, figure=figure, message=message, snapshot=manifest.get("snapshot_id"))

    @app.get("/backtest", response_class=HTMLResponse)
    def backtest(request: Request) -> HTMLResponse:
        figure, message = equity_figure(runs_dir)
        return page_response("backtest.html", request, figure=figure, message=message, runs=list_runs(runs_dir, limit=20))

    @app.get("/portfolio", response_class=HTMLResponse)
    def portfolio(request: Request) -> HTMLResponse:
        from .portfolio import PortfolioStore

        try:
            snapshot = PortfolioStore(db_path).snapshot(0.0)
        except ValueError:
            snapshot = None
        return page_response("portfolio.html", request, snapshot=snapshot, db_path=str(db_path))

    @app.get("/reconcile", response_class=HTMLResponse)
    def reconcile(request: Request) -> HTMLResponse:
        return page_response("reconcile.html", request)

    @app.get("/maintenance", response_class=HTMLResponse)
    def maintenance(request: Request) -> HTMLResponse:
        summary = history_summary(runs_dir=runs_dir, data_dir=data_dir, db_path=db_path)
        total_bytes = sum(
            section.get("bytes", 0)
            for key, section in summary.items()
            if key != "paths" and isinstance(section, dict)
        )
        return page_response(
            "maintenance.html",
            request,
            summary=summary,
            total_mb=format_mb(total_bytes),
            format_mb=format_mb,
        )

    @app.get("/suggestions", response_class=HTMLResponse)
    def suggestions(request: Request) -> HTMLResponse:
        rows, risk_status, risk_reasons = latest_orders(runs_dir)
        return page_response("suggestions.html", request, suggestions=rows, risk_status=risk_status, risk_reasons=risk_reasons)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(run_id: str, request: Request) -> HTMLResponse:
        record = find_run(run_id, runs_dir)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return page_response("run.html", request, run=record)

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str) -> JSONResponse:
        record = find_run(run_id, runs_dir)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return JSONResponse({"run_id": record.run_id, "command": record.command, "status": record.status, **record.payload})

    @app.post("/api/runs/{kind}")
    async def start_run(kind: str, request: Request, _csrf: str = Depends(ensure_csrf)) -> JSONResponse:
        try:
            body = await request.body()
            params = json.loads(body) if body else {}
        except ValueError:
            raise HTTPException(status_code=422, detail="request body must be a JSON object")
        if not isinstance(params, dict):
            raise HTTPException(status_code=422, detail="request body must be a JSON object")
        try:
            task = build_task(kind, params, data_dir=data_dir, configs_dir=configs_dir, db_path=db_path)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            context = manager.submit(kind, task, params=params)
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"run_id": context.run_id, "status": "QUEUED"}, status_code=202)

    @app.post("/api/reconcile/preview")
    async def reconcile_preview(
        request: Request,
        file: UploadFile | None = File(default=None),
        trade_id: str | None = Form(default=None),
        symbol: str | None = Form(default=None),
        side: str | None = Form(default=None),
        quantity: str | None = Form(default=None),
        price: str | None = Form(default=None),
        fee: str | None = Form(default=None),
        executed_at: str | None = Form(default=None),
        broker_reference: str | None = Form(default=None),
        note: str | None = Form(default=None),
        _csrf: str = Depends(ensure_csrf),
    ) -> JSONResponse:
        try:
            trades = parse_fill_rows((await file.read()).decode("utf-8-sig")) if file else parse_fill_fields(locals())
            return JSONResponse(preview_trades(trades, db_path))
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/reconcile/commit")
    async def reconcile_commit(
        request: Request,
        file: UploadFile | None = File(default=None),
        trade_id: str | None = Form(default=None),
        symbol: str | None = Form(default=None),
        side: str | None = Form(default=None),
        quantity: str | None = Form(default=None),
        price: str | None = Form(default=None),
        fee: str | None = Form(default=None),
        executed_at: str | None = Form(default=None),
        broker_reference: str | None = Form(default=None),
        note: str | None = Form(default=None),
        _csrf: str = Depends(ensure_csrf),
    ) -> JSONResponse:
        try:
            trades = parse_fill_rows((await file.read()).decode("utf-8-sig")) if file else parse_fill_fields(locals())
            return JSONResponse(commit_trades(trades, db_path))
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/history/summary")
    def history_status() -> JSONResponse:
        return JSONResponse(history_summary(runs_dir=runs_dir, data_dir=data_dir, db_path=db_path))

    @app.post("/api/history/clear")
    async def history_clear(request: Request, _csrf: str = Depends(ensure_csrf)) -> JSONResponse:
        try:
            body = await request.body()
            params = json.loads(body) if body else {}
        except ValueError:
            raise HTTPException(status_code=422, detail="request body must be a JSON object") from None
        if not isinstance(params, dict):
            raise HTTPException(status_code=422, detail="request body must be a JSON object")
        if params.get("confirm") != CONFIRM_TOKEN:
            raise HTTPException(status_code=422, detail=f'confirmation token must be "{CONFIRM_TOKEN}"')
        if manager.active_run_id:
            raise HTTPException(status_code=409, detail=f"a job is running: {manager.active_run_id}")
        scopes = params.get("scopes") or ["runs"]
        try:
            report = clear_history(
                runs_dir=runs_dir,
                data_dir=data_dir,
                db_path=db_path,
                runs="runs" in scopes,
                data="data" in scopes,
                database="database" in scopes,
                keep_days=params.get("keep_days"),
                confirm=True,
            )
        except MaintenanceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(report)

    @app.get("/static/plotly.min.js")
    def plotly_bundle() -> FileResponse:
        try:
            import plotly
        except ImportError as exc:  # pragma: no cover
            raise HTTPException(status_code=404, detail="install the web extra for Plotly") from exc
        bundle = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if bundle.exists():
            return FileResponse(bundle, media_type="application/javascript", headers={"Cache-Control": "no-store"})
        from plotly.offline import get_plotlyjs

        return Response(
            content=get_plotlyjs(),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/static/{asset_name}")
    def static_asset(asset_name: str) -> FileResponse:
        allowed = {"app.css": "text/css", "app.js": "application/javascript"}
        if asset_name not in allowed:
            raise HTTPException(status_code=404, detail="asset not found")
        path = static_root / asset_name
        return FileResponse(path, media_type=allowed[asset_name], headers={"Cache-Control": "no-store"})

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-server", description="Run the local quant dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--db", default="db/portfolio.sqlite")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--configs-dir", default="configs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install the web extra before starting the server: pip install -e '.[web]'") from exc
    uvicorn.run(
        create_app(
            runs_dir=args.runs_dir,
            db_path=args.db,
            data_dir=args.data_dir,
            configs_dir=args.configs_dir,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
