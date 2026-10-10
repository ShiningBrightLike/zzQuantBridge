"""Render a run directory into one self-contained HTML report.

Layout follows the plan (section 7.6): risk first, then actions, then reasons.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return '<p class="muted">无记录</p>'
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _status_class(status: str) -> str:
    mapping = {"PASS": "pass", "BLOCKED": "blocked", "ERROR": "error"}
    return mapping.get(status.upper(), "info")


def write_run_report(run_dir: str | Path, *, title: str | None = None) -> Path:
    """Write ``report.html`` into ``run_dir`` and return its path."""

    directory = Path(run_dir)
    manifest = _read_json(directory / "input_manifest.json")
    status = _read_json(directory / "status.json")
    risk = _read_json(directory / "risk_report.json")
    metrics = _read_json(directory / "backtest_metrics.json").get("metrics", {})
    sensitivity = _read_json(directory / "sensitivity.json")
    snapshot = _read_json(directory / "portfolio_snapshot.json")
    data_manifest = _read_json(directory / "data_manifest.json")
    orders = _read_csv(directory / "orders.csv")
    signals = _read_csv(directory / "signals.csv")
    trade_log = _read_csv(directory / "trades.csv")

    run_id = str(manifest.get("run_id", directory.name))
    command = str(manifest.get("command", status.get("command", "")))
    run_status = str(status.get("status", "UNKNOWN"))
    heading = title or f"{command or 'run'} · {run_id}"

    risk_rows = [{"check": reason, "result": "BLOCKED"} for reason in risk.get("reasons", [])]
    for symbol, flags in (risk.get("symbol_flags") or {}).items():
        for flag in flags:
            risk_rows.append({"check": f"{symbol}: {flag}", "result": "BLOCKED"})
    if not risk_rows:
        risk_rows = [{"check": "所有风险检查通过", "result": risk.get("status", run_status)}]

    metric_rows = [{"metric": str(key), "value": str(value)} for key, value in metrics.items()]
    sensitivity_rows = [
        {"scenario": str(key), "value": json.dumps(value, ensure_ascii=False)}
        for key, value in sensitivity.items()
    ]
    portfolio_rows = [
        {"item": str(key), "value": json.dumps(value, ensure_ascii=False)}
        for key, value in snapshot.items()
    ]
    data_rows = [
        {"item": str(key), "value": json.dumps(value, ensure_ascii=False)}
        for key, value in data_manifest.items()
    ]

    missing = [
        name
        for name, path in (
            ("backtest_metrics.json", directory / "backtest_metrics.json"),
            ("sensitivity.json", directory / "sensitivity.json"),
            ("orders.csv", directory / "orders.csv"),
            ("signals.csv", directory / "signals.csv"),
            ("portfolio_snapshot.json", directory / "portfolio_snapshot.json"),
        )
        if not path.exists()
    ]

    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(heading)}</title>
<style>
body{{background:#0b1020;color:#eef3ff;font:15px/1.6 system-ui,sans-serif;margin:0;padding:32px}}
main{{margin:0 auto;max-width:1100px}}
h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:17px;margin:28px 0 10px}}
.muted{{color:#93a2c4}} .meta{{color:#93a2c4;font-size:13px}}
.status{{border:1px solid currentColor;border-radius:999px;display:inline-block;font-size:12px;padding:3px 10px}}
.status-pass{{color:#2dd4a7}} .status-blocked{{color:#f6c85f}} .status-error{{color:#ff7d92}} .status-info{{color:#73a7ff}}
table{{border-collapse:collapse;margin-top:8px;width:100%}}
th,td{{border-bottom:1px solid #283654;padding:8px 10px;text-align:left}}
th{{color:#93a2c4;font-size:12px}} pre{{background:#0a0f1d;border:1px solid #283654;border-radius:8px;overflow:auto;padding:12px}}
section{{background:#131b2f;border:1px solid #283654;border-radius:12px;margin-top:18px;padding:18px}}
</style></head><body><main>
<h1>{html.escape(heading)}</h1>
<p class="meta">运行状态 <span class="status status-{_status_class(run_status)}">{html.escape(run_status)}</span>
 · 开始 {html.escape(str(manifest.get('started_at', '')))} · 运行目录 {html.escape(str(directory))}</p>
<section><h2>1. 风险</h2>{_table(risk_rows, ['check', 'result'])}</section>
<section><h2>2. 动作</h2>{_table(orders, ['symbol', 'action', 'quantity', 'target_weight', 'reference_price', 'status', 'risk_flags', 'reason'])}</section>
<section><h2>3. 原因</h2>{_table(signals, ['symbol', 'timestamp', 'target_weight', 'reason', 'model_version'])}</section>
<section><h2>4. 回测指标</h2>{_table(metric_rows, ['metric', 'value'])}</section>
<section><h2>5. 敏感性分析</h2>{_table(sensitivity_rows, ['scenario', 'value'])}</section>
<section><h2>6. 组合快照</h2>{_table(portfolio_rows, ['item', 'value'])}</section>
<section><h2>7. 输入数据</h2>{_table(data_rows, ['item', 'value'])}</section>
<section><h2>8. 逐笔交易</h2>{_table(trade_log, list(trade_log[0].keys()) if trade_log else ['trade'])}</section>
<p class="muted">本次运行未产出：{html.escape(', '.join(missing)) if missing else '无'}</p>
</main></body></html>
"""
    target = directory / "report.html"
    target.write_text(body, encoding="utf-8")
    return target
