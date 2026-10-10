import json

from quant.reports import write_run_report


def test_run_report_contains_risk_then_actions(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "input_manifest.json").write_text(
        json.dumps({"run_id": "123456-abcd", "command": "generate-signals"}), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(json.dumps({"status": "BLOCKED"}), encoding="utf-8")
    (run_dir / "risk_report.json").write_text(
        json.dumps(
            {
                "status": "BLOCKED",
                "reasons": ["turnover 0.9 exceeds limit 0.5"],
                "symbol_flags": {"AAA": ["cap"]},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "orders.csv").write_text(
        "symbol,action,quantity,target_weight,reference_price,status,risk_flags,reason\n"
        'AAA,BUY,100,0.2,10.0,BLOCKED,"cap, turnover",trend on\n',
        encoding="utf-8-sig",
    )

    path = write_run_report(run_dir)
    body = path.read_text(encoding="utf-8")

    assert path.name == "report.html"
    assert "123456-abcd" in body
    assert body.index("1. 风险") < body.index("2. 动作") < body.index("3. 原因")
    assert "turnover 0.9 exceeds limit 0.5" in body
    assert "cap, turnover" in body
