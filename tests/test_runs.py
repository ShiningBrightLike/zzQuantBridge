import json
from datetime import date

from quant.runs import create_run


def test_create_run_writes_manifest(tmp_path):
    context = create_run(tmp_path, date(2026, 10, 8), command="backtest", config_paths=["configs/strategy.yaml"])

    assert context.run_dir.parent.name == "2026-10-08"
    manifest = json.loads((context.run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == context.run_id
    assert manifest["command"] == "backtest"
    assert manifest["config_paths"] == ["configs/strategy.yaml"]
