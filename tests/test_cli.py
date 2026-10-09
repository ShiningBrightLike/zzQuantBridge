import json
from pathlib import Path

from quant.cli import main


def test_cli_initializes_run(tmp_path, capsys):
    assert main(["report", "--date", "2026-10-08", "--runs-dir", str(tmp_path)]) == 0

    output = capsys.readouterr().out.strip()
    run_dir = Path(output)
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "INITIALIZED"
