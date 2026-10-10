from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from quant.cli import main


def _write_fixture(path: Path, symbol: str, days: int = 90) -> Path:
    rows = ["symbol,timestamp,open,high,low,close,volume"]
    price = 10.0
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(days):
        price += 0.05
        day = (start + timedelta(days=index)).date().isoformat()
        rows.append(f"{symbol},{day},{price:.2f},{price + 0.2:.2f},{price - 0.2:.2f},{price:.2f},{1000 + index}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_download_command_creates_audited_run(tmp_path, capsys):
    snapshot = _write_fixture(tmp_path / "bars.csv", "600000")
    runs = tmp_path / "runs"

    code = main(
        [
            "download",
            "--provider",
            "local",
            "--file",
            str(snapshot),
            "--symbols",
            "600000",
            "--start",
            "2026-01-01",
            "--end",
            "2026-12-31",
            "--runs-dir",
            str(runs),
            "--data-dir",
            str(tmp_path / "data"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    manifests = list(runs.glob("*/*/input_manifest.json"))
    assert len(manifests) == 1
    status = json.loads(manifests[0].with_name("status.json").read_text(encoding="utf-8"))
    assert status["status"] == "PASS"
    assert status["rows"] == 90


def test_download_command_reports_missing_local_file(tmp_path):
    code = main(
        [
            "download",
            "--provider",
            "local",
            "--symbols",
            "600000",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--data-dir",
            str(tmp_path / "data"),
        ]
    )

    assert code == 2
    manifests = list((tmp_path / "runs").glob("*/*/status.json"))
    assert json.loads(manifests[0].read_text(encoding="utf-8"))["status"] == "ERROR"


def test_report_command_needs_an_existing_run(tmp_path, capsys):
    code = main(["report", "--runs-dir", str(tmp_path / "runs")])

    assert code == 2
    assert "no run available" in capsys.readouterr().err
