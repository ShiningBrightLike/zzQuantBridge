import json
from pathlib import Path

import pytest

from quant.cli import main
from quant.maintenance import MaintenanceError, clear_history, format_bytes, format_mb, history_summary


def _project(tmp_path: Path) -> dict[str, Path]:
    runs = tmp_path / "runs"
    data = tmp_path / "data"
    db = tmp_path / "db" / "portfolio.sqlite"
    for name, payload in (("2026-10-08", "a"), ("2026-10-09", "b")):
        run_dir = runs / name / payload
        run_dir.mkdir(parents=True)
        (run_dir / "input_manifest.json").write_text(json.dumps({"run_id": payload}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    for kind in ("raw", "processed"):
        (data / kind / "snapshot-1").mkdir(parents=True)
        (data / kind / "snapshot-1" / "bars.parquet").write_bytes(b"x" * 32)
    (data / "metadata").mkdir(parents=True)
    (data / "metadata" / "snapshot-1.json").write_text("{}", encoding="utf-8")
    db.parent.mkdir(parents=True)
    db.write_bytes(b"sqlite")
    return {"runs": runs, "data": data, "db": db}


def test_summary_counts_known_history(tmp_path):
    paths = _project(tmp_path)

    summary = history_summary(runs_dir=paths["runs"], data_dir=paths["data"], db_path=paths["db"])

    assert summary["runs"]["count"] == 2
    assert summary["snapshots"]["count"] == 2
    assert summary["metadata"]["count"] == 1
    assert summary["database"]["count"] == 1


def test_preview_deletes_nothing(tmp_path):
    paths = _project(tmp_path)

    report = clear_history(
        runs_dir=paths["runs"], data_dir=paths["data"], db_path=paths["db"], dry_run=True
    )

    assert report["dry_run"] is True
    assert report["total"] == 5
    assert (paths["runs"] / "2026-10-09" / "b" / "status.json").exists()
    assert (paths["data"] / "raw" / "snapshot-1").exists()
    assert paths["db"].exists()


def test_requires_confirmation_when_not_dry_run(tmp_path):
    paths = _project(tmp_path)

    with pytest.raises(MaintenanceError, match="confirm"):
        clear_history(runs_dir=paths["runs"], data_dir=paths["data"], db_path=paths["db"])


def test_clear_runs_keeps_data_and_database(tmp_path):
    paths = _project(tmp_path)

    report = clear_history(
        runs_dir=paths["runs"],
        data_dir=paths["data"],
        db_path=paths["db"],
        runs=True,
        data=False,
        database=False,
        confirm=True,
        audit=False,
    )

    assert report["counts"]["runs"] == 2
    assert list((paths["runs"]).glob("*/*")) == []
    assert (paths["data"] / "raw" / "snapshot-1").exists()
    assert paths["db"].exists()


def test_keep_days_preserves_recent_runs(tmp_path):
    paths = _project(tmp_path)
    old = paths["runs"] / "2026-01-01" / "old"
    old.mkdir(parents=True)
    (old / "input_manifest.json").write_text("{}", encoding="utf-8")
    import os
    import time

    stale = time.time() - 30 * 24 * 3600
    os.utime(old, (stale, stale))

    report = clear_history(
        runs_dir=paths["runs"],
        data_dir=paths["data"],
        db_path=paths["db"],
        runs=True,
        data=False,
        keep_days=7,
        confirm=True,
        audit=False,
    )

    assert report["counts"]["runs"] == 1
    assert not old.exists()
    assert (paths["runs"] / "2026-10-09" / "b").exists()


def test_unrelated_directories_are_left_alone(tmp_path):
    paths = _project(tmp_path)
    stray = paths["runs"] / "notes"
    stray.mkdir(parents=True)
    (stray / "todo.txt").write_text("keep me", encoding="utf-8")
    half_run = paths["runs"] / "2026-10-10" / "no-manifest"
    half_run.mkdir(parents=True)

    clear_history(
        runs_dir=paths["runs"],
        data_dir=paths["data"],
        db_path=paths["db"],
        runs=True,
        data=False,
        confirm=True,
        audit=False,
    )

    assert (stray / "todo.txt").exists()
    assert half_run.exists()


def test_database_scope_removes_sqlite(tmp_path):
    paths = _project(tmp_path)

    report = clear_history(
        runs_dir=paths["runs"],
        data_dir=paths["data"],
        db_path=paths["db"],
        runs=False,
        data=False,
        database=True,
        confirm=True,
        audit=False,
    )

    assert report["counts"]["database"] == 1
    assert not paths["db"].exists()


def test_format_bytes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KB"


def test_format_mb():
    assert format_mb(0) == "0.00 MB"
    assert format_mb(1024 * 1024) == "1.00 MB"
    assert format_mb(int(2.5 * 1024 * 1024)) == "2.50 MB"


def test_cli_preview_returns_zero_and_keeps_history(tmp_path, capsys):
    paths = _project(tmp_path)

    code = main(
        [
            "clear-history",
            "--runs-dir",
            str(paths["runs"]),
            "--data-dir",
            str(paths["data"]),
            "--db-path",
            str(paths["db"]),
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "preview only" in output
    assert (paths["runs"] / "2026-10-09" / "b" / "status.json").exists()
