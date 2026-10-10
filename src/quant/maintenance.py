"""Inspect and clear local run, market-data, and portfolio history."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any

from .runs import create_run


class MaintenanceError(RuntimeError):
    """Raised when a maintenance action is refused or cannot be completed."""


CONFIRM_TOKEN = "CLEAR"
RUN_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _guard(label: str, path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise MaintenanceError(f"refusing to clear a filesystem root for {label}: {resolved}")
    if resolved == Path.home().resolve():
        raise MaintenanceError(f"refusing to clear the home directory for {label}: {resolved}")
    return resolved


def _run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    found: list[Path] = []
    for day_dir in runs_dir.iterdir():
        if not day_dir.is_dir() or not RUN_DATE_PATTERN.match(day_dir.name):
            continue
        for run_dir in day_dir.iterdir():
            # Only real run folders are candidates; anything else stays put.
            if run_dir.is_dir() and (run_dir / "input_manifest.json").exists():
                found.append(run_dir)
    return sorted(found)


def _within(label: str, root: Path, candidate: Path) -> Path:
    """Resolve a deletion target, refusing anything outside its root."""

    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise MaintenanceError(f"refusing to touch a path outside {resolved_root} for {label}: {resolved}")
    return resolved


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_mb(value: int | float) -> str:
    """Render a byte count in megabytes for the Web UI."""

    return f"{float(value) / (1024 * 1024):.2f} MB"


def _snapshot_dirs(data_dir: Path) -> list[Path]:
    directories: list[Path] = []
    for name in ("raw", "processed"):
        base = data_dir / name
        if base.is_dir():
            directories.extend(sorted(item for item in base.iterdir() if item.is_dir()))
    return directories


def _metadata_files(data_dir: Path) -> list[Path]:
    base = data_dir / "metadata"
    if not base.is_dir():
        return []
    return sorted(item for item in base.glob("*.json") if item.is_file())


def _database_files(db_path: Path) -> list[Path]:
    candidates = [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]
    return [item for item in candidates if item.is_file()]


def _modified_at(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


def _describe(paths: list[Path], root: Path | None = None) -> dict[str, Any]:
    return {
        "count": len(paths),
        "bytes": sum(_tree_size(path) for path in paths),
        "items": [str(path.relative_to(root)) if root else str(path) for path in paths[:50]],
    }


def history_summary(
    *,
    runs_dir: str | Path = "runs",
    data_dir: str | Path = "data",
    db_path: str | Path = "db/portfolio.sqlite",
) -> dict[str, Any]:
    """Return a non-destructive summary of what a clear would remove."""

    runs_root = Path(runs_dir)
    data_root = Path(data_dir)
    database = Path(db_path)
    run_items = _run_dirs(runs_root)
    snapshot_items = _snapshot_dirs(data_root)
    metadata_items = _metadata_files(data_root)
    database_items = _database_files(database)
    return {
        "runs": _describe(run_items, runs_root),
        "snapshots": _describe(snapshot_items, data_root),
        "metadata": _describe(metadata_items, data_root),
        "database": _describe(database_items),
        "paths": {
            "runs_dir": str(runs_root.resolve()),
            "data_dir": str(data_root.resolve()),
            "db_path": str(database.resolve()),
        },
    }


def clear_history(
    *,
    runs_dir: str | Path = "runs",
    data_dir: str | Path = "data",
    db_path: str | Path = "db/portfolio.sqlite",
    runs: bool = True,
    data: bool = True,
    database: bool = False,
    keep_days: int | None = None,
    confirm: bool = False,
    dry_run: bool = False,
    audit: bool = True,
) -> dict[str, Any]:
    """Delete selected history.

    Only known history locations are removed: ``runs/<date>/<run_id>``,
    ``data/raw|processed/<snapshot>``, ``data/metadata/*.json`` and the
    portfolio database file. Parent directories are preserved.
    """

    if not confirm and not dry_run:
        raise MaintenanceError("clear_history requires confirm=True")
    if not any((runs, data, database)):
        raise MaintenanceError("select at least one of runs, data, or database")
    if keep_days is not None and keep_days < 0:
        raise MaintenanceError("keep_days must not be negative")

    runs_root = _guard("runs_dir", runs_dir)
    data_root = _guard("data_dir", data_dir)
    database_path = _guard("db_path", db_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days) if keep_days else None

    def keep(path: Path) -> bool:
        return cutoff is not None and _modified_at(path) >= cutoff

    removed: dict[str, list[str]] = {"runs": [], "snapshots": [], "metadata": [], "database": []}
    bytes_freed = 0
    if runs:
        for candidate in _run_dirs(runs_root):
            path = _within("runs", runs_root, candidate)
            if keep(path):
                continue
            bytes_freed += _tree_size(path)
            if not dry_run:
                shutil.rmtree(path)
            removed["runs"].append(path.relative_to(runs_root).as_posix())
    if data:
        for candidate in _snapshot_dirs(data_root):
            path = _within("data", data_root, candidate)
            if keep(path):
                continue
            bytes_freed += _tree_size(path)
            if not dry_run:
                shutil.rmtree(path)
            removed["snapshots"].append(path.relative_to(data_root).as_posix())
        for candidate in _metadata_files(data_root):
            path = _within("metadata", data_root, candidate)
            if keep(path):
                continue
            bytes_freed += _tree_size(path)
            if not dry_run:
                path.unlink()
            removed["metadata"].append(path.relative_to(data_root).as_posix())
    if database:
        for candidate in _database_files(database_path):
            path = _within("database", database_path.parent, candidate)
            bytes_freed += _tree_size(path)
            if not dry_run:
                path.unlink()
            removed["database"].append(path.name)

    report: dict[str, Any] = {
        "bytes_freed": bytes_freed,
        "removed": {key: sorted(value) for key, value in removed.items()},
        "counts": {key: len(value) for key, value in removed.items()},
        "total": sum(len(value) for value in removed.values()),
        "keep_days": keep_days,
        "dry_run": dry_run,
        "selection": {"runs": runs, "data": data, "database": database},
    }
    if audit and not dry_run:
        context = create_run(
            runs_root,
            command="clear-history",
            params={"keep_days": keep_days, "selection": report["selection"]},
        )
        (context.run_dir / "status.json").write_text(
            _dump_json({"status": "PASS", **report}), encoding="utf-8"
        )
        report["audit_run_id"] = context.run_id
    return report


def _dump_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
