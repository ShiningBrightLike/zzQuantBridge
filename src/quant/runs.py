"""Creation of reproducible run directories and manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from uuid import uuid4
from typing import Any


@dataclass(frozen=True)
class RunContext:
    """Filesystem locations and identity for one command invocation."""

    run_id: str
    run_dir: Path
    started_at: datetime


def _fingerprint(command: str, config_paths: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(command.encode("utf-8"))
    for name in config_paths:
        path = Path(name)
        digest.update(str(path).encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def create_run(
    runs_dir: str | Path = "runs",
    run_date: date | None = None,
    *,
    command: str,
    config_paths: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> RunContext:
    """Create a unique run directory and persist its input manifest."""

    started_at = datetime.now(timezone.utc)
    effective_date = run_date or started_at.date()
    run_id = f"{started_at.strftime('%H%M%S')}-{uuid4().hex[:8]}"
    run_dir = Path(runs_dir) / effective_date.isoformat() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "command": command,
        "started_at": started_at.isoformat(),
        "run_date": effective_date.isoformat(),
        "config_paths": config_paths or [],
        "params": params or {},
        "input_fingerprint": _fingerprint(command, config_paths or []),
    }
    (run_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return RunContext(run_id=run_id, run_dir=run_dir, started_at=started_at)


def configure_run_logging(context: RunContext) -> logging.Logger:
    """Configure a run-scoped logger without changing the application's root logger."""

    logger = logging.getLogger(f"quant.run.{context.run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(context.run_dir / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s run_id=%(name)s %(message)s"))
        logger.addHandler(handler)
    return logger
