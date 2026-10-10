"""Append-only storage for raw/processed market snapshots and manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .validation import normalize_ohlcv


class ArchiveError(RuntimeError):
    """Raised when a validated snapshot cannot be persisted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_snapshot(
    frame: Any,
    root: str | Path,
    *,
    source: str,
    request: dict[str, Any],
    adjustment: str,
) -> Path:
    """Write a canonical snapshot once and return its JSON manifest path.

    Files are partitioned by trading date and symbol. Existing files are never
    overwritten; callers can safely retry with a new snapshot id.
    """

    normalized = normalize_ohlcv(frame)
    root_path = Path(root)
    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    raw_dir = root_path / "raw" / snapshot_id
    processed_dir = root_path / "processed" / snapshot_id
    metadata_dir = root_path / "metadata"
    files: list[dict[str, str]] = []
    for symbol, group in normalized.groupby("symbol", sort=True):
        # One Parquet file per symbol per snapshot; per-day files explode into
        # hundreds of thousands of tiny files for a realistic universe.
        relative = Path(f"symbol={symbol}") / "bars.parquet"
        processed_path = processed_dir / relative
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            group.to_parquet(processed_path, index=False)
        except ImportError as exc:  # pragma: no cover - optional parquet backend
            raise ArchiveError("pyarrow is required to write Parquet snapshots; install the data extra") from exc
        files.append({"kind": "processed", "path": str(processed_path.relative_to(root_path)), "sha256": _sha256(processed_path)})

    # Keep the canonical provider response as the raw archive for now. A future
    # provider may add its untouched payload alongside this normalized copy.
    raw_path = raw_dir / "ohlcv.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        normalized.to_parquet(raw_path, index=False)
    except ImportError as exc:  # pragma: no cover - optional parquet backend
        raise ArchiveError("pyarrow is required to write Parquet snapshots; install the data extra") from exc
    files.append({"kind": "raw", "path": str(raw_path.relative_to(root_path)), "sha256": _sha256(raw_path)})

    manifest = {
        "snapshot_id": snapshot_id,
        "source": source,
        "request": request,
        "adjustment": adjustment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(normalized)),
        "files": files,
    }
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / f"{snapshot_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path
