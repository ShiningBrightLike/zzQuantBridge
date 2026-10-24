from datetime import datetime, timezone
import os

import pytest

pd = pytest.importorskip("pandas")

from quant.data.archive import archive_snapshot
from quant.tasks import latest_snapshot


def _frame(day: int):
    return pd.DataFrame(
        [
            {
                "symbol": "600000",
                "timestamp": datetime(2026, 10, day, tzinfo=timezone.utc),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000,
            }
        ]
    )


def test_archive_manifest_records_actual_data_cutoff(tmp_path):
    manifest_path = archive_snapshot(
        _frame(9),
        tmp_path,
        source="test",
        request={"start": "2026-10-01", "end": "2026-10-10"},
        adjustment="",
    )

    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_as_of"] == "2026-10-09"
    assert manifest["symbol_count"] == 1


def test_latest_snapshot_uses_data_cutoff_instead_of_file_mtime(tmp_path):
    older_data = archive_snapshot(
        _frame(9),
        tmp_path,
        source="test",
        request={"start": "2026-10-01", "end": "2026-10-10"},
        adjustment="",
    )
    newer_data = archive_snapshot(
        _frame(10),
        tmp_path,
        source="test",
        request={"start": "2026-10-01", "end": "2026-10-10"},
        adjustment="",
    )
    future = datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp()
    past = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(older_data, (future, future))
    os.utime(newer_data, (past, past))

    _, manifest = latest_snapshot(tmp_path)

    assert manifest["data_as_of"] == "2026-10-10"
