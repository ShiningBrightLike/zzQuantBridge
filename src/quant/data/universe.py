"""Point-in-time A-share universe construction and online candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from ..config import load_yaml_config


@dataclass(frozen=True)
class UniverseConfig:
    """The small, explicit set of universe filters used by the first model."""

    mode: str = "config"
    auto_download: bool = True
    max_symbols: int = 800
    candidate_pool_size: int = 0
    min_history_days: int = 120
    min_avg_dollar_volume: float = 50_000_000.0
    exclude_st: bool = True
    exclude_bj: bool = True
    include_chi_next: bool = True
    include_star: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"config", "all_a"}:
            raise ValueError("universe mode must be 'config' or 'all_a'")
        if self.max_symbols <= 0:
            raise ValueError("max_symbols must be positive")
        if self.candidate_pool_size < 0:
            raise ValueError("candidate_pool_size must not be negative")
        if self.min_history_days < 0:
            raise ValueError("min_history_days must not be negative")
        if self.min_avg_dollar_volume < 0:
            raise ValueError("min_avg_dollar_volume must not be negative")


def load_universe_config(configs_dir: str | Path, overrides: Mapping[str, Any] | None = None) -> UniverseConfig:
    """Merge ``configs/universe.yaml`` with a run-level override mapping."""

    path = Path(configs_dir) / "universe.yaml"
    payload = load_yaml_config(path) if path.is_file() else {}
    for key, value in dict(overrides or {}).items():
        if value not in (None, ""):
            payload[key] = value
    allowed = {
        "mode",
        "auto_download",
        "max_symbols",
        "candidate_pool_size",
        "min_history_days",
        "min_avg_dollar_volume",
        "exclude_st",
        "exclude_bj",
        "include_chi_next",
        "include_star",
    }
    return UniverseConfig(**{key: value for key, value in payload.items() if key in allowed})


def config_symbols(configs_dir: str | Path) -> list[str]:
    path = Path(configs_dir) / "universe.yaml"
    payload = load_yaml_config(path) if path.is_file() else {}
    return [str(item).strip() for item in payload.get("symbols", []) if str(item).strip()]


def config_names(configs_dir: str | Path) -> dict[str, str]:
    path = Path(configs_dir) / "universe.yaml"
    payload = load_yaml_config(path) if path.is_file() else {}
    return {str(key): str(value) for key, value in (payload.get("names") or {}).items()}


def _board(code: str) -> str:
    if code.startswith(("60", "00")):
        return "main"
    if code.startswith(("30", "301")):
        return "chinext"
    if code.startswith(("68", "689")):
        return "star"
    if code.startswith(("4", "8", "92")):
        return "bse"
    return "other"


def _column(columns: Any, *candidates: str) -> Any | None:
    lookup = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def _amount(value: Any) -> float:
    text = str(value or "").strip().replace(",", "")
    multiplier = 1.0
    if "亿" in text:
        multiplier = 1e8
        text = text.replace("亿", "")
    elif "万" in text:
        multiplier = 1e4
        text = text.replace("万", "")
    try:
        return float(text) * multiplier
    except (TypeError, ValueError):
        return 0.0


def _candidates_from_frame(raw: Any, source: str) -> Any:
    import pandas as pd

    code_column = _column(raw.columns, "代码", "code", "symbol", "证券代码")
    name_column = _column(raw.columns, "名称", "name", "证券简称")
    amount_column = _column(raw.columns, "成交额", "amount", "成交金额")
    if code_column is None:
        raise RuntimeError(f"{source} universe response is missing a code column")
    codes = (
        raw[code_column]
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
    )
    names = (
        raw[name_column].astype(str).str.strip().str.replace(r"^(XD|XR|DR)", "", regex=True)
        if name_column is not None
        else ""
    )
    frame = pd.DataFrame({"symbol": codes, "name": names})
    frame = frame.loc[frame["symbol"].notna()].copy()
    frame["board"] = frame["symbol"].map(_board)
    frame["amount"] = (
        raw.loc[frame.index, amount_column].map(_amount)
        if amount_column is not None
        else 0.0
    )
    frame["source"] = source
    return frame


def fetch_all_a_candidates(settings: UniverseConfig | None = None) -> Any:
    """Fetch the current listed A-share snapshot from AkShare.

    The returned frame contains ``symbol``, ``name``, ``board`` and ``amount``.
    The snapshot is only a candidate pool; historical bars still have to be
    downloaded and validated before they can become a research snapshot.
    """

    config = settings or UniverseConfig(mode="all_a")
    try:
        import akshare as ak
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional data extra
        raise RuntimeError("AkShare and pandas are required to build the online A-share universe") from exc
    errors: list[str] = []
    frame = None
    for source, fetch in (
        ("eastmoney", ak.stock_zh_a_spot_em),
        ("sina", ak.stock_zh_a_spot),
        ("tencent", ak.stock_zh_a_spot_tx),
    ):
        try:
            raw = fetch()
            if raw is not None and len(raw):
                frame = _candidates_from_frame(raw, source)
                break
            errors.append(f"{source}: empty response")
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    if frame is None:
        raise RuntimeError("failed to fetch the A-share universe: " + "; ".join(errors))
    frame = frame.loc[frame["board"].ne("other")]
    if config.exclude_bj:
        frame = frame.loc[frame["board"].ne("bse")]
    if not config.include_chi_next:
        frame = frame.loc[frame["board"].ne("chinext")]
    if not config.include_star:
        frame = frame.loc[frame["board"].ne("star")]
    if config.exclude_st:
        upper = frame["name"].str.upper()
        frame = frame.loc[~upper.str.contains("ST|退", regex=True, na=False)]
    if not frame["amount"].fillna(0).gt(0).any():
        raise RuntimeError(
            "universe endpoints did not return turnover data; "
            "refusing to build an arbitrary code-ordered pool"
        )
    frame = frame.loc[frame["amount"].fillna(0) > 0]
    frame = frame.sort_values(["amount", "symbol"], ascending=[False, True], kind="stable")
    target_size = config.candidate_pool_size or config.max_symbols * 2
    return frame.head(target_size).reset_index(drop=True)


def build_universe_membership(frame: Any, settings: UniverseConfig | None = None) -> Any:
    """Build a daily, point-in-time liquidity/history mask for all symbols.

    The membership uses only the bar at each date and prior bars. It does not
    know the future liquidity ranking or future listing status. Current-name
    ST filtering is applied separately because the canonical OHLCV schema does
    not contain historical security names.
    """

    config = settings or UniverseConfig()
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional data extra
        raise RuntimeError("pandas is required for universe construction; install the data extra") from exc
    work = frame.copy()
    work["symbol"] = work["symbol"].astype(str)
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    for column in ("close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
    work["dollar_volume"] = work["close"] * work["volume"]
    grouped = work.groupby("symbol", sort=False, group_keys=False)
    work["history_days"] = grouped.cumcount() + 1
    work["avg_dollar_volume_60"] = grouped["dollar_volume"].transform(
        lambda values: values.rolling(60, min_periods=20).mean()
    )
    work["liquidity_rank"] = work.groupby("timestamp", sort=False)["avg_dollar_volume_60"].rank(
        ascending=False, method="first"
    )
    work["tradable"] = (
        (work["history_days"] >= config.min_history_days)
        & (work["avg_dollar_volume_60"].fillna(0) >= config.min_avg_dollar_volume)
        & (work["liquidity_rank"] <= config.max_symbols)
    )
    return work[["symbol", "timestamp", "tradable", "history_days", "avg_dollar_volume_60"]]


def current_tradable_symbols(membership: Any, as_of: Any) -> list[str]:
    """Return symbols tradable at the latest observation on/before ``as_of``."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for universe construction") from exc
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    available = membership.loc[membership["timestamp"] <= cutoff]
    if available.empty:
        return []
    latest = available["timestamp"].max()
    return sorted(
        str(item)
        for item in available.loc[
            (available["timestamp"] == latest) & available["tradable"], "symbol"
        ].tolist()
    )
