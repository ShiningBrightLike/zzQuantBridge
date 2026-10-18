"""Cross-sectional price/volume factors with explicit point-in-time boundaries.

The functions in this module deliberately keep labels separate from features.
Forward returns are only created by :func:`compute_factor_panel` when a
``label_horizon`` is supplied, and :func:`fit_factor_model` refuses to use a
label whose end date is later than the model's ``as_of`` date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


FACTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "momentum": ("momentum_20", "momentum_60", "momentum_120"),
    "reversal": ("reversal_5",),
    "volatility": (
        "volatility_20",
        "volatility_60",
        "downside_vol_20",
        "max_drawdown_60",
    ),
    "liquidity": ("liquidity_20", "volume_change_5_60", "amihud_20"),
    "distribution": ("skew_60", "turnover_volatility_20"),
    "technical": (
        "rsi_14",
        "macd_hist",
        "bollinger_position_20",
        "distance_high_252",
        "price_volume_corr_20",
    ),
}
FACTOR_NAMES: tuple[str, ...] = tuple(
    name for names in FACTOR_GROUPS.values() for name in names
)


def _pd() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional data extra
        raise RuntimeError("pandas is required for multifactor research; install the data extra") from exc
    return pd


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional data extra
        raise RuntimeError("numpy is required for multifactor research; install the data extra") from exc
    return np


@dataclass(frozen=True)
class FactorConfig:
    """Window settings for the portable A-share price/volume factors."""

    momentum_windows: tuple[int, ...] = (20, 60, 120)
    volatility_windows: tuple[int, ...] = (20, 60)
    reversal_window: int = 5
    liquidity_window: int = 20
    long_liquidity_window: int = 60
    rsi_window: int = 14
    high_window: int = 252

    def __post_init__(self) -> None:
        windows = (*self.momentum_windows, *self.volatility_windows)
        if any(int(window) <= 0 for window in windows):
            raise ValueError("factor windows must be positive")
        if self.reversal_window <= 0 or self.liquidity_window <= 0 or self.long_liquidity_window <= 0:
            raise ValueError("factor windows must be positive")
        if self.rsi_window <= 1 or self.high_window <= 1:
            raise ValueError("rsi and high windows must be greater than one")
        if sorted(self.momentum_windows) != [20, 60, 120]:
            raise ValueError("the first version supports momentum windows 20, 60 and 120 only")
        if sorted(self.volatility_windows) != [20, 60]:
            raise ValueError("the first version supports volatility windows 20 and 60 only")


@dataclass(frozen=True)
class FactorTrainingConfig:
    """Rolling model settings; labels must be fully realized before ``as_of``."""

    lookback_days: int = 504
    label_horizon: int = 20
    min_cross_section: int = 30
    min_ic_days: int = 120
    min_abs_ic: float = 0.02
    min_abs_icir: float = 0.10
    weight_alpha: float = 0.5
    max_factors: int = 12

    def __post_init__(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if self.label_horizon <= 0:
            raise ValueError("label_horizon must be positive")
        if self.min_cross_section < 2:
            raise ValueError("min_cross_section must be at least two")
        if self.min_ic_days <= 0:
            raise ValueError("min_ic_days must be positive")
        if not 0 <= self.min_abs_ic <= 1:
            raise ValueError("min_abs_ic must be between zero and one")
        if self.min_abs_icir < 0:
            raise ValueError("min_abs_icir must not be negative")
        if not 0 < self.weight_alpha <= 2:
            raise ValueError("weight_alpha must be in (0, 2]")
        if self.max_factors <= 0:
            raise ValueError("max_factors must be positive")


@dataclass(frozen=True)
class FactorModel:
    """A frozen set of factor directions and weights for one training window."""

    model_version: str
    factor_names: tuple[str, ...]
    weights: Mapping[str, float]
    metrics: Any
    trained_until: datetime
    lookback_start: datetime | None
    label_horizon: int
    fallback_used: bool = False

    def as_weights(self) -> dict[str, float]:
        return {name: float(self.weights[name]) for name in self.factor_names}


def _timestamp(value: Any) -> Any:
    pd = _pd()
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def _normalize_frame(frame: Any) -> Any:
    pd = _pd()
    required = {"symbol", "timestamp", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"factor input missing columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.strip()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    for column in ("close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["symbol", "timestamp", "close", "volume"]].isna().any().any():
        raise ValueError("factor input contains null keys or prices")
    if (result["close"] <= 0).any() or (result["volume"] < 0).any():
        raise ValueError("factor input requires positive close and non-negative volume")
    return result.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)


def _rolling(grouped: Any, column: str, window: int, operation: str) -> Any:
    def apply(values: Any) -> Any:
        rolling = values.rolling(window, min_periods=window)
        return getattr(rolling, operation)()

    return grouped[column].transform(apply)


def compute_factor_panel(
    frame: Any,
    settings: FactorConfig | None = None,
    *,
    label_horizon: int | None = None,
) -> Any:
    """Compute the portable factor panel without looking into the future.

    ``label_horizon`` is optional and is intended only for training/evaluation.
    When supplied, the output gains ``forward_return`` and ``label_end`` columns;
    neither is included in :data:`FACTOR_NAMES`.
    """

    pd = _pd()
    np = _np()
    config = settings or FactorConfig()
    result = _normalize_frame(frame)
    grouped = result.groupby("symbol", sort=False, group_keys=False)
    result["return_1"] = grouped["close"].pct_change()
    result["dollar_volume"] = result["close"] * result["volume"]

    for window in config.momentum_windows:
        result[f"momentum_{window}"] = result["close"] / grouped["close"].shift(window) - 1.0
    result[f"reversal_{config.reversal_window}"] = (
        result["close"] / grouped["close"].shift(config.reversal_window) - 1.0
    )

    for window in config.volatility_windows:
        result[f"volatility_{window}"] = _rolling(grouped, "return_1", window, "std")
    result["downside_vol_20"] = grouped["return_1"].transform(
        lambda values: values.clip(upper=0).rolling(20, min_periods=20).std(ddof=0)
    )
    result["max_drawdown_60"] = grouped["close"].transform(
        lambda values: (
            values / values.rolling(60, min_periods=60).max() - 1.0
        ).rolling(60, min_periods=60).min()
    )

    result["liquidity_20"] = np.log1p(
        grouped["dollar_volume"].transform(lambda values: values.rolling(20, min_periods=20).mean())
    )
    volume_mean_5 = grouped["volume"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    volume_mean_long = grouped["volume"].transform(
        lambda values: values.rolling(config.long_liquidity_window, min_periods=config.long_liquidity_window).mean()
    )
    result["volume_change_5_60"] = volume_mean_5 / volume_mean_long - 1.0
    result["_illiq"] = result["return_1"].abs() / result["dollar_volume"].replace(0, np.nan)
    result["amihud_20"] = result.groupby("symbol", sort=False)["_illiq"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    ) * 1e10

    result["skew_60"] = _rolling(grouped, "return_1", 60, "skew")
    result["turnover_volatility_20"] = grouped["volume"].transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=0)
        / values.rolling(20, min_periods=20).mean()
    )

    delta = grouped["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    result["_gain"] = gain
    result["_loss"] = loss
    avg_gain = result.groupby("symbol", sort=False)["_gain"].transform(
        lambda values: values.rolling(config.rsi_window, min_periods=config.rsi_window).mean()
    )
    avg_loss = result.groupby("symbol", sort=False)["_loss"].transform(
        lambda values: values.rolling(config.rsi_window, min_periods=config.rsi_window).mean()
    )
    result["rsi_14"] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0, np.nan))
    result["rsi_14"] = result["rsi_14"].where(avg_loss.ne(0), 100.0)
    result.loc[avg_gain.eq(0) & avg_loss.eq(0), "rsi_14"] = 50.0

    def ema(values: Any, span: int) -> Any:
        return values.ewm(span=span, adjust=False, min_periods=span).mean()

    ema12 = grouped["close"].transform(lambda values: ema(values, 12))
    ema26 = grouped["close"].transform(lambda values: ema(values, 26))
    dif = ema12 - ema26
    result["_dif"] = dif
    dea = result.groupby("symbol", sort=False)["_dif"].transform(lambda values: ema(values, 9))
    result["macd_hist"] = (dif - dea) / result["close"]

    ma20 = grouped["close"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    std20 = grouped["close"].transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=0)
    )
    result["bollinger_position_20"] = (result["close"] - ma20) / (2.0 * std20.replace(0, np.nan))
    result["distance_high_252"] = (
        result["close"]
        / grouped["close"].transform(
            lambda values: values.rolling(config.high_window, min_periods=config.high_window).max()
        )
        - 1.0
    )
    result["_volume_change"] = grouped["volume"].pct_change()
    correlations = []
    for _, group in result.groupby("symbol", sort=False):
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations.append(
                group["return_1"].rolling(20, min_periods=20).corr(group["_volume_change"])
            )
    result["price_volume_corr_20"] = (
        pd.concat(correlations).sort_index() if correlations else pd.Series(index=result.index, dtype="float64")
    )
    result = result.drop(columns=["_illiq", "_gain", "_loss", "_dif", "_volume_change"])

    for factor in FACTOR_NAMES:
        result[factor] = result[factor].replace([np.inf, -np.inf], np.nan)

    if label_horizon is not None:
        if label_horizon <= 0:
            raise ValueError("label_horizon must be positive")
        result["forward_return"] = grouped["close"].shift(-label_horizon) / result["close"] - 1.0
        result["label_end"] = grouped["timestamp"].shift(-label_horizon)

    keep = [
        "symbol",
        "timestamp",
        "close",
        "dollar_volume",
        *FACTOR_NAMES,
    ]
    if label_horizon is not None:
        keep.extend(["forward_return", "label_end"])
    return result[keep].sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def _cross_sectional_zscore(values: Any) -> Any:
    pd = _pd()
    valid = values.dropna()
    if len(valid) < 2:
        return values.notna().astype("float64") * 0.0
    lower, upper = valid.quantile(0.01), valid.quantile(0.99)
    clipped = values.clip(lower, upper)
    mean = clipped.mean()
    std = clipped.std(ddof=0)
    if not math.isfinite(float(std)) or float(std) == 0.0:
        return values.notna().astype("float64") * 0.0
    return (clipped - mean) / std


def _ic_series(panel: Any, factor: str, dates: Sequence[Any], min_cross_section: int) -> Any:
    pd = _pd()
    work = panel.loc[
        panel["timestamp"].isin(dates) & panel[factor].notna() & panel["forward_return"].notna(),
        ["timestamp", factor, "forward_return"],
    ].copy()
    if work.empty:
        return pd.Series(dtype="float64")
    work["_f_rank"] = work.groupby("timestamp")[factor].rank(pct=True)
    work["_r_rank"] = work.groupby("timestamp")["forward_return"].rank(pct=True)
    counts = work.groupby("timestamp")["_f_rank"].transform("count")
    work = work.loc[counts >= min_cross_section]
    if work.empty:
        return pd.Series(dtype="float64")
    try:
        return work.groupby("timestamp").apply(
            lambda group: group["_f_rank"].corr(group["_r_rank"]),
            include_groups=False,
        )
    except TypeError:  # pandas < 2.2
        return work.groupby("timestamp").apply(
            lambda group: group["_f_rank"].corr(group["_r_rank"])
        )


def fit_factor_model(
    panel: Any,
    as_of: Any,
    training: FactorTrainingConfig | None = None,
    *,
    factor_names: Sequence[str] = FACTOR_NAMES,
) -> FactorModel:
    """Fit factor directions/weights using only labels realized by ``as_of``."""

    pd = _pd()
    np = _np()
    settings = training or FactorTrainingConfig()
    required = {"timestamp", "forward_return", "label_end", *factor_names}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"training panel missing columns: {', '.join(sorted(missing))}")
    cutoff = _timestamp(as_of)
    available = panel.loc[
        (panel["timestamp"] <= cutoff)
        & (panel["label_end"].notna())
        & (panel["label_end"] <= cutoff)
    ].copy()
    if "tradable" in available.columns:
        available = available.loc[available["tradable"].fillna(False)].copy()
    if available.empty:
        raise ValueError("no fully realized forward-return labels are available at the requested cutoff")
    dates = sorted(available["timestamp"].dropna().unique())
    if len(dates) > settings.lookback_days:
        dates = dates[-settings.lookback_days :]
    train = available.loc[available["timestamp"].isin(dates)].copy()
    if train.empty:
        raise ValueError("training window contains no factor observations")

    metrics_rows: list[dict[str, Any]] = []
    split_at = max(1, len(dates) // 2)
    first_dates = set(dates[:split_at])
    second_dates = set(dates[split_at:])
    for factor in factor_names:
        ic = _ic_series(train, factor, dates, settings.min_cross_section).dropna()
        if ic.empty:
            metrics_rows.append(
                {
                    "factor": factor,
                    "ic_mean": float("nan"),
                    "ic_std": float("nan"),
                    "icir": float("nan"),
                    "ic_win_rate": float("nan"),
                    "ic_days": 0,
                    "sign_stable": False,
                }
            )
            continue
        ic_mean = float(ic.mean())
        ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
        # A perfectly consistent IC series has zero sample dispersion. Treat it
        # as a strong signal rather than collapsing its weight to zero.
        effective_std = max(ic_std, 1e-6) if math.isfinite(ic_std) else 1e-6
        icir = max(-10.0, min(10.0, ic_mean / effective_std)) if ic_mean else 0.0
        first_mean = float(ic.loc[ic.index.isin(first_dates)].mean()) if first_dates else ic_mean
        second_mean = float(ic.loc[ic.index.isin(second_dates)].mean()) if second_dates else ic_mean
        sign_stable = bool(
            math.isfinite(first_mean)
            and math.isfinite(second_mean)
            and first_mean * second_mean > 0
        ) if len(dates) >= 40 else True
        metrics_rows.append(
            {
                "factor": factor,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "ic_win_rate": float((ic > 0).mean()),
                "ic_days": int(len(ic)),
                "sign_stable": sign_stable,
            }
        )

    metrics = pd.DataFrame(metrics_rows)
    eligible = metrics.loc[
        (metrics["ic_days"] >= settings.min_ic_days)
        & (metrics["ic_mean"].abs() >= settings.min_abs_ic)
        & (metrics["icir"].abs() >= settings.min_abs_icir)
        & metrics["sign_stable"]
    ].copy()
    fallback = False
    if eligible.empty:
        # A young or quiet universe may not clear the strict thresholds. Keep
        # the model usable, but mark the fallback so reports do not hide it.
        eligible = metrics.loc[
            (metrics["ic_days"] >= settings.min_ic_days)
            & (metrics["ic_mean"].notna())
            & (metrics["icir"].notna())
        ].copy()
        fallback = True
    if eligible.empty:
        raise ValueError("not enough cross-sectional history to estimate any factor weight")
    eligible = eligible.reindex(eligible["icir"].abs().sort_values(ascending=False).index)
    eligible = eligible.head(settings.max_factors)
    raw = np.sign(eligible["ic_mean"]) * eligible["icir"].abs().pow(settings.weight_alpha)
    denominator = float(raw.abs().sum())
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError("factor weights are degenerate")
    weights = raw / denominator
    selected = tuple(str(item) for item in eligible["factor"])
    fingerprint = {
        "factors": selected,
        "weights": {name: round(float(weights.loc[index]), 12) for index, name in zip(eligible.index, selected)},
        "label_horizon": settings.label_horizon,
        "lookback_days": settings.lookback_days,
        "trained_until": cutoff.isoformat(),
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return FactorModel(
        model_version=f"mf-{settings.label_horizon}d-{digest}",
        factor_names=selected,
        weights={name: float(weights.loc[index]) for index, name in zip(eligible.index, selected)},
        metrics=metrics,
        trained_until=cutoff.to_pydatetime(),
        lookback_start=_timestamp(dates[0]).to_pydatetime() if dates else None,
        label_horizon=settings.label_horizon,
        fallback_used=fallback,
    )


def score_cross_section(panel: Any, model: FactorModel, as_of: Any) -> Any:
    """Score the latest cross-section using only factors observed at/before ``as_of``."""

    pd = _pd()
    cutoff = _timestamp(as_of)
    observed = panel.loc[panel["timestamp"] <= cutoff, "timestamp"]
    if observed.empty:
        raise ValueError("no factor observations are available at the requested cutoff")
    latest = observed.max()
    if latest != cutoff:
        raise ValueError(f"no factor cross-section is available at {cutoff.isoformat()}")
    current = panel.loc[panel["timestamp"] == latest].copy()
    if current.empty:
        raise ValueError("latest factor cross-section is empty")
    missing = set(model.factor_names).difference(current.columns)
    if missing:
        raise ValueError(f"scoring panel missing model factors: {', '.join(sorted(missing))}")

    z_columns = []
    for factor in model.factor_names:
        z_name = f"z_{factor}"
        current[z_name] = _cross_sectional_zscore(current[factor])
        z_columns.append(z_name)
    current["valid_factor_count"] = current[z_columns].notna().sum(axis=1)
    weighted = current[z_columns].mul(
        [model.weights[factor] for factor in model.factor_names], axis=1
    )
    current["score"] = weighted.sum(axis=1, min_count=max(1, len(z_columns) // 2))
    current = current.loc[current["score"].notna()].copy()
    current["rank"] = current["score"].rank(ascending=False, method="first").astype(int)
    return current.sort_values(["rank", "symbol"], kind="stable").reset_index(drop=True)


def select_from_scores(
    scores: Any,
    *,
    top_n: int = 20,
    hold_buffer: int = 30,
    max_single_weight: float = 0.05,
    cash_buffer: float = 0.10,
    min_score: float | None = 0.0,
    previous_symbols: Iterable[str] = (),
) -> Any:
    """Apply top-N, hold-buffer and equal-weight rules to a score cross-section."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if hold_buffer < top_n:
        raise ValueError("hold_buffer must be at least top_n")
    if not 0 < max_single_weight <= 1:
        raise ValueError("max_single_weight must be between zero and one")
    if not 0 <= cash_buffer < 1:
        raise ValueError("cash_buffer must be in [0, 1)")
    ranked = scores.loc[scores["score"].notna()].sort_values(
        ["score", "symbol"], ascending=[False, True], kind="stable"
    ).copy()
    if min_score is not None:
        ranked = ranked.loc[ranked["score"] >= float(min_score)].copy()
    if ranked.empty:
        ranked["selected"] = False
        ranked["target_weight"] = 0.0
        return ranked
    if "rank" not in ranked.columns:
        ranked["rank"] = ranked["score"].rank(ascending=False, method="first").astype(int)
    previous = {str(symbol) for symbol in previous_symbols}
    held = ranked.loc[
        ranked["symbol"].astype(str).isin(previous) & (ranked["rank"] <= hold_buffer)
    ]
    selected_symbols = list(held["symbol"].astype(str))
    for symbol in ranked["symbol"].astype(str):
        if len(selected_symbols) >= top_n:
            break
        if symbol not in selected_symbols:
            selected_symbols.append(symbol)
    selected_set = set(selected_symbols)
    ranked["selected"] = ranked["symbol"].astype(str).isin(selected_set)
    gross = max(0.0, 1.0 - float(cash_buffer))
    weight = min(float(max_single_weight), gross / max(1, len(selected_symbols)))
    ranked["target_weight"] = weight if selected_symbols else 0.0
    ranked.loc[~ranked["selected"], "target_weight"] = 0.0
    return ranked
