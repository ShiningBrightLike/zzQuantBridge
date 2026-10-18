"""Shared run-artifact writers for multifactor selection and backtests."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backtest import (
    CrossSectionalBacktestConfig,
    performance_metrics,
    simulate_target_weights,
)
from .domain import TargetPosition
from .reports import write_run_report
from .runs import RunContext, write_status
from .strategies import MultifactorStrategyConfig, SelectionResult


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write CSV through the stdlib writer so commas or quotes cannot shift columns."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def write_target_signals(context: RunContext, targets: Sequence[TargetPosition]) -> None:
    write_csv(
        context.run_dir / "signals.csv",
        ("symbol", "timestamp", "target_weight", "reason", "model_version"),
        [
            {
                "symbol": target.symbol,
                "timestamp": target.timestamp.isoformat(),
                "target_weight": target.target_weight,
                "reason": target.reason,
                "model_version": target.model_version,
            }
            for target in targets
        ],
    )


def write_selection_artifacts(
    context: RunContext,
    selection: SelectionResult,
    *,
    names: Mapping[str, str] | None = None,
    top_n: int | None = None,
    affordability_filter: bool = False,
) -> None:
    """Persist the auditable cross-section and the frozen model weights."""

    name_map = dict(names or {})
    scores = selection.scores.copy()
    selected_map = (
        selection.selected.set_index("symbol") if len(selection.selected) else selection.selected
    )
    if len(selected_map):
        scores["selected"] = (
            scores["symbol"].map(selected_map["selected"]).fillna(False).astype(bool)
        )
        scores["target_weight"] = (
            scores["symbol"].map(selected_map["target_weight"]).fillna(0.0).astype(float)
        )
    else:
        scores["selected"] = False
        scores["target_weight"] = 0.0
    score_columns = [
        column
        for column in scores.columns
        if column not in {"timestamp", "label_end", "forward_return"}
    ]
    write_csv(
        context.run_dir / "factor_snapshot.csv",
        score_columns,
        [
            {column: json_value(row.get(column, "")) for column in score_columns}
            for row in scores.to_dict(orient="records")
        ],
    )
    metric_rows: list[dict[str, Any]] = []
    for row in selection.model.metrics.to_dict(orient="records"):
        factor = str(row["factor"])
        item = {key: json_value(value) for key, value in row.items()}
        item["weight"] = selection.model.weights.get(factor, 0.0)
        item["used"] = factor in selection.model.factor_names
        item["direction"] = 1 if item["weight"] > 0 else (-1 if item["weight"] < 0 else 0)
        metric_rows.append(item)
    write_csv(
        context.run_dir / "factor_weights.csv",
        [
            "factor",
            "weight",
            "direction",
            "used",
            "ic_mean",
            "ic_std",
            "icir",
            "ic_win_rate",
            "ic_days",
            "sign_stable",
        ],
        metric_rows,
    )
    selected_rows = []
    for row in selection.selected.loc[selection.selected["selected"]].to_dict(orient="records"):
        symbol = str(row.get("symbol", ""))
        selected_rows.append(
            {
                "symbol": symbol,
                "name": name_map.get(symbol, ""),
                "rank": row.get("rank", ""),
                "score": row.get("score", ""),
                "valid_factor_count": row.get("valid_factor_count", ""),
                "target_weight": row.get("target_weight", ""),
                "model_version": selection.model.model_version,
                "reason": (
                    f"multifactor rank {row.get('rank')} score "
                    f"{float(row.get('score', 0.0)):.4f}; "
                    f"factors {int(row.get('valid_factor_count', 0))}"
                ),
            }
        )
    write_csv(
        context.run_dir / "selections.csv",
        [
            "symbol",
            "name",
            "rank",
            "score",
            "valid_factor_count",
            "target_weight",
            "model_version",
            "reason",
        ],
        selected_rows,
    )
    write_csv(
        context.run_dir / "universe.csv",
        ("symbol", "name", "tradable"),
        [
            {"symbol": symbol, "name": name_map.get(symbol, ""), "tradable": 1}
            for symbol in selection.tradable_symbols
        ],
    )
    metadata = {
        "status": "PASS",
        "model_version": selection.model.model_version,
        "trained_until": json_value(selection.model.trained_until),
        "lookback_start": json_value(selection.model.lookback_start),
        "label_horizon": selection.model.label_horizon,
        "fallback_used": selection.model.fallback_used,
        "factor_names": list(selection.model.factor_names),
        "weights": {key: float(value) for key, value in selection.model.weights.items()},
        "universe_size": len(selection.tradable_symbols),
        "selected_count": len(selection.targets),
        "top_n": top_n,
        "affordability_filter": affordability_filter,
        "selection_limited": bool(top_n is not None and len(selection.targets) < top_n),
        "as_of": json_value(selection.as_of),
    }
    (context.run_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_multifactor_backtest_artifacts(
    context: RunContext,
    frame: Any,
    result: Any,
    settings: MultifactorStrategyConfig,
    backtest_config: CrossSectionalBacktestConfig,
) -> dict[str, Any]:
    metrics = dict(result.metrics)
    result_status = (
        "BLOCKED"
        if int(metrics.get("active_days", 0)) < backtest_config.min_oos_days
        else "PASS"
    )
    metrics.update(
        {
            "status": result_status,
            "model_version": (
                str(result.model_metrics["model_version"].iloc[-1])
                if len(result.model_metrics)
                else settings.model_version
            ),
            "universe_size": int(frame["symbol"].nunique()),
            "label_horizon": settings.training.label_horizon,
            "rebalance_days": settings.rebalance_days,
            "top_n": settings.top_n,
            "min_oos_days": backtest_config.min_oos_days,
            "skipped_rebalances": int(len(result.skipped_rebalances)),
        }
    )
    (context.run_dir / "backtest_metrics.json").write_text(
        json.dumps(
            {"status": result_status, "metrics": json_value(metrics)},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    result.equity.to_csv(context.run_dir / "equity_curve.csv")
    if len(result.trades):
        result.trades.to_csv(context.run_dir / "trades.csv", index=False)
    if len(result.model_metrics):
        result.model_metrics.to_csv(context.run_dir / "backtest_model_metrics.csv", index=False)
        write_csv(
            context.run_dir / "backtest_factor_weights.csv",
            (
                "rebalance_date",
                "factor",
                "weight",
                "used",
                "ic_mean",
                "ic_std",
                "icir",
                "ic_win_rate",
                "ic_days",
                "sign_stable",
                "model_version",
            ),
            result.model_metrics.to_dict(orient="records"),
        )
    write_csv(
        context.run_dir / "backtest_selections.csv",
        (
            "timestamp",
            "symbol",
            "rank",
            "score",
            "valid_factor_count",
            "target_weight",
            "model_version",
        ),
        [
            {key: json_value(value) for key, value in row.items()}
            for row in result.selections.to_dict(orient="records")
        ],
    )
    if len(result.skipped_rebalances):
        write_csv(
            context.run_dir / "skipped_rebalances.csv",
            ("timestamp", "reason"),
            [
                {key: json_value(value) for key, value in row.items()}
                for row in result.skipped_rebalances.to_dict(orient="records")
            ],
        )

    close = frame.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last").sort_index()
    open_prices = frame.pivot_table(index="timestamp", columns="symbol", values="open", aggfunc="last").sort_index()
    cost: dict[str, Any] = {}
    for multiplier in (1.0, 2.0, 3.0):
        scenario = replace(
            backtest_config,
            fees=backtest_config.fees * multiplier,
            slippage=backtest_config.slippage * multiplier,
        )
        equity, trades, _ = simulate_target_weights(close, open_prices, result.raw_target_weights, scenario)
        value = performance_metrics(equity, trades)
        cost[f"x{multiplier:g}"] = {
            "total_return": value["total_return"],
            "max_drawdown": value["max_drawdown"],
            "trade_count": value["trade_count"],
        }
    delay: dict[str, Any] = {}
    for value_delay in (0, 1, 2):
        scenario = replace(backtest_config, signal_delay=value_delay)
        equity, trades, _ = simulate_target_weights(close, open_prices, result.raw_target_weights, scenario)
        value = performance_metrics(equity, trades)
        delay[f"delay_{value_delay}"] = {
            "total_return": value["total_return"],
            "max_drawdown": value["max_drawdown"],
            "trade_count": value["trade_count"],
        }
    sensitivity = {"cost_multiplier": cost, "signal_delay": delay}
    (context.run_dir / "sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    artifact_names = [
        "data_quality.json",
        "backtest_metrics.json",
        "equity_curve.csv",
        "trades.csv",
        "backtest_model_metrics.csv",
        "backtest_factor_weights.csv",
        "backtest_selections.csv",
        "skipped_rebalances.csv",
        "sensitivity.json",
        "report.html",
    ]
    summary = {
        **metrics,
        "status": result_status,
        "data_as_of": str(frame["timestamp"].max().date()),
        "symbols": sorted(str(item) for item in frame["symbol"].unique()),
        "artifacts": [
            name
            for name in artifact_names
            if name == "report.html" or (context.run_dir / name).is_file()
        ],
    }
    write_run_report(context.run_dir)
    write_status(context.run_dir, result_status, summary)
    return summary
