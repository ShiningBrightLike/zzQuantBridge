"""SQLite-backed append-only fills and deterministic portfolio reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from ..domain import ExecutedTrade


class TradeConflictError(ValueError):
    """Raised when a trade id is reused with different details."""


@dataclass
class PortfolioSnapshot:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    average_cost: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0
    fees: float = 0.0

    def apply(self, trade: ExecutedTrade) -> None:
        quantity = trade.quantity
        old_quantity = self.positions.get(trade.symbol, 0)
        old_cost = self.average_cost.get(trade.symbol, 0.0)
        gross = quantity * trade.price
        if trade.side == "BUY":
            new_quantity = old_quantity + quantity
            self.cash -= gross + trade.fee
            self.positions[trade.symbol] = new_quantity
            self.average_cost[trade.symbol] = ((old_quantity * old_cost) + gross) / new_quantity
        else:
            if quantity > old_quantity:
                raise ValueError(f"cannot sell {quantity} shares of {trade.symbol}; only {old_quantity} held")
            self.cash += gross - trade.fee
            self.realized_pnl += (trade.price - old_cost) * quantity - trade.fee
            remaining = old_quantity - quantity
            if remaining:
                self.positions[trade.symbol] = remaining
            else:
                self.positions.pop(trade.symbol, None)
                self.average_cost.pop(trade.symbol, None)
        self.fees += trade.fee


class PortfolioStore:
    """Persist fills and rebuild state in execution order."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS executed_trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
                    quantity INTEGER NOT NULL CHECK (quantity > 0),
                    price REAL NOT NULL CHECK (price > 0),
                    fee REAL NOT NULL CHECK (fee >= 0),
                    executed_at TEXT NOT NULL,
                    broker_reference TEXT,
                    note TEXT,
                    inserted_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO portfolio_meta (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, value, datetime.now(timezone.utc).isoformat()),
            )

    def get_meta(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM portfolio_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_initial_cash(self, value: float) -> None:
        if value < 0:
            raise ValueError("initial_cash must not be negative")
        self.set_meta("initial_cash", repr(float(value)))

    def get_initial_cash(self) -> float | None:
        raw = self.get_meta("initial_cash")
        if raw is None:
            return None
        return float(raw)

    def append_trade(self, trade: ExecutedTrade) -> bool:
        """Append a fill; return False for an identical already-seen fill."""

        values = (
            trade.trade_id,
            trade.symbol,
            trade.side,
            trade.quantity,
            trade.price,
            trade.fee,
            trade.executed_at.isoformat(),
            trade.broker_reference,
            trade.note,
        )
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT symbol, side, quantity, price, fee, executed_at, broker_reference, note "
                "FROM executed_trades WHERE trade_id = ?",
                (trade.trade_id,),
            ).fetchone()
            if existing is not None:
                if tuple(existing) != values[1:]:
                    raise TradeConflictError(f"trade_id already exists with different details: {trade.trade_id}")
                return False
            connection.execute(
                "INSERT INTO executed_trades "
                "(trade_id, symbol, side, quantity, price, fee, executed_at, broker_reference, note, inserted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*values, datetime.now(timezone.utc).isoformat()),
            )
            return True

    def append_trades(self, trades: list[ExecutedTrade]) -> int:
        """Append a batch atomically, preserving idempotency and conflicts."""

        inserted = 0
        with self._connect() as connection:
            seen: set[str] = set()
            for trade in trades:
                if trade.trade_id in seen:
                    raise TradeConflictError(f"duplicate trade_id in batch: {trade.trade_id}")
                seen.add(trade.trade_id)
                values = (
                    trade.trade_id,
                    trade.symbol,
                    trade.side,
                    trade.quantity,
                    trade.price,
                    trade.fee,
                    trade.executed_at.isoformat(),
                    trade.broker_reference,
                    trade.note,
                )
                existing = connection.execute(
                    "SELECT symbol, side, quantity, price, fee, executed_at, broker_reference, note "
                    "FROM executed_trades WHERE trade_id = ?",
                    (trade.trade_id,),
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != values[1:]:
                        raise TradeConflictError(f"trade_id already exists with different details: {trade.trade_id}")
                    continue
                connection.execute(
                    "INSERT INTO executed_trades "
                    "(trade_id, symbol, side, quantity, price, fee, executed_at, broker_reference, note, inserted_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*values, datetime.now(timezone.utc).isoformat()),
                )
                inserted += 1
        return inserted

    def trades(self) -> list[ExecutedTrade]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM executed_trades ORDER BY executed_at, trade_id").fetchall()
        return [
            ExecutedTrade(
                trade_id=row["trade_id"],
                symbol=row["symbol"],
                side=row["side"],
                quantity=row["quantity"],
                price=row["price"],
                fee=row["fee"],
                executed_at=datetime.fromisoformat(row["executed_at"]),
                broker_reference=row["broker_reference"],
                note=row["note"],
            )
            for row in rows
        ]

    def snapshot(self, initial_cash: float) -> PortfolioSnapshot:
        if initial_cash < 0:
            raise ValueError("initial_cash must not be negative")
        snapshot = PortfolioSnapshot(cash=initial_cash)
        for trade in self.trades():
            snapshot.apply(trade)
        return snapshot
