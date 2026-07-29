"""SQLite trade journal: every recommendation you take is logged here,
so open positions and realized performance can be tracked."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "trades.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date_taken  TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            name        TEXT NOT NULL,
            sector      TEXT NOT NULL DEFAULT '',
            mode        TEXT NOT NULL,              -- 'daily' | 'weekly'
            entry_price REAL NOT NULL,
            qty         INTEGER NOT NULL,
            stop_loss   REAL,
            target      REAL,
            score       REAL,
            signals     TEXT DEFAULT '',
            status      TEXT NOT NULL DEFAULT 'open',  -- 'open' | 'closed'
            exit_price  REAL,
            exit_date   TEXT,
            pnl         REAL,
            pnl_pct     REAL,
            notes       TEXT DEFAULT ''
        )
    """)
    return conn


def add_trade(symbol: str, name: str, sector: str, mode: str,
              entry_price: float, qty: int, stop_loss: float,
              target: float, score: float, signals: str,
              notes: str = "") -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO trades
               (date_taken, symbol, name, sector, mode, entry_price, qty,
                stop_loss, target, score, signals, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (date.today().isoformat(), symbol, name, sector, mode,
             entry_price, qty, stop_loss, target, score, signals, notes),
        )


def close_trade(trade_id: int, exit_price: float, notes: str = "") -> None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT entry_price, qty, notes FROM trades WHERE id=?",
            (trade_id,)).fetchone()
        if row is None:
            return
        entry, qty, old_notes = row
        pnl = (exit_price - entry) * qty
        pnl_pct = (exit_price / entry - 1) * 100
        merged_notes = f"{old_notes} | {notes}".strip(" |") if notes else old_notes
        conn.execute(
            """UPDATE trades SET status='closed', exit_price=?, exit_date=?,
               pnl=?, pnl_pct=?, notes=? WHERE id=?""",
            (exit_price, date.today().isoformat(), round(pnl, 2),
             round(pnl_pct, 2), merged_notes, trade_id),
        )


def delete_trade(trade_id: int) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM trades WHERE id=?", (trade_id,))


def get_trades(status: str | None = None) -> pd.DataFrame:
    with _conn() as conn:
        if status:
            return pd.read_sql_query(
                "SELECT * FROM trades WHERE status=? ORDER BY id DESC",
                conn, params=(status,))
        return pd.read_sql_query(
            "SELECT * FROM trades ORDER BY id DESC", conn)
