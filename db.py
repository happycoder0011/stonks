"""SQLite trade journal.

Dates are stored as IST timestamps. Closing a trade that is not open is a
no-op. When STONKS_PASSWORD is set (environment or Streamlit secrets), writes
stay locked until the caller passes that password.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")


def db_path() -> Path:
    override = os.environ.get("STONKS_DB", "").strip()
    if override:
        return Path(override)
    return Path(__file__).parent / "trades.db"


def ist_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(IST)
    if now.tzinfo is None:
        return now.replace(tzinfo=IST)
    return now.astimezone(IST)


def ist_timestamp(now: datetime | None = None) -> str:
    return ist_now(now).isoformat(timespec="seconds")


def sessions_held(opened: str, today: date | None = None) -> int:
    """Completed NSE sessions since the entry date, using business days."""
    start = date.fromisoformat(str(opened)[:10])
    today = today or ist_now().date()
    if today <= start:
        return 0
    return max(0, len(pd.bdate_range(start, today)) - 1)


def configured_password() -> str | None:
    env = os.environ.get("STONKS_PASSWORD", "").strip()
    if env:
        return env
    try:
        import streamlit as st
        value = st.secrets["STONKS_PASSWORD"]
    except Exception:
        return None
    text = str(value).strip()
    return text or None


def password_ok(entered: str | None) -> bool:
    expected = configured_password()
    if not expected:
        return True
    return (entered or "") == expected


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date_taken  TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            name        TEXT NOT NULL,
            sector      TEXT NOT NULL DEFAULT '',
            mode        TEXT NOT NULL,
            entry_price REAL NOT NULL,
            qty         INTEGER NOT NULL,
            stop_loss   REAL,
            target      REAL,
            score       REAL,
            signals     TEXT DEFAULT '',
            status      TEXT NOT NULL DEFAULT 'open',
            exit_price  REAL,
            exit_date   TEXT,
            pnl         REAL,
            pnl_pct     REAL,
            notes       TEXT DEFAULT '',
            time_stop_sessions INTEGER
        )
    """)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    if "time_stop_sessions" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN time_stop_sessions INTEGER")
    return conn


def add_trade(symbol: str, name: str, sector: str, mode: str,
              entry_price: float, qty: int, stop_loss: float,
              target: float, score: float, signals: str,
              notes: str = "", time_stop_sessions: int | None = None,
              now: datetime | None = None) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO trades
               (date_taken, symbol, name, sector, mode, entry_price, qty,
                stop_loss, target, score, signals, notes, time_stop_sessions)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ist_timestamp(now), symbol, name, sector, mode,
             entry_price, qty, stop_loss, target, score, signals, notes,
             time_stop_sessions),
        )


def close_trade(trade_id: int, exit_price: float, notes: str = "",
                now: datetime | None = None) -> bool:
    """Close an open trade. Returns False when the id is missing or already closed."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT entry_price, qty, notes, status FROM trades WHERE id=?",
            (trade_id,),
        ).fetchone()
        if row is None or row[3] != "open":
            return False
        entry, qty, old_notes, _status = row
        pnl = (exit_price - entry) * qty
        pnl_pct = (exit_price / entry - 1) * 100 if entry else 0.0
        merged = f"{old_notes} | {notes}".strip(" |") if notes else old_notes
        cur = conn.execute(
            """UPDATE trades SET status='closed', exit_price=?, exit_date=?,
               pnl=?, pnl_pct=?, notes=? WHERE id=? AND status='open'""",
            (exit_price, ist_timestamp(now), round(pnl, 2),
             round(pnl_pct, 2), merged, trade_id),
        )
        return cur.rowcount == 1


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


def planned_open_risk(trades: pd.DataFrame | None = None) -> float:
    """Rupees at risk if every open trade hits its stop, from the entry."""
    frame = get_trades("open") if trades is None else trades
    if frame is None or frame.empty:
        return 0.0
    total = 0.0
    for _, row in frame.iterrows():
        if row.get("status", "open") != "open":
            continue
        distance = float(row["entry_price"]) - float(row["stop_loss"])
        if distance > 0:
            total += distance * int(row["qty"])
    return total
