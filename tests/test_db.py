"""Journal timestamps, the close guard, and the optional password."""

from datetime import date, datetime, timezone

import pandas as pd

from db import (
    add_trade,
    close_trade,
    configured_password,
    get_trades,
    ist_timestamp,
    password_ok,
    planned_open_risk,
    sessions_held,
)


def test_ist_timestamp_is_not_the_utc_date():
    now = datetime(2026, 11, 20, 20, 30, tzinfo=timezone.utc)
    assert ist_timestamp(now) == "2026-11-21T02:00:00+05:30"


def test_add_and_close_use_ist_and_refuse_a_second_close(tmp_path, monkeypatch):
    monkeypatch.setenv("STONKS_DB", str(tmp_path / "trades.db"))
    opened = datetime(2026, 11, 20, 20, 30, tzinfo=timezone.utc)
    add_trade(
        "INFY", "Infosys", "IT", "breakout",
        100.0, 10, 95.0, 110.0, 0.02, "breakout",
        time_stop_sessions=3, now=opened,
    )
    row = get_trades().iloc[0]
    assert row["date_taken"] == "2026-11-21T02:00:00+05:30"
    assert int(row["time_stop_sessions"]) == 3
    assert close_trade(int(row["id"]), 110.0, now=opened)
    closed = get_trades("closed").iloc[0]
    assert closed["pnl"] == 100.0
    assert closed["exit_date"].startswith("2026-11-21")
    assert close_trade(int(row["id"]), 50.0) is False
    assert get_trades("closed").iloc[0]["pnl"] == 100.0
    assert get_trades("open").empty


def test_sessions_held_and_open_risk():
    # Monday 16 Nov 2026 through Friday 20 Nov 2026 is four sessions after entry.
    assert sessions_held("2026-11-16T10:15:00+05:30", date(2026, 11, 20)) == 4
    assert sessions_held("2026-11-20T10:15:00+05:30", date(2026, 11, 20)) == 0
    book = pd.DataFrame([
        {"status": "open", "entry_price": 100.0, "stop_loss": 90.0, "qty": 10},
        {"status": "closed", "entry_price": 100.0, "stop_loss": 90.0, "qty": 10},
    ])
    assert planned_open_risk(book) == 100.0


def test_password_locks_writes_only_when_set(monkeypatch):
    monkeypatch.setenv("STONKS_PASSWORD", "secret")
    assert configured_password() == "secret"
    assert password_ok("secret")
    assert not password_ok("nope")
    assert not password_ok("")
    monkeypatch.setenv("STONKS_PASSWORD", "")
    assert configured_password() in (None, "")
    assert password_ok(None)
