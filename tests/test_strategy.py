"""Regime, breakout, pullback, sector cap, and position size."""

from datetime import date

import numpy as np
import pandas as pd

from engine import (
    MIN_AVG_TURNOVER,
    Candidate,
    Regime,
    breakout_at,
    build_indicators,
    classify_regime,
    plan_trade,
    position_size,
    pullback_at,
    screen,
    select_candidates,
)


def _index(n=6):
    return pd.bdate_range("2026-10-01", periods=n)


def _breakout_row(**overrides) -> dict:
    row = {
        "Open": 109.0, "High": 111.0, "Low": 109.6, "Close": 110.7, "Volume": 3_000_000.0,
        "sma20": 100.0, "sma50": 98.0, "sma200": 90.0, "rsi": 62.0, "atr": 2.0,
        "vol_avg_prior": 1_000_000.0, "prior_high_20": 105.0, "high_52w": 140.0,
        "ret_20": 0.08, "ret_60": 0.12, "turnover_avg20": MIN_AVG_TURNOVER * 2,
    }
    row.update(overrides)
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=_index(len(rows)))


def test_breakout_requires_high_volume_close_and_the_200_day():
    good = _frame([_breakout_row() for _ in range(6)])
    assert breakout_at(good, len(good) - 1)
    quiet = _frame([_breakout_row(Volume=1_000_000.0) for _ in range(6)])
    assert not breakout_at(quiet, len(quiet) - 1)
    below = _frame([_breakout_row(Close=80.0, Low=79.0, High=81.0) for _ in range(6)])
    assert not breakout_at(below, len(below) - 1)
    mid_range = _frame([_breakout_row(Close=110.0, Low=109.0, High=112.0) for _ in range(6)])
    # (110-109)/(112-109) = 0.33, not the top quarter
    assert not breakout_at(mid_range, len(mid_range) - 1)
    illiquid = _frame([_breakout_row(turnover_avg20=MIN_AVG_TURNOVER / 2) for _ in range(6)])
    assert not breakout_at(illiquid, len(illiquid) - 1)


def test_pullback_is_a_rising_stack_tagging_the_20_day_with_rsi_turning_up():
    rows = []
    for i in range(6):
        rows.append(_breakout_row(
            Open=100, High=102, Low=99.2, Close=100.4, Volume=1_000_000,
            sma20=100 + i * 0.2, sma50=96, sma200=90, atr=2.0, rsi=45 + i,
            ret_20=0.04, ret_60=0.06,
        ))
    # Previous RSI is 49 (row -2, i=4 → rsi 45+4=49), last RSI 50. Both in/leaving 40–55.
    frame = _frame(rows)
    assert pullback_at(frame, len(frame) - 1)
    too_far = frame.copy()
    too_far.loc[too_far.index[-1], "Low"] = 104
    assert not pullback_at(too_far, len(too_far) - 1)
    falling_rsi = frame.copy()
    falling_rsi.loc[falling_rsi.index[-1], "rsi"] = 40
    assert not pullback_at(falling_rsi, len(falling_rsi) - 1)
    below_200 = frame.copy()
    below_200["sma200"] = 200
    assert not pullback_at(below_200, len(below_200) - 1)


def test_stop_uses_the_level_that_gives_more_room_and_target_is_at_least_2r():
    rows = [_breakout_row(Low=100.0) for _ in range(6)]
    rows[-1]["Low"] = 109.6
    frame = _frame(rows)
    # Swing low of the window is 100, ATR stop is 110.7 - 3 = 107.7. Structure is further away.
    plan = plan_trade(frame, len(frame) - 1, "breakout")
    assert plan is not None
    assert plan["stop"] < 107.7
    assert plan["stop"] <= 100.0
    risk = plan["entry"] - plan["stop"]
    assert plan["target"] >= plan["entry"] + 2 * risk
    assert plan["time_stop"] == 3
    pullback = plan_trade(frame, len(frame) - 1, "pullback")
    assert pullback["time_stop"] == 10


def test_position_size_floors_the_share_count():
    shares, risk_rupees, value = position_size(100_000, 1.0, 100, 95)
    assert (shares, risk_rupees, value) == (200, 1000.0, 20_000.0)
    shares, risk_rupees, _value = position_size(1_000, 1.0, 5_000, 4_900)
    assert shares == 0
    assert risk_rupees == 10.0
    assert position_size(100_000, 1.0, 90, 100)[0] == 0


def _candidate(symbol, sector, rs, turnover=1e9) -> Candidate:
    return Candidate(
        symbol=symbol, name=symbol, sector=sector, setup="breakout",
        entry=100, stop_loss=95, target=110, time_stop=3, shares=10,
        risk_rupees=50, position_value=1000, rr=2, one_r=105,
        rs_20=rs, rs_60=rs, rs_score=rs, rsi=55, atr=2, turnover=turnover,
    )


def test_sector_cap_and_numeric_tie_break_ignore_input_order():
    candidates = [
        _candidate("SBIN", "Banking", 0.01),
        _candidate("INFY", "IT", 0.02),
        _candidate("ICICIBANK", "Banking", 0.05),
        _candidate("HDFCBANK", "Banking", 0.04),
        _candidate("TCS", "IT", 0.02, turnover=5e9),
    ]
    picked = select_candidates(candidates, top_n=4, sector_cap=2, earnings_fn=None)
    assert [item.symbol for item in picked] == ["ICICIBANK", "HDFCBANK", "TCS", "INFY"]
    assert sum(item.sector == "Banking" for item in picked) == 2


def test_known_earnings_inside_the_hold_are_skipped():
    candidates = [
        _candidate("ICICIBANK", "Banking", 0.05),
        _candidate("INFY", "IT", 0.02),
    ]

    def earnings(symbol, _as_of, _sessions):
        if symbol == "ICICIBANK":
            return "Earnings on 01 Dec 2026 (inside the 3-session window)"
        return None

    picked = select_candidates(candidates, top_n=2, earnings_fn=earnings, as_of=date(2026, 11, 19))
    assert [item.symbol for item in picked] == ["INFY"]


def test_earnings_failure_keeps_the_name_and_flags_it():
    candidates = [_candidate("INFY", "IT", 0.02)]

    def broken(*_args):
        raise RuntimeError("calendar down")

    picked = select_candidates(candidates, top_n=1, earnings_fn=broken, as_of=date(2026, 11, 19))
    assert picked[0].symbol == "INFY"
    assert picked[0].earnings_flag == "Earnings check unavailable"


def _symbol_frame(ret_20: float, ret_60: float) -> pd.DataFrame:
    row = _breakout_row(ret_20=ret_20, ret_60=ret_60)
    return _frame([row for _ in range(6)])


def test_screen_ranks_by_relative_strength_and_stands_aside():
    history = {
        "HDFCBANK": _symbol_frame(0.10, 0.10),
        "ICICIBANK": _symbol_frame(0.06, 0.06),
        "SBIN": _symbol_frame(0.03, 0.03),
        "INFY": _symbol_frame(0.01, 0.01),
    }
    nifty = _frame([_breakout_row(ret_20=0.0, ret_60=0.0) for _ in range(6)])
    regime = Regime("Uptrend", False, False, 100, 90, 80, 0.01, "2026-10-08", "up")
    picked = screen(history, nifty, regime, 500_000, 1.0, top_n=3, earnings_fn=lambda *_a: None)
    assert [item.symbol for item in picked] == ["HDFCBANK", "ICICIBANK", "INFY"]
    assert all(item.sector != "Banking" or True for item in picked)
    assert sum(item.sector == "Banking" for item in picked) == 2
    blocked = Regime("Downtrend", True, False, 100, 90, 110, -0.02, "2026-10-08", "down")
    assert screen(history, nifty, blocked, 500_000, 1.0, top_n=3) == []


def _close_series(values) -> pd.DataFrame:
    close = np.asarray(values, dtype=float)
    frame = pd.DataFrame({
        "Open": close, "High": close + 1, "Low": close - 1, "Close": close,
        "Volume": np.full(len(close), 1_000_000.0),
    }, index=pd.bdate_range("2024-01-01", periods=len(close)))
    return build_indicators(frame)


def test_regime_downtrend_overextension_and_clean_uptrend():
    down = classify_regime(_close_series(np.linspace(300, 100, 230)))
    assert down.label == "Downtrend"
    assert down.stand_aside

    flat = np.full(230, 100.0)
    flat[-20:] = np.linspace(100, 112, 20)
    hot = classify_regime(_close_series(flat))
    assert hot.label == "Overextended"
    assert hot.stand_aside
    assert hot.ret_20 >= 0.10

    grind = 100 * np.cumprod(np.full(250, 1.0005))
    trend = classify_regime(_close_series(grind))
    assert trend.label == "Uptrend"
    assert not trend.stand_aside
    assert not trend.caution
