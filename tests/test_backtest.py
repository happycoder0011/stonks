"""Walk-forward exits: stop, target, same-bar priority, breakeven, time, costs."""

import numpy as np
import pandas as pd

from backtest import run_backtest, summarize, walk_exit
from backtest import SimTrade
from engine import MIN_AVG_TURNOVER, build_indicators


def _ohlc(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])
    frame["Volume"] = 1_000_000.0
    frame.index = pd.bdate_range("2026-01-05", periods=len(frame))
    return frame


def _plan(stop=90.0, target=120.0):
    return {"stop": stop, "target": target, "entry": 100.0, "risk": 10.0, "atr": 2.0,
            "time_stop": 3, "swing_low": stop, "one_r": 110.0}


def test_target_stop_and_same_bar_priority():
    target_hit = _ohlc([
        (100, 105, 99, 104),
        (106, 121, 104, 120),
        (120, 122, 118, 119),
    ])
    fill, exit_i = walk_exit(target_hit, 0, _plan(), 0, 3)
    assert exit_i == 1
    assert fill["exit_reason"] == "target"
    assert fill["r_multiple"] == 2

    stopped = _ohlc([
        (100, 104, 89, 95),
        (95, 96, 94, 95),
        (95, 96, 94, 95),
    ])
    fill, _exit_i = walk_exit(stopped, 0, _plan(), 0, 3)
    assert fill["exit_reason"] == "stop"
    assert fill["r_multiple"] == -1

    both = _ohlc([
        (100, 130, 80, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
    ])
    fill, exit_i = walk_exit(both, 0, _plan(), 0, 3)
    assert exit_i == 0
    assert fill["exit_reason"] == "stop"
    assert fill["r_multiple"] == -1


def test_time_stop_and_breakeven_armed_for_the_next_session():
    quiet = _ohlc([
        (100, 104, 99, 101),
        (101, 104, 99, 102),
        (102, 104, 99, 103),
        (103, 104, 99, 104),
    ])
    fill, exit_i = walk_exit(quiet, 0, _plan(), 0, 3)
    assert exit_i == 2
    assert fill["exit_reason"] == "time"
    assert round(fill["r_multiple"], 2) == 0.3

    # Day 0 trades through +1R (110) without touching the original stop.
    # Day 1 comes back to the entry and the stop is now breakeven.
    armed = _ohlc([
        (100, 112, 99, 110),
        (108, 109, 100, 101),
        (101, 102, 100, 101),
    ])
    fill, exit_i = walk_exit(armed, 0, _plan(), 0, 3)
    assert exit_i == 1
    assert fill["exit_reason"] == "breakeven"
    assert fill["r_multiple"] == 0

    # A next open already through the stop is not a fill. Risk would be negative.
    gap = _ohlc([
        (85, 86, 80, 84),
        (84, 85, 83, 84),
        (84, 85, 83, 84),
    ])
    assert walk_exit(gap, 0, _plan(), 0, 3) is None


def test_costs_reduce_a_textbook_2r_win():
    target_hit = _ohlc([
        (100, 105, 99, 104),
        (100, 130, 99, 120),
        (120, 121, 119, 120),
    ])
    # Enter at 100, stop 90, risk 10 before costs. 100 bps each side.
    fill, _exit_i = walk_exit(target_hit, 0, _plan(), 100, 3)
    assert fill["entry"] == 101
    assert fill["r_multiple"] < 2


def test_summary_drawdown_counts_a_loss_from_the_start():
    trades = [
        SimTrade("A", "breakout", "2026-01-05", "2026-01-06", "2026-01-06",
                 100, 90, 90, 120, -1, "stop", 2026, "Uptrend"),
        SimTrade("B", "breakout", "2026-01-07", "2026-01-08", "2026-01-08",
                 100, 120, 90, 120, 2, "target", 2026, "Uptrend"),
    ]
    stats = summarize(trades)
    assert stats.trades == 2
    assert stats.expectancy_r == 0.5
    assert stats.win_rate == 0.5
    assert stats.profit_factor == 2
    assert stats.max_drawdown_r == 1
    assert stats.by_year[0]["name"] == "2026"
    assert stats.by_regime[0]["name"] == "Uptrend"


def _breakout_history():
    """Flat base, one breakout, then a rally that reaches the 2R target."""
    n = 230
    index = pd.bdate_range("2024-01-01", periods=n)
    close = np.full(n, 100.0)
    high = close + 1
    low = close - 1
    opened = close.copy()
    volume = np.full(n, 3_000_000.0)
    signal = n - 5
    close[signal] = 110.7
    high[signal] = 111.0
    low[signal] = 109.6
    opened[signal] = 109.0
    volume[signal] = 6_000_000.0
    # Next sessions rally through a wide structural stop's 2R target.
    for step, price in enumerate((112.0, 130.0, 160.0, 160.0), start=1):
        close[signal + step] = price
        opened[signal + step] = price - 1
        high[signal + step] = price + 2
        low[signal + step] = price - 2
    frame = pd.DataFrame(
        {"Open": opened, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )
    nifty_close = np.linspace(100, 130, n)
    nifty = pd.DataFrame({
        "Open": nifty_close, "High": nifty_close + 1, "Low": nifty_close - 1,
        "Close": nifty_close, "Volume": np.full(n, 1.0),
    }, index=index)
    stock = build_indicators(frame)
    bench = build_indicators(nifty)
    # Liquidity floor is met by the 3e6 * 100 base.
    assert stock["turnover_avg20"].iloc[signal] >= MIN_AVG_TURNOVER
    return stock, bench


def test_walk_forward_enters_the_next_open_and_does_not_use_the_future_bar_as_the_signal():
    stock, nifty = _breakout_history()
    result = run_backtest({"RELIANCE": stock}, nifty, cost_bps=0)
    breakouts = [trade for trade in result.trades if trade.setup == "breakout"]
    assert breakouts, "expected the crafted breakout to fill"
    trade = breakouts[0]
    assert trade.entry_date > trade.signal_date
    assert trade.exit_reason == "target"
    assert trade.r_multiple > 1
    # A future spike must not create a signal on the first bar of the series.
    assert trade.signal_date != stock.index[0].date().isoformat()
    assert result.breakout.trades >= 1
