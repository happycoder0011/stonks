"""Walk-forward backtest of the breakout and pullback setups.

A signal is read on the close. The fill is the next session's open. The
exit uses that day's high and low, with the stop checked before the target.
Touching +1R arms a breakeven stop for the following session. A trade that
is still open after the setup's time stop exits on that session's close.
Costs, when enabled, are charged in basis points on the way in and the way out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import (
    BREAKOUT_TIME_STOP,
    MIN_TARGET_R,
    PULLBACK_TIME_STOP,
    breakout_at,
    plan_trade,
    pullback_at,
    regime_table,
)

SETUPS = ("breakout", "pullback")


@dataclass
class SimTrade:
    symbol: str
    setup: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    stop: float
    target: float
    r_multiple: float
    exit_reason: str
    year: int
    regime: str


@dataclass
class SetupStats:
    trades: int
    expectancy_r: float
    win_rate: float
    profit_factor: float
    max_drawdown_r: float
    by_year: list[dict] = field(default_factory=list)
    by_regime: list[dict] = field(default_factory=list)

    def expectancy_label(self) -> str:
        if self.trades == 0:
            return "no completed trades in the cached history"
        pf = "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"
        return (
            f"{self.expectancy_r:+.2f}R expectancy · {self.win_rate * 100:.0f}% wins · "
            f"profit factor {pf} · {self.trades} trades"
        )


@dataclass
class BacktestResult:
    cost_bps: float
    breakout: SetupStats
    pullback: SetupStats
    trades: list[SimTrade]

    def for_setup(self, setup: str) -> SetupStats:
        return self.breakout if setup == "breakout" else self.pullback


def _empty_stats() -> SetupStats:
    return SetupStats(0, 0.0, 0.0, 0.0, 0.0, [], [])


def walk_exit(
    ind: pd.DataFrame,
    entry_i: int,
    plan: dict,
    cost_bps: float,
    time_stop: int,
) -> tuple[dict, int] | None:
    """Simulate one open trade. Returns the fill details and the exit bar."""
    if entry_i >= len(ind):
        return None
    cost = cost_bps / 10_000.0
    raw_open = float(ind["Open"].iloc[entry_i])
    if not math.isfinite(raw_open) or raw_open <= 0:
        return None
    entry = raw_open * (1 + cost)
    stop = float(plan["stop"])
    # An open already through the stop is not a buy. There is no positive risk.
    if raw_open <= stop:
        return None
    risk = entry - stop
    if risk <= 0:
        return None
    signal_target = float(plan["target"])
    two_r = entry + MIN_TARGET_R * risk
    target = two_r if signal_target <= entry else max(signal_target, two_r)
    one_r = entry + risk
    armed = False
    last_i = min(len(ind) - 1, entry_i + time_stop - 1)
    for k in range(entry_i, last_i + 1):
        opened = float(ind["Open"].iloc[k])
        high = float(ind["High"].iloc[k])
        low = float(ind["Low"].iloc[k])
        close = float(ind["Close"].iloc[k])
        if not all(math.isfinite(v) for v in (opened, high, low, close)):
            return None
        reason = None
        exit_raw = None
        if opened <= stop:
            reason = "breakeven" if armed else "stop"
            exit_raw = opened
        elif opened >= target:
            reason = "target"
            exit_raw = opened
        elif low <= stop:
            reason = "breakeven" if armed else "stop"
            exit_raw = stop
        elif high >= target:
            reason = "target"
            exit_raw = target
        if reason is not None and exit_raw is not None:
            return _filled(entry, exit_raw, stop, target, risk, cost, reason), k
        if not armed and high >= one_r:
            armed = True
            stop = entry
        if k == last_i:
            return _filled(entry, close, stop, target, risk, cost, "time"), k
    return None


def _filled(entry: float, exit_raw: float, stop: float, target: float,
            risk: float, cost: float, reason: str) -> dict:
    exit_fill = exit_raw * (1 - cost)
    return {
        "entry": entry,
        "exit": exit_fill,
        "stop": stop,
        "target": target,
        "r_multiple": (exit_fill - entry) / risk,
        "exit_reason": reason,
    }


def _signal(ind: pd.DataFrame, i: int, setup: str) -> bool:
    if setup == "breakout":
        return breakout_at(ind, i)
    return pullback_at(ind, i)


def simulate_symbol(
    symbol: str,
    ind: pd.DataFrame,
    setup: str,
    regimes: pd.DataFrame,
    cost_bps: float,
) -> list[SimTrade]:
    trades: list[SimTrade] = []
    time_stop = BREAKOUT_TIME_STOP if setup == "breakout" else PULLBACK_TIME_STOP
    i = 0
    last = len(ind) - 1
    while i < last:
        if not _signal(ind, i, setup):
            i += 1
            continue
        stamp = ind.index[i]
        if stamp not in regimes.index:
            i += 1
            continue
        state = regimes.loc[stamp]
        if bool(state["stand_aside"]):
            i += 1
            continue
        plan = plan_trade(ind, i, setup)
        if plan is None:
            i += 1
            continue
        walked = walk_exit(ind, i + 1, plan, cost_bps, time_stop)
        if walked is None:
            i += 1
            continue
        fill, exit_i = walked
        exit_stamp = ind.index[exit_i]
        trades.append(SimTrade(
            symbol=symbol,
            setup=setup,
            signal_date=pd.Timestamp(stamp).date().isoformat(),
            entry_date=pd.Timestamp(ind.index[i + 1]).date().isoformat(),
            exit_date=pd.Timestamp(exit_stamp).date().isoformat(),
            entry=fill["entry"],
            exit=fill["exit"],
            stop=fill["stop"],
            target=fill["target"],
            r_multiple=fill["r_multiple"],
            exit_reason=fill["exit_reason"],
            year=pd.Timestamp(exit_stamp).year,
            regime=str(state["label"]),
        ))
        i = exit_i + 1
    return trades


def _group_stats(trades: list[SimTrade], key) -> list[dict]:
    buckets: dict[str, list[SimTrade]] = {}
    for trade in trades:
        buckets.setdefault(str(key(trade)), []).append(trade)
    rows = []
    for name in sorted(buckets):
        subset = buckets[name]
        rs = np.array([item.r_multiple for item in subset], dtype=float)
        rows.append({
            "name": name,
            "trades": len(subset),
            "expectancy_r": float(rs.mean()),
            "win_rate": float((rs > 0).mean()),
        })
    return rows


def summarize(trades: list[SimTrade]) -> SetupStats:
    if not trades:
        return _empty_stats()
    ordered = sorted(trades, key=lambda item: (item.exit_date, item.symbol))
    rs = np.array([item.r_multiple for item in ordered], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0
    equity = np.concatenate([[0.0], np.cumsum(rs)])
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max()) if len(equity) else 0.0
    return SetupStats(
        trades=len(ordered),
        expectancy_r=float(rs.mean()),
        win_rate=float((rs > 0).mean()),
        profit_factor=profit_factor,
        max_drawdown_r=max_dd,
        by_year=_group_stats(ordered, lambda item: item.year),
        by_regime=_group_stats(ordered, lambda item: item.regime),
    )


def run_backtest(
    history: dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """Backtest both setups on indicator frames from the same daily cache."""
    if nifty is None or nifty.empty or "sma200" not in nifty.columns:
        return BacktestResult(cost_bps, _empty_stats(), _empty_stats(), [])
    regimes = regime_table(nifty)
    grouped: dict[str, list[SimTrade]] = {setup: [] for setup in SETUPS}
    for setup in SETUPS:
        for symbol, frame in history.items():
            grouped[setup].extend(simulate_symbol(symbol, frame, setup, regimes, cost_bps))
    trades = grouped["breakout"] + grouped["pullback"]
    return BacktestResult(
        cost_bps=cost_bps,
        breakout=summarize(grouped["breakout"]),
        pullback=summarize(grouped["pullback"]),
        trades=trades,
    )
