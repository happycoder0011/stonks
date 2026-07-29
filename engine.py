"""Recommendation engine: fetch free NSE data from Yahoo Finance,
compute technical indicators, and score stocks for daily or weekly trades.

All data is end-of-day / ~15-min delayed. Suitable for swing trading,
not intraday scalping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from universe import UNIVERSE, yahoo_symbol


# ---------------------------------------------------------------- data fetch

def fetch_history(period: str = "1y") -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for the whole universe in one batch call.

    Returns {symbol: DataFrame} with columns Open/High/Low/Close/Volume,
    skipping symbols that failed to download.
    """
    symbols = [yahoo_symbol(s) for s in UNIVERSE]
    raw = yf.download(
        symbols,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    out: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE:
        ysym = yahoo_symbol(sym)
        try:
            df = raw[ysym].dropna(how="all")
        except KeyError:
            continue
        if len(df) < 60:  # need enough history for indicators
            continue
        out[sym] = df
    return out


def fetch_last_prices(symbols: list[str]) -> dict[str, float]:
    """Latest available close for a list of universe symbols (for marking
    open positions to market)."""
    if not symbols:
        return {}
    ysyms = [yahoo_symbol(s) for s in symbols]
    raw = yf.download(ysyms, period="5d", interval="1d",
                      group_by="ticker", auto_adjust=True,
                      threads=True, progress=False)
    prices: dict[str, float] = {}
    for sym in symbols:
        try:
            closes = raw[yahoo_symbol(sym)]["Close"].dropna()
            if len(closes):
                prices[sym] = float(closes.iloc[-1])
        except KeyError:
            continue
    return prices


# ---------------------------------------------------------------- indicators

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()
    out["sma200"] = close.rolling(200).mean()
    out["rsi"] = rsi(close)
    out["macd"], out["macd_signal"] = macd(close)
    out["atr"] = atr(out)
    out["vol_avg20"] = out["Volume"].rolling(20).mean()
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)
    out["high_52w"] = close.rolling(min(252, len(out)), min_periods=60).max()
    return out


# ------------------------------------------------------------------- scoring

@dataclass
class Recommendation:
    symbol: str
    name: str
    sector: str
    close: float
    score: float
    signals: list[str] = field(default_factory=list)
    entry: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    rsi: float = 0.0
    ret_5d: float = 0.0
    ret_20d: float = 0.0
    vol_ratio: float = 0.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_daily(ind: pd.DataFrame) -> tuple[float, list[str]]:
    """Score for 1-3 day momentum trades: recent thrust, volume surge,
    healthy RSI, price above short-term average.  0-100."""
    last = ind.iloc[-1]
    signals: list[str] = []
    score = 0.0

    # Momentum thrust: today's move and 5-day move (30 pts)
    r1, r5 = last["ret_1d"], last["ret_5d"]
    if not math.isnan(r1):
        score += 15 * _clip01(r1 / 0.025)          # +2.5% day = full marks
        if r1 > 0.015:
            signals.append(f"Strong day +{r1 * 100:.1f}%")
    if not math.isnan(r5):
        score += 15 * _clip01(r5 / 0.05)
        if r5 > 0.03:
            signals.append(f"5-day momentum +{r5 * 100:.1f}%")

    # Volume surge vs 20-day average (25 pts)
    vol_ratio = last["Volume"] / last["vol_avg20"] if last["vol_avg20"] else 0
    score += 25 * _clip01((vol_ratio - 1) / 1.5)   # 2.5x volume = full marks
    if vol_ratio > 1.5:
        signals.append(f"Volume {vol_ratio:.1f}x average")

    # RSI in the momentum sweet spot 55-70 (20 pts)
    r = last["rsi"]
    if not math.isnan(r):
        if 55 <= r <= 70:
            score += 20
            signals.append(f"RSI {r:.0f} in momentum zone")
        elif 50 <= r < 55 or 70 < r <= 75:
            score += 10
        elif r > 80:
            score -= 10
            signals.append(f"Overbought RSI {r:.0f} — caution")

    # Above short-term trend (15 pts)
    if last["Close"] > last["sma20"]:
        score += 15
        signals.append("Above 20-day average")

    # Closed near day's high — buyers in control (10 pts)
    day_range = last["High"] - last["Low"]
    if day_range > 0:
        close_pos = (last["Close"] - last["Low"]) / day_range
        score += 10 * _clip01((close_pos - 0.5) / 0.4)
        if close_pos > 0.85:
            signals.append("Closed near day high")

    return max(0.0, score), signals


def score_weekly(ind: pd.DataFrame) -> tuple[float, list[str]]:
    """Score for 1-2 week swing trades: established trend, MACD momentum,
    breakout proximity, steady accumulation.  0-100."""
    last = ind.iloc[-1]
    signals: list[str] = []
    score = 0.0

    # Trend structure: price > SMA20 > SMA50, rising SMA50 (30 pts)
    if last["Close"] > last["sma20"]:
        score += 10
    if last["sma20"] > last["sma50"]:
        score += 10
    sma50_rising = ind["sma50"].iloc[-1] > ind["sma50"].iloc[-6]
    if sma50_rising:
        score += 10
    if last["Close"] > last["sma20"] > last["sma50"] and sma50_rising:
        signals.append("Uptrend intact (price > 20d > 50d avg)")

    # MACD bullish and recent crossover (20 pts)
    macd_hist = ind["macd"] - ind["macd_signal"]
    if macd_hist.iloc[-1] > 0:
        score += 10
        recent = macd_hist.iloc[-6:]
        if (recent <= 0).any():
            score += 10
            signals.append("Fresh MACD bullish crossover")
        else:
            signals.append("MACD bullish")

    # 20-day momentum, capped so we don't chase parabolic moves (20 pts)
    r20 = last["ret_20d"]
    if not math.isnan(r20) and r20 > 0:
        score += 20 * _clip01(r20 / 0.10)
        if r20 > 0.20:
            score -= 10  # extended, mean-reversion risk
            signals.append(f"Extended +{r20 * 100:.0f}% in a month — caution")
        elif r20 > 0.05:
            signals.append(f"1-month momentum +{r20 * 100:.1f}%")

    # Near 52-week high = breakout territory (15 pts)
    dist_high = last["Close"] / last["high_52w"] - 1  # <= 0
    if dist_high > -0.03:
        score += 15
        signals.append("Within 3% of 52-week high")
    elif dist_high > -0.08:
        score += 8

    # RSI healthy, not overbought (15 pts)
    r = last["rsi"]
    if not math.isnan(r):
        if 50 <= r <= 65:
            score += 15
            signals.append(f"RSI {r:.0f} healthy")
        elif 45 <= r < 50 or 65 < r <= 72:
            score += 8
        elif r > 78:
            score -= 10
            signals.append(f"Overbought RSI {r:.0f} — caution")

    return max(0.0, score), signals


def recommend(history: dict[str, pd.DataFrame], mode: str,
              top_n: int = 10) -> list[Recommendation]:
    """Rank the universe for the given mode ('daily' or 'weekly')."""
    scorer = score_daily if mode == "daily" else score_weekly
    # Daily trades: tighter stop, closer target. Weekly: wider room.
    sl_mult, tgt_mult = (1.5, 2.5) if mode == "daily" else (2.0, 4.0)

    recs: list[Recommendation] = []
    for sym, df in history.items():
        ind = compute_indicators(df)
        last = ind.iloc[-1]
        if math.isnan(last["atr"]) or math.isnan(last["sma20"]):
            continue
        score, signals = scorer(ind)
        if score <= 0:
            continue
        name, sector = UNIVERSE[sym]
        entry = float(last["Close"])
        a = float(last["atr"])
        vol_ratio = (float(last["Volume"] / last["vol_avg20"])
                     if last["vol_avg20"] else 0.0)
        recs.append(Recommendation(
            symbol=sym, name=name, sector=sector,
            close=entry, score=round(score, 1), signals=signals,
            entry=round(entry, 2),
            stop_loss=round(entry - sl_mult * a, 2),
            target=round(entry + tgt_mult * a, 2),
            rsi=round(float(last["rsi"]), 1) if not math.isnan(last["rsi"]) else 0,
            ret_5d=round(float(last["ret_5d"]) * 100, 2) if not math.isnan(last["ret_5d"]) else 0,
            ret_20d=round(float(last["ret_20d"]) * 100, 2) if not math.isnan(last["ret_20d"]) else 0,
            vol_ratio=round(vol_ratio, 2),
        ))

    recs.sort(key=lambda r: r.score, reverse=True)
    return recs[:top_n]
