"""NSE swing engine: completed-session data, two long setups, and risk levels.

Signals use the last completed NSE session. A bar dated today is ignored
until 15:30 IST, so a forming intraday candle cannot change the plan.
Prices are unadjusted so a missing close does not wipe the day's range.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from universe import UNIVERSE, yahoo_symbol

IST = ZoneInfo("Asia/Kolkata")
NIFTY_SYMBOL = "^NSEI"
HISTORY_PERIOD = "5y"
MIN_BARS = 200
MIN_AVG_TURNOVER = 20_00_00_000  # ₹20 crore
OVEREXTEND_CAUTION = 0.06
OVEREXTEND_STAND_ASIDE = 0.10
BREAKOUT_TIME_STOP = 3
PULLBACK_TIME_STOP = 10
BREAKOUT_ATR_MULT = 1.5
PULLBACK_ATR_MULT = 2.0
MIN_TARGET_R = 2.0
SECTOR_CAP = 2
PULLBACK_TAG_ATR = 0.5
NSE_OPEN = time(9, 15)
NSE_CLOSE = time(15, 30)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def format_session_day(day: date) -> str:
    return f"{_WEEKDAYS[day.weekday()]} {day.day:02d} {_MONTHS[day.month - 1]} {day.year}"

OHLCV = ("Open", "High", "Low", "Close", "Volume")


# ----------------------------------------------------------------- clock

def nse_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(IST)
    if now.tzinfo is None:
        return now.replace(tzinfo=IST)
    return now.astimezone(IST)


def session_is_forming(now: datetime | None = None) -> bool:
    """True on a weekday before the 15:30 IST close.

    The cash session runs 9:15–15:30. Before the open, today's bar is not a
    completed session either, so both windows are excluded.
    """
    current = nse_now(now)
    if current.weekday() >= 5:
        return False
    return current.time() < NSE_CLOSE


def in_nse_session(now: datetime | None = None) -> bool:
    current = nse_now(now)
    if current.weekday() >= 5:
        return False
    return NSE_OPEN <= current.time() < NSE_CLOSE


def session_has_opened(now: datetime | None = None) -> bool:
    """True once today's cash session has begun, including after the close."""
    current = nse_now(now)
    if current.weekday() >= 5:
        return False
    return current.time() >= NSE_OPEN


def _naive_dates(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    return idx.normalize()


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = _naive_dates(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def freeze_completed_session(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Drop a bar dated today when that session has not closed."""
    if df is None or df.empty or not session_is_forming(now):
        return df
    today = pd.Timestamp(nse_now(now).date())
    dates = _naive_dates(df.index)
    kept = df.loc[dates != today].copy()
    kept.index = dates[dates != today]
    return kept


def cache_bucket(now: datetime | None = None) -> str:
    current = nse_now(now)
    phase = "forming" if session_is_forming(current) else "closed"
    return f"{current:%Y-%m-%d}-{phase}"


# --------------------------------------------------------------- repair

def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number) and number > 0:
        return number
    return None


def valid_range_volume(row) -> bool:
    high, low, volume = _finite(row.get("High")), _finite(row.get("Low")), _finite(row.get("Volume"))
    if high is None or low is None or volume is None:
        return False
    return high >= low and low > 0 and volume > 0


def choose_quote(
    last_price: float | None,
    previous_close: float | None,
    bar_date: date,
    bar_low: float,
    bar_high: float,
    today: date,
    session_opened: bool,
) -> float | None:
    """Pick a quote that sits inside the bar's high/low.

    Before the next session opens, Yahoo's last price is still this bar's
    close. After the open, that close moves to previous_close and last price
    is the new session. Anything outside the printed range is rejected.
    """

    def inside(price: float | None) -> bool:
        if price is None or not math.isfinite(price) or price <= 0:
            return False
        return bar_low * 0.95 <= price <= bar_high * 1.05

    last = _finite(last_price)
    prev = _finite(previous_close)
    if bar_date < today and session_opened and inside(prev):
        return prev
    if bar_date < today and not session_opened and inside(last):
        return last
    if inside(last):
        return last
    if inside(prev):
        return prev
    return None


def repair_ohlc(df: pd.DataFrame, quote_last: float | None = None) -> tuple[pd.DataFrame, str | None]:
    """Backfill a null last close from a quote, or drop the incomplete bar.

    Interior bars with a null close are removed so one missing print cannot
    turn the 20-day and 200-day averages into NaN.
    """
    if df is None or df.empty:
        return df, "empty"
    out = normalize_frame(df)
    for col in OHLCV:
        if col not in out.columns:
            out[col] = np.nan
    note: str | None = None
    if len(out) and pd.isna(out["Close"].iloc[-1]):
        row = out.iloc[-1]
        quote = _finite(quote_last)
        low = _finite(row.get("Low"))
        high = _finite(row.get("High"))
        in_range = (
            valid_range_volume(row) and quote is not None and low is not None and high is not None
            and low * 0.95 <= quote <= high * 1.05
        )
        if in_range:
            closes = out["Close"].to_numpy(dtype=float, copy=True)
            opens = out["Open"].to_numpy(dtype=float, copy=True)
            closes[-1] = quote
            if not math.isfinite(opens[-1]):
                opens[-1] = quote
            out["Close"] = closes
            out["Open"] = opens
            note = "repaired"
        else:
            out = out.iloc[:-1]
            note = "dropped"
    if len(out):
        keep = out["Close"].notna() & out["High"].notna() & out["Low"].notna()
        keep &= out["High"] >= out["Low"]
        out = out.loc[keep].copy()
        out["Open"] = out["Open"].fillna(out["Close"])
        out["Volume"] = out["Volume"].fillna(0.0)
    return out, note


@dataclass
class Quote:
    last: float | None = None
    previous_close: float | None = None


def quote_for_bar(
    quote: Quote | None, bar_date: date, low: float, high: float, today: date, session_opened: bool,
) -> float | None:
    if quote is None:
        return None
    return choose_quote(quote.last, quote.previous_close, bar_date, low, high, today, session_opened)


def close_for_missing_session(quote: Quote | None, now: datetime | None, last_close: float) -> float | None:
    """Official close of the session Yahoo omitted, taken from the quote."""
    if quote is None or not math.isfinite(last_close) or last_close <= 0:
        return None
    if session_has_opened(now):
        primary, secondary = quote.previous_close, quote.last
    else:
        primary, secondary = quote.last, quote.previous_close
    for price in (primary, secondary):
        px = _finite(price)
        if px is not None and 0.7 * last_close <= px <= 1.3 * last_close:
            return px
    return None


def fill_missing_session(
    frame: pd.DataFrame, quote: Quote | None, as_of: pd.Timestamp, now: datetime | None,
) -> tuple[pd.DataFrame, str | None]:
    """Insert the benchmark session when Yahoo dropped that bar entirely.

    The inserted bar uses the quote close for open, high, low, and close, and
    zero volume, so it can update the trend without inventing a breakout.
    """
    if frame is None or frame.empty or frame.index[-1] >= as_of:
        return frame, None
    if (as_of.normalize() - frame.index[-1]).days > 5:
        return frame, None
    price = close_for_missing_session(quote, now, float(frame["Close"].iloc[-1]))
    if price is None:
        return frame, None
    out = frame.copy()
    out.loc[pd.Timestamp(as_of).normalize(), ["Open", "High", "Low", "Close"]] = price
    out.loc[pd.Timestamp(as_of).normalize(), "Volume"] = 0.0
    return out.sort_index(), "repaired"


def clean_symbol(df: pd.DataFrame, quote: Quote | None, now: datetime | None) -> tuple[pd.DataFrame, str | None]:
    """Drop a forming bar, then repair the last completed close."""
    if df is None or df.empty:
        return df, "empty"
    framed = freeze_completed_session(normalize_frame(df), now)
    if framed is None or framed.empty:
        return framed, "empty"
    fill: float | None = None
    if pd.isna(framed["Close"].iloc[-1]) and valid_range_volume(framed.iloc[-1]):
        fill = quote_for_bar(
            quote,
            framed.index[-1].date(),
            float(framed["Low"].iloc[-1]),
            float(framed["High"].iloc[-1]),
            nse_now(now).date(),
            session_has_opened(now),
        )
    return repair_ohlc(framed, fill)


# ------------------------------------------------------------- indicators

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False).mean()


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_frame(df)
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float)
    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()
    out["sma200"] = close.rolling(200).mean()
    out["rsi"] = rsi(close)
    out["atr"] = atr(out)
    out["vol_avg_prior"] = volume.shift(1).rolling(20).mean()
    out["prior_high_20"] = high.shift(1).rolling(20).max()
    out["high_52w"] = high.rolling(252, min_periods=60).max()
    out["ret_20"] = close.pct_change(20)
    out["ret_60"] = close.pct_change(60)
    out["turnover_avg20"] = (close * volume).rolling(20).mean()
    return out


# ----------------------------------------------------------------- regime

@dataclass
class Regime:
    label: str
    stand_aside: bool
    caution: bool
    close: float
    sma50: float
    sma200: float
    ret_20: float
    as_of: str
    reason: str


def _regime_row(close: float, sma50: float, sma200: float, ret20: float, as_of: str) -> Regime:
    if not all(math.isfinite(v) for v in (close, sma50, sma200, ret20)):
        return Regime(
            "Unknown", True, True, close, sma50, sma200, ret20, as_of,
            "Nifty history is too short to classify the trend. No new longs.",
        )
    if close < sma200:
        return Regime(
            "Downtrend", True, False, close, sma50, sma200, ret20, as_of,
            f"Nifty is below its 200-day average ({sma200:,.0f}). No new longs.",
        )
    if ret20 >= OVEREXTEND_STAND_ASIDE:
        return Regime(
            "Overextended", True, True, close, sma50, sma200, ret20, as_of,
            f"Nifty is up {ret20 * 100:.1f}% in 20 sessions. Stand aside from new longs.",
        )
    if close > sma50 > sma200 and ret20 < OVEREXTEND_CAUTION:
        return Regime(
            "Uptrend", False, False, close, sma50, sma200, ret20, as_of,
            "Nifty is above a stacked 50- and 200-day average, without a 20-session surge.",
        )
    details: list[str] = []
    if sma50 <= sma200:
        details.append("the 50-day average is still below the 200-day")
    if close <= sma50:
        details.append("price is below the 50-day average")
    if ret20 >= OVEREXTEND_CAUTION:
        details.append(f"the 20-session gain is {ret20 * 100:.1f}%")
    detail = "; ".join(details) if details else "the trend stack is mixed"
    return Regime(
        "Cautious uptrend", False, True, close, sma50, sma200, ret20, as_of,
        f"Nifty is above its 200-day average, with caution: {detail}.",
    )


def regime_table(nifty: pd.DataFrame) -> pd.DataFrame:
    """Per-session regime. Stand-aside days are not traded."""
    close = nifty["Close"].astype(float)
    sma50 = nifty["sma50"].astype(float)
    sma200 = nifty["sma200"].astype(float)
    ret20 = nifty["ret_20"].astype(float)
    unknown = sma200.isna() | sma50.isna() | ret20.isna() | close.isna()
    close_np = close.to_numpy()
    sma50_np = sma50.to_numpy()
    sma200_np = sma200.to_numpy()
    ret_np = ret20.to_numpy()
    stacked = (close_np > sma50_np) & (sma50_np > sma200_np) & (ret_np < OVEREXTEND_CAUTION)
    # Later assignments win, in the same order as classify_regime.
    labels = np.full(len(nifty), "Cautious uptrend", dtype=object)
    labels = np.where(stacked, "Uptrend", labels)
    labels = np.where(ret_np >= OVEREXTEND_STAND_ASIDE, "Overextended", labels)
    labels = np.where(close_np < sma200_np, "Downtrend", labels)
    labels = np.where(unknown.to_numpy(), "Unknown", labels)
    stand = unknown.to_numpy() | np.isin(labels, ("Downtrend", "Overextended", "Unknown"))
    caution = (
        unknown.to_numpy()
        | (ret20.to_numpy() >= OVEREXTEND_CAUTION)
        | (sma50.to_numpy() <= sma200.to_numpy())
        | (close.to_numpy() <= sma50.to_numpy())
        | (labels == "Overextended")
    )
    # A clean uptrend is not a caution day.
    caution = np.where(labels == "Uptrend", False, caution)
    return pd.DataFrame(
        {"label": labels, "stand_aside": stand, "caution": caution},
        index=nifty.index,
    )


def classify_regime(nifty: pd.DataFrame) -> Regime:
    if nifty is None or nifty.empty or "sma200" not in nifty.columns:
        return Regime("Unknown", True, True, math.nan, math.nan, math.nan, math.nan, "",
                      "Nifty data is unavailable. No new longs.")
    last = nifty.iloc[-1]
    as_of = pd.Timestamp(nifty.index[-1]).date().isoformat()
    return _regime_row(
        float(last["Close"]),
        float(last["sma50"]) if pd.notna(last["sma50"]) else math.nan,
        float(last["sma200"]) if pd.notna(last["sma200"]) else math.nan,
        float(last["ret_20"]) if pd.notna(last["ret_20"]) else math.nan,
        as_of,
    )


# ----------------------------------------------------------------- setups

def _ready(row, columns: tuple[str, ...]) -> bool:
    return all(pd.notna(row[col]) for col in columns)


def breakout_at(ind: pd.DataFrame, i: int) -> bool:
    row = ind.iloc[i]
    cols = ("Close", "High", "Low", "Volume", "sma200", "prior_high_20", "vol_avg_prior", "turnover_avg20")
    if not _ready(row, cols):
        return False
    if float(row["vol_avg_prior"]) <= 0 or float(row["turnover_avg20"]) < MIN_AVG_TURNOVER:
        return False
    if float(row["Close"]) <= float(row["sma200"]):
        return False
    if float(row["Close"]) <= float(row["prior_high_20"]):
        return False
    if float(row["Volume"]) < 1.5 * float(row["vol_avg_prior"]):
        return False
    span = float(row["High"]) - float(row["Low"])
    if span <= 0:
        return False
    close_pos = (float(row["Close"]) - float(row["Low"])) / span
    return close_pos >= 0.75


def pullback_at(ind: pd.DataFrame, i: int) -> bool:
    if i < 5:
        return False
    row = ind.iloc[i]
    prev = ind.iloc[i - 1]
    earlier = ind.iloc[i - 5]
    cols = ("Close", "Low", "sma20", "sma50", "sma200", "atr", "rsi", "turnover_avg20")
    if not _ready(row, cols) or pd.isna(prev["rsi"]) or pd.isna(earlier["sma20"]):
        return False
    if float(row["turnover_avg20"]) < MIN_AVG_TURNOVER:
        return False
    if not (float(row["sma20"]) > float(row["sma50"]) > float(row["sma200"])):
        return False
    if float(row["sma20"]) <= float(earlier["sma20"]):
        return False
    if float(row["Close"]) <= float(row["sma200"]):
        return False
    if float(row["atr"]) <= 0:
        return False
    tag = PULLBACK_TAG_ATR * float(row["atr"])
    if float(row["Low"]) > float(row["sma20"]) + tag:
        return False
    if float(row["Close"]) < float(row["sma20"]) - tag:
        return False
    prev_rsi = float(prev["rsi"])
    if not (40 <= prev_rsi <= 55):
        return False
    return float(row["rsi"]) > prev_rsi


def plan_trade(ind: pd.DataFrame, i: int, setup: str) -> dict | None:
    row = ind.iloc[i]
    entry = float(row["Close"])
    bar_atr = float(row["atr"]) if pd.notna(row.get("atr")) else math.nan
    if not math.isfinite(entry) or entry <= 0 or not math.isfinite(bar_atr) or bar_atr <= 0:
        return None
    multiple = BREAKOUT_ATR_MULT if setup == "breakout" else PULLBACK_ATR_MULT
    lookback = 10 if setup == "breakout" else 15
    window = ind["Low"].iloc[max(0, i - lookback + 1): i + 1]
    if window.isna().all():
        return None
    swing_low = float(window.min())
    atr_stop = entry - multiple * bar_atr
    structural = swing_low - 0.05 * bar_atr
    stop = min(atr_stop, structural)
    if not math.isfinite(stop) or stop >= entry:
        return None
    risk = entry - stop
    two_r = entry + MIN_TARGET_R * risk
    resistance = [two_r]
    for col in ("prior_high_20", "high_52w"):
        if col in ind.columns and pd.notna(row[col]) and float(row[col]) > entry:
            resistance.append(float(row[col]))
    target = max(resistance)
    sessions = BREAKOUT_TIME_STOP if setup == "breakout" else PULLBACK_TIME_STOP
    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "atr": bar_atr,
        "time_stop": sessions,
        "swing_low": swing_low,
        "one_r": entry + risk,
    }


def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> tuple[int, float, float]:
    """Shares, rupees at risk, and position value. Shares is a floor."""
    if capital <= 0 or risk_pct <= 0 or entry <= stop or entry <= 0:
        return 0, 0.0, 0.0
    risk_rupees = capital * (risk_pct / 100.0)
    shares = int(math.floor(risk_rupees / (entry - stop)))
    if shares < 0:
        shares = 0
    return shares, risk_rupees, shares * entry


def relative_strength(stock: pd.DataFrame, nifty: pd.DataFrame, i: int) -> tuple[float, float, float] | None:
    stamp = stock.index[i]
    if stamp not in nifty.index:
        return None
    bench = nifty.loc[stamp]
    row = stock.iloc[i]
    if any(pd.isna(v) for v in (row["ret_20"], row["ret_60"], bench["ret_20"], bench["ret_60"])):
        return None
    rs_20 = float(row["ret_20"] - bench["ret_20"])
    rs_60 = float(row["ret_60"] - bench["ret_60"])
    return rs_20, rs_60, 0.5 * rs_20 + 0.5 * rs_60


def setup_signals(ind: pd.DataFrame, i: int, setup: str, rs_20: float, rs_60: float) -> list[str]:
    row = ind.iloc[i]
    if setup == "breakout":
        baseline = float(row["vol_avg_prior"])
        ratio = float(row["Volume"]) / baseline if baseline else 0.0
        return [
            "Close above the prior 20-day high",
            f"Volume {ratio:.1f}× the prior 20-day average",
            "Close in the top quarter of the day's range",
            "Above the 200-day average",
            f"Excess return vs Nifty {rs_20 * 100:+.1f}% (20d) / {rs_60 * 100:+.1f}% (60d)",
        ]
    prev_rsi = float(ind["rsi"].iloc[i - 1])
    return [
        "20-day average is rising and stacked above the 50- and 200-day",
        "Pullback tagged the 20-day average",
        f"RSI turned up from {prev_rsi:.0f} to {float(row['rsi']):.0f}",
        "Above the 200-day average",
        f"Excess return vs Nifty {rs_20 * 100:+.1f}% (20d) / {rs_60 * 100:+.1f}% (60d)",
    ]


# -------------------------------------------------------------- ranking

@dataclass
class Candidate:
    symbol: str
    name: str
    sector: str
    setup: str
    entry: float
    stop_loss: float
    target: float
    time_stop: int
    shares: int
    risk_rupees: float
    position_value: float
    rr: float
    one_r: float
    rs_20: float
    rs_60: float
    rs_score: float
    rsi: float
    atr: float
    turnover: float
    signals: list[str] = field(default_factory=list)
    earnings_flag: str | None = None

    @property
    def rank_key(self) -> tuple[float, float, float, float]:
        # Higher RS first. Turnover is the numeric tie-break, never universe order.
        return (-self.rs_score, -self.rs_60, -self.rs_20, -self.turnover)


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda item: item.rank_key)


def select_candidates(
    candidates: list[Candidate],
    top_n: int,
    sector_cap: int = SECTOR_CAP,
    earnings_fn=None,
    as_of: date | None = None,
) -> list[Candidate]:
    """Walk the RS ranking, keep at most `sector_cap` names per sector, skip known earnings."""
    selected: list[Candidate] = []
    counts: dict[str, int] = {}
    for candidate in rank_candidates(candidates):
        if counts.get(candidate.sector, 0) >= sector_cap:
            continue
        flag = None
        if earnings_fn is not None and as_of is not None:
            try:
                flag = earnings_fn(candidate.symbol, as_of, candidate.time_stop)
            except Exception:
                flag = "Earnings check unavailable"
        if flag and flag.startswith("Earnings on"):
            continue
        candidate.earnings_flag = flag
        selected.append(candidate)
        counts[candidate.sector] = counts.get(candidate.sector, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def screen(
    history: dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
    regime: Regime,
    capital: float,
    risk_pct: float,
    top_n: int = 10,
    setups: tuple[str, ...] = ("breakout", "pullback"),
    earnings_fn=None,
    sector_cap: int = SECTOR_CAP,
) -> list[Candidate]:
    if regime.stand_aside or nifty is None or nifty.empty:
        return []
    found: list[Candidate] = []
    seen: set[str] = set()
    for symbol, ind in history.items():
        if symbol not in UNIVERSE or len(ind) < 2:
            continue
        i = len(ind) - 1
        rs = relative_strength(ind, nifty, i)
        if rs is None:
            continue
        rs_20, rs_60, rs_score = rs
        name, sector = UNIVERSE[symbol]
        # One card per symbol. A fresh breakout outranks a pullback on the same bar.
        chosen: str | None = None
        for setup in ("breakout", "pullback"):
            if setup not in setups:
                continue
            fired = breakout_at(ind, i) if setup == "breakout" else pullback_at(ind, i)
            if fired:
                chosen = setup
                break
        if chosen is None or symbol in seen:
            continue
        plan = plan_trade(ind, i, chosen)
        if plan is None:
            continue
        entry = round(plan["entry"], 2)
        stop = round(plan["stop"], 2)
        target = round(plan["target"], 2)
        if stop >= entry:
            continue
        risk = entry - stop
        shares, risk_rupees, position_value = position_size(capital, risk_pct, entry, stop)
        row = ind.iloc[i]
        found.append(Candidate(
            symbol=symbol,
            name=name,
            sector=sector,
            setup=chosen,
            entry=entry,
            stop_loss=stop,
            target=target,
            time_stop=plan["time_stop"],
            shares=shares,
            risk_rupees=round(risk_rupees, 2),
            position_value=round(position_value, 2),
            rr=risk and (target - entry) / risk,
            one_r=round(entry + risk, 2),
            rs_20=rs_20,
            rs_60=rs_60,
            rs_score=rs_score,
            rsi=round(float(row["rsi"]), 1) if pd.notna(row["rsi"]) else 0.0,
            atr=float(row["atr"]),
            turnover=float(row["turnover_avg20"]),
            signals=setup_signals(ind, i, chosen, rs_20, rs_60),
        ))
        seen.add(symbol)
    as_of = date.fromisoformat(regime.as_of) if regime.as_of else None
    return select_candidates(found, top_n, sector_cap, earnings_fn, as_of)


# ----------------------------------------------------------- earnings

def _as_date(value) -> date | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


def _calendar_dates(symbol: str) -> list[date]:
    ticker = yf.Ticker(yahoo_symbol(symbol))
    found: list[date] = []
    calendar = None
    try:
        calendar = ticker.calendar
    except Exception:
        calendar = None
    raw_dates = []
    if isinstance(calendar, dict):
        raw = calendar.get("Earnings Date", calendar.get("earnings_date", []))
        raw_dates = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    elif isinstance(calendar, pd.DataFrame) and len(calendar):
        if "Earnings Date" in calendar.columns:
            raw_dates = list(calendar["Earnings Date"])
        elif "Earnings Date" in calendar.index:
            raw_dates = list(calendar.loc["Earnings Date"].tolist())
    for item in raw_dates:
        parsed = _as_date(item)
        if parsed:
            found.append(parsed)
    if found:
        return found
    frame = ticker.get_earnings_dates(limit=12)
    if frame is None or len(frame) == 0:
        return []
    for stamp in frame.index:
        parsed = _as_date(stamp)
        if parsed:
            found.append(parsed)
    return found


def earnings_flag(symbol: str, as_of: date, hold_sessions: int) -> str | None:
    """Flag a name when a known earnings date falls inside the holding window.

    Raises only when the calendar cannot be read; callers treat that as
    'Earnings check unavailable' and keep the name.
    """
    dates = _calendar_dates(symbol)
    if not dates:
        return None
    horizon = (pd.Timestamp(as_of) + pd.tseries.offsets.BDay(hold_sessions)).date()
    upcoming = sorted(d for d in dates if as_of <= d <= horizon)
    if not upcoming:
        return None
    shown = upcoming[0].strftime("%d %b %Y")
    return f"Earnings on {shown} (inside the {hold_sessions}-session window)"


# --------------------------------------------------------------- fetch

@dataclass
class ScanHealth:
    universe: int
    scanned: int
    skipped: list[tuple[str, str]] = field(default_factory=list)
    repaired: list[tuple[str, float]] = field(default_factory=list)
    as_of: str | None = None

    @property
    def as_of_pretty(self) -> str:
        if not self.as_of:
            return "unavailable"
        return format_session_day(date.fromisoformat(self.as_of))

    def banner(self) -> str:
        return (
            f"Scanned {self.scanned} / {self.universe} · "
            f"Skipped {len(self.skipped)} · "
            f"Signal as of {self.as_of_pretty} IST"
        )


@dataclass
class MarketData:
    history: dict[str, pd.DataFrame]
    nifty: pd.DataFrame
    health: ScanHealth
    regime: Regime


def _extract(raw: pd.DataFrame | None, ysym: str) -> pd.DataFrame | None:
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return None
    frame: pd.DataFrame | None = None
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        if ysym in level0:
            frame = raw[ysym].copy()
        elif ysym in level1:
            frame = raw.xs(ysym, axis=1, level=1).copy()
    else:
        frame = raw.copy()
    if frame is None or frame.empty:
        return None
    keep = [col for col in OHLCV if col in frame.columns]
    if "Close" not in keep or "High" not in keep or "Low" not in keep:
        return None
    frame = frame.loc[:, keep].dropna(how="all")
    if frame.empty:
        return None
    return normalize_frame(frame)


def _download(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        period=HISTORY_PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    if not isinstance(raw, pd.DataFrame):
        return pd.DataFrame()
    return raw


def _fetch_quote(symbol: str) -> Quote:
    ysym = symbol if symbol.startswith("^") else yahoo_symbol(symbol)
    try:
        info = yf.Ticker(ysym).fast_info
        return Quote(_finite(getattr(info, "last_price", None)), _finite(getattr(info, "previous_close", None)))
    except Exception:
        return Quote()


def _quotes(symbols: list[str], quote_fn) -> dict[str, Quote]:
    if not symbols:
        return {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(quote_fn, symbols))
    return dict(zip(symbols, values))


def fetch_market(
    now: datetime | None = None,
    downloader=None,
    quote_fn=None,
    quotes: dict[str, Quote] | None = None,
) -> MarketData:
    """Download the universe and Nifty, repair null closes, and freeze the session."""
    ymap = {symbol: yahoo_symbol(symbol) for symbol in UNIVERSE}
    tickers = list(ymap.values()) + [NIFTY_SYMBOL]
    raw = (downloader or _download)(tickers)
    health = ScanHealth(universe=len(UNIVERSE), scanned=0)
    extracted: dict[str, pd.DataFrame] = {}
    needs_quote: list[str] = []
    for symbol, ysym in ymap.items():
        frame = _extract(raw, ysym)
        if frame is None or frame.empty:
            health.skipped.append((symbol, "no price data"))
            continue
        extracted[symbol] = frame

    nifty_raw = _extract(raw, NIFTY_SYMBOL)
    nifty_frozen = freeze_completed_session(nifty_raw, now) if nifty_raw is not None else None
    nifty_mark = nifty_frozen.index[-1] if nifty_frozen is not None and len(nifty_frozen) else None
    if (
        nifty_frozen is not None and len(nifty_frozen)
        and pd.isna(nifty_frozen["Close"].iloc[-1]) and valid_range_volume(nifty_frozen.iloc[-1])
    ):
        needs_quote.append(NIFTY_SYMBOL)
    for symbol, frame in extracted.items():
        frozen = freeze_completed_session(frame, now)
        if frozen is None or frozen.empty:
            continue
        null_close = pd.isna(frozen["Close"].iloc[-1]) and valid_range_volume(frozen.iloc[-1])
        behind = nifty_mark is not None and frozen.index[-1] < nifty_mark
        if null_close or behind:
            needs_quote.append(symbol)

    if quotes is None:
        quotes = _quotes(needs_quote, quote_fn or _fetch_quote)
    else:
        quotes = dict(quotes)

    cleaned: dict[str, pd.DataFrame] = {}
    notes: dict[str, str | None] = {}
    for symbol, frame in extracted.items():
        ready, note = clean_symbol(frame, quotes.get(symbol), now)
        notes[symbol] = note
        if ready is None or len(ready) < MIN_BARS:
            reason = "not enough history for a 200-day average"
            if note == "dropped":
                reason = "incomplete last bar dropped and history is too short"
            health.skipped.append((symbol, reason))
            continue
        cleaned[symbol] = ready

    if nifty_raw is None or nifty_raw.empty:
        regime = classify_regime(pd.DataFrame())
        health.as_of = None
        return MarketData({}, pd.DataFrame(), health, regime)
    nifty_ready, _nifty_note = clean_symbol(nifty_raw, quotes.get(NIFTY_SYMBOL), now)
    if nifty_ready is None or len(nifty_ready) < MIN_BARS:
        regime = classify_regime(pd.DataFrame())
        return MarketData({}, pd.DataFrame(), health, regime)

    as_of = nifty_ready.index[-1]
    history: dict[str, pd.DataFrame] = {}
    for symbol, frame in cleaned.items():
        if frame.index[-1] != as_of:
            frame, fill_note = fill_missing_session(frame, quotes.get(symbol), as_of, now)
            if fill_note:
                notes[symbol] = fill_note
        if frame.index[-1] != as_of:
            health.skipped.append((symbol, f"stale last bar {frame.index[-1].date().isoformat()}"))
            continue
        if notes.get(symbol) == "repaired":
            health.repaired.append((symbol, float(frame["Close"].iloc[-1])))
        history[symbol] = build_indicators(frame)

    nifty = build_indicators(nifty_ready)
    regime = classify_regime(nifty)
    health.scanned = len(history)
    health.as_of = as_of.date().isoformat()
    regime.as_of = health.as_of
    return MarketData(history, nifty, health, regime)


def fetch_last_prices(symbols: list[str], now: datetime | None = None) -> dict[str, float]:
    """Last completed close, replaced by the live quote while the session is open."""
    if not symbols:
        return {}
    ysyms = [yahoo_symbol(symbol) for symbol in symbols]
    try:
        raw = yf.download(
            ysyms, period="10d", interval="1d", group_by="ticker",
            auto_adjust=False, threads=True, progress=False,
        )
    except Exception:
        raw = pd.DataFrame()
    prices: dict[str, float] = {}
    for symbol in symbols:
        frame = _extract(raw, yahoo_symbol(symbol))
        if frame is None:
            continue
        ready, _note = clean_symbol(frame, None, now)
        if ready is not None and len(ready) and pd.notna(ready["Close"].iloc[-1]):
            prices[symbol] = float(ready["Close"].iloc[-1])
    if in_nse_session(now):
        live = _quotes(symbols, _fetch_quote)
        for symbol, quote in live.items():
            if quote.last:
                prices[symbol] = quote.last
    return prices


def format_inr(value: float) -> str:
    return f"₹{value:,.2f}"
