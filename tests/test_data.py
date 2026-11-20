"""Null-close repair, quote selection, and the completed-session freeze."""

from datetime import datetime

import numpy as np
import pandas as pd

from engine import (
    IST,
    Quote,
    ScanHealth,
    build_indicators,
    choose_quote,
    clean_symbol,
    fetch_market,
    freeze_completed_session,
    in_nse_session,
    repair_ohlc,
    session_is_forming,
)


def _frame(index, close, high=None, low=None, volume=None, open_=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    high = close + 1 if high is None else np.asarray(high, dtype=float)
    low = close - 1 if low is None else np.asarray(low, dtype=float)
    opened = close if open_ is None else np.asarray(open_, dtype=float)
    volume = np.full(len(close), 1_000_000.0) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame(
        {"Open": opened, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def test_backfill_null_close_from_in_range_quote():
    index = pd.bdate_range("2026-11-18", periods=2)
    frame = _frame(index, [101.0, np.nan], high=[102.0, 105.0], low=[99.0, 100.5],
                   open_=[100.0, 101.0], volume=[1_000_000, 2_000_000])
    repaired, note = repair_ohlc(frame, 104.0)
    assert note == "repaired"
    assert repaired["Close"].iloc[-1] == 104.0
    assert repaired["Close"].notna().all()


def test_previous_close_fills_a_past_bar_when_the_live_price_has_moved():
    index = pd.to_datetime(["2026-11-19"])
    frame = _frame(index, [np.nan], high=[105.0], low=[100.5], open_=[101.0], volume=[2_000_000])
    # Add a completed prior session so dropping is not the only option we assert on.
    prior = _frame(pd.to_datetime(["2026-11-18"]), [101.0])
    frame = pd.concat([prior, frame])
    quote = Quote(last=130.0, previous_close=104.0)
    now = datetime(2026, 11, 20, 10, 0, tzinfo=IST)
    cleaned, note = clean_symbol(frame, quote, now)
    assert note == "repaired"
    assert cleaned["Close"].iloc[-1] == 104.0
    assert cleaned.index[-1] == pd.Timestamp("2026-11-19")


def test_quote_outside_the_range_drops_the_bar():
    index = pd.to_datetime(["2026-11-18", "2026-11-19"])
    frame = _frame(index, [101.0, np.nan], high=[102.0, 105.0], low=[99.0, 100.5])
    repaired, note = repair_ohlc(frame, 130.0)
    assert note == "dropped"
    assert list(repaired.index) == [pd.Timestamp("2026-11-18")]
    assert repaired["Close"].iloc[-1] == 101.0


def test_volume_only_bar_is_dropped_so_it_cannot_poison_the_average():
    index = pd.bdate_range("2026-01-01", periods=30)
    close = np.full(30, 100.0)
    close[-5] = np.nan
    close[-1] = np.nan
    high = np.where(np.isnan(close), np.nan, close + 1)
    low = np.where(np.isnan(close), np.nan, close - 1)
    # Last bar has volume but no range: the auto-adjust failure mode.
    volume = np.full(30, 1_000_000.0)
    volume[-1] = 5_000_000.0
    frame = _frame(index, close, high=high, low=low, volume=volume)
    repaired, note = repair_ohlc(frame, 100.0)
    assert note == "dropped"
    assert repaired["Close"].notna().all()
    assert pd.Timestamp(index[-1]) not in repaired.index
    assert pd.notna(build_indicators(repaired)["sma20"].iloc[-1])


def test_choose_quote_follows_the_clock_when_both_prices_sit_in_range():
    bar = pd.Timestamp("2026-11-19").date()
    today = pd.Timestamp("2026-11-20").date()
    # Live price is outside the completed bar. Previous close is the one that fits.
    assert choose_quote(130.0, 104.0, bar, 100.5, 105.0, today, True) == 104.0
    # Before the next open, last price is still this session's close.
    assert choose_quote(2074.0, 1993.0, bar, 1981.0, 2075.0, today, False) == 2074.0
    # After the open, that close has moved to previous_close.
    assert choose_quote(2056.0, 2074.0, bar, 1981.0, 2075.0, today, True) == 2074.0


def test_freeze_drops_today_until_the_close_and_keeps_it_after():
    index = pd.to_datetime(["2026-11-19", "2026-11-20"])
    frame = _frame(index, [100.0, 101.0])
    during = freeze_completed_session(frame, datetime(2026, 11, 20, 10, 0, tzinfo=IST))
    preopen = freeze_completed_session(frame, datetime(2026, 11, 20, 9, 14, tzinfo=IST))
    after = freeze_completed_session(frame, datetime(2026, 11, 20, 15, 30, tzinfo=IST))
    weekend = freeze_completed_session(frame, datetime(2026, 11, 21, 11, 0, tzinfo=IST))
    assert list(during.index) == [pd.Timestamp("2026-11-19")]
    assert list(preopen.index) == [pd.Timestamp("2026-11-19")]
    assert list(after.index) == [pd.Timestamp("2026-11-19"), pd.Timestamp("2026-11-20")]
    assert list(weekend.index) == list(after.index)


def test_session_clock_boundaries():
    assert in_nse_session(datetime(2026, 11, 20, 9, 15, tzinfo=IST))
    assert in_nse_session(datetime(2026, 11, 20, 15, 29, tzinfo=IST))
    assert not in_nse_session(datetime(2026, 11, 20, 9, 14, tzinfo=IST))
    assert not in_nse_session(datetime(2026, 11, 20, 15, 30, tzinfo=IST))
    assert not in_nse_session(datetime(2026, 11, 21, 10, 0, tzinfo=IST))
    assert session_is_forming(datetime(2026, 11, 20, 9, 14, tzinfo=IST))
    assert not session_is_forming(datetime(2026, 11, 20, 15, 30, tzinfo=IST))
    assert not session_is_forming(datetime(2026, 11, 21, 10, 0, tzinfo=IST))


def test_forming_bar_is_not_the_signal_even_if_its_close_can_be_filled():
    index = pd.to_datetime(["2026-11-19", "2026-11-20"])
    frame = _frame(index, [101.0, np.nan], high=[102.0, 106.0], low=[99.0, 103.0])
    now = datetime(2026, 11, 20, 11, 0, tzinfo=IST)
    cleaned, note = clean_symbol(frame, Quote(last=104.5, previous_close=101.0), now)
    assert list(cleaned.index) == [pd.Timestamp("2026-11-19")]
    assert cleaned["Close"].iloc[-1] == 101.0
    assert note is None


def _history(index, last_close, last_high, last_low, last_volume) -> pd.DataFrame:
    close = np.full(len(index), 100.0)
    high = close + 1
    low = close - 1
    opened = close.copy()
    volume = np.full(len(index), 1_000_000.0)
    close[-1] = last_close
    high[-1] = last_high
    low[-1] = last_low
    volume[-1] = last_volume
    return _frame(index, close, high=high, low=low, open_=opened, volume=volume)


def _batch(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for symbol, frame in frames.items():
        piece = frame.copy()
        piece.columns = pd.MultiIndex.from_product([[symbol], list(piece.columns)])
        pieces.append(piece)
    return pd.concat(pieces, axis=1)


def test_fetch_repairs_a_null_close_and_skips_a_bar_with_no_range():
    index = pd.bdate_range(end="2026-11-19", periods=210)
    reliance = _history(index, np.nan, 105.0, 100.5, 2_000_000)
    tcs = _history(index, np.nan, np.nan, np.nan, 2_000_000)
    nifty = _history(index, 100.0, 101.0, 99.0, 1.0)
    raw = _batch({"RELIANCE.NS": reliance, "TCS.NS": tcs, "^NSEI": nifty})
    now = datetime(2026, 11, 20, 10, 0, tzinfo=IST)
    market = fetch_market(
        now=now,
        downloader=lambda _tickers: raw,
        quotes={
            "RELIANCE": Quote(last=130.0, previous_close=104.0),
            "TCS": Quote(),
            "^NSEI": Quote(last=100.0, previous_close=100.0),
        },
    )
    assert "RELIANCE" in market.history
    assert market.history["RELIANCE"]["Close"].iloc[-1] == 104.0
    assert "TCS" not in market.history
    assert ("RELIANCE", 104.0) in market.health.repaired
    assert any(symbol == "TCS" and reason.startswith("stale") for symbol, reason in market.health.skipped)
    assert market.health.as_of == "2026-11-19"
    assert market.health.banner() == "Scanned 1 / 109 · Skipped 108 · Signal as of Thu 19 Nov 2026 IST"


def test_null_close_under_todays_candle_is_repaired_after_the_freeze():
    index = pd.to_datetime(["2026-11-19", "2026-11-20"])
    frame = _frame(
        index,
        [np.nan, 2056.0],
        high=[2075.0, 2063.0],
        low=[1981.0, 2053.0],
        open_=[1981.0, 2055.0],
        volume=[2_800_000, 50_000],
    )
    now = datetime(2026, 11, 20, 11, 0, tzinfo=IST)
    cleaned, note = clean_symbol(frame, Quote(last=2056.0, previous_close=2074.0), now)
    assert list(cleaned.index) == [pd.Timestamp("2026-11-19")]
    assert cleaned["Close"].iloc[-1] == 2074.0
    assert note == "repaired"


def test_missing_completed_bar_is_filled_from_the_previous_close():
    index = pd.bdate_range(end="2026-11-18", periods=210)
    today = pd.bdate_range("2026-11-20", periods=1)
    reliance = _history(index, 100.0, 101.0, 99.0, 1_000_000)
    forming = _frame(today, [110.0], high=[111.0], low=[109.0], open_=[109.5], volume=[10_000])
    reliance = pd.concat([reliance, forming])
    nifty_index = pd.bdate_range(end="2026-11-19", periods=210)
    nifty = _history(nifty_index, 100.0, 101.0, 99.0, 1.0)
    nifty_today = _frame(today, [101.0], high=[102.0], low=[100.0], volume=[1.0])
    nifty = pd.concat([nifty, nifty_today])
    raw = _batch({"RELIANCE.NS": reliance, "^NSEI": nifty})
    now = datetime(2026, 11, 20, 11, 0, tzinfo=IST)
    market = fetch_market(
        now=now,
        downloader=lambda _tickers: raw,
        quotes={"RELIANCE": Quote(last=110.0, previous_close=104.0), "^NSEI": Quote(last=101.0, previous_close=100.0)},
    )
    assert market.health.as_of == "2026-11-19"
    assert market.history["RELIANCE"]["Close"].iloc[-1] == 104.0
    assert market.history["RELIANCE"].index[-1] == pd.Timestamp("2026-11-19")
    assert market.history["RELIANCE"]["Volume"].iloc[-1] == 0
    assert ("RELIANCE", 104.0) in market.health.repaired


def test_banner_counts():
    health = ScanHealth(universe=109, scanned=79, skipped=[("X", "stale")], as_of="2026-11-19")
    assert health.banner() == "Scanned 79 / 109 · Skipped 1 · Signal as of Thu 19 Nov 2026 IST"
