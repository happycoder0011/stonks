# 📈 Stonks — NSE Swing-Trade Recommendation Engine & Journal

A local Streamlit app that screens ~105 liquid NSE stocks for **daily (1–3 day)**
and **weekly (1–2 week)** trade ideas using free Yahoo Finance data, and keeps a
journal + performance dashboard of the trades you actually take.

> ⚠️ **Not investment advice.** This is a rule-based technical screener for
> educational use. Always do your own research and manage risk.

## Run it

```bash
./run.sh
# or manually:
.venv/bin/streamlit run app.py
```

Then open http://localhost:8501

## What's inside

| Tab | What it does |
|---|---|
| 🎯 Recommendations | Top-scored ideas for the selected style, each with entry, ATR-based stop-loss & target, risk:reward, the signals that fired, and a price chart. One click logs the trade. |
| 💼 My Trades | Open positions marked to the latest price, with stop-loss/target alerts; close positions to record realized P&L; full trade history. |
| 📊 Dashboard | Realized P&L, win rate, average return, cumulative P&L curve, per-trade P&L, and breakdowns by style and sector. |

## How scoring works (0–100)

**Daily mode** favors short bursts: 1-day and 5-day momentum, volume surge vs
20-day average, RSI in the 55–70 momentum zone, price above the 20-day average,
close near the day's high. Stop = entry − 1.5×ATR, target = entry + 2.5×ATR.

**Weekly mode** favors established trends: price > 20d > 50d averages with a
rising 50d, MACD bullish (fresh crossovers score extra), steady 1-month momentum
(parabolic moves are penalized), proximity to the 52-week high, healthy RSI.
Stop = entry − 2×ATR, target = entry + 4×ATR.

## Data

- **Source:** Yahoo Finance via `yfinance` — free, no API key. End-of-day /
  ~15-min delayed, refreshed every 15 minutes while the app is open.
- **Universe:** Nifty-100-ish list in `universe.py` — edit it to add/remove stocks.
- **Journal:** SQLite file `trades.db` in this folder (created on first trade).

### Upgrading to live data (optional)

Delayed data is fine for daily/weekly swing trades. If you later want live ticks
or in-app order placement, Zerodha's [Kite Connect](https://kite.trade) API
(₹2,000/month) can be plugged into `engine.py` by replacing the two fetch
functions — the rest of the app doesn't care where prices come from.

## Files

- `app.py` — Streamlit UI (three tabs)
- `engine.py` — data fetch, indicators (RSI, MACD, ATR, SMAs), scoring
- `universe.py` — the stock list
- `db.py` — SQLite trade journal
- `trades.db` — your trade history (back this up!)
