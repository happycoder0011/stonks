# Stonks — NSE swing-trade ideas and journal

A Streamlit app that screens about 109 liquid NSE stocks for **breakout** (about 3 sessions) and **pullback** (1–2 weeks) ideas, sizes the position from your capital, and keeps a journal of the trades you actually take.

Not investment advice. It does not place orders.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh
```

Or `streamlit run app.py`. Then open http://localhost:8501.

## Deploy (Streamlit Community Cloud)

No API keys are required.

1. This repo is the app. Main file: `app.py`. Branch: `main`.
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → this repo, `main`, `app.py`.
3. Leave secrets empty unless you want to lock the journal.
4. Deploy. The build installs the pinned `requirements.txt`.

Optional journal lock: set `STONKS_PASSWORD` in the Cloud app's secrets (or as an environment variable). When it is unset, the journal is open. When it is set, the book stays readable and writes wait for that password. On Cloud, add it in Settings → Secrets:

```toml
STONKS_PASSWORD = "choose-a-long-password"
```

The journal is a SQLite file next to the app. A redeploy replaces the container and deletes `trades.db`. Back it up, or point `STONKS_DB` at a path you persist.

## What a card means

Signals use the **last completed NSE session**. A bar dated today is ignored until 15:30 IST.

- **Breakout:** close above the prior 20-day high, volume at least 1.5× the prior 20-day average, close in the top quarter of the day's range, above the 200-day average.
- **Pullback:** 20-day average rising and stacked above the 50- and 200-day, the bar tags the 20-day average within 0.5 ATR, RSI(14) turns up from 40–55.
- Both require average daily turnover of at least ₹20 crore.
- Names are ordered by 20-day and 60-day return minus the Nifty 50. At most two names per sector.
- A known earnings date inside the holding window removes the name. If the calendar cannot be read, the name stays and the card says so.
- **No new longs** when the Nifty is below its 200-day average, or when the Nifty is up 10% or more in 20 sessions.

The stop is the further of the recent swing low and 1.5× ATR (breakout) or 2× ATR (pullback). The target is prior resistance or at least 2R. After price trades +1R, the plan moves the stop to entry. Time stop is 3 sessions or 10 sessions.

Shares = floor(capital × risk% / (entry − stop)).

## Backtest

The Backtest tab replays both setups on the same daily cache: signal on the close, enter the next open, exit on stop, target, the +1R breakeven (armed for the next session), or the time stop. Brokerage and slippage default to 10 bps per side and can be turned off.

## Tests

```bash
.venv/bin/pytest
```
