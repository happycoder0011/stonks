"""Stonks — NSE swing-trade recommendation engine + trade journal.

Run with:  streamlit run app.py
Data: free Yahoo Finance end-of-day NSE data (~15 min delayed).
Not investment advice — a screening and journaling tool.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import db
from engine import Recommendation, compute_indicators, fetch_history, fetch_last_prices, recommend

st.set_page_config(page_title="Stonks — NSE Trade Ideas", page_icon="📈",
                   layout="wide")

# ---- palette tokens (validated reference palette) ----
C_SERIES = "#2a78d6"      # primary series (blue)
C_SERIES2 = "#1baf7a"     # secondary series (aqua)
C_MUTED = "#898781"       # axis / muted ink
C_GRID = "#e1e0d9"        # hairline grid
C_GOOD = "#0ca30c"        # profit
C_CRIT = "#d03b3b"        # loss
C_TEXT = "#0b0b0b"


def _style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                  color=C_MUTED, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=C_GRID, linecolor=C_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=C_GRID, linecolor=C_GRID, zeroline=False)
    return fig


# ------------------------------------------------------------- cached data

@st.cache_data(ttl=900, show_spinner="Fetching NSE data (free, ~15 min delayed)…")
def load_history_cached() -> tuple[dict[str, pd.DataFrame], datetime]:
    return fetch_history(period="1y"), datetime.now()


def load_history() -> dict[str, pd.DataFrame]:
    return load_history_cached()[0]


@st.cache_data(ttl=900, show_spinner=False)
def load_recommendations(mode: str, top_n: int) -> list[Recommendation]:
    return recommend(load_history(), mode, top_n)


@st.cache_data(ttl=600, show_spinner="Marking positions to market…")
def load_last_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    return fetch_last_prices(list(symbols))


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("📈 Stonks")
    st.caption("NSE swing-trade screener & journal")

    mode_label = st.radio("Trade style", ["Daily (1–3 days)", "Weekly (1–2 weeks)"])
    mode = "daily" if mode_label.startswith("Daily") else "weekly"

    top_n = st.slider("Number of ideas", 5, 20, 10)

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(
        "Data: Yahoo Finance (free, end-of-day / ~15 min delayed). "
        "Universe: ~100 liquid NSE large/mid caps — edit `universe.py` to change it."
    )
    st.warning(
        "⚠️ Educational tool, **not investment advice**. Recommendations are "
        "rule-based technical screens. Always do your own research and size "
        "positions so a stop-loss hit costs ≤1–2% of capital.",
        icon="⚠️",
    )

tab_recs, tab_trades, tab_dash = st.tabs(
    ["🎯 Recommendations", "💼 My Trades", "📊 Dashboard"])


# ----------------------------------------------------- tab 1: recommendations

with tab_recs:
    st.subheader(f"Top ideas — {mode_label}")

    try:
        history, fetched_at = load_history_cached()
        recs = load_recommendations(mode, top_n)
    except Exception as e:
        st.error(f"Data fetch failed — check your internet connection. ({e})")
        history, fetched_at, recs = {}, None, []

    if fetched_at is not None and history:
        last_candle = max(df.index[-1] for df in history.values())
        st.caption(
            f"Prices as of **{last_candle:%a %d %b %Y}** (last trading day) · "
            f"data fetched at **{fetched_at:%H:%M:%S}**, auto-refreshes every "
            f"15 min · use 🔄 in the sidebar to force a refresh."
        )

    with st.expander("ℹ️ How recommendations are scored (0–100)"):
        st.markdown(
            """
**Daily mode (1–3 day holds)** — catches short-term momentum bursts:
| Component | Points | Full marks when |
|---|---|---|
| Momentum thrust | 30 | +2.5% today (15) and +5% over 5 days (15) |
| Volume surge | 25 | Volume ≥ 2.5× the 20-day average |
| RSI sweet spot | 20 | RSI 55–70; RSI > 80 **subtracts** 10 |
| Above short-term trend | 15 | Price above 20-day average |
| Close near day's high | 10 | Buyers in control into the close |

**Weekly mode (1–2 week holds)** — established trends with fresh fuel:
| Component | Points | Full marks when |
|---|---|---|
| Trend structure | 30 | Price > 20d avg > 50d avg, 50d rising |
| MACD | 20 | Bullish (10), +10 if crossover was within 5 sessions |
| 1-month momentum | 20 | +10% in 20 days; > +20% **loses** 10 (parabolic) |
| Near 52-week high | 15 | Within 3% of the high (breakout zone) |
| RSI health | 15 | RSI 50–65; > 78 subtracts 10 |

**Levels:** stop-loss and target scale with each stock's ATR (typical daily
range) — daily: entry − 1.5×ATR / entry + 2.5×ATR; weekly: − 2×ATR / + 4×ATR.
"""
        )

    if not recs:
        st.info("No candidates scored above zero right now — quiet market or data issue.")

    for rec in recs:
        rr = ((rec.target - rec.entry) / (rec.entry - rec.stop_loss)
              if rec.entry > rec.stop_loss else 0)
        header = (f"**{rec.symbol}** · {rec.name} · score **{rec.score:.0f}**"
                  f" · ₹{rec.close:,.2f}")
        with st.expander(header, expanded=False):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Entry", f"₹{rec.entry:,.2f}")
            c2.metric("Stop-loss", f"₹{rec.stop_loss:,.2f}",
                      f"{(rec.stop_loss / rec.entry - 1) * 100:.1f}%")
            c3.metric("Target", f"₹{rec.target:,.2f}",
                      f"{(rec.target / rec.entry - 1) * 100:.1f}%")
            c4.metric("Risk : Reward", f"1 : {rr:.1f}")
            c5.metric("RSI", f"{rec.rsi:.0f}")

            st.markdown("**Why:** " + " · ".join(rec.signals)
                        if rec.signals else "_No standout signals_")
            st.caption(f"Sector: {rec.sector} · 5-day {rec.ret_5d:+.1f}% · "
                       f"1-month {rec.ret_20d:+.1f}% · volume {rec.vol_ratio:.1f}× avg")

            # price chart: close + moving averages
            hist = load_history().get(rec.symbol)
            if hist is not None:
                ind = compute_indicators(hist).iloc[-120:]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"],
                                         name="Close", mode="lines",
                                         line=dict(color=C_SERIES, width=2)))
                fig.add_trace(go.Scatter(x=ind.index, y=ind["sma20"],
                                         name="20-day avg", mode="lines",
                                         line=dict(color=C_SERIES2, width=1.5)))
                fig.add_trace(go.Scatter(x=ind.index, y=ind["sma50"],
                                         name="50-day avg", mode="lines",
                                         line=dict(color=C_MUTED, width=1.5,
                                                   dash="dot")))
                fig.add_hline(y=rec.target, line_color=C_GOOD, line_width=1,
                              line_dash="dash",
                              annotation_text=f"Target ₹{rec.target:,.0f}",
                              annotation_font_color=C_GOOD)
                fig.add_hline(y=rec.stop_loss, line_color=C_CRIT, line_width=1,
                              line_dash="dash",
                              annotation_text=f"Stop ₹{rec.stop_loss:,.0f}",
                              annotation_font_color=C_CRIT)
                fig.update_layout(title=f"{rec.symbol} — last 120 sessions")
                st.plotly_chart(_style_fig(fig), width="stretch",
                                key=f"chart_{mode}_{rec.symbol}")

            # take-this-trade form
            with st.form(key=f"take_{mode}_{rec.symbol}"):
                fc1, fc2, fc3 = st.columns([1, 2, 1])
                qty = fc1.number_input("Quantity", min_value=1, value=1,
                                       key=f"qty_{mode}_{rec.symbol}")
                notes = fc2.text_input("Notes (optional)",
                                       key=f"notes_{mode}_{rec.symbol}")
                if fc3.form_submit_button("✅ Take this trade",
                                          type="primary",
                                          use_container_width=True):
                    db.add_trade(rec.symbol, rec.name, rec.sector, mode,
                                 rec.entry, int(qty), rec.stop_loss,
                                 rec.target, rec.score,
                                 " · ".join(rec.signals), notes)
                    st.success(f"Logged: {qty} × {rec.symbol} @ ₹{rec.entry:,.2f}")


# --------------------------------------------------------- tab 2: my trades

with tab_trades:
    st.subheader("Open positions")
    open_trades = db.get_trades("open")

    if open_trades.empty:
        st.info("No open trades. Take a recommendation from the first tab to start tracking.")
    else:
        prices = load_last_prices(tuple(sorted(open_trades["symbol"].unique())))
        rows = []
        for _, t in open_trades.iterrows():
            last = prices.get(t["symbol"], t["entry_price"])
            unreal = (last - t["entry_price"]) * t["qty"]
            rows.append({
                "id": t["id"], "Symbol": t["symbol"], "Mode": t["mode"],
                "Taken": t["date_taken"], "Qty": t["qty"],
                "Entry ₹": t["entry_price"], "Last ₹": round(last, 2),
                "Stop ₹": t["stop_loss"], "Target ₹": t["target"],
                "Unrealized ₹": round(unreal, 2),
                "Unrealized %": round((last / t["entry_price"] - 1) * 100, 2),
            })
        pos = pd.DataFrame(rows)
        st.dataframe(
            pos.drop(columns=["id"]),
            width="stretch", hide_index=True,
            column_config={
                "Unrealized ₹": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        # stop-loss / target alerts
        for _, p in pos.iterrows():
            if p["Last ₹"] <= p["Stop ₹"]:
                st.error(f"🛑 {p['Symbol']} is at/below stop-loss "
                         f"(₹{p['Last ₹']:,.2f} ≤ ₹{p['Stop ₹']:,.2f}) — consider exiting.")
            elif p["Last ₹"] >= p["Target ₹"]:
                st.success(f"🎯 {p['Symbol']} hit its target "
                           f"(₹{p['Last ₹']:,.2f} ≥ ₹{p['Target ₹']:,.2f}) — consider booking profit.")

        st.markdown("##### Close a position")
        with st.form("close_trade"):
            cc1, cc2, cc3, cc4 = st.columns([2, 1, 2, 1])
            labels = {
                int(t["id"]): f"#{t['id']} {t['symbol']} ({t['qty']} @ ₹{t['entry_price']:,.2f})"
                for _, t in open_trades.iterrows()
            }
            sel = cc1.selectbox("Trade", list(labels), format_func=labels.get)
            default_px = float(prices.get(
                open_trades.set_index("id").loc[sel, "symbol"],
                open_trades.set_index("id").loc[sel, "entry_price"]))
            exit_px = cc2.number_input("Exit price ₹", min_value=0.01,
                                       value=round(default_px, 2), format="%.2f")
            close_notes = cc3.text_input("Notes (optional)", key="close_notes")
            if cc4.form_submit_button("Close trade", type="primary",
                                      use_container_width=True):
                db.close_trade(int(sel), float(exit_px), close_notes)
                st.success("Trade closed and P&L recorded.")
                st.rerun()

    st.divider()
    st.subheader("Trade history")
    closed = db.get_trades("closed")
    if closed.empty:
        st.caption("No closed trades yet.")
    else:
        hist_view = closed[["id", "date_taken", "exit_date", "symbol", "mode",
                            "qty", "entry_price", "exit_price", "pnl",
                            "pnl_pct", "notes"]].rename(columns={
            "date_taken": "Taken", "exit_date": "Exited", "symbol": "Symbol",
            "mode": "Mode", "qty": "Qty", "entry_price": "Entry ₹",
            "exit_price": "Exit ₹", "pnl": "P&L ₹", "pnl_pct": "P&L %",
            "notes": "Notes", "id": "#"})
        st.dataframe(hist_view, width="stretch", hide_index=True)

        with st.expander("Delete a trade record"):
            del_id = st.number_input("Trade #", min_value=1, step=1)
            if st.button("Delete", type="secondary"):
                db.delete_trade(int(del_id))
                st.rerun()


# ---------------------------------------------------------- tab 3: dashboard

with tab_dash:
    st.subheader("Performance")
    all_trades = db.get_trades()
    closed = all_trades[all_trades["status"] == "closed"]

    if all_trades.empty:
        st.info("Your dashboard appears once you take your first trade.")
    else:
        wins = closed[closed["pnl"] > 0]
        total_pnl = closed["pnl"].sum() if not closed.empty else 0.0
        win_rate = len(wins) / len(closed) * 100 if len(closed) else 0.0
        avg_ret = closed["pnl_pct"].mean() if not closed.empty else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Realized P&L", f"₹{total_pnl:,.0f}")
        m2.metric("Win rate", f"{win_rate:.0f}%",
                  help="Share of closed trades with positive P&L")
        m3.metric("Closed trades", f"{len(closed)}")
        m4.metric("Open positions",
                  f"{len(all_trades[all_trades['status'] == 'open'])}")
        m5.metric("Avg return / trade", f"{avg_ret:+.2f}%")

        if not closed.empty:
            g1, g2 = st.columns(2)

            with g1:
                # cumulative realized P&L over time (single series)
                eq = (closed.sort_values("exit_date")
                      .groupby("exit_date")["pnl"].sum().cumsum())
                fig = go.Figure(go.Scatter(
                    x=eq.index, y=eq.values, mode="lines+markers",
                    name="Cumulative P&L",
                    line=dict(color=C_SERIES, width=2),
                    marker=dict(size=8, color=C_SERIES),
                ))
                fig.add_hline(y=0, line_color=C_MUTED, line_width=1)
                fig.update_layout(title="Cumulative realized P&L (₹)")
                st.plotly_chart(_style_fig(fig), width="stretch")

            with g2:
                # per-trade P&L, colored by outcome (status colors + labels)
                per = closed.sort_values("id")
                labels = [f"#{i} {s}" for i, s in zip(per["id"], per["symbol"])]
                colors = [C_GOOD if p > 0 else C_CRIT for p in per["pnl"]]
                fig = go.Figure(go.Bar(
                    x=labels, y=per["pnl"], marker_color=colors,
                    marker=dict(cornerradius=4),
                    text=[f"₹{p:,.0f}" for p in per["pnl"]],
                    textposition="outside",
                ))
                fig.add_hline(y=0, line_color=C_MUTED, line_width=1)
                fig.update_layout(title="P&L per closed trade (green = profit, red = loss)")
                st.plotly_chart(_style_fig(fig), width="stretch")

            b1, b2 = st.columns(2)
            with b1:
                st.markdown("##### By trade style")
                by_mode = (closed.groupby("mode")
                           .agg(trades=("id", "count"), pnl=("pnl", "sum"),
                                win_rate=("pnl", lambda s: (s > 0).mean() * 100),
                                avg_pct=("pnl_pct", "mean"))
                           .round(2).reset_index()
                           .rename(columns={"mode": "Style", "trades": "Trades",
                                            "pnl": "P&L ₹", "win_rate": "Win %",
                                            "avg_pct": "Avg %"}))
                st.dataframe(by_mode, hide_index=True, width="stretch")
            with b2:
                st.markdown("##### By sector")
                by_sec = (closed.groupby("sector")
                          .agg(trades=("id", "count"), pnl=("pnl", "sum"))
                          .round(2).sort_values("pnl", ascending=False)
                          .reset_index()
                          .rename(columns={"sector": "Sector",
                                           "trades": "Trades", "pnl": "P&L ₹"}))
                st.dataframe(by_sec, hide_index=True, width="stretch")
