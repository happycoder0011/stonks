"""Stonks — NSE swing-trade ideas, journal, and walk-forward backtest.

Run with:  streamlit run app.py
Signals use the last completed NSE session. Not investment advice.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import db
from backtest import run_backtest
from engine import (
    Candidate,
    cache_bucket,
    earnings_flag,
    fetch_last_prices,
    fetch_market,
    screen,
)

st.set_page_config(page_title="Stonks — NSE Swing Trades", page_icon="📈", layout="wide")

C_SERIES = "#2a78d6"
C_SERIES2 = "#1baf7a"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_GOOD = "#0ca30c"
C_CRIT = "#d03b3b"


def _style_fig(fig: go.Figure, height: int = 460) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                  color=C_MUTED, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor=C_GRID, linecolor=C_GRID, zeroline=False, rangeslider_visible=False)
    fig.update_yaxes(gridcolor=C_GRID, linecolor=C_GRID, zeroline=False)
    return fig


def trade_chart(hist: pd.DataFrame, rec: Candidate) -> go.Figure:
    view = hist.iloc[-120:]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.74, 0.26],
    )
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
        name="Price",
        increasing_line_color=C_GOOD, decreasing_line_color=C_CRIT,
        increasing_fillcolor=C_GOOD, decreasing_fillcolor=C_CRIT,
        showlegend=False,
    ), row=1, col=1)
    for column, name, color, dash in (
        ("sma20", "20-day", C_SERIES2, "solid"),
        ("sma50", "50-day", C_SERIES, "solid"),
        ("sma200", "200-day", C_MUTED, "dot"),
    ):
        fig.add_trace(go.Scatter(
            x=view.index, y=view[column], name=name, mode="lines",
            line=dict(color=color, width=1.4, dash=dash),
        ), row=1, col=1)
    for level, label, color in (
        (rec.entry, f"Entry ₹{rec.entry:,.0f}", C_SERIES),
        (rec.stop_loss, f"Stop ₹{rec.stop_loss:,.0f}", C_CRIT),
        (rec.target, f"Target ₹{rec.target:,.0f}", C_GOOD),
    ):
        fig.add_hline(
            y=level, line_color=color, line_dash="dash", line_width=1,
            annotation_text=label, annotation_font_color=color,
            annotation_position="right", row=1, col=1,
        )
    volume_colors = [
        C_GOOD if close >= opened else C_CRIT
        for close, opened in zip(view["Close"], view["Open"])
    ]
    fig.add_trace(go.Bar(
        x=view.index, y=view["Volume"], marker_color=volume_colors,
        name="Volume", showlegend=False,
    ), row=2, col=1)
    fig.update_layout(title=f"{rec.symbol} — last {len(view)} completed sessions")
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return _style_fig(fig)


@st.cache_data(ttl=900, show_spinner="Fetching NSE data (last completed session)…")
def load_market(bucket: str):
    del bucket  # cache key: forming vs closed session
    return fetch_market()


@st.cache_data(ttl=900, show_spinner="Running the walk-forward backtest…")
def load_backtest(bucket: str, cost_bps: float):
    market = load_market(bucket)
    return run_backtest(market.history, market.nifty, cost_bps)


@st.cache_data(ttl=900, show_spinner="Ranking setups and checking earnings dates…")
def load_ideas(bucket: str, capital: float, risk_pct: float, top_n: int, setups: tuple[str, ...]):
    market = load_market(bucket)
    return screen(
        market.history, market.nifty, market.regime,
        capital, risk_pct, top_n, setups, earnings_fn=earnings_flag,
    )


def _day(value: str) -> str:
    return str(value)[:10]


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("Stonks")
    st.caption("NSE swing ideas for the next few days to two weeks")
    setup_label = st.radio("Setups", ["Both", "Breakout (3 sessions)", "Pullback (10 sessions)"])
    if setup_label.startswith("Breakout"):
        setups: tuple[str, ...] = ("breakout",)
    elif setup_label.startswith("Pullback"):
        setups = ("pullback",)
    else:
        setups = ("breakout", "pullback")
    top_n = st.slider("Number of ideas", 3, 12, 8)
    capital = st.number_input("Capital (₹)", min_value=1000.0, value=500_000.0, step=50_000.0)
    risk_pct = st.number_input("Risk % per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    charge_costs = st.checkbox("Include brokerage and slippage in the backtest", value=True)
    cost_bps = st.number_input(
        "Cost per side (bps)", min_value=0.0, max_value=50.0, value=10.0, step=1.0,
        disabled=not charge_costs,
    )
    journal_password = ""
    if db.configured_password():
        journal_password = st.text_input("Journal password", type="password")
        if db.password_ok(journal_password):
            st.caption("Journal writes are unlocked.")
        else:
            st.caption("Journal writes are locked. Open trades stay visible.")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption(
        "Data: Yahoo Finance daily bars, unadjusted. "
        "Signals ignore today's candle until 15:30 IST."
    )
    st.warning(
        "Educational tool, not investment advice. "
        "Size every position so a stop-out costs the risk percent above.",
        icon="⚠️",
    )

bucket = cache_bucket()
active_cost = float(cost_bps) if charge_costs else 0.0

try:
    market = load_market(bucket)
    backtest = load_backtest(bucket, active_cost)
    ideas = load_ideas(bucket, float(capital), float(risk_pct), int(top_n), setups)
except Exception as exc:
    st.error(f"Data fetch failed. Check the connection and refresh. ({exc})")
    st.stop()

open_book = db.get_trades("open")
open_risk = db.planned_open_risk(open_book)
with st.sidebar:
    heat = (open_risk / capital * 100) if capital else 0.0
    st.metric("Open risk to stops", f"₹{open_risk:,.0f}", f"{heat:.1f}% of capital")

tab_today, tab_trades, tab_dash, tab_backtest = st.tabs(
    ["Today's Trade", "My Trades", "Dashboard", "Backtest"])


def _regime_banner() -> None:
    regime = market.regime
    if regime.stand_aside:
        st.error(f"**{regime.label}.** {regime.reason}")
    elif regime.caution:
        st.warning(f"**{regime.label}.** {regime.reason}")
    else:
        st.success(f"**{regime.label}.** {regime.reason}")


with tab_today:
    st.info(market.health.banner())
    _regime_banner()
    with st.expander("Scan details"):
        if market.health.repaired:
            st.markdown("**Close backfilled from the quote**")
            for symbol, price in market.health.repaired:
                st.write(f"{symbol} — close filled at ₹{price:,.2f}")
        if market.health.skipped:
            st.markdown("**Skipped**")
            for symbol, reason in market.health.skipped:
                st.write(f"{symbol} — {reason}")
        else:
            st.write("No symbols skipped.")
        if market.regime.close == market.regime.close:  # not NaN
            st.caption(
                f"Nifty {market.regime.close:,.0f} · 50-day {market.regime.sma50:,.0f} · "
                f"200-day {market.regime.sma200:,.0f} · 20-session {market.regime.ret_20 * 100:+.1f}%"
            )

    if market.regime.stand_aside:
        st.info("No new long is suggested while the regime is stand-aside. The backtest tab still shows history.")
    elif not ideas:
        st.info("No breakout or pullback cleared the trend, liquidity, and earnings gates on this session.")

    for rec in ideas:
        stats = backtest.for_setup(rec.setup)
        stop_pct = (rec.stop_loss / rec.entry - 1) * 100
        target_pct = (rec.target / rec.entry - 1) * 100
        header = (
            f"**{rec.symbol}** · {rec.name} · {rec.setup.title()} · "
            f"RS {rec.rs_score * 100:+.1f}%"
        )
        with st.expander(header, expanded=False):
            badges = [market.regime.label, rec.setup.title(), f"{rec.time_stop}-session time stop"]
            st.caption(" · ".join(badges))
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Entry", f"₹{rec.entry:,.2f}")
            c2.metric("Stop", f"₹{rec.stop_loss:,.2f}", f"{stop_pct:.1f}%")
            c3.metric("Target", f"₹{rec.target:,.2f}", f"{target_pct:.1f}%")
            c4.metric("Shares", f"{rec.shares:,}")
            c5.metric("Reward : risk", f"{rec.rr:.1f}R")
            c6.metric("Historical expectancy", f"{stats.expectancy_r:+.2f}R", f"{stats.trades} trades")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Time stop", f"{rec.time_stop} sessions")
            d2.metric("+1R, then stop to entry", f"₹{rec.one_r:,.2f}")
            d3.metric("Risk on this trade", f"₹{rec.shares * (rec.entry - rec.stop_loss):,.0f}")
            d4.metric("Position value", f"₹{rec.position_value:,.0f}")
            st.caption(
                f"Sector {rec.sector} · RSI {rec.rsi:.0f} · "
                f"20-day excess {rec.rs_20 * 100:+.1f}% · 60-day excess {rec.rs_60 * 100:+.1f}% · "
                f"avg turnover ₹{rec.turnover / 1e7:.0f} Cr"
            )
            st.markdown("**Plan:** " + " · ".join(rec.signals))
            st.caption(
                "Entry is the signal close. Place the order at the next open or as a limit. "
                f"If price trades ₹{rec.one_r:,.2f} (+1R), move the stop to ₹{rec.entry:,.2f}. "
                f"Exit at the close of session {rec.time_stop} if neither stop nor target has hit."
            )
            st.caption("Backtest: " + stats.expectancy_label())
            if rec.earnings_flag:
                st.warning(rec.earnings_flag)
            hist = market.history.get(rec.symbol)
            if hist is not None:
                st.plotly_chart(
                    trade_chart(hist, rec), width="stretch",
                    key=f"chart_{rec.setup}_{rec.symbol}",
                )
            if rec.shares < 1:
                st.warning("This risk budget does not cover one share at the planned stop. Raise capital or lower the risk only if you accept a larger percent.")
            else:
                with st.form(key=f"take_{rec.setup}_{rec.symbol}"):
                    fc1, fc2, fc3 = st.columns([1, 2, 1])
                    qty = fc1.number_input(
                        "Quantity", min_value=1, value=int(rec.shares),
                        key=f"qty_{rec.setup}_{rec.symbol}",
                    )
                    notes = fc2.text_input("Notes (optional)", key=f"notes_{rec.setup}_{rec.symbol}")
                    if fc3.form_submit_button("Take this trade", type="primary", use_container_width=True):
                        if not db.password_ok(journal_password):
                            st.error("Enter the journal password in the sidebar before recording a trade.")
                        else:
                            db.add_trade(
                                rec.symbol, rec.name, rec.sector, rec.setup,
                                rec.entry, int(qty), rec.stop_loss, rec.target,
                                rec.rs_score, " · ".join(rec.signals), notes,
                                time_stop_sessions=rec.time_stop,
                            )
                            st.success(f"Logged {qty} × {rec.symbol} @ ₹{rec.entry:,.2f}")
                            st.rerun()


with tab_trades:
    st.subheader("Open positions")
    if open_book.empty:
        st.info("No open trades. A recommendation from Today's Trade is the way to start the journal.")
    else:
        prices = fetch_last_prices(sorted(open_book["symbol"].astype(str).unique()))
        rows = []
        today = date.fromisoformat(db.ist_timestamp()[:10])
        for _, trade in open_book.iterrows():
            last = prices.get(trade["symbol"], trade["entry_price"])
            held = db.sessions_held(trade["date_taken"], today)
            one_r = trade["entry_price"] + (trade["entry_price"] - trade["stop_loss"])
            rows.append({
                "id": trade["id"],
                "Symbol": trade["symbol"],
                "Setup": trade["mode"],
                "Taken": _day(trade["date_taken"]),
                "Sessions": held,
                "Time stop": trade["time_stop_sessions"],
                "Qty": trade["qty"],
                "Entry ₹": trade["entry_price"],
                "Last ₹": round(float(last), 2),
                "Stop ₹": trade["stop_loss"],
                "Target ₹": trade["target"],
                "+1R ₹": round(float(one_r), 2),
                "Unrealized ₹": round((last - trade["entry_price"]) * trade["qty"], 2),
                "Unrealized %": round((last / trade["entry_price"] - 1) * 100, 2),
            })
        positions = pd.DataFrame(rows)
        st.dataframe(positions.drop(columns=["id"]), width="stretch", hide_index=True)
        st.caption(f"Planned open risk to stops: ₹{open_risk:,.0f} ({heat:.1f}% of capital).")
        for _, pos in positions.iterrows():
            if pos["Last ₹"] <= pos["Stop ₹"]:
                st.error(
                    f"{pos['Symbol']} is at or through its stop "
                    f"(₹{pos['Last ₹']:,.2f} ≤ ₹{pos['Stop ₹']:,.2f})."
                )
            elif pos["Last ₹"] >= pos["Target ₹"]:
                st.success(
                    f"{pos['Symbol']} has reached its target "
                    f"(₹{pos['Last ₹']:,.2f} ≥ ₹{pos['Target ₹']:,.2f})."
                )
            elif pos["Last ₹"] >= pos["+1R ₹"]:
                st.info(
                    f"{pos['Symbol']} has traded through +1R. "
                    f"The plan moves the stop to the entry at ₹{pos['Entry ₹']:,.2f}."
                )
            time_stop = pos["Time stop"]
            if pd.notna(time_stop) and pos["Sessions"] >= int(time_stop):
                st.warning(
                    f"{pos['Symbol']} has used its {int(time_stop)}-session time stop. "
                    "The plan is to exit on this close."
                )

        st.markdown("##### Close a position")
        with st.form("close_trade"):
            cc1, cc2, cc3, cc4 = st.columns([2, 1, 2, 1])
            labels = {
                int(trade["id"]): (
                    f"#{int(trade['id'])} {trade['symbol']} "
                    f"({int(trade['qty'])} @ ₹{trade['entry_price']:,.2f})"
                )
                for _, trade in open_book.iterrows()
            }
            selected = cc1.selectbox("Trade", list(labels), format_func=labels.get)
            symbol = open_book.set_index("id").loc[selected, "symbol"]
            default_px = float(prices.get(symbol, open_book.set_index("id").loc[selected, "entry_price"]))
            exit_px = cc2.number_input("Exit price ₹", min_value=0.01, value=round(default_px, 2), format="%.2f")
            close_notes = cc3.text_input("Notes (optional)", key="close_notes")
            if cc4.form_submit_button("Close trade", type="primary", use_container_width=True):
                if not db.password_ok(journal_password):
                    st.error("Enter the journal password in the sidebar before closing a trade.")
                elif db.close_trade(int(selected), float(exit_px), close_notes):
                    st.success("Trade closed and P&L recorded.")
                    st.rerun()
                else:
                    st.error("That trade is already closed.")

    st.divider()
    st.subheader("Trade history")
    closed = db.get_trades("closed")
    if closed.empty:
        st.caption("No closed trades yet.")
    else:
        view = closed[["id", "date_taken", "exit_date", "symbol", "mode", "qty",
                       "entry_price", "exit_price", "pnl", "pnl_pct", "notes"]].copy()
        view["date_taken"] = view["date_taken"].map(_day)
        view["exit_date"] = view["exit_date"].map(_day)
        view = view.rename(columns={
            "id": "#", "date_taken": "Taken", "exit_date": "Exited", "symbol": "Symbol",
            "mode": "Setup", "qty": "Qty", "entry_price": "Entry ₹", "exit_price": "Exit ₹",
            "pnl": "P&L ₹", "pnl_pct": "P&L %", "notes": "Notes",
        })
        st.dataframe(view, width="stretch", hide_index=True)
        with st.expander("Delete a trade record"):
            delete_id = st.number_input("Trade #", min_value=1, step=1)
            if st.button("Delete"):
                if not db.password_ok(journal_password):
                    st.error("Enter the journal password in the sidebar before deleting a trade.")
                else:
                    db.delete_trade(int(delete_id))
                    st.rerun()


with tab_dash:
    st.subheader("Performance")
    all_trades = db.get_trades()
    closed = all_trades[all_trades["status"] == "closed"] if not all_trades.empty else all_trades
    if all_trades.empty:
        st.info("The dashboard fills in after the first logged trade.")
    else:
        wins = closed[closed["pnl"] > 0] if not closed.empty else closed
        total_pnl = float(closed["pnl"].sum()) if not closed.empty else 0.0
        win_rate = (len(wins) / len(closed) * 100) if len(closed) else 0.0
        avg_ret = float(closed["pnl_pct"].mean()) if not closed.empty else 0.0
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Realized P&L", f"₹{total_pnl:,.0f}")
        m2.metric("Win rate", f"{win_rate:.0f}%")
        m3.metric("Closed trades", f"{len(closed)}")
        m4.metric("Open positions", f"{len(open_book)}")
        m5.metric("Avg return / trade", f"{avg_ret:+.2f}%")
        if not closed.empty:
            g1, g2 = st.columns(2)
            with g1:
                closed = closed.copy()
                closed["exit_day"] = closed["exit_date"].map(_day)
                equity = closed.sort_values("exit_day").groupby("exit_day")["pnl"].sum().cumsum()
                fig = go.Figure(go.Scatter(
                    x=equity.index, y=equity.values, mode="lines+markers",
                    name="Cumulative P&L", line=dict(color=C_SERIES, width=2),
                ))
                fig.add_hline(y=0, line_color=C_MUTED, line_width=1)
                fig.update_layout(title="Cumulative realized P&L (₹)")
                fig.update_xaxes(title_text="Exit date")
                fig.update_yaxes(title_text="P&L (₹)")
                st.plotly_chart(_style_fig(fig, 320), width="stretch")
            with g2:
                per = closed.sort_values("id")
                labels = [f"#{i} {s}" for i, s in zip(per["id"], per["symbol"])]
                colors = [C_GOOD if pnl > 0 else C_CRIT for pnl in per["pnl"]]
                fig = go.Figure(go.Bar(
                    x=labels, y=per["pnl"], marker_color=colors,
                    text=[f"₹{pnl:,.0f}" for pnl in per["pnl"]], textposition="outside",
                ))
                fig.add_hline(y=0, line_color=C_MUTED, line_width=1)
                fig.update_layout(title="P&L per closed trade")
                fig.update_xaxes(title_text="Trade")
                fig.update_yaxes(title_text="P&L (₹)")
                st.plotly_chart(_style_fig(fig, 320), width="stretch")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("##### By setup")
                by_mode = (closed.groupby("mode")
                           .agg(trades=("id", "count"), pnl=("pnl", "sum"),
                                win_rate=("pnl", lambda s: (s > 0).mean() * 100),
                                avg_pct=("pnl_pct", "mean"))
                           .round(2).reset_index()
                           .rename(columns={"mode": "Setup", "trades": "Trades",
                                            "pnl": "P&L ₹", "win_rate": "Win %",
                                            "avg_pct": "Avg %"}))
                st.dataframe(by_mode, hide_index=True, width="stretch")
            with b2:
                st.markdown("##### By sector")
                by_sec = (closed.groupby("sector")
                          .agg(trades=("id", "count"), pnl=("pnl", "sum"))
                          .round(2).sort_values("pnl", ascending=False)
                          .reset_index()
                          .rename(columns={"sector": "Sector", "trades": "Trades", "pnl": "P&L ₹"}))
                st.dataframe(by_sec, hide_index=True, width="stretch")


with tab_backtest:
    st.subheader("Walk-forward results")
    st.caption(
        "Signal on the completed close, enter the next open, exit on stop, target, "
        "+1R breakeven (armed for the next session), or the time stop. "
        "Same daily cache as the ideas above. Stand-aside Nifty sessions are not traded. "
        f"Costs: {active_cost:.0f} bps per side."
    )
    for setup, stats in (("Breakout", backtest.breakout), ("Pullback", backtest.pullback)):
        st.markdown(f"##### {setup}")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Expectancy", f"{stats.expectancy_r:+.2f}R")
        s2.metric("Win rate", f"{stats.win_rate * 100:.0f}%")
        pf = "∞" if stats.profit_factor == float("inf") else f"{stats.profit_factor:.2f}"
        s3.metric("Profit factor", pf)
        s4.metric("Max drawdown", f"{stats.max_drawdown_r:.2f}R")
        s5.metric("Trades", f"{stats.trades}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("By year")
            if stats.by_year:
                year_frame = pd.DataFrame(stats.by_year).rename(columns={
                    "name": "Year", "trades": "Trades",
                    "expectancy_r": "Expectancy (R)", "win_rate": "Win rate",
                })
                year_frame["Win rate"] = (year_frame["Win rate"] * 100).round(1)
                year_frame["Expectancy (R)"] = year_frame["Expectancy (R)"].round(2)
                st.dataframe(year_frame, hide_index=True, width="stretch")
            else:
                st.caption("No trades.")
        with c2:
            st.markdown("By Nifty regime at the signal")
            if stats.by_regime:
                regime_frame = pd.DataFrame(stats.by_regime).rename(columns={
                    "name": "Regime", "trades": "Trades",
                    "expectancy_r": "Expectancy (R)", "win_rate": "Win rate",
                })
                regime_frame["Win rate"] = (regime_frame["Win rate"] * 100).round(1)
                regime_frame["Expectancy (R)"] = regime_frame["Expectancy (R)"].round(2)
                st.dataframe(regime_frame, hide_index=True, width="stretch")
            else:
                st.caption("No trades.")
    if backtest.trades:
        curve = pd.DataFrame({
            "exit": [trade.exit_date for trade in backtest.trades],
            "setup": [trade.setup for trade in backtest.trades],
            "r": [trade.r_multiple for trade in backtest.trades],
        }).sort_values("exit")
        fig = go.Figure()
        for setup, color in (("breakout", C_SERIES), ("pullback", C_SERIES2)):
            subset = curve[curve["setup"] == setup]
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset["exit"], y=subset["r"].cumsum(), mode="lines",
                name=setup.title(), line=dict(color=color, width=2),
            ))
        fig.add_hline(y=0, line_color=C_MUTED, line_width=1)
        fig.update_layout(title="Cumulative R by setup")
        fig.update_xaxes(title_text="Exit date")
        fig.update_yaxes(title_text="Cumulative R")
        st.plotly_chart(_style_fig(fig, 320), width="stretch")
