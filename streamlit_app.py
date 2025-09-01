
import os, sqlite3, pandas as pd, numpy as np, matplotlib.pyplot as plt, streamlit as st

DB_PATH = os.getenv("DB_PATH", "simple_bot.db")

st.title("Multi-Factor Bot — Dashboard")

if not os.path.exists(DB_PATH):
    st.warning("No DB found yet. Run the bot once to create it.")
    st.stop()

conn = sqlite3.connect(DB_PATH)
trades = pd.read_sql_query("SELECT * FROM trades ORDER BY ts ASC", conn)
positions = pd.read_sql_query("SELECT * FROM positions", conn)

st.subheader("Open Positions")
if positions.empty:
    st.write("None")
else:
    st.dataframe(positions)

st.subheader("Trades")
st.dataframe(trades)

def pairs(trades_df):
    out = []
    buy = None
    for _, r in trades_df.iterrows():
        if r['action'] == 'BUY' and buy is None:
            buy = r
        elif r['action'] == 'SELL' and buy is not None:
            out.append((buy, r))
            buy = None
    return out

ps = pairs(trades)

if ps:
    pnls, Rs = [], []
    for b,s in ps:
        qty = min(b['qty'], s['qty'])
        pnl = (s['price'] - b['price']) * qty
        risk = (b['price'] - (b['stop'] if not pd.isna(b['stop']) else b['price'])) * qty
        R = pnl / risk if risk>0 else np.nan
        pnls.append(pnl)
        Rs.append(R)

    st.subheader("Summary")
    st.write(f"Closed P&L: {np.nansum(pnls):.2f}")
    st.write(f"Win rate: {np.mean([1 if p>0 else 0 for p in pnls]):.2%}")
    st.write(f"Avg R: {np.nanmean(Rs):.2f}")

    eq = np.nancumsum(pnls)
    dd = eq - np.maximum.accumulate(eq)

    fig1, ax1 = plt.subplots()
    ax1.plot(eq)
    ax1.set_title("Equity (Closed P&L)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(dd)
    ax2.set_title("Drawdown")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    clean_R = [r for r in Rs if np.isfinite(r)]
    if clean_R:
        ax3.hist(clean_R, bins=20)
    ax3.set_title("R-multiple Distribution")
    st.pyplot(fig3)
else:
    st.info("No closed trades yet.")
