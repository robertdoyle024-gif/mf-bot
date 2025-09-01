# streamlit_app.py
from __future__ import annotations

import io
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import warnings

# Quiet the "Could not infer format..." user warnings from pandas when reading CSV date columns
warnings.filterwarnings('ignore', message='Could not infer format.*', category=UserWarning)

# Optional: load .env if present (does nothing if absent)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional but useful for grouped charts in WF tab
try:
    import altair as alt
    _HAVE_ALT = True
except Exception:
    _HAVE_ALT = False

# -------------------------
# Page setup (DIM THEME via CSS, not Streamlit theme toggle)
# -------------------------
st.set_page_config(
    page_title="MF Bot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dim, accessible palette (keeps contrast high; clarifies tabs & cards)
st.markdown(
    """
<style>
/* Base background & text */
:root {
  --bg: #0f172a;           /* slate-900 */
  --panel: #111827;        /* gray-900 */
  --panel-2: #0b1220;      /* deep slate */
  --text: #e5e7eb;         /* gray-200 */
  --muted: #94a3b8;        /* slate-400 */
  --accent: #7c3aed;       /* violet-600 */
  --accent-2: #a78bfa;     /* violet-300 */
  --pos: #22c55e;          /* green-500 */
  --neg: #ef4444;          /* red-500 */
  --border: rgba(148,163,184,.24);
}

/* Streamlit containers */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
}
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* Headings & text */
h1, h2, h3, h4, h5, h6, p, span, label, div, code, pre, small {
  color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--panel-2) !important;
  border-right: 1px solid var(--border) !important;
}

/* Inputs */
.stTextInput, .stDateInput, .stMultiSelect, .stSelectbox {
  color: var(--text) !important;
}
.stTextInput>div>div>input,
.stDateInput input,
.stMultiSelect input,
.stSelectbox div[role="combobox"] {
  background: #0b1220 !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}

/* Buttons */
.stButton>button {
  background: var(--accent) !important;
  color: white !important;
  border: 1px solid transparent !important;
  border-radius: 8px !important;
}
.stDownloadButton>button {
  background: #1f2937 !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* Tabs: clear highlight + readable labels */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.25rem !important;
  border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: #0b1220 !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-bottom: none !important;
  border-top-left-radius: 8px !important;
  border-top-right-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
  background: #1f2937 !important;
  color: var(--accent-2) !important;
  font-weight: 600 !important;
  border-bottom: 2px solid transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }

/* KPI card */
.kpi-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 12px;
  background: #0b1220;
}
.small { color: var(--muted); font-size: 0.85rem; }
.caption-note { color: var(--muted); font-size:0.9rem; margin-top:.25rem; }

/* DataFrames */
[data-testid="stDataFrame"] {
  background: #0b1220 !important;
}
[data-testid="stDataFrame"] div,
[data-testid="stTable"] div {
  color: var(--text) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Config
# -------------------------
@dataclass
class AppConfig:
    db_path: str = os.getenv("DB_PATH", "simple_bot.db")

CFG = AppConfig()

# -------------------------
# SQLite helpers
# -------------------------
def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    try:
        row = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None
    except Exception:
        return False

def _load_table(con: sqlite3.Connection, name: str) -> pd.DataFrame:
    if not _table_exists(con, name):
        return pd.DataFrame()
    try:
        return pd.read_sql(f"SELECT * FROM {name}", con)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_db(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (equity, trades, positions). Empty frames if tables missing."""
    if not os.path.exists(db_path):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    con = sqlite3.connect(db_path)
    try:
        equity = _load_table(con, "equity")
        trades = _load_table(con, "trades")
        positions = _load_table(con, "positions")
    finally:
        con.close()

    # Normalize columns and dtypes
    if not equity.empty:
        equity = equity.copy()
        equity["ts"] = pd.to_datetime(equity["ts"], errors="coerce", utc=True).dt.tz_convert(None)
        equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
        equity = equity.dropna(subset=["ts", "equity"]).sort_values("ts")

    if not trades.empty:
        trades = trades.copy()
        trades["ts"] = pd.to_datetime(trades["ts"], errors="coerce", utc=True).dt.tz_convert(None)
        if "side" not in trades.columns and "action" in trades.columns:
            trades["side"] = trades["action"]
        for col in ("price", "qty", "stop", "take_profit"):
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors="coerce")
        if "symbol" in trades.columns:
            trades["symbol"] = trades["symbol"].astype(str).str.upper()
        trades = trades.dropna(subset=["ts", "symbol", "side"]).sort_values("ts")

    if not positions.empty:
        positions = positions.copy()
        if "symbol" in positions.columns:
            positions["symbol"] = positions["symbol"].astype(str).str.upper()
        for col in ("entry", "qty", "init_stop", "trail_stop", "highest_close"):
            if col in positions.columns:
                positions[col] = pd.to_numeric(positions[col], errors="coerce")

    return equity, trades, positions

# -------------------------
# Metrics & transforms
# -------------------------
def compute_drawdown(equity_series: pd.Series) -> Tuple[pd.Series, float]:
    """Return (drawdown_series %, max_drawdown %)."""
    if equity_series.empty:
        return pd.Series(dtype=float), np.nan
    roll_max = equity_series.cummax()
    dd = equity_series / roll_max - 1.0
    return dd, float(dd.min())

def annualize_from_dates(first: pd.Timestamp, last: pd.Timestamp) -> float:
    days = max((last - first).days, 1)
    return days / 365.25

def equity_metrics(equity_df: pd.DataFrame) -> Dict[str, float]:
    """Compute Total Return, CAGR, Sharpe, MaxDD, Vol."""
    if equity_df.empty or len(equity_df) < 2:
        return dict(total_return=np.nan, cagr=np.nan, sharpe=np.nan, max_dd=np.nan, vol=np.nan)

    eq = equity_df.set_index("ts")["equity"].astype(float)
    ret = eq.pct_change().dropna()
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    years = annualize_from_dates(eq.index[0], eq.index[-1])
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan

    vol = float(ret.std() * np.sqrt(252)) if ret.std() > 0 else np.nan
    sharpe = float(ret.mean() * np.sqrt(252) / ret.std()) if ret.std() > 0 else np.nan

    _, max_dd = compute_drawdown(eq)
    return dict(total_return=total_return, cagr=cagr, sharpe=sharpe, max_dd=max_dd, vol=vol)

def pair_roundtrips(trades: pd.DataFrame) -> pd.DataFrame:
    """
    FIFO pair BUY->SELL per symbol/qty. Returns a DataFrame of closed trades with:
    symbol, entry_ts, exit_ts, qty, entry_price, exit_price, pnl, ret_pct, hold_days
    """
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "symbol","entry_ts","exit_ts","qty","entry_price","exit_price","pnl","ret_pct","hold_days"
            ]
        )

    t = trades.sort_values("ts").copy()
    t["side"] = t["side"].str.upper()
    out_rows: List[Dict] = []
    fifo: Dict[str, List[Tuple[pd.Timestamp, float, float]]] = {}

    for _, row in t.iterrows():
        sym = str(row["symbol"]).upper()
        side = row["side"]
        ts = row["ts"]
        price = float(row.get("price", np.nan))
        qty = float(row.get("qty", 0))
        if not np.isfinite(price) or qty <= 0:
            continue

        fifo.setdefault(sym, [])
        if side == "BUY":
            fifo[sym].append((ts, price, qty))
        elif side == "SELL":
            remaining = qty
            while remaining > 0 and fifo[sym]:
                ent_ts, ent_px, ent_qty = fifo[sym][0]
                match_qty = min(remaining, ent_qty)
                pnl = (price - ent_px) * match_qty
                ret_pct = (price / ent_px - 1.0) if ent_px > 0 else np.nan
                hold_days = ((ts - ent_ts).days
                             if isinstance(ts, pd.Timestamp) and isinstance(ent_ts, pd.Timestamp)
                             else np.nan)
                out_rows.append(
                    dict(symbol=sym, entry_ts=ent_ts, exit_ts=ts, qty=match_qty,
                         entry_price=ent_px, exit_price=price, pnl=pnl, ret_pct=ret_pct, hold_days=hold_days)
                )
                ent_qty -= match_qty
                remaining -= match_qty
                if ent_qty <= 0:
                    fifo[sym].pop(0)
                else:
                    fifo[sym][0] = (ent_ts, ent_px, ent_qty)
            # ignore excess sells (no shorting)

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "symbol","entry_ts","exit_ts","qty","entry_price","exit_price","pnl","ret_pct","hold_days"
            ]
        )
    return pd.DataFrame(out_rows).sort_values("exit_ts")

def symbol_stats(roundtrips: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by symbol."""
    if roundtrips.empty:
        return pd.DataFrame(
            columns=["Symbol","Trades","Win Rate","Avg Return","Total PnL","Avg Hold (d)"]
        )

    g = roundtrips.copy()
    g["win"] = g["pnl"] > 0
    stats = (
        g.groupby("symbol")
         .agg(Trades=("pnl","count"),
              WinRate=("win","mean"),
              AvgReturn=("ret_pct","mean"),
              TotalPnL=("pnl","sum"),
              AvgHold=("hold_days","mean"))
         .reset_index()
    )
    for c in ["WinRate","AvgReturn","TotalPnL","AvgHold"]:
        if c in stats.columns:
            stats[c] = pd.to_numeric(stats[c], errors="coerce")

    stats = stats.rename(columns={
        "symbol":"Symbol",
        "WinRate":"Win Rate",
        "AvgReturn":"Avg Return",
        "TotalPnL":"Total PnL",
        "AvgHold":"Avg Hold (d)",
    })
    return stats.sort_values("Total PnL", ascending=False)

def kpi_block(label: str, value: Optional[float], fmt: str, help_text: str = "") -> None:
    with st.container():
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            st.metric(label, value="—", help=help_text or None)
        elif fmt == "pct":
            st.metric(label, f"{value*100:,.2f}%", help=help_text or None)
        elif fmt == "int":
            st.metric(label, f"{int(value):,}", help=help_text or None)
        elif fmt == "float":
            st.metric(label, f"{value:,.2f}", help=help_text or None)
        else:
            st.metric(label, str(value), help=help_text or None)
        st.markdown("</div>", unsafe_allow_html=True)

def style_sign(obj, cols: List[str], fmt: Optional[Dict[str, str]] = None):
    """
    Return a Styler with +/- coloring. Works whether obj is DataFrame or Styler.
    Safe across pandas versions (uses map or applymap as available).
    """
    is_df = isinstance(obj, pd.DataFrame)
    sty = obj.style if is_df else obj
    if fmt:
        sty = sty.format(fmt)

    # determine subset cols that exist
    try:
        existing_cols = list(sty.data.columns)
    except Exception:
        # older pandas Styler might not expose .data
        try:
            existing_cols = list(obj.columns) if is_df else []
        except Exception:
            existing_cols = []
    subset = [c for c in cols if c in existing_cols]
    if not subset:
        return sty

    def _color(v):
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        return "color: #22c55e;" if v > 0 else ("color: #ef4444;" if v < 0 else "")

    try:
        return sty.map(_color, subset=subset)  # pandas >=2.2
    except Exception:
        return sty.applymap(_color, subset=subset)  # pandas <2.2

def make_excel_bytes(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    roundtrips: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: Dict[str, float],
) -> Optional[bytes]:
    """Create a multi-sheet Excel workbook as bytes. Returns None if no engine is available."""
    buf = io.BytesIO()
    engine_used = None
    for engine in ("xlsxwriter", "openpyxl", None):  # None lets pandas guess
        try:
            with pd.ExcelWriter(buf, engine=engine) as writer:
                if not equity.empty:
                    eout = equity.copy()
                    eout["ts"] = pd.to_datetime(eout["ts"])
                    eout.to_excel(writer, index=False, sheet_name="Equity")
                if not trades.empty:
                    trades.to_excel(writer, index=False, sheet_name="Trades")
                if not roundtrips.empty:
                    roundtrips.to_excel(writer, index=False, sheet_name="Round-trips")
                if not positions.empty:
                    positions.to_excel(writer, index=False, sheet_name="Positions")
                if metrics:
                    met_df = (
                        pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
                        .rename_axis("Metric")
                        .reset_index()
                    )
                    met_df.to_excel(writer, index=False, sheet_name="Metrics")
            engine_used = engine
            break
        except Exception:
            buf.seek(0)
            buf.truncate(0)
            continue

    if engine_used is None:
        return None
    return buf.getvalue()

# ---------- WF loaders ----------
@st.cache_data(ttl=30)
def load_wf_equity(path: str = "wf_equity.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)  # don't parse dates here; normalize below
        ts_col = df.columns[0]  # first column is time index written by pandas
        out = df.rename(columns={ts_col: "ts"})
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
        out["equity"] = pd.to_numeric(out.get("equity", np.nan), errors="coerce")
        out = out.dropna(subset=["ts", "equity"]).sort_values("ts")
        return out
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_wf_windows(path: str = "wf_windows.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # ensure expected columns exist even if missing
        for col in ["train_MAR", "train_Sharpe", "test_MAR", "test_Sharpe"]:
            if col not in df.columns:
                df[col] = np.nan
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------
# Sidebar (manual refresh only)
# -------------------------
st.sidebar.header("Settings")
db_path = st.sidebar.text_input("SQLite DB path", value=CFG.db_path)
refresh = st.sidebar.button("Refresh data")

# Manual cache bust (no auto-refresh loops)
if refresh:
    st.cache_data.clear()
    st.rerun()

equity, trades, positions = load_db(db_path)

# Symbol filter from trades/positions
all_symbols = sorted(
    list(
        set(trades["symbol"].unique().tolist() if "symbol" in trades else [])
        | set(positions["symbol"].unique().tolist() if "symbol" in positions else [])
    )
)
sel_symbols = st.sidebar.multiselect("Symbols", options=all_symbols, default=all_symbols)

# Date filter from equity span
if not equity.empty:
    dmin = equity["ts"].min().date()
    dmax = equity["ts"].max().date()
else:
    today = pd.Timestamp.today().date()
    dmin = dmax = today
date_range = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)

# Apply filters
if sel_symbols and not trades.empty:
    trades = trades[trades["symbol"].isin(sel_symbols)].copy()
if not equity.empty and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    a = pd.to_datetime(str(date_range[0]))
    b = pd.to_datetime(str(date_range[1])) + pd.Timedelta(days=1)
    equity = equity[(equity["ts"] >= a) & (equity["ts"] < b)].copy()

# -------------------------
# Header & status
# -------------------------
st.title("📈 MF Bot Dashboard")

top = st.columns([4, 2, 2, 2, 2])
with top[0]:
    st.write(f"**DB:** `{os.path.abspath(db_path)}`")
    if not equity.empty:
        st.caption(f"Last update: {equity['ts'].max().strftime('%Y-%m-%d %H:%M:%S')}")
with top[1]:
    st.write(f"**Equity points:** {len(equity):,}" if not equity.empty else "**Equity points:** —")
with top[2]:
    st.write(f"**Trades:** {len(trades):,}" if not trades.empty else "**Trades:** —")
with top[3]:
    st.write(f"**Open positions:** {len(positions):,}" if not positions.empty else "**Open positions:** —")
with top[4]:
    st.caption("Use the sidebar to filter. Click **Refresh data** to reload.")

st.divider()

# -------------------------
# Tabs
# -------------------------
tab_overview, tab_positions, tab_trades, tab_roundtrips, tab_wf, tab_export = st.tabs(
    ["Overview", "Positions", "Trades", "Round-trips", "WF Results", "Export"]
)

# =========================
# Overview
# =========================
with tab_overview:
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

    eq_kpis = (
        equity_metrics(equity)
        if not equity.empty
        else dict(total_return=np.nan, cagr=np.nan, sharpe=np.nan, max_dd=np.nan, vol=np.nan)
    )
    with k1:
        kpi_block("Total Return", eq_kpis["total_return"], "pct")
    with k2:
        kpi_block("CAGR", eq_kpis["cagr"], "pct")
    with k3:
        kpi_block("Max Drawdown", eq_kpis["max_dd"], "pct")
    with k4:
        kpi_block("Sharpe (r=0)", eq_kpis["sharpe"], "float")
    with k5:
        kpi_block("Volatility (ann.)", eq_kpis["vol"], "pct")

    # Round-trip KPIs
    roundtrips = pair_roundtrips(trades) if not trades.empty else pd.DataFrame(
        columns=["pnl", "ret_pct", "hold_days"]
    )
    if not roundtrips.empty:
        wins = roundtrips["pnl"] > 0
        win_rate = float(wins.mean()) if len(roundtrips) > 0 else np.nan
        avg_win = float(roundtrips.loc[wins, "pnl"].mean()) if wins.any() else np.nan
        avg_loss = float(roundtrips.loc[~wins, "pnl"].mean()) if (~wins).any() else np.nan
        loss_sum = float(abs(roundtrips.loc[~wins, "pnl"].sum())) if (~wins).any() else 0.0
        profit_factor = float(roundtrips.loc[wins, "pnl"].sum() / loss_sum) if loss_sum > 0 else np.nan
    else:
        win_rate = avg_win = avg_loss = profit_factor = np.nan

    with k6:
        kpi_block("Win Rate", win_rate, "pct")
    with k7:
        kpi_block("Profit Factor", profit_factor, "float")

    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Equity Curve")
        if equity.empty:
            st.info("No equity data yet. Run a backtest or paper session to populate the `equity` table.")
        else:
            show = equity[["ts", "equity"]].set_index("ts")
            st.line_chart(show)
    with c2:
        st.subheader("Drawdown")
        if equity.empty:
            st.info("No drawdown available.")
        else:
            eq = equity.set_index("ts")["equity"].astype(float)
            dd, _ = compute_drawdown(eq)
            st.area_chart(dd.rename("Drawdown"))

    st.divider()

    # Per-symbol summary (from round-trips)
    st.subheader("Per-Symbol Summary")
    symtbl = symbol_stats(roundtrips) if not roundtrips.empty else pd.DataFrame()
    if symtbl.empty:
        st.info("No closed trades yet to summarize by symbol.")
    else:
        fmt = {"Win Rate":"{:.2%}","Avg Return":"{:.2%}","Total PnL":"{:,.2f}","Avg Hold (d)":"{:.1f}"}
        styled = style_sign(symtbl.style.format(fmt), ["Win Rate","Avg Return","Total PnL"])
        st.dataframe(styled, use_container_width=True, height=320)

# =========================
# Positions
# =========================
with tab_positions:
    st.subheader("Open Positions")
    if positions.empty:
        st.info("No open positions.")
    else:
        pos = positions.copy()
        pos = pos[
            ["symbol", "qty", "entry", "init_stop", "trail_stop", "highest_close"]
        ].rename(
            columns={
                "symbol":"Symbol","qty":"Qty","entry":"Entry","init_stop":"Init SL",
                "trail_stop":"Trail SL","highest_close":"Highest Close",
            }
        )
        st.dataframe(
            pos.style.format({"Qty":"{:,.0f}","Entry":"{:,.2f}","Init SL":"{:,.2f}",
                              "Trail SL":"{:,.2f}","Highest Close":"{:,.2f}"}),
            use_container_width=True,
            height=300,
        )
        st.download_button(
            "Download Positions (CSV)",
            data=pos.to_csv(index=False),
            file_name="positions.csv",
            mime="text/csv",
        )

# =========================
# Trades
# =========================
with tab_trades:
    st.subheader("Recent Trades")
    if trades.empty:
        st.info("No trades yet.")
    else:
        t_show = trades.copy()
        cols = ["ts","side","symbol","qty","price","stop","take_profit","note"]
        have = [c for c in cols if c in t_show.columns]
        t_show = t_show[have].tail(500).rename(columns={
            "ts":"Timestamp","side":"Side","symbol":"Symbol","qty":"Qty",
            "price":"Price","stop":"Stop","take_profit":"Take Profit","note":"Note",
        })
        st.dataframe(
            t_show.style.format({"Qty":"{:,.0f}","Price":"{:,.2f}","Stop":"{:,.2f}","Take Profit":"{:,.2f}"}),
            use_container_width=True,
            height=360,
        )
        st.download_button(
            "Download Trades (CSV)",
            data=t_show.to_csv(index=False),
            file_name="trades.csv",
            mime="text/csv",
        )

# =========================
# Round-trips
# =========================
with tab_roundtrips:
    st.subheader("Closed Round-trips")
    rt = pair_roundtrips(trades) if not trades.empty else pd.DataFrame()
    if rt.empty:
        st.info("No closed trades to summarize yet.")
    else:
        rt = rt.rename(columns={
            "entry_ts":"Entry","exit_ts":"Exit","symbol":"Symbol","qty":"Qty",
            "entry_price":"Entry Px","exit_price":"Exit Px","pnl":"PnL","ret_pct":"Return","hold_days":"Hold (d)",
        })
        rt = rt[["Entry","Exit","Symbol","Qty","Entry Px","Exit Px","PnL","Return","Hold (d)"]]
        styled = style_sign(
            rt.style.format({
                "Qty":"{:,.0f}","Entry Px":"{:,.2f}","Exit Px":"{:,.2f}",
                "PnL":"{:,.2f}","Return":"{:.2%}","Hold (d)":"{:,.0f}",
            }),
            ["PnL","Return"]
        )
        st.dataframe(styled, use_container_width=True, height=360)
        st.download_button(
            "Download Round-trips (CSV)",
            data=rt.to_csv(index=False),
            file_name="roundtrips.csv",
            mime="text/csv",
        )

# =========================
# WF Results
# =========================
with tab_wf:
    st.subheader("Walk-Forward Results")

    colA, colB = st.columns([2, 1])
    wf_eq = load_wf_equity("wf_equity.csv")
    wf_win = load_wf_windows("wf_windows.csv")

    with colA:
        st.markdown("**Stitched OOS Equity**")
        if wf_eq.empty:
            st.info("Run `backtest-wf` to generate `wf_equity.csv`.")
        else:
            st.line_chart(wf_eq.set_index("ts")[["equity"]])

    with colB:
        st.markdown("**OOS Drawdown**")
        if wf_eq.empty:
            st.info("—")
        else:
            dd, _ = compute_drawdown(wf_eq.set_index("ts")["equity"].astype(float))
            st.area_chart(dd.rename("Drawdown"))

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    oos_kpis = (
        equity_metrics(wf_eq.rename(columns={"ts":"ts", "equity":"equity"}))
        if not wf_eq.empty else dict(total_return=np.nan, cagr=np.nan, sharpe=np.nan, max_dd=np.nan, vol=np.nan)
    )
    with col1: kpi_block("OOS Total Return", oos_kpis["total_return"], "pct")
    with col2: kpi_block("OOS CAGR", oos_kpis["cagr"], "pct")
    with col3: kpi_block("OOS Max DD", oos_kpis["max_dd"], "pct")
    with col4: kpi_block("OOS Sharpe", oos_kpis["sharpe"], "float")

    st.divider()
    st.markdown("### Per-Window KPIs & Picks")
    if wf_win.empty:
        st.info("Run `backtest-wf` to generate `wf_windows.csv` (per-window train/test KPIs).")
    else:
        pretty = wf_win.copy()
        fmt_map = {
            "train_CAGR":"{:.2%}", "test_CAGR":"{:.2%}",
            "train_Sharpe":"{:.2f}", "test_Sharpe":"{:.2f}",
            "train_MAR":"{:.2f}", "test_MAR":"{:.2f}",
            "train_MaxDD":"{:.2%}", "test_MaxDD":"{:.2%}",
        }
        styled = style_sign(
            pretty.style.format(fmt_map),
            ["train_CAGR","test_CAGR","train_Sharpe","test_Sharpe","train_MAR","test_MAR"]
        )
        st.dataframe(styled, use_container_width=True, height=360)
        st.download_button(
            "Download WF Windows (CSV)",
            data=wf_win.to_csv(index=False),
            file_name="wf_windows.csv",
            mime="text/csv",
        )

        if _HAVE_ALT:
            # Build a readable "Window" label
            df_plot = wf_win.copy()
            df_plot["Window"] = df_plot.get("test_start", "").astype(str).fillna("") + " → " + df_plot.get("test_end", "").astype(str).fillna("")
            # MAR bars
            mar_melt = df_plot.melt(
                id_vars=["Window"], value_vars=["train_MAR","test_MAR"],
                var_name="Phase", value_name="MAR"
            )
            mar_melt["Phase"] = mar_melt["Phase"].map({"train_MAR":"Train","test_MAR":"Test"})
            # Sharpe bars
            shr_melt = df_plot.melt(
                id_vars=["Window"], value_vars=["train_Sharpe","test_Sharpe"],
                var_name="Phase", value_name="Sharpe"
            )
            shr_melt["Phase"] = shr_melt["Phase"].map({"train_Sharpe":"Train","test_Sharpe":"Test"})

            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Train vs Test MAR by Window**")
                chart_mar = (
                    alt.Chart(mar_melt)
                       .mark_bar()
                       .encode(x=alt.X("Window:N", sort=None),
                               y=alt.Y("MAR:Q"),
                               color="Phase:N",
                               tooltip=["Window","Phase", alt.Tooltip("MAR:Q", format=".2f")])
                       .properties(height=280)
                )
                st.altair_chart(chart_mar, use_container_width=True)
            with cB:
                st.markdown("**Train vs Test Sharpe by Window**")
                chart_sh = (
                    alt.Chart(shr_melt)
                       .mark_bar()
                       .encode(x=alt.X("Window:N", sort=None),
                               y=alt.Y("Sharpe:Q"),
                               color="Phase:N",
                               tooltip=["Window","Phase", alt.Tooltip("Sharpe:Q", format=".2f")])
                       .properties(height=280)
                )
                st.altair_chart(chart_sh, use_container_width=True)
        else:
            st.info("Install `altair` for grouped bar charts, or use the table above.")

# =========================
# Export
# =========================
with tab_export:
    st.subheader("Export data")
    eq_kpis = (
        equity_metrics(equity)
        if not equity.empty
        else dict(total_return=np.nan, cagr=np.nan, sharpe=np.nan, max_dd=np.nan, vol=np.nan)
    )

    excel_bytes = make_excel_bytes(
        equity=equity.copy(),
        trades=trades.copy(),
        roundtrips=(pair_roundtrips(trades) if not trades.empty else pd.DataFrame()),
        positions=positions.copy(),
        metrics={
            "Total Return": eq_kpis.get("total_return", np.nan),
            "CAGR": eq_kpis.get("cagr", np.nan),
            "Max Drawdown": eq_kpis.get("max_dd", np.nan),
            "Sharpe": eq_kpis.get("sharpe", np.nan),
            "Vol (ann.)": eq_kpis.get("vol", np.nan),
        },
    )

    if excel_bytes is None:
        st.warning(
            "Excel export needs an engine. Install one of: `openpyxl` or `xlsxwriter`:\n\n"
            "`python -m pip install openpyxl xlsxwriter`"
        )
    else:
        st.download_button(
            "📥 Download Excel (all tabs)",
            data=excel_bytes,
            file_name="mf_bot_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

st.caption(
    "Tip: If data looks stale, click **Refresh data** in the sidebar. "
    "Equity/Trades populate after running a backtest or a paper trading session. "
    "WF files (`wf_equity.csv`, `wf_windows.csv`) are produced by the `backtest-wf` command."
)
