# bot/strategy_legacy.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List

# --- Helpers ---
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Wilder's RSI
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # Requires High, Low, Close (case-insensitive); falls back to 'Close' if needed
    cols = {c.lower(): c for c in df.columns}
    high = df[cols.get("high", "Close")]
    low  = df[cols.get("low",  "Close")]
    close= df[cols.get("close","Close")]
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low  - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Simple rolling mean ATR (fine for backtests)
    return tr.rolling(period, min_periods=period).mean().bfill()

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

# -----------------------------------------------------------------------------
# PUBLIC API expected by your runner/backtest/trade
# -----------------------------------------------------------------------------
def compute(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    rsi_period: int,
    roc_period: int,
    trend_sma: int,
    atr_period: int,
) -> pd.DataFrame:
    """
    Returns a DataFrame with at least these columns:
      - 'rsi'       : float
      - 'roc'       : float (rate of change)
      - 'trend_up'  : bool  (trend filter)
      - 'mom_ok'    : bool  (momentum condition used by K/M confirmation)
      - 'atr'       : float (average true range)
    Input df must have 'Close' and ideally 'High','Low'. Index must be datetime.
    """
    # Normalize column names (handle both Close and close)
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close", "Close")
    close = df[close_col].astype(float)

    # Indicators
    rsi = _rsi(close, rsi_period)
    roc = close.pct_change(roc_period)  # simple ROC as % change
    atr = _atr(df, atr_period)

    # Trend: price above SMA(trend_sma)
    trend_ma = _sma(close, trend_sma)
    trend_up = (close > trend_ma).fillna(False)

    # Momentum condition used for confirm K-in-M:
    # Keep it simple & robust: positive ROC **and** short SMA > long SMA (if windows valid)
    mom_ok = pd.Series(False, index=df.index)
    if long_window and short_window and long_window > 0 and short_window > 0:
        sma_s = _sma(close, short_window)
        sma_l = _sma(close, long_window)
        mom_ok = ((roc > 0) & (sma_s > sma_l)).fillna(False)
    else:
        # Fallback: just positive ROC
        mom_ok = (roc > 0).fillna(False)

    out = pd.DataFrame(
        {
            "rsi": rsi.astype(float),
            "roc": roc.astype(float).fillna(0.0),
            "trend_up": trend_up.astype(bool),
            "mom_ok": mom_ok.astype(bool),
            "atr": atr.astype(float).fillna(0.0),
        },
        index=df.index,
    )
    return out

def select_rotation(
    hist: Dict[str, pd.DataFrame],
    rotation_lookback: int,
    rotation_top_n: int,
) -> List[str]:
    """
    Pick the top-N symbols by simple momentum over the lookback window.
    hist: dict[symbol] -> DataFrame with 'Close'.
    Returns a list of symbols (length <= rotation_top_n).
    """
    if rotation_lookback <= 0:
        # Degenerate: just return all keys up to top_n
        return list(hist.keys())[: int(rotation_top_n)]

    mom = []
    for sym, df in hist.items():
        if df is None or df.empty:
            continue
        close_col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
        if close_col is None:
            continue
        c = df[close_col].astype(float)
        # Momentum = % change over lookback; require enough data
        if len(c) < rotation_lookback + 1:
            continue
        m = c.pct_change(rotation_lookback).iloc[-1]
        if pd.isna(m):
            continue
        mom.append((sym, float(m)))

    # Rank desc by momentum and pick top-N
    mom.sort(key=lambda x: x[1], reverse=True)
    picks = [s for s, _ in mom[: int(rotation_top_n)]]
    return picks
