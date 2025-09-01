
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

def ensure_dt_index(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    out = s.copy()
    out.index = pd.to_datetime(out.index)
    return out

def daily_returns(equity: pd.Series) -> pd.Series:
    equity = ensure_dt_index(equity).astype(float)
    return equity.pct_change().fillna(0.0)

def drawdown(equity: pd.Series) -> pd.Series:
    equity = ensure_dt_index(equity).astype(float)
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd

def kpis(equity: pd.Series, rf_daily: float = 0.0) -> dict:
    equity = ensure_dt_index(equity).astype(float)
    rets = daily_returns(equity)
    ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / len(equity)) - 1.0
    excess = rets - rf_daily
    vol = excess.std() * np.sqrt(TRADING_DAYS)
    sharpe = (excess.mean() * TRADING_DAYS) / (vol + 1e-12)
    dd = drawdown(equity)
    maxdd = dd.min()
    mar = ann_ret / (abs(maxdd) + 1e-12) if maxdd != 0 else np.nan
    return {
        "CAGR": ann_ret,
        "Sharpe": sharpe,
        "MaxDD": float(maxdd),
        "Vol": vol,
        "MAR": mar,
        "FinalEquity": float(equity.iloc[-1]),
    }

def rolling_kpi(equity: pd.Series, window_days: int = 126) -> pd.DataFrame:
    equity = ensure_dt_index(equity).astype(float)
    rets = daily_returns(equity)
    roll = pd.DataFrame(index=equity.index)
    roll["Sharpe"] = (rets.rolling(window_days).mean() / (rets.rolling(window_days).std() + 1e-12)) * np.sqrt(252)
    roll["Drawdown"] = drawdown(equity)
    return roll
