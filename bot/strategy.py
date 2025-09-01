# bot/strategy.py
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

# ---------- Data structures ----------
@dataclass(frozen=True)
class Params:
    ATR_PCT_MIN: float = 0.02
    MAX_POSITIONS: int = 5
    MOM_CONFIRM_K: int = 63
    MOM_CONFIRM_M: int = 21
    RISK_FRACTION: float = 0.01
    TAKE_PROFIT_PCT: float = 0.10
    REGIME_FILTER: str = "SMA200"  # or "none", "200d-momentum", "vol-switch"

@dataclass
class BacktestResult:
    equity: pd.Series              # index: date, value: equity curve
    trades: pd.DataFrame           # each executed trade
    kpis: Dict[str, float]         # calculated metrics

# ---------- Utilities ----------
def _safe_pct_change(s: pd.Series, periods=1):
    return s.pct_change(periods=periods).fillna(0.0)

def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # df must have columns: 'high','low','close'
    tr1 = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    return atr

def regime_mask(prices: pd.Series, kind: str) -> pd.Series:
    if kind == "none":
        return pd.Series(True, index=prices.index)
    if kind == "SMA200":
        sma = prices.rolling(200, min_periods=200).mean()
        return prices > sma
    if kind == "200d-momentum":
        mom = prices.pct_change(200)
        return mom > 0
    if kind == "vol-switch":
        vol = prices.pct_change().rolling(30).std()
        med = vol.rolling(252).median()
        return vol <= med  # trade only in lower vol regime
    return pd.Series(True, index=prices.index)

# ---------- KPI calc ----------
def calc_kpis(equity: pd.Series) -> Dict[str, float]:
    ret = equity.pct_change().fillna(0.0)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252/len(equity)) - 1 if len(equity) > 1 else 0.0
    vol = ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0.0
    # Max drawdown
    roll_max = equity.cummax()
    dd_series = equity / roll_max - 1.0
    max_dd = dd_series.min()
    # MAR ratio
    mar = cagr / abs(max_dd) if max_dd < 0 else float('inf')
    # Pain/Ulcer
    ulcer = np.sqrt((dd_series.pow(2).mean())) if len(dd_series) else 0.0
    return {
        "CAGR": float(cagr),
        "Vol": float(vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(max_dd),
        "MAR": float(mar if np.isfinite(mar) else 0.0),
        "Ulcer": float(ulcer),
    }

# ---------- Core strategy (vectorized, long-only rotation) ----------
def backtest_rotation(
    data: Dict[str, pd.DataFrame],
    params: Params,
    start: str,
    end: str,
    cost_bps: float = 2.5,
) -> BacktestResult:
    """
    data: dict of symbol -> DataFrame[date, open, high, low, close]
    Returns daily equity curve and log of trades (simple rotation with caps).
    """
    # Align close prices
    closes = pd.DataFrame({sym: df['close'] for sym, df in data.items()}).dropna()
    closes = closes.loc[start:end]
    if closes.empty:
        raise ValueError("No price data in selected range")

    # Indicators
    atrs = {}
    for sym, df in data.items():
        atr = compute_atr(df).reindex(closes.index)
        atrs[sym] = atr
    atr_df = pd.DataFrame(atrs)

    # Momentum confirm: rank by M-K (e.g., 63d minus 21d)
    k = params.MOM_CONFIRM_K
    m = params.MOM_CONFIRM_M
    mom_k = closes.pct_change(k)
    mom_m = closes.pct_change(m)
    mom = mom_k - mom_m

    # ATR floor mask
    atr_pct = atr_df / closes
    atr_ok = atr_pct >= params.ATR_PCT_MIN

    # Regime mask per symbol (based on each symbol's close)
    regimes = {sym: regime_mask(closes[sym], params.REGIME_FILTER) for sym in closes.columns}
    regime_df = pd.DataFrame(regimes)

    # Select tradable universe each day
    tradable = atr_ok & regime_df
    # Rank momentum
    ranks = mom.rank(axis=1, ascending=False)

    # Target weights: top N among tradable
    top_n = params.MAX_POSITIONS
    long_mask = (ranks <= top_n) & tradable
    weights = long_mask.div(long_mask.sum(axis=1), axis=0).fillna(0.0)  # equal weight among selected

    # Simple daily rebal at close to next day open (approx): use close-to-close with cost
    ret = closes.pct_change().fillna(0.0)
    gross_ret = (weights.shift().fillna(0.0) * ret).sum(axis=1)

    # Turnover (approx) → costs
    w_prev = weights.shift().fillna(0.0)
    turnover = (weights - w_prev).abs().sum(axis=1)
    cost = turnover * (cost_bps / 10000.0)

    net_ret = gross_ret - cost
    equity = (1 + net_ret).cumprod()
    equity.name = "equity"

    # Basic trade log (entries/exits based on weight changes)
    changes = (weights - w_prev)
    trades = []
    for dt, row in changes.iterrows():
        for sym, delta in row[row != 0].items():
            side = "BUY" if delta > 0 else "SELL"
            trades.append({
                "dt": dt,
                "symbol": sym,
                "side": side,
                "weight_delta": float(delta),
            })
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["dt","symbol","side","weight_delta"])
    kpis = calc_kpis(equity)
    return BacktestResult(equity=equity, trades=trades_df, kpis=kpis)
