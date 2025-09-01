
import pandas as pd
import numpy as np
from .indicators import sma, rsi, roc, atr

def _as_series(obj: pd.Series | pd.DataFrame, index: pd.Index) -> pd.Series:
    """Coerce a column selection into a 1-D Series aligned to index."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(index=index, dtype=float)
        # If duplicate 'Close' columns exist, take the first
        ser = obj.iloc[:, 0]
    else:
        ser = obj
    # Ensure name and index alignment
    ser = ser.reindex(index)
    return pd.to_numeric(ser, errors='coerce')

def compute(df: pd.DataFrame, short:int, long:int, rsi_period:int, roc_period:int, trend_sma:int, atr_period:int) -> pd.DataFrame:
    out = df.copy()

    # Force 1-D Series for prices (avoid alignment issues)
    close = _as_series(out.get('Close'), out.index)

    # Indicators based on coerced close series
    out['sma_s'] = sma(close, short)
    out['sma_l'] = sma(close, long)
    out['trend'] = sma(close, trend_sma)
    out['rsi'] = rsi(close, rsi_period)
    out['roc'] = roc(close, roc_period)
    out['atr'] = atr(out, atr_period)  # uses High/Low/Close from frame

    # Robust boolean logic using numpy arrays
    sma_ok = (out['sma_s'] > out['sma_l']).fillna(False).to_numpy(dtype=bool)
    trend_ok = (close > out['trend']).fillna(False).to_numpy(dtype=bool)
    out['trend_up'] = pd.Series(sma_ok & trend_ok, index=out.index)

    out['pullback_ok'] = (out['rsi'] <= 60).fillna(False).astype(bool)
    out['mom_ok'] = (out['roc'] > 0).fillna(False).astype(bool)
    return out

def select_rotation(symbol_to_df: dict, lookback:int, top_n:int) -> list[str]:
    perf = []
    for sym, df in symbol_to_df.items():
        if len(df) < lookback + 2:
            continue
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:,0]
        ret = close.iloc[-1] / close.iloc[-lookback] - 1.0
        perf.append((sym, ret))
    perf.sort(key=lambda x: x[1], reverse=True)
    return [s for s,_ in perf[:top_n]]
