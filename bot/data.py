from __future__ import annotations
import contextlib, io
import datetime as dt
import os
from typing import Optional

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

# small parquet cache to avoid repeated downloads
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(symbol: str, start: str, end: Optional[str]) -> str:
    end_str = end or "today"
    safe = f"{symbol}_{start}_{end_str}".replace(":", "-")
    return os.path.join(CACHE_DIR, f"{safe}.parquet")

def _read_cache(symbol: str, start: str, end: Optional[str]) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol, start, end)
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                return df
        except Exception:
            pass
    return None

def _write_cache(symbol: str, start: str, end: Optional[str], df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(symbol, start, end))
    except Exception:
        pass

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for name in ("close", "high", "low"):
        for c in df.columns:
            if c.lower() == name:
                out[name.title()] = df[c]
                break
    if out.empty or any(col not in out.columns for col in ["Close","High","Low"]):
        raise RuntimeError("Price frame missing Close/High/Low after standardize")
    return out.sort_index()

def _try_yahoo(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    # Mute stdout/stderr from yfinance so we don't see "Failed download" noise
    end = end or dt.date.today().isoformat()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance empty")
    return _standardize(df)

def _try_stooq(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    stq = symbol if "." in symbol else f"{symbol}.US"
    end_dt = dt.date.fromisoformat(end) if end else dt.date.today()
    start_dt = dt.date.fromisoformat(start)
    df = pdr.DataReader(stq, "stooq", start_dt, end_dt)
    if df is None or df.empty:
        raise RuntimeError("stooq empty")
    df = df.sort_index()
    return _standardize(df)

def get_history(symbol: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Return OHLC subset with Close/High/Low (title case), ascending index; cache -> Yahoo -> Stooq."""
    cached = _read_cache(symbol, start, end)
    if cached is not None and not cached.empty:
        return cached
    try:
        df = _try_yahoo(symbol, start, end)
    except Exception:
        df = _try_stooq(symbol, start, end)
    _write_cache(symbol, start, end, df)
    return df

def latest_close(symbol: str) -> float:
    df = get_history(symbol, (dt.date.today() - dt.timedelta(days=14)).isoformat())
    return float(df["Close"].iloc[-1])
