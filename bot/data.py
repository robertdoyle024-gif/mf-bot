# bot/data.py
import os
import time
import logging
import warnings
import pandas as pd
import yfinance as yf

# ── Quiet noisy libs (logs + warnings) ───────────────────────────────────────────
for name in ("yfinance", "urllib3", "requests"):
    logging.getLogger(name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*possibly delisted.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*delisted.*")
warnings.filterwarnings("ignore", message=".*JSONDecodeError.*")

# Stooq fallback (daily) via pandas-datareader
try:
    from pandas_datareader import data as pdr
    HAVE_PDR = True
except Exception:
    HAVE_PDR = False

# Optional: bypass Yahoo and prefer Stooq for historical pulls
PREFER_STOOQ = os.getenv("PREFER_STOOQ", "0").lower() in ("1", "true", "yes", "on")

# ── Parquet cache helpers (best effort; requires pyarrow; will silently skip if not available) ──
def _cache_enabled() -> bool:
    return os.getenv("DATA_CACHE_ENABLED", "1").lower() in ("1", "true", "yes", "on")

def _cache_dir() -> str:
    d = os.getenv("DATA_CACHE_DIR", os.path.join(".cache", "prices"))
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d

def _cache_days() -> int:
    try:
        return int(os.getenv("DATA_CACHE_DAYS", "3"))
    except Exception:
        return 3

def _cache_path(symbol: str, interval: str = "1d") -> str:
    return os.path.join(_cache_dir(), f"{symbol.upper()}_{interval}.parquet")

def _read_cache(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            return _normalize_index(df)
    except Exception:
        pass
    return pd.DataFrame()

def _write_cache(path: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    try:
        tmp = path + ".tmp"
        df.to_parquet(tmp, index=True)
        os.replace(tmp, path)
    except Exception:
        # No pyarrow or filesystem issue: skip caching silently
        pass

# ── Normalize columns/index ───────────────────────────────────────────────────────
def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex and standard OHLC columns; dedupe & sort."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([c for c in tup if c]).strip() for tup in df.columns]

    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "open" in lc and "open" not in colmap:
            colmap[c] = "Open"
        elif "high" in lc and "high" not in colmap:
            colmap[c] = "High"
        elif "low" in lc and "low" not in colmap:
            colmap[c] = "Low"
        elif ("close" in lc or "adj close" in lc or lc == "adjclose") and "close" not in colmap:
            colmap[c] = "Close"
        elif lc == "volume":
            colmap[c] = "Volume"

    if colmap:
        df = df.rename(columns=colmap)

    keep = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    if keep:
        df = df[keep]

    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

# ── Raw download funcs ───────────────────────────────────────────────────────────
def _try_yf_download(ticker: str, *, start=None, period=None, interval="1d", auto_adjust=True) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=start,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,   # more stable on Windows
        prepost=False,
        group_by="column",
    )
    return _normalize_index(df)

def _try_yf_ticker(ticker: str, *, start=None, period=None, interval="1d", auto_adjust=True) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    if start:
        df = t.history(start=start, interval=interval, auto_adjust=auto_adjust)
    else:
        df = t.history(period=period or "5d", interval=interval, auto_adjust=auto_adjust)
    return _normalize_index(df)

def _try_stooq(ticker: str, *, start=None) -> pd.DataFrame:
    if not HAVE_PDR:
        raise RuntimeError("pandas-datareader not installed for Stooq fallback")
    df = pdr.DataReader(ticker, "stooq", start=start)
    return _normalize_index(df)

# ── Retry wrapper + .US suffix fallback ──────────────────────────────────────────
def _download_with_retry_one_symbol(
    sym: str, *, start=None, period=None, interval="1d",
    auto_adjust=True, retries: int | None = None, backoff: float | None = None
) -> pd.DataFrame | None:
    if retries is None:
        retries = int(os.getenv("YF_RETRIES", "3"))
    if backoff is None:
        backoff = float(os.getenv("YF_BACKOFF", "0.8"))

    last_err = None

    # Prefer Stooq first if requested
    if PREFER_STOOQ:
        try:
            stq = _try_stooq(sym, start=start)
            if stq is not None and not stq.empty:
                return stq
        except Exception as e:
            last_err = e  # keep going

    # 1) yfinance.download
    for k in range(retries):
        try:
            df = _try_yf_download(sym, start=start, period=period, interval=interval, auto_adjust=auto_adjust)
            if df is not None and not df.empty:
                return df
            last_err = RuntimeError("Empty dataframe from Yahoo (download)")
        except Exception as e:
            last_err = e
        time.sleep(backoff * (k + 1))

    # 2) yfinance.Ticker().history
    try:
        df = _try_yf_ticker(sym, start=start, period=period, interval=interval, auto_adjust=auto_adjust)
        if df is not None and not df.empty:
            return df
        last_err = RuntimeError("Empty dataframe from Yahoo (Ticker.history)")
    except Exception as e:
        last_err = e

    # 3) Stooq fallback (daily)
    try:
        stq = _try_stooq(sym, start=start)
        if stq is not None and not stq.empty:
            return stq
        last_err = RuntimeError("Empty dataframe from Stooq")
    except Exception as e:
        last_err = e

    return None  # let caller try suffix before raising

def _download_with_retry(
    ticker: str, *, start=None, period=None, interval="1d",
    auto_adjust=True, retries: int | None = None, backoff: float | None = None
) -> pd.DataFrame:
    df = _download_with_retry_one_symbol(
        ticker, start=start, period=period, interval=interval,
        auto_adjust=auto_adjust, retries=retries, backoff=backoff
    )
    if df is not None and not df.empty:
        return df

    # Stooq-style .US suffix as a second try
    if not ticker.endswith(".US"):
        df = _download_with_retry_one_symbol(
            ticker + ".US", start=start, period=period, interval=interval,
            auto_adjust=auto_adjust, retries=retries, backoff=backoff
        )
        if df is not None and not df.empty:
            return df

    raise RuntimeError(f"Failed to download data for {ticker}")

# ── Public API (with cache) ──────────────────────────────────────────────────────
def get_history(symbol: str, start: str, interval: str = "1d") -> pd.DataFrame:
    sym = str(symbol).upper()
    if not _cache_enabled():
        return _download_with_retry(sym, start=start, interval=interval, auto_adjust=True)

    path = _cache_path(sym, interval)
    cached = _read_cache(path)

    # If cache empty or doesn't cover requested start → full pull
    if cached.empty or (len(cached.index) and pd.Timestamp(start) < cached.index.min()):
        df = _download_with_retry(sym, start=start, interval=interval, auto_adjust=True)
        _write_cache(path, df)
        return df

    # Refresh tail only
    tail_start = (cached.index.max() - pd.Timedelta(days=_cache_days() + 2)).date().isoformat()
    try:
        fresh = _download_with_retry(sym, start=tail_start, interval=interval, auto_adjust=True)
    except Exception:
        fresh = pd.DataFrame()

    merged = pd.concat([cached, fresh]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    _write_cache(path, merged)

    return merged.loc[pd.to_datetime(start):] if start else merged

def latest_close(symbol: str) -> float:
    sym = str(symbol).upper()
    # Try a tiny online pull first
    try:
        df = _download_with_retry(sym, period="5d", interval="1d", auto_adjust=True)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    # Cache fallback
    if _cache_enabled():
        path = _cache_path(sym, "1d")
        dfc = _read_cache(path)
        if not dfc.empty:
            return float(dfc["Close"].iloc[-1])
    raise RuntimeError(f"latest_close: no data for {symbol}")
