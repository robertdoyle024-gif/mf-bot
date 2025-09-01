
# bot/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict

# -------- helpers (robust env parsing) --------
def _float_env(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        # tolerate accidental junk after the number (e.g., "0 python -c ...")
        tok = str(v).strip().split()[0]
        return float(tok)
    except Exception:
        return default

def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        tok = str(v).strip().split()[0]
        return int(tok)
    except Exception:
        return default

def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _list_env(name: str, default: List[str]) -> List[str]:
    v = os.getenv(name)
    if not v:
        return list(default)
    # support comma or semicolon; trim spaces
    sep = ";" if ";" in v and "," not in v else ","
    out = [t.strip() for t in v.split(sep) if t.strip()]
    return out or list(default)


@dataclass
class Settings:
    # --- API / broker ---
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # --- storage ---
    db_path: str = os.getenv("DB_PATH", "simple_bot.db")

    # --- trading universe ---
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IEF"])
    history_start: str = os.getenv("HISTORY_START", "2000-01-01")

    # --- execution ---
    execution_mode: str = os.getenv("EXEC_MODE", "daily")  # "daily" or "realtime"
    poll_seconds: int = _int_env("POLL_SECONDS", 60)

    # --- strategy params ---
    short_window: int = _int_env("SHORT_WINDOW", 20)
    long_window: int = _int_env("LONG_WINDOW", 100)
    rsi_period: int = _int_env("RSI_PERIOD", 14)
    roc_period: int = _int_env("ROC_PERIOD", 12)
    trend_sma: int = _int_env("TREND_SMA", 50)
    atr_period: int = _int_env("ATR_PERIOD", 14)

    rotation_lookback: int = _int_env("ROTATION_LOOKBACK", 60)
    rotation_top_n: int = _int_env("ROTATION_TOP_N", 3)

    atr_mult_sl: float = _float_env("ATR_MULT_SL", 2.0)
    atr_trail_mult: float = _float_env("ATR_TRAIL_MULT", 2.0)
    atr_pct_min: float = _float_env("ATR_PCT_MIN", 0.005)

    rsi_floor: float = _float_env("RSI_FLOOR", 25.0)
    rsi_pullback_max: float = _float_env("RSI_PULLBACK_MAX", 70.0)
    time_stop_days: int = _int_env("TIME_STOP_DAYS", 0)  # 0 = disabled

    take_profit_pct: float = _float_env("TAKE_PROFIT_PCT", 0.0)
    risk_fraction: float = _float_env("RISK_FRACTION", 0.01)
    max_positions: int = _int_env("MAX_POSITIONS", 10)
    portfolio_risk_cap: float = _float_env("PORTFOLIO_RISK_CAP", 0.0)

    mom_confirm_m: int = _int_env("MOM_CONFIRM_M", 5)
    mom_confirm_k: int = _int_env("MOM_CONFIRM_K", 3)

    # --- regime filter ---
    regime_filter: bool = _bool_env("REGIME_FILTER", False)
    regime_symbol: str = os.getenv("REGIME_SYMBOL", "SPY")
    regime_sma: int = _int_env("REGIME_SMA", 200)

    # --- risk controls ---
    kill_switch_equity_dd: float = _float_env("KILL_SWITCH_DD", 0.25)
    max_daily_new_pos: int = _int_env("MAX_DAILY_NEW_POS", 999999)
    max_day_risk_fraction: float = _float_env("MAX_DAY_RISK_FRACTION", 1.0)

    # --- alerts ---
    alerts_enabled: bool = _bool_env("ALERTS_ENABLED", True)
    alert_embeds: bool = _bool_env("ALERT_EMBEDS", False)
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    alert_cooldown_sec: int = _int_env("ALERT_COOLDOWN", 60)
    alert_min_notional: float = _float_env("ALERT_MIN_NOTIONAL", 0.0)

    # --- NEW knobs (safer trading) ---
    market_hours_only: bool = _bool_env("MARKET_HOURS_ONLY", True)
    min_order_notional: float = _float_env("MIN_ORDER_NOTIONAL", 0.0)
    min_order_notional_by_symbol: Dict[str, float] = field(default_factory=dict)

    # Per-position notional cap (fraction of equity). Used in runner.py sizing.
    position_notional_cap_pct: float = _float_env("POSITION_NOTIONAL_CAP_PCT", 0.33)

    cost_bps_by_symbol: Dict[str, float] = field(default_factory=dict)
    slippage_bps_by_symbol: Dict[str, float] = field(default_factory=dict)

    # --- data cache (optional) ---
    data_cache_dir: str = os.getenv('DATA_CACHE_DIR', os.path.join('.cache', 'prices'))
    data_cache_days: int = _int_env('DATA_CACHE_DAYS', 3)  # refresh this many recent days
    data_cache_enabled: bool = _bool_env('DATA_CACHE_ENABLED', True)

    def __post_init__(self):
        # Allow environment SYMBOLS to override defaults, robust parsing
        env_syms = _list_env("SYMBOLS", self.symbols)
        self.symbols = env_syms or ["SPY", "QQQ", "IEF"]

        # One-line startup summary
        regime_str = f"ON ({self.regime_symbol}/{self.regime_sma})" if self.regime_filter else "OFF"
        print(
            f"[CFG] symbols={self.symbols} (n={len(self.symbols)}) | "
            f"top_n={self.rotation_top_n} max_pos={self.max_positions} | "
            f"risk={self.risk_fraction} tp={self.take_profit_pct} atr_min={self.atr_pct_min} | "
            f"regime={regime_str} | start={self.history_start} exec={self.execution_mode}"
        )
