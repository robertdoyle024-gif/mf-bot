
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _to_bool(v: str, default=False):
    if v is None: return default
    return v.strip().lower() in ("1","true","yes","y","on")

@dataclass
class Settings:
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    symbols: list[str] = tuple([s.strip() for s in os.getenv("SYMBOLS","SPY,GLD,TLT").split(",") if s.strip()])
    rotation_lookback: int = int(os.getenv("ROTATION_LOOKBACK_DAYS", "60"))
    rotation_top_n: int = int(os.getenv("ROTATION_TOP_N", "2"))
    long_only: bool = _to_bool(os.getenv("LONG_ONLY","true"))

    short_window: int = int(os.getenv("SHORT_WINDOW","10"))
    long_window: int = int(os.getenv("LONG_WINDOW","30"))
    rsi_period: int = int(os.getenv("RSI_PERIOD","14"))
    rsi_pullback_max: float = float(os.getenv("RSI_PULLBACK_MAX","60"))
    roc_period: int = int(os.getenv("ROC_PERIOD","20"))
    trend_sma: int = int(os.getenv("TREND_SMA","200"))

    atr_period: int = int(os.getenv("ATR_PERIOD","14"))
    atr_mult_sl: float = float(os.getenv("ATR_MULT_SL","2.0"))
    atr_trail_mult: float = float(os.getenv("ATR_TRAIL_MULT","2.75"))
    take_profit_pct: float = float(os.getenv("TAKE_PROFIT_PCT","0.00"))

    risk_fraction: float = float(os.getenv("RISK_FRACTION","0.0075"))

    execution_mode: str = os.getenv("EXECUTION_MODE","daily")
    poll_seconds: int = int(os.getenv("POLL_SECONDS","60"))
    history_start: str = os.getenv("HISTORY_START","2018-01-01")

    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL","")
    log_csv_path: str = os.getenv("LOG_CSV_PATH","trades.csv")
    db_path: str = os.getenv("DB_PATH","simple_bot.db")
