# MF Bot (Momentum + Filter Rotation)

A pragmatic momentum-rotation trading bot with risk controls, optimizer (grid sweep & walk-forward), Streamlit dashboard, and Alpaca paper/live trading.

> **Elevator pitch**  
> MF Bot hunts for the strongest ETFs/stocks, rotates into leaders only when the market regime agrees, and caps risk per-trade and at the portfolio. It can tune itself via grid sweeps and walk-forward validation, then trade those settings live (or paper) with alerts and kill-switches. Clean CMD commands, a readable dashboard, and a hardened data loader mean you spend time on ideas, not plumbing.

---

## Features

- **Momentum + Filters**
  - Multi-symbol rotation with top‑N selection
  - Regime filter (e.g., SPY over SMA200)
  - Momentum confirmation windows (M/K), RSI/time exits, ATR floor
- **Risk Controls**
  - Per‑trade risk fraction and position caps
  - Portfolio risk cap and kill‑switch on drawdown
  - Market‑hours only toggle, min notional, symbol‑level overrides
- **Data Layer**
  - Robust Yahoo loader with retries → Ticker().history → Stooq fallback
  - Normalized OHLC; caching with rolling refresh days
- **Optimizers**
  - **Sweep** grid over hyper‑params; CSV results
  - **Walk‑Forward** rolling/expanding windows; KPIs & picks CSVs
- **Execution**
  - **Backtest** (daily bars), **Paper/Live** trading via Alpaca
  - Discord alerts (entries, errors) with cooldowns
- **Dashboard (Streamlit)**
  - Dim theme by default, smoother refresh, clearer panels
  - Overview KPIs, equity vs benchmark, drawdowns, signals table, optimizer results

---

## Project layout (key files)

```
bot/
  config.py         # Settings dataclass (ENV-driven)
  data.py           # Robust price loader + cache
  strategy.py       # Signals & rotation logic
  runner.py         # CLI: backtest | sweep | backtest-wf | trade
  broker.py         # Alpaca REST wrapper
  alerts.py         # Discord webhook
  utils.py          # Helpers
grids/mf_default.json   # Example grid for sweep/WF
streamlit_app.py        # Dashboard
requirements.txt        # Pinned deps (freeze from your venv)
```

---

## Install

```bat
cd C:\path\to\mf_bot
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you don’t have `requirements.txt` yet, create from your working venv:
```bat
python -m pip freeze > requirements.txt
```

---

## Configuration

All knobs live in `bot/config.py` (ENV first, then defaults). Key ENV vars:

```bat
:: Alpaca (Paper) — required for trade mode
set ALPACA_API_KEY=PKXXXX...
set ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXX
set ALPACA_BASE_URL=https://paper-api.alpaca.markets

:: Discord alerts (optional)
set ALERTS_ENABLED=1
set DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

:: Strategy / risk (examples)
set SYMBOLS=SPY,QQQ,IWM,EFA,EEM,VNQ,GLD,SLV,TLT,IEF,LQD,HYG
set ROTATION_TOP_N=3
set MAX_POSITIONS=3
set RISK_FRACTION=0.01
set POSITION_NOTIONAL_CAP_PCT=0.30
set REGIME_FILTER=1
set REGIME_SYMBOL=SPY
set REGIME_SMA=200

:: Data cache
set DATA_CACHE_ENABLED=1
set DATA_CACHE_DIR=.cache\prices
set DATA_CACHE_DAYS=3

:: Execution
set EXEC_MODE=realtime
set POLL_SECONDS=60
set MARKET_HOURS_ONLY=1
```

Store these in a local `.env` that is git‑ignored.

---

## Quickstart (CMD)

### Backtest
```bat
call .venv\Scripts\activate.bat
python -m bot.runner backtest --symbols SPY,GLD,TLT --start 2018-01-01 --cost_bps 2.5
```

### Grid sweep
```bat
python -m bot.runner sweep --symbols SPY,GLD,TLT --start 2018-01-01 --end 2022-12-31 --grid_path .\grids\mf_default.json
type sweep_results.csv
```

### Walk‑forward
```bat
python -m bot.runner backtest-wf ^
  --symbols SPY,GLD,TLT ^
  --full_start 2010-01-01 ^
  --full_end 2024-12-31 ^
  --train_years 3 ^
  --test_years 1 ^
  --step_years 1 ^
  --scheme rolling ^
  --score MAR ^
  --grid_path .\grids\mf_default.json
```

### Trade (paper)
```bat
set ALPACA_API_KEY=PK... & set ALPACA_SECRET_KEY=... & set ALPACA_BASE_URL=https://paper-api.alpaca.markets
set ALERTS_ENABLED=1 & set DISCORD_WEBHOOK_URL=your_webhook
set SYMBOLS=SPY,QQQ,IWM,EFA,EEM,VNQ,GLD,SLV,TLT,IEF,LQD,HYG
python -m bot.runner trade
```

Stop the loop with `Ctrl+C`.

### Dashboard
```bat
streamlit run streamlit_app.py
```

---

## Data reliability

- Yahoo sometimes throws `JSONDecodeError` or “possibly delisted”. The loader retries, then falls back to `Ticker().history`, then **Stooq**, and tries `.US` suffix if needed.  
- Cache: recent N days refreshed (`DATA_CACHE_DAYS`). Delete `.cache\prices\TICKER_1d.parquet` to force a clean pull.

---

## Troubleshooting

**Alpaca `403 forbidden`** — use paper keys against `https://paper-api.alpaca.markets`.  
**YF `No timezone found`** — stick to `yfinance==0.2.40`, `websockets<11`, `urllib3<2`.  
**Streamlit loops** — one browser tab, latest `streamlit>=1.28,<2`, our app uses `st.cache_data` with short TTL.  
**No Discord alerts** — check webhook URL, `ALERTS_ENABLED=1`, and `ALERT_MIN_NOTIONAL` threshold.

---

## Packaging (optional)

**PyInstaller (runner exe):**
```bat
pip install pyinstaller
echo from bot.runner import main > run_bot.py
echo if __name__ == "__main__": main() >> run_bot.py
pyinstaller --onefile --name mfbot-runner --add-data "grids;grids" run_bot.py
dist\mfbot-runner.exe backtest --symbols SPY,GLD,TLT --start 2018-01-01
```

---

## License

See `LICENSE` (MIT).

---

## Changelog

Use GitHub Releases. Summarize: data loader, optimizers, dashboard UX, risk fixes.
