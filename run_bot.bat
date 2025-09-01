@echo off
call .venv\Scripts\activate.bat
set SYMBOLS=SPY,QQQ,IWM,EFA,EEM,VNQ,GLD,SLV,TLT,IEF,LQD,HYG
set RISK_FRACTION=0.005
set POSITION_NOTIONAL_CAP_PCT=0.30
set MARKET_HOURS_ONLY=
set ALERTS_ENABLED=
python -m bot.runner trade
