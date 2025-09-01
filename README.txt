
Multi-Factor Bot (Simple) — Full Package

Quick start (Windows):
1) Open Command Prompt:
   cd %HOMEPATH%\Downloads\mf_bot_simple_full2
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env

2) Open .env in Notepad, paste your preset + Alpaca PAPER keys.

3) Backtest:
   python -m bot.runner backtest --symbols SPY,GLD,TLT --start 2018-01-01 --cost_bps 2.5

4) Paper trade (daily close):
   python -m bot.runner trade

5) Dashboard:
   streamlit run streamlit_app.py
