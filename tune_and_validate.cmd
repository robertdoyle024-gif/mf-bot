@echo off
setlocal ENABLEDELAYEDEXPANSION

REM === MF Bot: Tune & Validate (Windows CMD) ===
REM This script runs: Grid Sweep -> Walk-Forward -> Extract best params -> Full Backtest

REM Change to this script's folder
cd /d "%~dp0"

REM Activate virtualenv
if not exist ".venv\Scripts\activate.bat" (
  echo [ERR] .venv not found. Create/activate your venv first.
  exit /b 1
)
call .venv\Scripts\activate.bat

REM ----- Config -----
set SYMBOLS=SPY,QQQ,IWM,EFA,EEM,VNQ,GLD,SLV,TLT,IEF,LQD,HYG
set START_SWEEP=2015-01-01
set END_SWEEP=2024-12-31
set WF_FULL_START=2010-01-01
set WF_FULL_END=2024-12-31
set GRID_PATH=.\grids\mf_default.json
set COST_BPS=2.5

echo [1/4] Sweep on %SYMBOLS% from %START_SWEEP% to %END_SWEEP% ...
python -m bot.runner sweep --symbols %SYMBOLS% --start %START_SWEEP% --end %END_SWEEP% --grid_path "%GRID_PATH%"
if errorlevel 1 goto :err

echo [2/4] Walk-forward validation ...
python -m bot.runner backtest-wf --symbols %SYMBOLS% --full_start %WF_FULL_START% --full_end %WF_FULL_END% --train_years 3 --test_years 1 --step_years 1 --scheme rolling --score MAR --grid_path "%GRID_PATH%"
if errorlevel 1 goto :err

echo [3/4] Extracting best params from sweep_results.csv -> best_params.env ...
python -c "import pandas as pd; df=pd.read_csv('sweep_results.csv'); row=df.iloc[0]; \
params={'ATR_PCT_MIN':row.get('ATR_PCT_MIN',0.005),'MAX_POSITIONS':int(row.get('MAX_POSITIONS',3)),'MOM_CONFIRM_K':int(row.get('MOM_CONFIRM_K',1)),'MOM_CONFIRM_M':int(row.get('MOM_CONFIRM_M',3)),'RISK_FRACTION':row.get('RISK_FRACTION',0.01),'TAKE_PROFIT_PCT':row.get('TAKE_PROFIT_PCT',0.1),'REGIME_FILTER':row.get('REGIME_FILTER','SMA200')}; \
open('best_params.env','w',newline='').write('\r\n'.join([f'set {k}={v}' for k,v in params.items()])+'\r\n'); \
print('Wrote best_params.env with:',params)"
if errorlevel 1 goto :err

echo.
type best_params.env
echo.

echo [4/4] Backtest full period with extracted params ...
call best_params.env
python -m bot.runner backtest --symbols %SYMBOLS% --start %WF_FULL_START% --cost_bps %COST_BPS%
if errorlevel 1 goto :err

echo.
echo Done.
echo Outputs:
echo   - sweep_results.csv
echo   - wf_windows.csv, wf_equity.csv, wf_picks.csv
echo   - best_params.env  (you can CALL this later to reuse)
echo.
goto :eof

:err
echo.
echo [ERR] A step failed. Check logs above.
exit /b 1
