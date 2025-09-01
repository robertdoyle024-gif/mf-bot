# Release Checklist (MF Bot)

**Goal:** Cut a clean, reproducible release with tested params + artifacts.

## 0) Prep
- [ ] Pull latest `main`: `git pull`
- [ ] Branch: `git checkout -b release/vX.Y.Z`

## 1) Bump version
- [ ] Update `pyproject.toml` with new version (e.g., `0.2.0`)
- [ ] `git commit -am "Bump version to vX.Y.Z"`

## 2) Dependency freeze
- [ ] Activate venv, then: `python -m pip freeze > requirements.txt`
- [ ] `git add requirements.txt && git commit -m "Freeze deps for vX.Y.Z"`

## 3) Smoke tests (data)
- [ ] Warm cache:  
      `python -c "from bot.data import get_history; [get_history(s,'2018-01-01') for s in 'SPY,GLD,TLT'.split(',')]; print('ok')"`
- [ ] Verify no crashes, last rows look sane

## 4) Backtest
- [ ] `python -m bot.runner backtest --symbols SPY,GLD,TLT --start 2018-01-01 --cost_bps 2.5`
- [ ] Note KPIs

## 5) Sweep
- [ ] `python -m bot.runner sweep --symbols SPY,GLD,TLT --start 2018-01-01 --end 2022-12-31 --grid_path .\grids\mf_default.json`
- [ ] Archive `sweep_results.csv` in release assets

## 6) Walk-Forward
- [ ] `python -m bot.runner backtest-wf --symbols SPY,GLD,TLT --full_start 2010-01-01 --full_end 2024-12-31 --train_years 3 --test_years 1 --step_years 1 --scheme rolling --score MAR --grid_path .\grids\mf_default.json`
- [ ] Archive `wf_equity.csv`, `wf_windows.csv`, `wf_picks.csv`

## 7) Dashboard
- [ ] `streamlit run streamlit_app.py` → no loops, KPIs render, tabs readable

## 8) Packaging (optional)
- [ ] Build exe:  
      `pyinstaller --onefile --name mfbot-runner --add-data "grids;grids" run_bot.py`
- [ ] Sanity run exe backtest/sweep

## 9) Tag & Release
- [ ] `git commit -m "Release vX.Y.Z"` (or fast-forward)
- [ ] `git tag -a vX.Y.Z -m "MF Bot vX.Y.Z"`
- [ ] `git push && git push origin vX.Y.Z`
- [ ] Create GitHub Release
  - [ ] Paste headline KPIs + changelog
  - [ ] Attach CSV artifacts and `mfbot-runner.exe` (if built)

## 10) Post
- [ ] Open an issue with “Next ideas” (TODO stack)
