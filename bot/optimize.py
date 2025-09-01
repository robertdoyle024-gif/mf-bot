# bot/optimize.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple
import pandas as pd
import numpy as np
from itertools import product

try:
    from joblib import Parallel, delayed  # type: ignore
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False
    def Parallel(*_, **__):  # type: ignore
        class _P:
            def __init__(self, *_a, **_k): pass
            def __call__(self, it): return [f() for f in it]
        return _P()
    def delayed(fn):  # type: ignore
        return fn

from .strategy import Params, backtest_rotation, calc_kpis

@dataclass
class WFWindow:
    train_start: str
    train_end: str
    test_start: str
    test_end: str

def _add_years(date_str: str, years: int) -> str:
    return (pd.Timestamp(date_str) + pd.offsets.DateOffset(years=years)).date().isoformat()

# CHANGED: add scheme= 'rolling' | 'expanding'
def make_walkforward_windows(
    full_start: str,
    full_end: str,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    scheme: str = "rolling",
) -> List[WFWindow]:
    fs, fe = pd.Timestamp(full_start), pd.Timestamp(full_end)
    if fs >= fe:
        raise ValueError("full_start must be < full_end")

    windows: List[WFWindow] = []
    cursor = fs

    if scheme not in ("rolling", "expanding"):
        raise ValueError("scheme must be 'rolling' or 'expanding'")

    while True:
        if scheme == "rolling":
            tr_start = cursor.date().isoformat()
        else:  # expanding
            tr_start = fs.date().isoformat()

        tr_end = _add_years(cursor.date().isoformat(), train_years)  # exclusive
        te_start = tr_end
        te_end = _add_years(te_start, test_years)

        if pd.Timestamp(te_end) > fe:
            break

        windows.append(WFWindow(
            train_start=tr_start, train_end=tr_end,
            test_start=te_start,  test_end=te_end
        ))

        cursor = pd.Timestamp(_add_years(cursor.date().isoformat(), step_years))

    if not windows:
        raise ValueError("No walk-forward windows produced. Adjust train/test/step or the full period.")
    return windows

def expand_grid(grid: Dict[str, Iterable]) -> List[Params]:
    keys = list(grid.keys()); vals = [list(v) for v in grid.values()]
    return [Params(**dict(zip(keys, tup))) for tup in product(*vals)]

def objective_value(kpis: Dict[str, float], score: str) -> float:
    return float(kpis.get(score, kpis.get("MAR", 0.0))) if score in ("MAR","Sharpe","CAGR") else float(kpis.get("MAR",0.0))

def sweep_once(
    data: Dict[str, pd.DataFrame],
    start: str,
    end: str,
    grid: Dict[str, Iterable],
    score: str = "MAR",
    cost_bps: float = 2.5,
    n_jobs: int = -1,
):
    params_list = expand_grid(grid)
    filtered: List[Params] = []
    for p in params_list:
        if hasattr(p, "MOM_CONFIRM_K") and hasattr(p, "MOM_CONFIRM_M") and p.MOM_CONFIRM_K > p.MOM_CONFIRM_M:
            continue
        if hasattr(p, "MAX_POSITIONS") and p.MAX_POSITIONS < 1:
            continue
        if hasattr(p, "ATR_PCT_MIN") and p.ATR_PCT_MIN < 0:
            continue
        filtered.append(p)
    if filtered: params_list = filtered

    def _run(p: Params):
        res = backtest_rotation(data, p, start, end, cost_bps)
        return p, res.kpis, res

    if _HAVE_JOBLIB and (n_jobs != 1):
        out = Parallel(n_jobs=n_jobs, prefer="threads")([delayed(_run)(p) for p in params_list])
    else:
        out = [_run(p) for p in params_list]

    ranked = sorted(out, key=lambda x: objective_value(x[1], score), reverse=True)
    best_params, best_kpis, best_result = ranked[0]
    return {"ranked": ranked, "best_params": best_params, "best_kpis": best_kpis, "best_result": best_result}

def _chain_equity_segments(segments: List[pd.Series]) -> pd.Series:
    if not segments: return pd.Series(dtype=float)
    chained: List[pd.Series] = []; level = 1.0; first = True
    for seg in segments:
        seg = seg.sort_index()
        ret = seg.pct_change().fillna(0.0)
        eq = (1.0 + ret).cumprod() * level
        level = float(eq.iloc[-1])
        chained.append(eq if first else eq.iloc[1:])
        first = False
    return pd.concat(chained).sort_index()

def backtest_kpis_from_equity(equity: pd.Series) -> Dict[str, float]:
    return calc_kpis(equity)

def walkforward(
    data: Dict[str, pd.DataFrame],
    windows: List[WFWindow],
    grid: Dict[str, Iterable],
    score: str = "MAR",
    cost_bps: float = 2.5,
    n_jobs: int = -1,
):
    test_segments: List[pd.Series] = []
    picks: List[Dict] = []

    for w in windows:
        sw = sweep_once(data, w.train_start, w.train_end, grid, score, cost_bps, n_jobs)
        pstar: Params = sw["best_params"]

        # test eval
        test_res = backtest_rotation(data, pstar, w.test_start, w.test_end, cost_bps)

        # record both train and test KPIs
        picks.append({
            "window": w,
            "params": pstar,
            "train_kpis": sw["best_kpis"],
            "test_kpis": test_res.kpis,                   # <-- NEW
        })

        test_segments.append(test_res.equity)

    equity = _chain_equity_segments(test_segments)
    kpis = backtest_kpis_from_equity(equity)
    return {"equity": equity, "picks": picks, "kpis": kpis}
