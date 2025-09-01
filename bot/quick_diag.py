# bot/quick_diag.py
import sys
from bot.data import get_history

def get_compute():
    # mimic runner._get_legacy_funcs() behavior
    try:
        from bot.strategy_legacy import compute  # type: ignore
        return compute
    except Exception:
        from bot.strategy import compute  # type: ignore
        return compute

def main():
    syms = (sys.argv[1] if len(sys.argv) > 1 else "SPY").split(",")
    start = sys.argv[2] if len(sys.argv) > 2 else "2018-01-01"
    compute = get_compute()

    for s in syms:
        df = get_history(s, start)
        print(f"{s} history shape:", df.shape)
        inds = compute(df, 10, 30, 14, 20, 200, 14)  # (sma_s, sma_l, rsi, roc, trend_sma, atr_period)
        cols = [c for c in ["sma_s","sma_l","trend","rsi","roc","atr","trend_up","mom_ok"] if c in inds.columns]
        print(f"{s} last signals:\n{inds[cols].tail(5)}\n")

if __name__ == "__main__":
    main()
