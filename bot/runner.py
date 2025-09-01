# bot/runner.py
import os, argparse, time
import datetime as dt
import pandas as pd
from datetime import datetime

from alpaca_trade_api.rest import REST
from .config import Settings
from .data import get_history, latest_close
from .strategy import compute, select_rotation
from .risk import PositionSizer, init_db
from .broker import Broker
from .alerts import ping
from .db_log import DBLog

def backtest(symbols: list[str], start: str, cfg: Settings, cost_bps: float = 0.0) -> float:
    """Runs bar-by-bar backtest and logs trades+equity to SQLite. Returns final equity."""
    print(f"[BT] Logging to DB_PATH={os.getenv('DB_PATH','simple_bot.db')}")
    hist = {s: get_history(s, start) for s in symbols}
    if not hist:
        print("No data downloaded.")
        return 0.0
    first = next(iter(hist.values()))
    dates = first.index

    equity = 10000.0
    cash = equity
    positions: dict[str, dict] = {}
    costs = cost_bps / 10000.0

    db = DBLog(os.getenv("DB_PATH", "simple_bot.db"))
    eq_points = []

    for t in dates:
        inds = {
            s: compute(df.loc[:t], cfg.short_window, cfg.long_window, cfg.rsi_period,
                       cfg.roc_period, cfg.trend_sma, cfg.atr_period)
            for s, df in hist.items()
        }
        universe = select_rotation({s: df.loc[:t] for s, df in hist.items()},
                                   cfg.rotation_lookback, cfg.rotation_top_n)

        # exits / trailing
        to_close = []
        for s, pos in list(positions.items()):
            price = float(hist[s].loc[t]["Close"])
            last = inds[s].iloc[-1]
            atr = float(last["atr"]) if "atr" in last else 0.0

            highest_close = max(float(pos["highest_close"]), price)
            trail = max(float(pos["init_stop"]), highest_close - cfg.atr_trail_mult * atr)
            pos["highest_close"] = highest_close
            pos["trail_stop"] = trail

            take_profit = float(pos["entry"]) * (1 + cfg.take_profit_pct) if cfg.take_profit_pct > 0 else None

            if take_profit and price >= take_profit:
                cash += pos["qty"] * price * (1 - costs)
                db.log_trade(ts=pd.Timestamp(t).to_pydatetime(), symbol=s, side="SELL",
                             qty=int(pos["qty"]), price=price, note="backtest-tp")
                to_close.append(s); continue

            if price <= trail:
                cash += pos["qty"] * price * (1 - costs)
                db.log_trade(ts=pd.Timestamp(t).to_pydatetime(), symbol=s, side="SELL",
                             qty=int(pos["qty"]), price=price, note="backtest-ts")
                to_close.append(s)

        for s in to_close:
            positions.pop(s, None)

        # entries
        for s in universe:
            if s in positions: continue
            df_s = hist[s].loc[:t]
            if df_s.empty: continue
            last = inds[s].iloc[-1]

            trend_up = bool(last.get("trend_up", False))
            mom_ok = bool(last.get("mom_ok", False))
            rsi_val = float(last["rsi"]) if ("rsi" in last and last["rsi"] == last["rsi"]) else 100.0
            price = float(df_s.iloc[-1]["Close"])
            atr = float(last["atr"]) if "atr" in last else 0.0

            if trend_up and mom_ok and rsi_val <= cfg.rsi_pullback_max:
                init_stop = price - cfg.atr_mult_sl * atr
                risk_per_share = max(price - init_stop, 0.0)
                risk_dollars = equity * cfg.risk_fraction
                qty = int(risk_dollars // risk_per_share) if risk_per_share > 0 else 0
                if qty > 0:
                    cash -= qty * price * (1 + costs)
                    positions[s] = dict(entry=price, qty=qty,
                                        init_stop=init_stop, trail_stop=init_stop,
                                        highest_close=price)
                    db.log_trade(ts=pd.Timestamp(t).to_pydatetime(), symbol=s, side="BUY",
                                 qty=int(qty), price=price, note="backtest-entry",
                                 stop=float(init_stop),
                                 take_profit=float(price * (1 + cfg.take_profit_pct)) if cfg.take_profit_pct > 0 else None)

        mtm = sum(float(hist[s].loc[t]["Close"]) * pos["qty"] for s, pos in positions.items())
        equity = cash + mtm
        eq_points.append((pd.Timestamp(t).to_pydatetime(), float(equity)))

    if eq_points:
        db.log_equity_series(eq_points)
    db.close()

    print(f"Backtest done. Final equity (start 10k): ${equity:,.2f}")
    return float(equity)

def decision_once_per_day(conn, symbol: str, today: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT ts FROM trades WHERE symbol=? ORDER BY ts DESC LIMIT 1", (symbol,))
    r = cur.fetchone()
    return (not r) or (not r[0].startswith(today))

def trade(cfg: Settings):
    conn = init_db(cfg.db_path)
    cur = conn.cursor()
    api = REST(cfg.alpaca_api_key, cfg.alpaca_secret_key, base_url=cfg.alpaca_base_url)
    broker = Broker(api)

    hist = {s: get_history(s, cfg.history_start) for s in cfg.symbols}
    ping(cfg.discord_webhook_url, f"🤖 MF Bot starting for {', '.join(cfg.symbols)} mode={cfg.execution_mode}")

    while True:
        today = dt.datetime.utcnow().strftime("%Y-%m-%d")
        try:
            inds = { s: compute(df, cfg.short_window, cfg.long_window, cfg.rsi_period,
                                cfg.roc_period, cfg.trend_sma, cfg.atr_period)
                     for s, df in hist.items() }
            universe = select_rotation(hist, cfg.rotation_lookback, cfg.rotation_top_n)

            for sym in universe:
                if cfg.execution_mode == "daily" and not decision_once_per_day(conn, sym, today):
                    continue
                df = hist[sym]
                last = inds[sym].iloc[-1]

                trend_up = bool(last.get("trend_up", False))
                mom_ok = bool(last.get("mom_ok", False))
                rsi_val = float(last["rsi"]) if ("rsi" in last and last["rsi"] == last["rsi"]) else 100.0
                price = float(latest_close(sym))
                atr = float(last["atr"]) if "atr" in last else 0.0

                cur.execute("SELECT entry, qty, init_stop, trail_stop, highest_close FROM positions WHERE symbol=?", (sym,))
                row = cur.fetchone()
                have_pos = row is not None

                if have_pos:
                    entry, qty, init_stop, trail_stop, highest_close = row
                    highest_close = max(float(highest_close), price)
                    new_trail = max(float(init_stop), highest_close - cfg.atr_trail_mult * atr)
                    if price <= new_trail or (cfg.take_profit_pct > 0 and price >= float(entry) * (1 + cfg.take_profit_pct)):
                        broker.market_sell(sym, int(qty))
                        cur.execute("DELETE FROM positions WHERE symbol=?", (sym,))
                        conn.commit()
                        cur.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
                                    (dt.datetime.utcnow().isoformat(),"SELL",sym,price,int(qty),None,None,"exit"))
                        conn.commit()
                        ping(cfg.discord_webhook_url, f"SELL {sym} qty={int(qty)} @~{price:.2f}")
                    else:
                        cur.execute("UPDATE positions SET trail_stop=?, highest_close=? WHERE symbol=?",
                                    (float(new_trail), float(highest_close), sym))
                        conn.commit()
                    continue

                if trend_up and mom_ok and rsi_val <= cfg.rsi_pullback_max:
                    init_stop = price - cfg.atr_mult_sl * atr
                    sizer = PositionSizer(broker.equity(), cfg.risk_fraction)
                    qty = sizer.shares_for_risk(price, init_stop)
                    if qty > 0:
                        broker.market_buy(sym, int(qty))
                        cur.execute("INSERT INTO positions VALUES (?,?,?,?,?)",
                                    (sym, float(price), int(qty), float(init_stop), float(init_stop), float(price)))
                        conn.commit()
                        cur.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
                                    (dt.datetime.utcnow().isoformat(),"BUY",sym,float(price),int(qty),float(init_stop),
                                     float(price*(1+cfg.take_profit_pct) if cfg.take_profit_pct>0 else None), "entry"))
                        conn.commit()
                        ping(cfg.discord_webhook_url, f"BUY {sym} qty={int(qty)} @~{price:.2f} SL={init_stop:.2f}")
        except Exception as e:
            ping(cfg.discord_webhook_url, f"⚠️ Error: {e}")

        if cfg.execution_mode == "daily":
            break
        time.sleep(cfg.poll_seconds)

def main():
    cfg = Settings()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest")
    bt.add_argument("--symbols", default=None)
    bt.add_argument("--start", default=None)
    bt.add_argument("--cost_bps", default="2.5")

    sub.add_parser("trade")

    args = ap.parse_args()
    if args.cmd == "backtest":
        syms = [s.strip() for s in (args.symbols or ",".join(cfg.symbols)).split(",") if s.strip()]
        final_eq = backtest(syms, args.start or cfg.history_start, cfg, float(args.cost_bps))
        print(f"Backtest final equity (starting 10k): ${final_eq:,.2f}")
    elif args.cmd == "trade":
        trade(cfg)

if __name__ == "__main__":
    main()
