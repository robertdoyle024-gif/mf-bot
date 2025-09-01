# bot/runner.py
from __future__ import annotations

import os
import argparse
import time
import json
import datetime as dt
from datetime import datetime, time as dtime
from typing import Dict, List, Tuple

import pandas as pd
from alpaca_trade_api.rest import REST

from .config import Settings
from .data import get_history, latest_close
# (Legacy funcs are lazy-imported; see _get_legacy_funcs)
from .risk import PositionSizer, init_db
from .broker import Broker
from .alerts import ping, ping_embed, allow_alert
from .db_log import DBLog

# Optimizer imports (do not depend on legacy strategy)
from .optimize import make_walkforward_windows, sweep_once, walkforward


# ============================== JSON/GRID UTILITIES ==============================

def _parse_grid_arg(grid_json: str | None, grid_path: str | None) -> dict:
    """
    Accept a JSON string (possibly wrapped in quotes by the shell) or a path to a JSON file.
    Try json.loads first, then ast.literal_eval as a fallback for PowerShell here-strings.
    """
    import ast

    # Prefer a file if provided
    if grid_path:
        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"--grid_path not found: {grid_path}")
        with open(grid_path, "r", encoding="utf-8") as f:
            txt = f.read()
    else:
        if grid_json is None:
            raise ValueError("Provide either --grid_json or --grid_path")
        txt = grid_json

    s = txt.strip()

    # Strip accidental outer quotes from the shell ('{...}' or "{...}")
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()

    # First try strict JSON
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("Grid must be a JSON object (dict).")
        return obj
    except json.JSONDecodeError as je:
        # Fallback: allow Python literals (here-strings in PowerShell often behave)
        try:
            obj = ast.literal_eval(s)
            if not isinstance(obj, dict):
                raise ValueError("Grid must evaluate to a dict.")
            return obj
        except Exception as e:
            preview = s[:300].replace("\\n", "\\\\n")
            raise ValueError(
                f"Failed to parse grid. JSON error at pos {je.pos}: {je.msg}. "
                f"literal_eval fallback error: {e}\\nInput preview: {preview}"
            )


# ======================= LEGACY STRATEGY (LAZY IMPORT SHIM) ======================

def _get_legacy_funcs():
    """
    Import compute/select_rotation ONLY when backtest/trade need them.
    Looks in bot.strategy_legacy first, then bot.strategy.
    """
    try:
        from .strategy_legacy import compute, select_rotation  # type: ignore
        return compute, select_rotation
    except Exception:
        try:
            from .strategy import compute, select_rotation  # type: ignore
            return compute, select_rotation
        except Exception as e:
            raise ImportError(
                "Legacy functions compute/select_rotation not found.\\n"
                "Add them to bot/strategy_legacy.py (recommended) or bot/strategy.py."
            ) from e


# ================================ UTILITIES =====================================

def _get_bps(mapping: dict | None, symbol: str, default_bps: float) -> float:
    """Return basis points for symbol from mapping (symbol->bps) with a default."""
    try:
        if isinstance(mapping, dict) and symbol in mapping:
            return float(mapping[symbol])
    except Exception:
        pass
    return float(default_bps)


def _min_notional_ok(cfg: Settings, symbol: str, price: float, qty: int) -> bool:
    """Enforce min notional (price*qty) based on global or per-symbol config."""
    notional = float(price) * int(qty)
    per_sym = getattr(cfg, "min_order_notional_by_symbol", None)
    sym_min = 0.0
    try:
        if isinstance(per_sym, dict):
            sym_min = float(per_sym.get(symbol, 0.0))
    except Exception:
        sym_min = 0.0
    global_min = float(getattr(cfg, "min_order_notional", 0.0))
    threshold = max(sym_min, global_min, float(getattr(cfg, "alert_min_notional", 0.0)))
    return notional >= threshold


def _is_market_hours_now(cfg: Settings) -> bool:
    """
    Hard market-hours guard for US equities.
    - Only Mon-Fri
    - Between 09:30 and 16:00 America/New_York
    - Can be disabled with cfg.market_hours_only = False
    """
    if not bool(getattr(cfg, "market_hours_only", True)):
        return True
    try:
        now_et = pd.Timestamp.now(tz="America/New_York")
        if now_et.weekday() >= 5:  # 5=Sat, 6=Sun
            return False
        start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        end = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now_et <= end
    except Exception:
        # Fail-closed (safer): if tz check fails, disallow
        return False


def _total_open_risk_positions_dict(positions: dict) -> float:
    """
    For backtest (in-memory positions dict):
    Sum of (entry - init_stop) * qty for all open positions.
    """
    risk_sum = 0.0
    for pos in positions.values():
        entry = float(pos.get("entry", 0.0))
        init_stop = float(pos.get("init_stop", entry))
        qty = int(pos.get("qty", 0))
        risk_sum += max(entry - init_stop, 0.0) * qty
    return risk_sum


def _total_open_risk_db(conn) -> float:
    """
    For live/paper trade (positions table in SQLite):
    Sum of (entry - init_stop) * qty across all rows.
    """
    cur = conn.cursor()
    cur.execute("SELECT entry, qty, init_stop FROM positions")
    rows = cur.fetchall()
    risk_sum = 0.0
    for entry, qty, init_stop in rows:
        entry = float(entry)
        qty = int(qty)
        init_stop = float(init_stop)
        risk_sum += max(entry - init_stop, 0.0) * qty
    return risk_sum


# ============================ OPT DATA LOADING HELPERS ===========================

def _lower_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a price DataFrame to lowercase open/high/low/close without collision
    if both 'Close' and 'close' exist. Falls back to 'Close' when needed.
    """
    if df is None or df.empty:
        raise ValueError("History DataFrame is empty.")

    # Keep first occurrence for each lowered name
    first_by_lower: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if lc not in first_by_lower:
            first_by_lower[lc] = c

    src_open = first_by_lower.get("open", first_by_lower.get("close", "Close"))
    src_high = first_by_lower.get("high", first_by_lower.get("close", "Close"))
    src_low = first_by_lower.get("low", first_by_lower.get("close", "Close"))
    src_close = first_by_lower.get("close", "Close")

    try:
        out = df[[src_open, src_high, src_low, src_close]].copy()
    except KeyError:
        raise ValueError("Missing required OHLC columns in history DataFrame.")

    out.columns = ["open", "high", "low", "close"]
    return out.sort_index()[["open", "high", "low", "close"]]


def _load_data_for_opt(symbols: List[str], start: str | None = None) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = get_history(s, start or "2000-01-01")
        if df is None or df.empty:
            raise ValueError(f"No history for {s}")
        out[s] = _lower_ohlc(df)
    return out


# ================================== BACKTEST ===================================

def backtest(symbols: List[str], start: str, cfg: Settings, cost_bps: float = 0.0) -> float:
    """Runs bar-by-bar backtest and logs trades+equity to SQLite. Returns final equity."""
    compute, select_rotation = _get_legacy_funcs()

    # If universe collapses, try a robust default set
    fallback_syms = ["SPY", "QQQ", "IEF"]

    print(f"[BT] Logging to DB_PATH={os.getenv('DB_PATH', 'simple_bot.db')}")

    # History load (with shapes)
    raw_hist = {s: get_history(s, start) for s in symbols}

    # Log shapes and drop empties
    hist: Dict[str, pd.DataFrame] = {}
    for s, df in raw_hist.items():
        n = 0 if (df is None) else len(df)
        print(f"[BT] {s}: {n} rows")
        if df is not None and not df.empty:
            hist[s] = df

    # Fallback universe if too few symbols survived
    if len(hist) < 2:
        print("[BT] Too few usable symbols; switching to fallback universe:", ",".join(fallback_syms))
        raw_hist = {s: get_history(s, start) for s in fallback_syms}
        hist = {s: df for s, df in raw_hist.items() if df is not None and not df.empty}
        if not hist:
            print("[BT] Fallback universe also empty. Exiting.")
            return 0.0

    # Build a common date index across all symbols (intersection),
    # so we don't reference a date that one of the symbols doesn't have.
    common_idx = None
    for df in hist.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) == 0:
        print("[BT] No overlapping dates across symbols. Exiting.")
        return 0.0

    dates = common_idx.sort_values()

    # Optional regime gating time series
    regime_ok_by_date: Dict[pd.Timestamp, bool] = {}
    slope_ok_by_date: Dict[pd.Timestamp, bool] = {}
    if getattr(cfg, "regime_filter", False):
        try:
            reg_hist = get_history(cfg.regime_symbol, start)
            if reg_hist is not None and not reg_hist.empty:
                sma = reg_hist["Close"].rolling(int(getattr(cfg, "regime_sma", 200))).mean()
                slope = sma.diff(10)
                common_reg = reg_hist.index.intersection(dates)
                for t in common_reg:
                    c = float(reg_hist.loc[t, "Close"])
                    s = float(sma.loc[t]) if pd.notna(sma.loc[t]) else c
                    sl = float(slope.loc[t]) if pd.notna(slope.loc[t]) else 0.0
                    regime_ok_by_date[t] = (c > s)
                    slope_ok_by_date[t] = (sl > 0)
            else:
                regime_ok_by_date = {t: True for t in dates}
                slope_ok_by_date = {t: True for t in dates}
        except Exception:
            regime_ok_by_date = {t: True for t in dates}
            slope_ok_by_date = {t: True for t in dates}
    else:
    # No regime filter
        regime_ok_by_date = {t: True for t in dates}
        slope_ok_by_date = {t: True for t in dates}

    equity = 10000.0
    cash = equity
    positions: Dict[str, Dict] = {}
    default_costs = float(cost_bps)
    default_slip = float(getattr(cfg, "slippage_bps", 0.0))
    costs_by_symbol = getattr(cfg, "cost_bps_by_symbol", None)  # dict[str, bps]
    slip_by_symbol = getattr(cfg, "slippage_bps_by_symbol", None)  # dict[str, bps]

    # exits config
    rsi_floor = float(getattr(cfg, "rsi_floor", 25.0))
    time_stop_days = int(getattr(cfg, "time_stop_days", 0))  # 0 disables
    atr_trail_mult = float(getattr(cfg, "atr_trail_mult", 2.0))
    take_profit_pct = float(getattr(cfg, "take_profit_pct", 0.0))
    atr_pct_min = float(getattr(cfg, "atr_pct_min", 0.0))
    max_positions = int(getattr(cfg, "max_positions", 9999))
    rsi_pullback_max = float(getattr(cfg, "rsi_pullback_max", 100.0))
    risk_fraction = float(getattr(cfg, "risk_fraction", 0.01))
    portfolio_risk_cap = float(getattr(cfg, "portfolio_risk_cap", 0.0))

    db = DBLog(os.getenv("DB_PATH", "simple_bot.db"))
    eq_points: List[Tuple[datetime, float]] = []

    for t in dates:
        # Indicators per symbol up to date t
        inds: Dict[str, pd.DataFrame] = {}
        for s, df in hist.items():
            df_t = df.loc[:t]
            if df_t.empty:
                continue
            inds[s] = compute(
                df_t,
                cfg.short_window,
                cfg.long_window,
                cfg.rsi_period,
                cfg.roc_period,
                cfg.trend_sma,
                cfg.atr_period,
            )

        # Rotation universe on data up to t
        rot_hist = {s: df.loc[:t] for s, df in hist.items() if not df.loc[:t].empty}
        universe = select_rotation(rot_hist, cfg.rotation_lookback, cfg.rotation_top_n)

        # exits / trailing
        to_close: List[str] = []
        for s, pos in list(positions.items()):
            price = float(hist[s].loc[t]["Close"])
            last = inds.get(s)
            if last is None or last.empty:
                continue
            last = last.iloc[-1]
            atr = float(last["atr"]) if ("atr" in last and pd.notna(last["atr"])) else 0.0
            rsi_val = float(last["rsi"]) if ("rsi" in last and pd.notna(last["rsi"])) else 50.0

            highest_close = max(float(pos["highest_close"]), price)
            trail = max(float(pos["init_stop"]), highest_close - atr_trail_mult * atr)
            pos["highest_close"] = highest_close
            pos["trail_stop"] = trail

            take_profit = float(pos["entry"]) * (1 + take_profit_pct) if take_profit_pct > 0 else None

            reason = None
            if (take_profit is not None) and price >= take_profit:
                reason = "tp"
            elif price <= trail:
                reason = "ts"
            elif rsi_val <= rsi_floor:
                reason = "rsi"
            elif time_stop_days > 0 and (pd.Timestamp(t).to_pydatetime() - pos["entry_ts"]).days >= time_stop_days:
                reason = "time"

            if reason:
                costs = _get_bps(costs_by_symbol, s, default_costs) / 10000.0
                slip = _get_bps(slip_by_symbol, s, default_slip) / 10000.0
                exec_px = price * (1 - (costs + slip))
                cash += pos["qty"] * exec_px
                db.log_trade(
                    ts=pd.Timestamp(t).to_pydatetime(),
                    symbol=s,
                    side="SELL",
                    qty=int(pos["qty"]),
                    price=exec_px,
                    note=f"backtest-{reason}",
                )
                to_close.append(s)

        for s in to_close:
            positions.pop(s, None)

        # entries
        for s in universe:
            if s in positions:
                continue
            df_s = hist[s].loc[:t]
            if df_s.empty or s not in inds:
                continue
            last = inds[s].iloc[-1]

            trend_up = bool(last.get("trend_up", False))
            mom_ok = bool(last.get("mom_ok", False))
            rsi_val = float(last["rsi"]) if ("rsi" in last and pd.notna(last["rsi"])) else 100.0
            price = float(df_s.iloc[-1]["Close"])
            atr = float(last["atr"]) if "atr" in last and pd.notna(last["atr"]) else 0.0

            atr_pct = (atr / price) if price > 0 else 0.0
            if atr_pct < atr_pct_min:
                continue
            if len(positions) >= max_positions:
                continue

            # Optional global regime gate
            if not (regime_ok_by_date.get(t, True) and slope_ok_by_date.get(t, True)):
                continue

            # Momentum confirmation window
            try:
                tail = inds[s].tail(int(getattr(cfg, "mom_confirm_m", 1)))
                pos_days = int((tail["mom_ok"].astype(bool)).sum())
                if pos_days < int(getattr(cfg, "mom_confirm_k", 0)):
                    continue
            except Exception:
                pass

            if not (trend_up and mom_ok and rsi_val <= rsi_pullback_max):
                continue

            # Entry + sizing
            init_stop = price - float(getattr(cfg, "atr_mult_sl", 2.0)) * atr
            per_share = max(price - init_stop, 0.0)
            if per_share <= 0:
                continue

            # Base sizing (per-trade risk)
            base_risk = equity * risk_fraction
            qty_base = int(base_risk // per_share)

            # Portfolio risk cap
            remaining = 10**18  # effectively infinite if cap not set
            if portfolio_risk_cap > 0:
                total_cap = equity * portfolio_risk_cap
                used = _total_open_risk_positions_dict(positions)
                remaining = max(total_cap - used, 0.0)

            qty_cap = int(remaining // per_share) if remaining > 0 else 0
            qty = max(min(qty_base, qty_cap), 0)

            # Enforce min notional before placing "virtual" order
            if qty <= 0 or not _min_notional_ok(cfg, s, price, qty):
                continue

            costs = _get_bps(costs_by_symbol, s, default_costs) / 10000.0
            slip = _get_bps(slip_by_symbol, s, default_slip) / 10000.0

            exec_px = price * (1 + (costs + slip))
            cash -= qty * exec_px
            positions[s] = dict(
                entry=exec_px,
                qty=qty,
                init_stop=float(init_stop),
                trail_stop=float(init_stop),
                highest_close=exec_px,
                entry_ts=pd.Timestamp(t).to_pydatetime(),
            )
            tp = exec_px * (1 + take_profit_pct) if take_profit_pct > 0 else None
            db.log_trade(
                ts=pd.Timestamp(t).to_pydatetime(),
                symbol=s,
                side="BUY",
                qty=int(qty),
                price=exec_px,
                note="backtest-entry",
                stop=float(init_stop),
                take_profit=tp,
            )

        # mark-to-market and equity curve point after handling all entries/exits for t
        mtm = sum(float(hist[s].loc[t]["Close"]) * pos["qty"] for s, pos in positions.items())
        equity = cash + mtm
        eq_points.append((pd.Timestamp(t).to_pydatetime(), float(equity)))

    if eq_points:
        db.log_equity_series(eq_points)
    db.close()

    print(f"Backtest done. Final equity (start 10k): ${equity:,.2f}")
    return float(equity)


# =================================== TRADE ====================================

def decision_once_per_day(conn, symbol: str, today: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT ts FROM trades WHERE symbol=? ORDER BY ts DESC LIMIT 1", (symbol,))
    r = cur.fetchone()
    return (not r) or (not r[0].startswith(today))


def _send_startup_alert(cfg: Settings) -> None:
    if not getattr(cfg, "alerts_enabled", True):
        return
    msg = f"🤖 MF Bot starting for {', '.join(cfg.symbols)} mode={cfg.execution_mode}"
    if allow_alert("START", cfg.alert_cooldown_sec):
        if getattr(cfg, "alert_embeds", False):
            ping_embed(cfg.discord_webhook_url, title="Bot Start", description=msg, color=0x5865F2)
        else:
            ping(cfg.discord_webhook_url, msg)


def _send_buy_alert(cfg: Settings, sym: str, qty: int, price: float, init_stop: float) -> None:
    if not getattr(cfg, "alerts_enabled", True):
        return
    if (price * qty) < getattr(cfg, "alert_min_notional", 0.0):
        return
    key = f"BUY:{sym}"
    if not allow_alert(key, cfg.alert_cooldown_sec):
        return
    if getattr(cfg, "alert_embeds", False):
        ping_embed(
            cfg.discord_webhook_url,
            title=f"BUY {sym}",
            fields=[("Price", f"{price:.2f}", True), ("Qty", f"{qty}", True), ("Init SL", f"{init_stop:.2f}", True)],
            color=0x2ECC71,
        )
    else:
        ping(cfg.discord_webhook_url, f"BUY {sym} qty={qty} @~{price:.2f} SL={init_stop:.2f}")


def _send_sell_alert(cfg: Settings, sym: str, qty: int, price: float, reason: str) -> None:
    if not getattr(cfg, "alerts_enabled", True):
        return
    if (price * qty) < getattr(cfg, "alert_min_notional", 0.0):
        return
    key = f"SELL:{sym}"
    if not allow_alert(key, cfg.alert_cooldown_sec):
        return
    msg_suffix = f" ({reason})" if reason else ""
    if getattr(cfg, "alert_embeds", False):
        ping_embed(
            cfg.discord_webhook_url,
            title=f"SELL {sym}{msg_suffix}",
            fields=[("Price", f"{price:.2f}", True), ("Qty", f"{qty}", True)],
            color=0xE74C3C,
        )
    else:
        ping(cfg.discord_webhook_url, f"SELL {sym} qty={qty} @~{price:.2f}{msg_suffix}")


def trade(cfg: Settings) -> None:
    compute, select_rotation = _get_legacy_funcs()

    conn = init_db(cfg.db_path)
    cur = conn.cursor()
    api = REST(cfg.alpaca_api_key, cfg.alpaca_secret_key, base_url=cfg.alpaca_base_url)
    broker = Broker(api)

    hist: Dict[str, pd.DataFrame] = {s: get_history(s, cfg.history_start) for s in cfg.symbols}
    hist = {k: v for k, v in hist.items() if v is not None and not v.empty}

    # Regime latest gates
    regime_ok_latest = True
    slope_ok_latest = True
    if getattr(cfg, "regime_filter", False):
        try:
            reg_hist = get_history(cfg.regime_symbol, cfg.history_start)
            sma = reg_hist["Close"].rolling(int(getattr(cfg, "regime_sma", 200))).mean()
            slope = sma.diff(10)
            regime_ok_latest = bool(float(reg_hist["Close"].iloc[-1]) > float(sma.iloc[-1]))
            slope_ok_latest = bool(float(slope.iloc[-1]) > 0)
        except Exception:
            regime_ok_latest, slope_ok_latest = True, True  # fail-open

    _send_startup_alert(cfg)

    start_equity = float(broker.equity())
    peak_equity = start_equity
    day_key = None
    risk_today = 0.0

    rsi_floor = float(getattr(cfg, "rsi_floor", 25.0))
    time_stop_days = int(getattr(cfg, "time_stop_days", 0))
    atr_pct_min = float(getattr(cfg, "atr_pct_min", 0.0))

    while True:
        today = dt.datetime.utcnow().strftime("%Y-%m-%d")

        if today != day_key:
            day_key = today
            risk_today = 0.0

        cur_eq = float(broker.equity())
        peak_equity = max(peak_equity, cur_eq)
        if peak_equity > 0:
            dd = (peak_equity - cur_eq) / peak_equity
            if dd >= getattr(cfg, "kill_switch_equity_dd", 0.0):
                msg = f"🛑 Kill-switch: DD {dd:.1%} >= {getattr(cfg, 'kill_switch_equity_dd', 0.0):.1%}. Trading halted."
                if getattr(cfg, "alerts_enabled", True) and allow_alert("KILL", cfg.alert_cooldown_sec):
                    ping(cfg.discord_webhook_url, msg)
                break

        try:
            inds = {
                s: compute(
                    df,
                    cfg.short_window,
                    cfg.long_window,
                    cfg.rsi_period,
                    cfg.roc_period,
                    cfg.trend_sma,
                    cfg.atr_period,
                )
                for s, df in hist.items()
            }
            universe = select_rotation(hist, cfg.rotation_lookback, cfg.rotation_top_n)

            cur.execute("SELECT COUNT(*) FROM positions")
            open_count = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM trades WHERE action='BUY' AND ts LIKE ?", (today + "%",))
            buys_today = int(cur.fetchone()[0])

            for sym in universe:
                # Hard market-hours guard
                if not _is_market_hours_now(cfg):
                    continue

                # DAILY mode guard
                if cfg.execution_mode == "daily" and not decision_once_per_day(conn, sym, today):
                    continue

                # Global regime filter applies to ALL symbols
                if getattr(cfg, "regime_filter", False) and not (regime_ok_latest and slope_ok_latest):
                    continue

                df_sym = hist.get(sym)
                if df_sym is None or df_sym.empty:
                    continue
                last = inds[sym].iloc[-1]

                trend_up = bool(last.get("trend_up", False))
                mom_ok = bool(last.get("mom_ok", False))
                rsi_val = float(last["rsi"]) if ("rsi" in last and pd.notna(last["rsi"])) else 50.0
                price = float(latest_close(sym))
                atr = float(last["atr"]) if "atr" in last and pd.notna(last["atr"]) else 0.0

                atr_pct = (atr / price) if price > 0 else 0.0
                if atr_pct < atr_pct_min:
                    continue

                if open_count >= cfg.max_positions:
                    continue

                # Momentum confirmation window
                try:
                    tail = inds[sym].tail(int(getattr(cfg, "mom_confirm_m", 1)))
                    pos_days = int((tail["mom_ok"].astype(bool)).sum())
                    if pos_days < int(getattr(cfg, "mom_confirm_k", 0)):
                        continue
                except Exception:
                    pass  # fail-open

                # If already have a position, update trailing/exit logic
                cur.execute(
                    "SELECT entry, qty, init_stop, trail_stop, highest_close FROM positions WHERE symbol=?",
                    (sym,),
                )
                row = cur.fetchone()
                have_pos = row is not None

                if have_pos:
                    entry, qty, init_stop, trail_stop, highest_close = row
                    highest_close = max(float(highest_close), price)
                    new_trail = max(float(init_stop), highest_close - float(getattr(cfg, "atr_trail_mult", 2.0)) * atr)

                    time_exit = False
                    if time_stop_days > 0:
                        cur.execute(
                            "SELECT ts FROM trades WHERE symbol=? AND action='BUY' ORDER BY ts DESC LIMIT 1",
                            (sym,),
                        )
                        r = cur.fetchone()
                        if r and r[0]:
                            try:
                                ent_ts = datetime.fromisoformat(r[0])
                                if (datetime.utcnow() - ent_ts).days >= time_stop_days:
                                    time_exit = True
                            except Exception:
                                pass

                    reason = None
                    if price <= new_trail:
                        reason = "ts"
                    elif getattr(cfg, "take_profit_pct", 0.0) > 0 and price >= float(entry) * (1 + float(cfg.take_profit_pct)):
                        reason = "tp"
                    elif rsi_val <= rsi_floor:
                        reason = "rsi"
                    elif time_exit:
                        reason = "time"

                    if reason:
                        # Enforce min notional on exits? Usually not necessary; always sell.
                        broker.market_sell(sym, int(qty))
                        cur.execute("DELETE FROM positions WHERE symbol=?", (sym,))
                        conn.commit()
                        cur.execute(
                            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
                            (dt.datetime.utcnow().isoformat(), "SELL", sym, price, int(qty), None, None, f"exit-{reason}"),
                        )
                        conn.commit()
                        open_count = max(open_count - 1, 0)
                        _send_sell_alert(cfg, sym, int(qty), price, reason)
                    else:
                        cur.execute(
                            "UPDATE positions SET trail_stop=?, highest_close=? WHERE symbol=?",
                            (float(new_trail), float(highest_close), sym),
                        )
                        conn.commit()
                    continue  # handled existing pos; go next symbol

                # New entry path
                if buys_today >= getattr(cfg, "max_daily_new_pos", 10**9):
                    continue

                if not (trend_up and mom_ok and rsi_val <= float(getattr(cfg, "rsi_pullback_max", 100.0))):
                    continue

                init_stop = price - float(getattr(cfg, "atr_mult_sl", 2.0)) * atr
                unit_risk = max(price - init_stop, 0.0)
                if unit_risk <= 0:
                    continue

                # Base sizing from your per-trade risk
                sizer = PositionSizer(broker.equity(), float(getattr(cfg, "risk_fraction", 0.01)))
                qty = sizer.shares_for_risk(price, init_stop)

                # Portfolio risk cap (database-backed positions)
                try:
                    port_cap = float(getattr(cfg, "portfolio_risk_cap", 0.0) or 0.0)
                except Exception:
                    port_cap = 0.0
                if port_cap > 0:
                    total_cap = float(broker.equity()) * port_cap
                    used = _total_open_risk_db(conn)
                    remaining = max(total_cap - used, 0.0)
                    qty_cap = int(remaining // unit_risk) if remaining > 0 else 0
                    qty = max(min(int(qty), qty_cap), 0)

                proposed_risk = float(qty) * unit_risk
                day_risk_cap = float(broker.equity()) * float(getattr(cfg, "max_day_risk_fraction", 1.0))
                if proposed_risk <= 0:
                    continue

                # Enforce min notional before sending the order
                if not _min_notional_ok(cfg, sym, price, int(qty)):
                    continue

                if (risk_today + proposed_risk) > day_risk_cap:
                    continue

                if qty > 0:
                    broker.market_buy(sym, int(qty))
                    cur.execute(
                        "INSERT INTO positions VALUES (?,?,?,?,?,?)",
                        (sym, float(price), int(qty), float(init_stop), float(init_stop), float(price)),
                    )
                    conn.commit()
                    open_count += 1
                    buys_today += 1
                    risk_today += proposed_risk

                    tp = price * (1 + float(getattr(cfg, "take_profit_pct", 0.0))) if float(getattr(cfg, "take_profit_pct", 0.0)) > 0 else None
                    cur.execute(
                        "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
                        (
                            dt.datetime.utcnow().isoformat(),
                            "BUY",
                            sym,
                            float(price),
                            int(qty),
                            float(init_stop),
                            tp,
                            "entry",
                        ),
                    )
                    conn.commit()
                    _send_buy_alert(cfg, sym, int(qty), price, init_stop)

        except Exception as e:
            if getattr(cfg, "alerts_enabled", True) and allow_alert("ERROR", cfg.alert_cooldown_sec):
                ping(cfg.discord_webhook_url, f"⚠️ Error: {e}")

        if cfg.execution_mode == "daily":
            break
        time.sleep(cfg.poll_seconds)


# ========================= SWEEP & WALK-FORWARD COMMANDS ========================

def run_sweep(symbols: List[str], start: str, end: str,
              grid_json: str | None, grid_path: str | None,
              score: str, cost_bps: float, jobs: int) -> None:
    data = _load_data_for_opt(symbols, start)
    grid = _parse_grid_arg(grid_json, grid_path)

    sw = sweep_once(
        data=data,
        start=start,
        end=end,
        grid=grid,
        score=score,
        cost_bps=cost_bps,
        n_jobs=jobs,
    )
    print("Best params:", sw["best_params"])
    print("Best KPIs:", json.dumps(sw["best_kpis"], indent=2))
    rows = []
    for p, k, _ in sw["ranked"]:
        rows.append({**k, **vars(p)})
    pd.DataFrame(rows).to_csv("sweep_results.csv", index=False)
    print("Saved sweep_results.csv")


def run_walkforward(symbols: List[str],
                    full_start: str, full_end: str,
                    train_years: int, test_years: int, step_years: int,
                    grid_json: str | None, grid_path: str | None,
                    score: str, cost_bps: float, jobs: int,
                    scheme: str = "rolling") -> None:
    data = _load_data_for_opt(symbols, full_start)
    grid = _parse_grid_arg(grid_json, grid_path)
    windows = make_walkforward_windows(
        full_start, full_end,
        train_years=train_years, test_years=test_years, step_years=step_years,
        scheme=scheme,
    )
    wf = walkforward(
        data=data, windows=windows, grid=grid,
        score=score, cost_bps=cost_bps, n_jobs=jobs
    )
    pd.DataFrame({"equity": wf["equity"]}).to_csv("wf_equity.csv", index=False)

    rows = []
    for pkt in wf["picks"]:
        w = pkt["window"]; p = pkt["params"]
        trk = pkt["train_kpis"]; tsk = pkt["test_kpis"]
        rows.append({
            "train_start": w.train_start, "train_end": w.train_end,
            "test_start": w.test_start,   "test_end": w.test_end,
            **vars(p),
            # train KPIs
            "train_CAGR": trk.get("CAGR", 0.0), "train_Sharpe": trk.get("Sharpe", 0.0),
            "train_MAR": trk.get("MAR", 0.0), "train_MaxDD": trk.get("MaxDD", 0.0),
            # test KPIs
            "test_CAGR": tsk.get("CAGR", 0.0), "test_Sharpe": tsk.get("Sharpe", 0.0),
            "test_MAR": tsk.get("MAR", 0.0), "test_MaxDD": tsk.get("MaxDD", 0.0),
        })
    pd.DataFrame(rows).to_csv("wf_windows.csv", index=False)
    pd.DataFrame(rows).to_csv("wf_picks.csv", index=False)

    print("Walk-forward KPIs:", json.dumps(wf["kpis"], indent=2))
    print("Saved wf_equity.csv, wf_windows.csv, wf_picks.csv")


# ==================================== CLI =====================================

def main() -> None:
    cfg = Settings()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Backtest
    bt = sub.add_parser("backtest")
    bt.add_argument("--symbols", default=None)
    bt.add_argument("--start", default=None)
    bt.add_argument("--cost_bps", type=float, default=2.5)

    # Trade
    sub.add_parser("trade")

    # Parameter sweep
    sw = sub.add_parser("sweep")
    sw.add_argument("--symbols", required=True, help="comma-separated symbols, e.g. SPY,GLD,TLT")
    sw.add_argument("--start", required=True)
    sw.add_argument("--end", required=True)
    sw.add_argument("--grid_json", default=None,
                    help='JSON dict or here-string; e.g. {"ATR_PCT_MIN":[0.01,0.02],"MAX_POSITIONS":[3,5]}')
    sw.add_argument("--grid_path", default=None, help="Path to a JSON file with the grid")
    sw.add_argument("--score", default="MAR", choices=["MAR", "Sharpe", "CAGR"])
    sw.add_argument("--cost_bps", type=float, default=2.5)
    sw.add_argument("--jobs", type=int, default=-1)

    # Walk-forward
    wf = sub.add_parser("backtest-wf")
    wf.add_argument("--symbols", required=True)
    wf.add_argument("--full_start", required=True)
    wf.add_argument("--full_end", required=True)
    wf.add_argument("--train_years", type=int, default=3)
    wf.add_argument("--test_years", type=int, default=1)
    wf.add_argument("--step_years", type=int, default=1)
    wf.add_argument("--grid_json", default=None)
    wf.add_argument("--grid_path", default=None, help="Path to a JSON file with the grid")
    wf.add_argument("--scheme", default="rolling", choices=["rolling", "expanding"])
    wf.add_argument("--score", default="MAR", choices=["MAR", "Sharpe", "CAGR"])
    wf.add_argument("--cost_bps", type=float, default=2.5)
    wf.add_argument("--jobs", type=int, default=-1)

    args = ap.parse_args()

    if args.cmd == "backtest":
        syms = [s.strip() for s in (args.symbols or ",".join(cfg.symbols)).split(",") if s.strip()]
        final_eq = backtest(syms, args.start or cfg.history_start, cfg, args.cost_bps)
        print(f"Backtest final equity (starting 10k): ${final_eq:,.2f}")

    elif args.cmd == "trade":
        trade(cfg)

    elif args.cmd == "sweep":
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        run_sweep(
            symbols=syms,
            start=args.start,
            end=args.end,
            grid_json=args.grid_json,
            grid_path=args.grid_path,
            score=args.score,
            cost_bps=args.cost_bps,
            jobs=int(args.jobs),
        )

    elif args.cmd == "backtest-wf":
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        run_walkforward(
            symbols=syms,
            full_start=args.full_start,
            full_end=args.full_end,
            train_years=int(args.train_years),
            test_years=int(args.test_years),
            step_years=int(args.step_years),
            grid_json=args.grid_json,
            grid_path=args.grid_path,
            score=args.score,
            cost_bps=args.cost_bps,
            jobs=int(args.jobs),
            scheme=args.scheme,
        )


if __name__ == "__main__":
    main()
