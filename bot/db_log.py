# bot/db_log.py
import os, sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, Iterable
import pandas as pd

def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None

def _get_columns(con: sqlite3.Connection, name: str) -> list[str]:
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info('{name}')").fetchall()]
    except Exception:
        return []

class DBLog:
    """
    Logs backtests to SQLite and adapts to either trade schema:
      A) trades(ts, action, symbol, price, qty, stop, take_profit, note)
      B) trades(ts, symbol, side, qty, price, pnl, note)
    Always creates 'equity(ts,equity)' if missing.
    """
    def __init__(self, path: Optional[str] = None):
        env_path = os.getenv("DB_PATH", "simple_bot.db")
        self.path: str = (path or env_path) or "simple_bot.db"
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self._ensure_tables()
        # cache columns
        self.trades_cols = _get_columns(self.con, "trades")
        self.equity_cols = _get_columns(self.con, "equity")

    def _ensure_tables(self):
        if not _table_exists(self.con, "trades"):
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    ts TEXT,
                    action TEXT,
                    symbol TEXT,
                    price REAL,
                    qty REAL,
                    stop REAL,
                    take_profit REAL,
                    note TEXT
                )
            """)
        if not _table_exists(self.con, "equity"):
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS equity (
                    ts TEXT,
                    equity REAL
                )
            """)
        self.con.commit()

    def _insert_row(self, table: str, row: Dict[str, Any]):
        cols = _get_columns(self.con, table)
        if not cols:
            return
        payload = {k: v for k, v in row.items() if k in cols}
        if not payload:
            return
        placeholders = ",".join(["?"] * len(payload))
        columns = ",".join(payload.keys())
        self.con.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(payload.values()),
        )
        self.con.commit()

    def log_trade(
        self,
        ts: datetime,
        symbol: str,
        side: str,  # 'BUY'/'SELL'
        qty: float,
        price: float,
        pnl: float = 0.0,
        note: str = "backtest",
        stop: float | None = None,
        take_profit: float | None = None,
    ):
        row = {
            "ts": ts.isoformat(),
            "symbol": symbol,
            "qty": float(qty),
            "price": float(price),
            "note": note,
        }

        if "action" in self.trades_cols:
            row["action"] = side
            if "stop" in self.trades_cols:
                row["stop"] = None if stop is None else float(stop)
            if "take_profit" in self.trades_cols:
                row["take_profit"] = None if take_profit is None else float(take_profit)
        if "side" in self.trades_cols:
            row["side"] = side
        if "pnl" in self.trades_cols:
            row["pnl"] = float(pnl)

        self._insert_row("trades", row)

    def log_equity_point(self, ts: datetime, equity: float):
        self._insert_row("equity", {"ts": ts.isoformat(), "equity": float(equity)})

    def log_equity_series(self, series: pd.Series | Iterable):
        if isinstance(series, pd.Series):
            df = pd.DataFrame({"ts": pd.to_datetime(series.index).astype(str), "equity": series.values})
        else:
            df = pd.DataFrame(series, columns=["ts", "equity"])
            df["ts"] = pd.to_datetime(df["ts"]).astype(str)
            df["equity"] = df["equity"].astype(float)
        if not df.empty:
            df.to_sql("equity", self.con, if_exists="append", index=False)
            self.con.commit()

    def close(self):
        self.con.close()
