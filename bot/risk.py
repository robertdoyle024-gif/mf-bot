
from dataclasses import dataclass
import sqlite3, os

@dataclass
class PositionSizer:
    equity: float
    risk_fraction: float

    def shares_for_risk(self, entry: float, stop: float) -> int:
        risk_per_share = max(entry - stop, 0.0)
        if risk_per_share <= 0:
            return 0
        dollars = max(self.equity * self.risk_fraction, 0.0)
        return max(int(dollars // risk_per_share), 1)

def init_db(path: str):
    import sqlite3
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades(
        ts TEXT, action TEXT, symbol TEXT, price REAL, qty INTEGER, stop REAL, take_profit REAL, note TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS positions(
        symbol TEXT PRIMARY KEY, entry REAL, qty INTEGER, init_stop REAL, trail_stop REAL, highest_close REAL
    )""")
    conn.commit()
    return conn
