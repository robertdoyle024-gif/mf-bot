# bot/count_trades.py
import sqlite3, os
db = os.getenv("DB_PATH", "simple_bot.db")
conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM trades")
n = cur.fetchone()[0]
print("trades rows:", n)
cur.execute("SELECT COUNT(*) FROM positions")
m = cur.fetchone()[0]
print("positions rows:", m)
conn.close()
