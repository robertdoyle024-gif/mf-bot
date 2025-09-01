import sqlite3, pandas as pd
con = sqlite3.connect("simple_bot.db")
print("Tables:", con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
for tbl in ("trades","equity","positions"):
    try:
        df = pd.read_sql(f"SELECT * FROM {tbl} ORDER BY rowid DESC LIMIT 10", con)
        print(f"\n=== {tbl} ==="); print(df.iloc[::-1].to_string(index=False))
    except Exception: pass
con.close()
