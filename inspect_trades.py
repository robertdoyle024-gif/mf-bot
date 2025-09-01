import glob, os, sqlite3, pandas as pd

cands = glob.glob("**/*.db", recursive=True) + glob.glob("**/*.sqlite*", recursive=True)
assert cands, "No DB found. Run a backtest/trade first."

db = max(cands, key=os.path.getmtime)
con = sqlite3.connect(db)

for tbl in ("trades","orders","executions","trade_log","positions"):
    try:
        df = pd.read_sql(f"SELECT * FROM {tbl} ORDER BY rowid DESC LIMIT 15", con)
        print(f"\n=== {tbl} ({db}) ===")
        print(df.iloc[::-1].to_string(index=False))
    except Exception:
        pass

con.close()
