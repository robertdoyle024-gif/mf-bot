import sqlite3

db = "simple_bot.db"

con = sqlite3.connect(db)
tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables in", db, ":", tables)

for (name,) in tables:
    try:
        rows = con.execute(f"SELECT * FROM {name} LIMIT 5").fetchall()
        print(f"\n=== {name} (showing up to 5 rows) ===")
        for row in rows:
            print(row)
    except Exception as e:
        print(f"\n{name} error:", e)

con.close()
