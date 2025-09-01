import os, sqlite3
from dotenv import load_dotenv

load_dotenv()
db = os.getenv("DB_PATH", "simple_bot.db")
print("DB_PATH =", db, "| Exists:", os.path.exists(db))
if os.path.exists(db):
    con = sqlite3.connect(db)
    print("Tables:", con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
    con.close()
