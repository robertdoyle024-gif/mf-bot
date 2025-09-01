
import requests

def ping(url: str, msg: str):
    if not url: return
    try:
        requests.post(url, json={"content": msg}, timeout=10)
    except Exception:
        pass
