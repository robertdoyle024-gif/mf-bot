# bot/alerts.py
from __future__ import annotations
import time
from typing import Optional, Sequence, Tuple

import requests  # type: ignore[import-not-found]

_LAST_SEND: dict[str, float] = {}


def allow_alert(key: str, cooldown_sec: int) -> bool:
    """Return True if enough time elapsed since last alert with this key."""
    now = time.monotonic()
    last = _LAST_SEND.get(key, 0.0)
    if now - last < max(cooldown_sec, 0):
        return False
    _LAST_SEND[key] = now
    return True


def _post_with_retry(url: str, payload: dict, retries: int = 3, timeout: int = 7) -> None:
    backoff = 1.0
    for _ in range(max(retries, 1)):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if 200 <= r.status_code < 300 or r.status_code == 204:  # Discord returns 204 on success
                return
            if r.status_code == 429:
                try:
                    retry_after = float(r.json().get("retry_after", 1.0))
                except Exception:
                    retry_after = 1.0
                time.sleep(retry_after + 0.25)
                continue
        except Exception:
            pass
        time.sleep(backoff)
        backoff *= 1.6


def ping(webhook_url: Optional[str], text: str, username: str = "MF Bot") -> None:
    if not webhook_url:
        return
    _post_with_retry(webhook_url, {"content": text, "username": username})


def ping_embed(
    webhook_url: Optional[str],
    title: str,
    description: Optional[str] = None,
    fields: Optional[Sequence[Tuple[str, str, bool]]] = None,
    color: int = 0x2ECC71,
    username: str = "MF Bot",
) -> None:
    if not webhook_url:
        return
    embed = {"title": title, "color": color}
    if description:
        embed["description"] = description
    if fields:
        embed["fields"] = [{"name": n, "value": v, "inline": bool(i)} for n, v, i in fields]
    _post_with_retry(webhook_url, {"username": username, "embeds": [embed]})
