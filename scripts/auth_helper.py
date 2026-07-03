"""Shared JWT login helper for CLI and eval scripts."""

from __future__ import annotations

import requests

DEFAULT_EVAL_PHONE = "13900000001"
DEFAULT_EVAL_PASSWORD = "eval-pass-123"


def login(base_url: str, phone: str, password: str, *, timeout: float = 30) -> str:
    """POST /auth/login and return access_token."""
    url = base_url.rstrip("/")
    r = requests.post(
        f"{url}/auth/login",
        json={"phone": phone, "password": password},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def register_or_login(
    base_url: str,
    phone: str,
    password: str,
    *,
    timeout: float = 30,
    display_name: str | None = None,
) -> str:
    """Login; on 401 register then return access_token."""
    url = base_url.rstrip("/")
    try:
        return login(url, phone, password, timeout=timeout)
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 401:
            raise
    payload: dict = {"phone": phone, "password": password}
    if display_name:
        payload["display_name"] = display_name
    r = requests.post(f"{url}/auth/register", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["access_token"]


def bearer_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def authed_session(
    base_url: str,
    phone: str,
    password: str,
    *,
    timeout: float = 30,
    display_name: str | None = None,
    proxies: dict | None = None,
) -> tuple[requests.Session, str, str]:
    """Return (session with Bearer header, access_token, normalized phone from /auth/me)."""
    url = base_url.rstrip("/")
    token = register_or_login(url, phone, password, timeout=timeout, display_name=display_name)
    sess = requests.Session()
    sess.trust_env = False
    if proxies is not None:
        sess.proxies.update(proxies)
    sess.headers.update(bearer_headers(token))
    me = sess.get(f"{url}/auth/me", timeout=timeout)
    me.raise_for_status()
    normalized_phone = me.json()["phone"]
    return sess, token, normalized_phone
