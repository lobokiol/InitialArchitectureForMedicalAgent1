"""Shared JWT login helper for CLI and eval scripts."""

from __future__ import annotations

import requests


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
