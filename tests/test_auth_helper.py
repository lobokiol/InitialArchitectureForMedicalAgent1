"""Tests for scripts.auth_helper."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from scripts.auth_helper import bearer_headers, register_or_login


def test_bearer_headers():
    assert bearer_headers("tok") == {"Authorization": "Bearer tok"}


def test_register_or_login_uses_login_when_ok(monkeypatch):
    calls: list[str] = []

    def fake_post(url, json=None, timeout=30):
        calls.append(url)
        resp = MagicMock()
        resp.json.return_value = {"access_token": "abc"}
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    token = register_or_login("http://localhost:8000", "13800138000", "secret123")
    assert token == "abc"
    assert calls == ["http://localhost:8000/auth/login"]


def test_register_or_login_registers_on_401(monkeypatch):
    def fake_post(url, json=None, timeout=30):
        resp = MagicMock()
        if url.endswith("/auth/login"):
            err = requests.HTTPError()
            err.response = MagicMock(status_code=401)
            resp.raise_for_status.side_effect = err
            return resp
        resp.json.return_value = {"access_token": "new"}
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    token = register_or_login("http://localhost:8000", "13800138000", "secret123")
    assert token == "new"
