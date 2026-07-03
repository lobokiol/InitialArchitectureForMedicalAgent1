import pytest

from app.gateway.phone import normalize_phone


def test_normalize_cn_mobile():
    assert normalize_phone("13800138000") == "+8613800138000"


def test_normalize_already_e164():
    assert normalize_phone("+8613800138000") == "+8613800138000"


def test_normalize_rejects_invalid():
    with pytest.raises(ValueError):
        normalize_phone("123")
