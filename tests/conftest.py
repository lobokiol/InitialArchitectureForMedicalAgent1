import pytest

from app.gateway.rate_limit import limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    limiter.reset()
    yield
    limiter.reset()
