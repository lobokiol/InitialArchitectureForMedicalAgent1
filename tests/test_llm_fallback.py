import pytest
from langchain_core.messages import AIMessage

from app.core.llm import FallbackChatModel


class _FakeLLM:
    def __init__(self, name: str, *, fail: bool = False, reply: str = "ok") -> None:
        self.name = name
        self._fail = fail
        self._reply = reply
        self.invoke_calls = 0

    def invoke(self, _msgs):
        self.invoke_calls += 1
        if self._fail:
            raise RuntimeError(f"{self.name} unavailable")
        return AIMessage(content=self._reply)

    def with_structured_output(self, _schema):
        return self


def test_invoke_uses_primary_when_ok():
    primary = _FakeLLM("primary", reply="from-primary")
    fallback = _FakeLLM("fallback", reply="from-fallback")
    model = FallbackChatModel(
        primary_name="primary",
        fallback_name="fallback",
        primary=primary,  # type: ignore[arg-type]
        fallback=fallback,  # type: ignore[arg-type]
    )

    result = model.invoke([])

    assert result.content == "from-primary"
    assert primary.invoke_calls == 1
    assert fallback.invoke_calls == 0


def test_invoke_falls_back_on_primary_failure():
    primary = _FakeLLM("primary", fail=True)
    fallback = _FakeLLM("fallback", reply="from-fallback")
    model = FallbackChatModel(
        primary_name="primary",
        fallback_name="fallback",
        primary=primary,  # type: ignore[arg-type]
        fallback=fallback,  # type: ignore[arg-type]
    )

    result = model.invoke([])

    assert result.content == "from-fallback"
    assert primary.invoke_calls == 1
    assert fallback.invoke_calls == 1


def test_invoke_raises_when_primary_fails_and_no_fallback():
    primary = _FakeLLM("primary", fail=True)
    model = FallbackChatModel(
        primary_name="primary",
        fallback_name=None,
        primary=primary,  # type: ignore[arg-type]
        fallback=None,
    )

    with pytest.raises(RuntimeError, match="primary unavailable"):
        model.invoke([])


def test_with_structured_output_falls_back():
    primary = _FakeLLM("primary", fail=True)
    fallback = _FakeLLM("fallback", reply="structured-fallback")
    model = FallbackChatModel(
        primary_name="primary",
        fallback_name="fallback",
        primary=primary,  # type: ignore[arg-type]
        fallback=fallback,  # type: ignore[arg-type]
    )

    structured = model.with_structured_output(dict)
    result = structured.invoke([])

    assert result.content == "structured-fallback"
    assert primary.invoke_calls == 1
    assert fallback.invoke_calls == 1
