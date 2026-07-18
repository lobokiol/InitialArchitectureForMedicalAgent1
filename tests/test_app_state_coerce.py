"""Lock AppState checkpoint coercion helpers."""

from app.domain.models import AppState, RetrievedDoc


def test_coerce_medical_docs_from_plain_dicts():
    state = AppState(medical_docs=[{"id": "1", "content": "发烧", "source": "medical"}])
    assert len(state.medical_docs) == 1
    assert isinstance(state.medical_docs[0], RetrievedDoc)
    assert state.medical_docs[0].content == "发烧"


def test_coerce_docs_unwraps_kwargs_checkpoint_shape():
    state = AppState(
        process_docs=[{"kwargs": {"id": "p1", "content": "挂号", "source": "process"}}]
    )
    assert len(state.process_docs) == 1
    assert state.process_docs[0].id == "p1"


def test_coerce_docs_skips_garbage_items():
    state = AppState(medical_docs=[None, "bad", {"id": "ok", "content": "x"}])
    assert len(state.medical_docs) == 1
    assert state.medical_docs[0].id == "ok"


def test_coerce_docs_none_becomes_empty_list():
    state = AppState(medical_docs=None, process_docs=None)  # type: ignore[arg-type]
    assert state.medical_docs == []
    assert state.process_docs == []
