from app.domain.models import AppState
from app.domain.routing import route_after_rag
from app.graph.nodes.rag_symptom_recall import _prefer_symptom_clarify


def _cl(id_: str, score: float, aliases: list[str] | None = None) -> dict:
    return {
        "id": id_,
        "type": "symptomClarify",
        "aliases": aliases or [],
        "_score": score,
    }


def test_alias_hit_ignores_threshold():
    hits = [_cl("CL0015", 0.1, ["神经疼", "神经痛"])]
    out = _prefer_symptom_clarify(hits, "神经疼", "神经疼")
    assert out is not None
    assert out["id"] == "CL0015"


def test_no_alias_clustered_scores_returns_none():
    hits = [
        _cl("CL0007", 1.07),
        _cl("CL0006", 1.06),
        _cl("CL0008", 1.04),
    ]
    assert _prefer_symptom_clarify(hits, "未知症状", "未知症状") is None


def test_no_alias_high_score_wide_margin_returns_top():
    hits = [_cl("CL0099", 3.0), _cl("CL0006", 0.5)]
    out = _prefer_symptom_clarify(hits, "未知症状", "未知症状")
    assert out is not None
    assert out["id"] == "CL0099"


def test_no_alias_high_score_narrow_margin_returns_none():
    hits = [_cl("CL0007", 3.0), _cl("CL0006", 2.95)]
    assert _prefer_symptom_clarify(hits, "未知症状", "未知症状") is None


def test_route_after_rag_empty_chunk():
    assert route_after_rag(AppState(rag_chunk=None)) == "rag_miss_reject"
    assert route_after_rag(AppState()) == "rag_miss_reject"


def test_route_after_rag_symptom_clarify():
    state = AppState(rag_chunk={"type": "symptomClarify", "id": "CL0001"})
    assert route_after_rag(state) == "symptom_clarify"


def test_route_after_rag_symptom_rk():
    state = AppState(rag_chunk={"type": "symptom", "id": "RK0001"})
    assert route_after_rag(state) == "dept_disambiguation"
