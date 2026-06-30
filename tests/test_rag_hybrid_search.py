from app.infra.rag_hybrid_search import (
    BM25_WEIGHT,
    KNN_WEIGHT,
    bm25_clause,
    hybrid_pipeline_body,
    hybrid_search_body,
    keyword_search_body,
)


def test_bm25_clause_has_expected_fields_and_boosts():
    clause = bm25_clause("肚子疼")
    should = clause["bool"]["should"]
    multi = should[0]["multi_match"]
    assert multi["query"] == "肚子疼"
    fields = multi["fields"]
    assert "canonical_symptom^5" in fields
    assert "alliance^4" in fields
    assert should[1]["term"]["alliance"]["boost"] == 8


def test_hybrid_pipeline_weights():
    body = hybrid_pipeline_body()
    proc = body["phase_results_processors"][0]["normalization-processor"]
    assert proc["normalization"]["technique"] == "min_max"
    weights = proc["combination"]["parameters"]["weights"]
    assert weights == [BM25_WEIGHT, KNN_WEIGHT]


def test_hybrid_search_body_structure():
    body = hybrid_search_body("脚疼", [0.1, 0.2], k=5)
    queries = body["query"]["hybrid"]["queries"]
    assert queries[0] == bm25_clause("脚疼")
    assert queries[1]["knn"]["embedding"]["k"] == 5
    assert body["size"] == 5


def test_keyword_search_body_matches_bm25_clause():
    body = keyword_search_body("神经疼", k=3)
    assert body == {"query": bm25_clause("神经疼"), "size": 3}
