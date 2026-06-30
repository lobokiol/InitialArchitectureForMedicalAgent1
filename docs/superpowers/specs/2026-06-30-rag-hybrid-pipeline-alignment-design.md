# RAG Hybrid Pipeline Alignment

**Date:** 2026-06-30  
**Status:** Approved (approach B)  
**Scope:** Align production `search_rag_knowledge` with indexing/acceptance hybrid search: shared BM25 clause, `search_pipeline` with min-max normalization, 0.4 BM25 + 0.6 kNN weights, and recalibrated clarify gate thresholds.

---

## 1. Problem

`app/infra/opensearch_rag.py` runs OpenSearch `hybrid` queries without `search_pipeline`. `sourceData/opensearch_rag_kb.py` defines min-max + weighted fusion and passes `search_pipeline`, but production never uses it. BM25 field boosts also differ between the two paths.

`RAG_CLARIFY_MIN_SCORE=1.2` was calibrated on raw (non-normalized) hybrid scores (~1.0x). Enabling min-max moves scores into ~0–1; thresholds must be retuned.

---

## 2. Decisions

| Topic | Choice |
|-------|--------|
| Scope | **B** — full alignment + shared module |
| Shared module | `app/infra/rag_hybrid_search.py` (imported by production + `opensearch_rag_kb.py`) |
| Pipeline name | `RAG_KB_HYBRID_PIPELINE` env, default `rag-knowledge-hybrid-pipeline` |
| Fusion | min-max normalization; arithmetic mean weights `[0.4, 0.6]` (BM25, kNN) |
| BM25 | Shared `_bm25_clause`: `canonical_symptom^5`, `alliance^4`, `description^2`, `search_text`, `term alliance boost:8` |
| Keyword fallback | Same `_bm25_clause` when embedding or hybrid fails |
| Pipeline bootstrap | Lazy `ensure_hybrid_pipeline` on first production hybrid search (idempotent PUT) |
| Thresholds | Re-run acceptance queries; update defaults + unit-test fixture scores to normalized scale |
| Alias pass | Unchanged — no score check when alias matches |

---

## 3. Architecture

```
rag_symptom_recall_node
  → search_rag_knowledge (opensearch_rag.py)
      → rag_hybrid_search.hybrid_search_body + ensure_hybrid_pipeline
      → client.search(..., params={search_pipeline})
  → rerank_by_alliance
  → _prefer_symptom_clarify (threshold gate)
```

---

## 4. Files

| File | Change |
|------|--------|
| `app/infra/rag_hybrid_search.py` | **New** — bm25_clause, pipeline body, ensure, hybrid body builder |
| `app/infra/opensearch_rag.py` | Use shared module; pass pipeline |
| `app/core/config.py` | `RAG_KB_HYBRID_PIPELINE`; update clarify threshold defaults |
| `sourceData/opensearch_rag_kb.py` | Import shared module; remove duplicates |
| `tests/test_rag_hybrid_search.py` | **New** — unit tests for clause/body shape |
| `tests/test_rag_symptom_recall.py` | Update mock scores to normalized scale |

---

## 5. Threshold retuning

Acceptance run (2026-06-30, OpenSearch 2.19.1, pipeline enabled):

| Query | Top1 | Score | #2 Score | Margin |
|-------|------|-------|----------|--------|
| 肚子疼 | CL0001 | 1.0000 | 0.0655 | 0.9345 |
| 脚疼 | CL0007 | 1.0000 | 0.1795 | 0.8205 |
| 神经疼 | CL0015 | 1.0000 | 0.0405 | 0.9595 |

Unknown probe `未知症状xyz`: top CL0012 **0.6391**, #2 **0.6000**, margin **0.0391** → fails margin gate.

**Chosen defaults:** `RAG_CLARIFY_MIN_SCORE=0.55`, `RAG_CLARIFY_MIN_MARGIN=0.10`

---

## 6. Error handling

- Embedding failure → keyword-only `_bm25_clause` (no pipeline needed).
- Hybrid + pipeline failure → log warning, retry keyword-only fallback.
- Pipeline PUT failure → log once; search may still fail; keyword fallback on next error.

---

## 7. Out of scope

- Changing `rerank_by_alliance` or alias-matching logic.
- Merging `rag_department_rules` into `rag_knowledge`.
- Index rebuild automation in app startup.
