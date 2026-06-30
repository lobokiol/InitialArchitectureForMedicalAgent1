# RAG Hybrid Pipeline Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align production `search_rag_knowledge` with indexing hybrid search (shared BM25, min-max pipeline, 0.4/0.6 weights) and retune clarify thresholds.

**Architecture:** Extract `app/infra/rag_hybrid_search.py`; production and `opensearch_rag_kb.py` import it; lazy pipeline ensure on first hybrid search.

**Tech Stack:** OpenSearch 2.x hybrid query, normalization-processor, DashScope embeddings, Python 3.11

## Global Constraints

- Pipeline name: `RAG_KB_HYBRID_PIPELINE` default `rag-knowledge-hybrid-pipeline`
- Fusion weights: BM25 0.4, kNN 0.6, min-max normalization
- Threshold defaults: `RAG_CLARIFY_MIN_SCORE=0.55`, `RAG_CLARIFY_MIN_MARGIN=0.10`

---

## Task 1: Shared hybrid module — **done**

- [x] Create `app/infra/rag_hybrid_search.py`
- [x] Unit tests in `tests/test_rag_hybrid_search.py`

## Task 2: Wire production + indexing — **done**

- [x] Update `app/infra/opensearch_rag.py`
- [x] Refactor `sourceData/opensearch_rag_kb.py`
- [x] Add `RAG_KB_HYBRID_PIPELINE` to `app/core/config.py`

## Task 3: Threshold + tests — **done**

- [x] Update `RAG_CLARIFY_*` defaults
- [x] Update `tests/test_rag_symptom_recall.py` fixture scores
- [x] Run `pytest tests/test_rag_hybrid_search.py tests/test_rag_symptom_recall.py`
- [x] Run `opensearch_rag_kb.py --acceptance` (3/3 pass)
