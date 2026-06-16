"""OpenSearch index mappings for demo knowledge bases."""

from __future__ import annotations

from typing import Any

# DashScope text-embedding-v2
EMBEDDING_DIM = 1536

_KNN_VECTOR = {
    "type": "knn_vector",
    "dimension": EMBEDDING_DIM,
    "method": {
        "name": "hnsw",
        "space_type": "cosinesimil",
        "engine": "lucene",
        "parameters": {"ef_construction": 128, "m": 16},
    },
}

_KNN_INDEX_SETTINGS: dict[str, Any] = {
    "index": {
        "knn": True,
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }
}


def rag_knowledge_index_body() -> dict[str, Any]:
    """
    Mapping for demo/data/rag_knowledge.jsonl — keyword + semantic (hybrid-ready).

    - keyword / text: BM25 on alliance, canonical_symptom, search_text
    - embedding: kNN for semantic recall
    - object fields: stored for routing, not full-text indexed by default
    """
    return {
        "settings": _KNN_INDEX_SETTINGS,
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "body_part": {"type": "keyword"},
                "gender": {"type": "keyword"},
                "age": {"type": "keyword"},
                "version": {"type": "keyword"},
                "canonical_symptom": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "raw_question": {"type": "text"},
                "description": {"type": "text"},
                "alliance": {"type": "keyword"},
                "raw_json": {"type": "text", "index": False},
                "search_text": {"type": "text"},
                "accompanying_symptom_keywords": {"type": "keyword"},
                "differential_exclusion_terms": {"type": "keyword"},
                "embedding": _KNN_VECTOR,
                "department_recommendation": {
                    "type": "nested",
                    "properties": {
                        "department": {"type": "keyword"},
                        "priority": {"type": "integer"},
                        "condition": {"type": "text"},
                    },
                },
                "emergency_flag": {
                    "properties": {
                        "suggestion": {"type": "keyword"},
                        "condition": {"type": "text"},
                    }
                },
            }
        },
    }


def triage_templates_index_body() -> dict[str, Any]:
    """Keyword-only mapping (no vectors) for demo/data/triage_templates.jsonl."""
    return {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "canonical_symptom": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "raw_question": {"type": "text"},
                "description": {"type": "text"},
                "alliance": {"type": "keyword"},
                "template": {"type": "keyword"},
                "version": {"type": "keyword"},
                "search_text": {"type": "text"},
            }
        },
    }


def disease_kb_index_body() -> dict[str, Any]:
    """Keyword mapping for demo/data/disease_kb.jsonl — alias / canonical → departments."""
    return {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "canonical_disease": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "aliases": {"type": "keyword"},
                "description": {"type": "text"},
                "departments": {"type": "keyword"},
                "search_text": {"type": "text"},
                "version": {"type": "keyword"},
            }
        },
    }


def hospital_procedures_index_body() -> dict[str, Any]:
    """Keyword-only mapping for hospital process / FAQ docs."""
    return {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "hospital": {"type": "keyword"},
                "scene": {"type": "text"},
                "department": {"type": "keyword"},
                "process_type": {"type": "keyword"},
                "raw_text": {"type": "text"},
                "source_file": {"type": "keyword"},
                "page_range": {"type": "keyword"},
            }
        },
    }
