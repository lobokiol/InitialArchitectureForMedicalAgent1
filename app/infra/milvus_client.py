from typing import List, Optional

from langchain_milvus import Milvus

from app.core import config
from app.core.logging import logger
from app.core.llm import get_embedding_model
from app.domain.models import RetrievedDoc

_milvus_store: Optional[Milvus] = None
_milvus_unavailable = False


def _get_milvus_store() -> Optional[Milvus]:
    global _milvus_store, _milvus_unavailable
    if _milvus_unavailable:
        return None
    if _milvus_store is not None:
        return _milvus_store
    try:
        _milvus_store = Milvus(
            embedding_function=get_embedding_model(),
            connection_args={"uri": config.MILVUS_URI},
            collection_name=config.MILVUS_COLLECTION,
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 16,
                    "efConstruction": 200,
                },
            },
        )
        return _milvus_store
    except Exception:
        logger.exception("Milvus 不可用，症状向量检索将跳过: %s", config.MILVUS_URI)
        _milvus_unavailable = True
        return None


def search_medical_docs(query: str) -> List[RetrievedDoc]:
    store = _get_milvus_store()
    if store is None:
        return []

    try:
        docs_and_scores = store.similarity_search_with_score(
            query,
            k=config.MILVUS_TOP_K,
        )
    except Exception:
        logger.exception("Milvus 查询失败")
        return []

    if not docs_and_scores:
        logger.info("search_medical_docs: no docs returned from Milvus")
        return []

    converted: List[RetrievedDoc] = []
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        rid = doc.metadata.get("id", "") if getattr(doc, "metadata", None) else ""
        title = doc.metadata.get("title") if getattr(doc, "metadata", None) else None

        logger.info(
            "search_medical_docs doc[%d]: id=%s, score=%s, snippet=%s",
            i,
            rid,
            score,
            doc.page_content[:50].replace("\n", " "),
        )

        converted.append(
            RetrievedDoc(
                id=rid,
                source="medical",
                title=title,
                content=doc.page_content,
                score=float(score) if score is not None else None,
            )
        )

    filtered = [
        d for d in converted
        if (d.score is None) or (d.score >= config.MILVUS_MIN_SIM)
    ]

    if not filtered and converted:
        filtered = converted[:1]

    filtered.sort(
        key=lambda d: (d.score is None, -(d.score or 0.0))
    )

    if len(filtered) > config.MILVUS_MAX_DOCS:
        filtered = filtered[:config.MILVUS_MAX_DOCS]

    logger.info(
        "search_medical_docs: got %d docs after filter & top%d (before=%d)",
        len(filtered),
        config.MILVUS_MAX_DOCS,
        len(converted),
    )
    return filtered
