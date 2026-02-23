from typing import List

from langchain_milvus import Milvus

from app.core import config
from app.core.logging import logger
from app.core.llm import get_embedding_model
from app.domain.models import RetrievedDoc


milvus_store = Milvus(
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


def search_medical_docs(query: str) -> List[RetrievedDoc]:
    docs_and_scores = milvus_store.similarity_search_with_score(
        query,
        k=config.MILVUS_TOP_K,
    )

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
