"""LLM 失败兜底：从 disease_kb / triage_templates 扫描 query 内子串。"""

from functools import lru_cache

from app.infra.disease_kb_store import disease_catalog_terms, symptom_catalog_terms
from app.ner.models import NERExtractOutput


@lru_cache(maxsize=1)
def load_entity_catalog() -> dict[str, list[str]]:
    return {
        "主症": symptom_catalog_terms(),
        "疾病": disease_catalog_terms(),
    }


def scan_catalog_substrings(
    query: str,
    symptom_terms: list[str],
    disease_terms: list[str],
) -> NERExtractOutput:
    symptoms = [t for t in symptom_terms if t and t in query]
    diseases = [t for t in disease_terms if t and t in query]
    symptoms.sort(key=lambda s: query.index(s))
    diseases.sort(key=lambda s: query.index(s))
    return NERExtractOutput(symptom_spans=symptoms, disease_spans=diseases)
