"""实体提取管线：span 列表 → EntityExtractResult。"""

from app.ner.models import EntityExtractResult, NERExtractOutput
from app.ner.span_utils import process_spans


def build_entity_result(query: str, raw: NERExtractOutput) -> EntityExtractResult:
    primary_symptom, companion_symptoms = process_spans(raw.symptom_spans, query)
    primary_disease, companion_diseases = process_spans(raw.disease_spans, query)
    return EntityExtractResult(
        query=query,
        primary_symptom=primary_symptom,
        companion_symptoms=companion_symptoms,
        primary_disease=primary_disease,
        companion_diseases=companion_diseases,
    )
