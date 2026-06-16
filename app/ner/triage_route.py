"""导诊路由：实体提取结果 → disease / symptom / reject。"""

from app.domain.triage_intent import TriageRoute
from app.ner.models import EntityExtractResult


def resolve_triage_route(result: EntityExtractResult) -> TriageRoute:
    """
    1. 有疾病（primary 或 companion）→ disease
    2. 无疾病、有症状 → symptom
    3. 否则 → reject
    """
    if result.has_disease:
        return "disease"
    if result.has_symptom:
        return "symptom"
    return "reject"
