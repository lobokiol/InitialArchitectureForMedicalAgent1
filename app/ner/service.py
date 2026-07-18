from langchain_core.messages import HumanMessage

from app.core.llm import get_chat_llm
from app.core.logging import logger
from app.ner.catalog_scan import load_entity_catalog, scan_catalog_substrings
from app.ner.extract import build_entity_result
from app.ner.models import EntityExtractResult, NERExtractOutput
from app.ner.prompts import NER_EXTRACT_PROMPT


def _llm_extract(query: str) -> NERExtractOutput | None:
    try:
        llm = get_chat_llm().with_structured_output(NERExtractOutput)
        prompt = NER_EXTRACT_PROMPT.format(query=query)
        result = llm.invoke([HumanMessage(content=prompt)])
        if isinstance(result, NERExtractOutput):
            return result
        return NERExtractOutput.model_validate(result)
    except Exception:
        logger.exception("NER LLM extract failed, fallback to catalog scan")
        return None


def extract_entity_tags(
    query: str,
    catalog: dict[str, list[str]] | None = None,
) -> EntityExtractResult:
    """严格子串实体提取：LLM span + 规则选主项；失败则词典子串扫描兜底。"""
    q = (query or "").strip()
    cat = catalog if catalog is not None else load_entity_catalog()

    raw = _llm_extract(q)
    if raw is None:
        raw = scan_catalog_substrings(q, cat["主症"], cat["疾病"])

    result = build_entity_result(q, raw)
    logger.info(
        "NER extract query=%r primary_symptom=%s primary_disease=%s companions_s=%s companions_d=%s",
        q,
        result.primary_symptom,
        result.primary_disease,
        result.companion_symptoms,
        result.companion_diseases,
    )
    return result
