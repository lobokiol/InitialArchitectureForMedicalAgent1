import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.core.llm import get_chat_llm
from app.core.logging import logger
from app.ner.catalog_scan import load_entity_catalog, scan_catalog_substrings
from app.ner.extract import build_entity_result
from app.ner.models import EntityExtractResult, NERExtractOutput

NER_EXTRACT_PROMPT = """
你是医疗导诊的实体抽取模块。从用户描述中摘出 **原句连续子串**，归入两类：

1. **symptom_spans**：症状、不适、体征（如「心慌」「手抖」「肚脐上方疼」）
2. **disease_spans**：用户明确提到的病名（如「胃炎」）

## 硬性规则
- 每个条目必须是用户原句中 **逐字连续出现** 的子串，禁止改写、同义词替换、合并或补充
- 不要把疾病放进 symptom_spans，不要把症状放进 disease_spans
- 同一子串不要重复
- 不要输出主/伴分类，只输出两个列表
- 用户未提到的实体不要输出
- 只返回 JSON

## 用户描述
{query}
"""


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
