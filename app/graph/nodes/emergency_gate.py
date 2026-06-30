from __future__ import annotations

from app.core.logging import logger
from app.domain.dept_disambiguation import DeptDisambiguationState
from app.domain.models import AppState
from app.triage.emergency_rules import match_emergency


def emergency_gate_node(state: AppState) -> dict:
    logger.info(">>> Enter node: emergency_gate")
    text = (state.ner_result.query if state.ner_result else "") or ""
    hit = match_emergency(text)
    if hit is None:
        logger.info("emergency_gate: no match")
        return {"emergency_gate_passed": True}
    logger.info("emergency_gate: hit keyword=%r em_id=%s", hit.keyword, hit.em_id)
    return {
        "emergency_gate_passed": False,
        "locked_department": "急诊",
        "dept_state": DeptDisambiguationState(
            status="emergency",
            candidate_departments=[{"department": "急诊"}],
        ),
        "emergency_match": {"keyword": hit.keyword, "em_id": hit.em_id},
        "rag_chunk": hit.entry,
    }
