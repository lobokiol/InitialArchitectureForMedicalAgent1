from __future__ import annotations

import json

from app.domain.models import AppState


def slots_for_confidence(state: AppState) -> dict[str, str]:
    cs = state.clarify_state
    if not cs:
        return {}
    allow = {"age", "sex", "pain_location", "differential"}
    return {k: v for k, v in (cs.filled_slots or {}).items() if k in allow}


def build_confidence_prompt(state: AppState) -> str:
    slots = slots_for_confidence(state)
    table = state.slot_table
    cs = state.clarify_state
    rule = cs.dept_rule_chunk if cs else None
    payload = {
        "locked_department": state.locked_department,
        "primary_symptom": getattr(table, "primary_symptom", None) if table else None,
        "companion_symptoms": getattr(table, "companion_symptoms", None) if table else None,
        "filled_slots": slots,
        "dept_rule": {
            "symptom_id": (rule or {}).get("symptom_id"),
            "location": (rule or {}).get("location"),
            "candidate_departments": (rule or {}).get("candidate_departments"),
        }
        if rule
        else None,
    }
    return f"""你是医疗导诊质控模块。评估推荐科室与用户槽位信息的一致性，输出 0-100 的置信度分数。

规则：
- 槽位越完整、症状与科室匹配越强，分数越高
- 信息不足、仅依赖「都没有」或模糊描述时分数应偏低
- 不要根据 red_flags 评分（输入中不包含 red_flags）

输入 JSON：
{json.dumps(payload, ensure_ascii=False, indent=2)}

仅返回 JSON：score（0-100 浮点数）、reason（简短中文）、slot_alignment（科室与槽位一致性简述）。"""
