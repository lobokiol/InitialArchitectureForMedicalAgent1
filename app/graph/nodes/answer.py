from langchain_core.messages import AIMessage, HumanMessage

from app.core.logging import logger
from app.domain.models import AppState


def _new_intake_prefix(state: AppState) -> str:
    msgs = state.messages or []
    if len(msgs) < 3:
        return ""
    prior_humans = [
        m.content for m in msgs[:-1] if isinstance(m, HumanMessage) and isinstance(m.content, str)
    ]
    cur = (state.ner_result.query if state.ner_result else "") or ""
    if prior_humans and cur and cur.strip() != str(prior_humans[-1]).strip():
        return "（已按您本轮新描述重新评估）"
    return ""


def answer_generate_node(state: AppState) -> dict:
    logger.info(">>> Enter node: answer_generate")
    prefix = _new_intake_prefix(state)
    if prefix:
        prefix = prefix + "\n"

    if state.locked_department:
        chunk = state.rag_chunk or {}
        canonical = chunk.get("canonical_symptom") or (state.slot_table.primary_symptom if state.slot_table else "")
        dept = state.locked_department
        ds = state.dept_state
        status = getattr(ds, "status", None) if ds else None
        if dept == "急诊":
            from app.triage.emergency_rules import DEFAULT_EMERGENCY_REPLY

            detail = chunk.get("emergency_reply") or DEFAULT_EMERGENCY_REPLY
            full_content = f"建议尽快就诊：**急诊**。\n{detail}"
            return {"messages": [AIMessage(content=full_content)]}
        elif status == "fallback":
            full_content = (
                f"{prefix}根据您描述的症状（{canonical}），建议首选就诊：**{dept}**。"
                "如与实际情况不符，请补充更多细节以便更准确推荐。"
            )
        else:
            full_content = f"{prefix}根据您描述的症状（{canonical}），建议就诊科室：**{dept}**。"
        return {"messages": [AIMessage(content=full_content)]}

    ddr = state.disease_dept_result
    if ddr and ddr.departments:
        names = "、".join(ddr.diseases) if ddr.diseases else "您的描述"
        dept_names: list[str] = []
        for item in ddr.departments[:3]:
            if isinstance(item, dict):
                d = item.get("dept") or item.get("department") or ""
                if d:
                    dept_names.append(str(d))
            elif item:
                dept_names.append(str(item))
        dept_str = "、".join(dept_names) if dept_names else "导诊台"
        full_content = f"{prefix}根据您提到的「{names}」，建议就诊科室：**{dept_str}**。"
        return {"messages": [AIMessage(content=full_content)]}

    logger.info(
        "answer_generate_node: no locked dept / disease depts, using fixed fallback (LLM disabled)"
    )
    full_content = (
        f"{prefix}根据现有资料无法匹配导诊结果，建议先到**导诊台**咨询，或补充更具体的症状描述。"
    )
    return {"messages": [AIMessage(content=full_content)]}
