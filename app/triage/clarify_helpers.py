from __future__ import annotations

from app.domain.symptom_clarify import ClarifyChoice


def choices_for_slot(cl_chunk: dict, slot: str) -> tuple[str, list[ClarifyChoice]]:
    q = (cl_chunk.get("questions") or {}).get(slot) or {}
    text = q.get("text") or ""
    options = q.get("options") or []
    choices = [ClarifyChoice(id=f"c{i}", label=lb, slot=slot) for i, lb in enumerate(options, 1)]
    return text, choices


def format_clarify_message(question: str, choices: list[ClarifyChoice]) -> str:
    lines = [question, ""]
    for i, c in enumerate(choices, 1):
        lines.append(f"{i}. {c.label}")
    return "\n".join(lines)


_SLOT_ORDER = ["age", "sex", "pain_location", "red_flags"]


def ordered_required_slots(required_slots: list[str]) -> list[str]:
    return [s for s in _SLOT_ORDER if s in required_slots]


def next_slot_phase(current: str, required_slots: list[str]) -> str | None:
    slots = ordered_required_slots(required_slots)
    if current not in slots:
        return slots[0] if slots else None
    idx = slots.index(current)
    return slots[idx + 1] if idx + 1 < len(slots) else None
