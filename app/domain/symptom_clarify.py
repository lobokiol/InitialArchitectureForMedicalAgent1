from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class ClarifyChoice(BaseModel):
    id: str
    label: str
    slot: str | None = None
    scores: dict[str, float] | None = None

    model_config = ConfigDict(extra="ignore")


class SymptomClarifyState(BaseModel):
    status: Literal["asking", "done"] = "asking"
    clarify_chunk_id: str | None = None
    symptom_id: str | None = None
    phase: Literal["age", "sex", "pain_location", "red_flags", "done"] = "age"
    filled_slots: dict[str, str] = Field(default_factory=dict)
    last_question: str | None = None
    last_choices: list[ClarifyChoice] = Field(default_factory=list)
    multi_select: bool = False
    dept_rule_id: str | None = None
    dept_rule_chunk: dict | None = None

    model_config = ConfigDict(extra="ignore")
