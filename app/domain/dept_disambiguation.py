from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class DeptChoice(BaseModel):
    id: str
    label: str
    target_departments: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class DeptDisambiguationState(BaseModel):
    candidate_departments: list[dict] = Field(default_factory=list)
    dept_scores: dict[str, float] = Field(default_factory=dict)
    round: int = 0
    status: Literal["scoring", "asking", "locked", "emergency", "fallback"] = "scoring"
    last_question: str | None = None
    last_choices: list[DeptChoice] = Field(default_factory=list)
    asked_choice_ids: list[str] = Field(default_factory=list)
    margin: float | None = None
    multi_select: bool = False
    choice_mode: Literal["accompany", "differential"] = "accompany"

    model_config = ConfigDict(extra="ignore")
