from typing import Annotated, Any, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DiseaseDeptResult(BaseModel):
    """疾病链输出：疾病 → 推荐科室。"""

    diseases: list[str] = Field(default_factory=list)
    departments: list[dict] = Field(default_factory=list)
    route: Literal["disease"] = "disease"

    model_config = ConfigDict(extra="ignore")


class SymptomSlotResult(BaseModel):
    """症状链入口输出：主症 + 槽位表代号。"""

    route: Literal["symptom"] = "symptom"
    chief_symptom: str | None = None
    chief_symptom_canonical: str | None = None
    slot_table_code: str | None = None
    symptom_candidates: list[str] = Field(default_factory=list)
    companion_symptoms: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class IntentResult(BaseModel):
    """解析用户意图的结果（LLM structured output）。"""

    has_symptom: bool = False
    has_process: bool = False
    main_intent: Literal["symptom", "process", "mixed", "non_medical"] = "non_medical"
    symptom_query: Optional[str] = None
    process_query: Optional[str] = None
    need_symptom_search: bool = False
    need_process_search: bool = False
    need_tool_call: bool = False
    triage_route: Literal["disease", "symptom", "reject"] | None = None
    model_config = ConfigDict(extra="ignore")


class RetrievedDoc(BaseModel):
    """从向量数据库或工具调用中检索到的文档。"""

    id: str = ""
    source: Literal["medical", "process", "tool"] = "medical"
    title: Optional[str] = None
    content: str = ""
    score: Optional[float] = None

    model_config = ConfigDict(extra="ignore")


class OnCallDoctor(BaseModel):
    name: str
    time: str
    slots: int

    model_config = ConfigDict(extra="ignore")


class DepartmentIntro(BaseModel):
    department: str
    summary: str
    scope: list[str] = Field(default_factory=list)
    visit_tips: str = ""
    floor: str = ""
    phone: str = ""

    model_config = ConfigDict(extra="ignore")


class DepartmentRoute(BaseModel):
    department: str
    from_location: str
    to: str
    estimated_minutes: int
    steps: list[str] = Field(default_factory=list)
    landmarks: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class RelevanceResult(BaseModel):
    """LLM 判断检索到的文档与用户问题的相关性结果。"""

    can_answer_overall: bool = False
    need_rewrite_symptom: bool = False
    need_rewrite_process: bool = False
    reason: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


def _coerce_retrieved_docs(v: Any) -> list[RetrievedDoc]:
    """Checkpoint-safe coercion for medical/process doc lists."""
    if v is None:
        return []
    if not isinstance(v, list):
        return []
    out: list[RetrievedDoc] = []
    for item in v:
        if isinstance(item, RetrievedDoc):
            out.append(item)
        elif isinstance(item, dict):
            if "kwargs" in item and isinstance(item["kwargs"], dict):
                item = item["kwargs"]
            try:
                out.append(RetrievedDoc(**item))
            except Exception:
                continue
    return out


class AppState(BaseModel):
    """应用状态，在图中流转，包含整个对话的上下文信息。"""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    intent_result: Optional[IntentResult] = None
    ner_result: Optional[Any] = None  # NERExtractResult，避免循环 import
    disease_dept_result: Optional[DiseaseDeptResult] = None
    symptom_slot_result: Optional[SymptomSlotResult] = None
    medical_docs: List[RetrievedDoc] = Field(default_factory=list)
    process_docs: List[RetrievedDoc] = Field(default_factory=list)
    relevance_result: Optional[RelevanceResult] = None
    rewrite_attempts: int = 0
    need_tool_call: bool = False
    tool_call_result: Any = None
    slot_table: Optional[Any] = None
    slot_gate_passed: bool = False
    rag_chunk_id: str | None = None
    rag_chunk: dict | None = None
    dept_state: Optional[Any] = None
    locked_department: str | None = None
    clarify_state: Optional[Any] = None
    dept_confidence_result: Optional[Any] = None
    dept_confidence_passed: bool | None = None
    emergency_gate_passed: bool | None = None
    emergency_match: dict | None = None
    oncall_appointments: list[OnCallDoctor] = Field(default_factory=list)
    oncall_fetch_error: str | None = None
    last_recommended_department: str | None = None

    @field_validator("medical_docs", mode="before")
    @classmethod
    def _coerce_medical_docs(cls, v):
        return _coerce_retrieved_docs(v)

    @field_validator("process_docs", mode="before")
    @classmethod
    def _coerce_process_docs(cls, v):
        return _coerce_retrieved_docs(v)

    @field_validator("relevance_result", mode="before")
    @classmethod
    def _coerce_relevance_result(cls, v):
        if v is None:
            return None
        if isinstance(v, RelevanceResult):
            return v
        if isinstance(v, dict):
            if "kwargs" in v and isinstance(v["kwargs"], dict):
                v = v["kwargs"]
            try:
                return RelevanceResult(**v)
            except Exception:
                return None
        return None
