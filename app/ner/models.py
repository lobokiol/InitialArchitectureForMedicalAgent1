from pydantic import BaseModel, Field, ConfigDict


class NERExtractOutput(BaseModel):
    """LLM 结构化输出：原句子串列表（未分主/伴）。"""

    symptom_spans: list[str] = Field(
        default_factory=list,
        description="用户原句中的症状/不适连续子串",
    )
    disease_spans: list[str] = Field(
        default_factory=list,
        description="用户原句中的疾病名连续子串",
    )

    model_config = ConfigDict(extra="ignore")


class EntityExtractResult(BaseModel):
    """decision NER 四字段输出。"""

    query: str
    primary_symptom: str | None = None
    companion_symptoms: list[str] = Field(default_factory=list)
    primary_disease: str | None = None
    companion_diseases: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @property
    def has_symptom(self) -> bool:
        return self.primary_symptom is not None or bool(self.companion_symptoms)

    @property
    def has_disease(self) -> bool:
        return self.primary_disease is not None or bool(self.companion_diseases)

    @property
    def all_symptoms(self) -> list[str]:
        out: list[str] = []
        if self.primary_symptom:
            out.append(self.primary_symptom)
        out.extend(self.companion_symptoms)
        return out

    @property
    def all_diseases(self) -> list[str]:
        out: list[str] = []
        if self.primary_disease:
            out.append(self.primary_disease)
        out.extend(self.companion_diseases)
        return out

    # --- 下游兼容别名 ---
    @property
    def chief_symptom(self) -> str | None:
        return self.primary_symptom

    @property
    def symptom_candidates(self) -> list[str]:
        return self.all_symptoms

    @property
    def diseases(self) -> list[str]:
        return self.all_diseases


# 兼容旧名
NERExtractResult = EntityExtractResult


class NERExtractState(BaseModel):
    """独立 NER 测试图状态，与 AppState 隔离。"""

    query: str = ""
    primary_symptom: str | None = None
    companion_symptoms: list[str] = Field(default_factory=list)
    primary_disease: str | None = None
    companion_diseases: list[str] = Field(default_factory=list)
    error: str | None = None

    model_config = ConfigDict(extra="ignore")
