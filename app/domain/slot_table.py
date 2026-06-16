from pydantic import BaseModel, Field, ConfigDict


class TriageSlotTable(BaseModel):
    gender: str = "男"
    age: str = "30岁"
    primary_symptom: str | None = None
    companion_symptoms: list[str] = Field(default_factory=list)
    primary_disease: str | None = None
    companion_diseases: list[str] = Field(default_factory=list)
    trigger: str | None = None
    duration: str | None = None
    emergency: str | None = None
    model_config = ConfigDict(extra="ignore")


def default_slot_table() -> TriageSlotTable:
    return TriageSlotTable()


def slot_gate_passes(table: TriageSlotTable) -> bool:
    return bool(table.primary_symptom or table.primary_disease)
