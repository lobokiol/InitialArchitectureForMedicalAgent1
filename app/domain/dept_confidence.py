from pydantic import BaseModel, ConfigDict


class DeptConfidenceResult(BaseModel):
    score: float
    reason: str = ""
    slot_alignment: str = ""

    model_config = ConfigDict(extra="ignore")
