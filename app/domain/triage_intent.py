"""
导诊意图规范（一期）

用户输入 → NER 实体提取 → 三分类路由：

| triage_route | 条件 | 后续 |
|--------------|------|------|
| disease      | 提到病名（可伴随症状） | disease_dept → answer |
| symptom      | 无病名、有症状       | symptom_slot → answer |
| reject       | 无病名、无症状       | 固定拒答，不调 LLM |

拒答文案：请输入症状？
"""

from typing import Literal

TriageRoute = Literal["disease", "symptom", "reject"]

REJECT_MESSAGE = "请输入症状？"
