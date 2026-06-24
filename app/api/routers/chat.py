from typing import Any, Optional, List

import asyncio
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.logging import logger
from app.domain.dept_disambiguation import DeptChoice
from app.domain.symptom_clarify import ClarifyChoice
from app.domain.models import IntentResult, RetrievedDoc
from app.services import chat_service


class ChatRequest(BaseModel):
    user_id: str
    thread_id: Optional[str] = None
    message: str


class UsedDocs(BaseModel):
    medical: List[RetrievedDoc]
    process: List[RetrievedDoc]


class ChatResponse(BaseModel):
    user_id: str
    thread_id: str
    reply: str
    intent_result: Optional[IntentResult] = None
    used_docs: UsedDocs
    awaiting_dept_choice: bool = False
    dept_choices: List[DeptChoice] = []
    awaiting_clarify: bool = False
    clarify_phase: Optional[str] = None
    clarify_choices: List[ClarifyChoice] = []
    multi_select: bool = False
    dept_confidence: Optional[float] = None
    dept_confidence_passed: Optional[bool] = None
    dept_confidence_reason: Optional[str] = None
    locked_department: Optional[str] = None
    node_trace: List[str] = []
    app_state: Optional[dict[str, Any]] = None


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    logger.info('POST /chat user_id=%s thread_id=%s message=%r', body.user_id, body.thread_id, body.message)
    # Run sync chat_once in thread to avoid blocking event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        chat_service.chat_once,
        body.user_id,
        body.thread_id,
        body.message,
    )
    return result
