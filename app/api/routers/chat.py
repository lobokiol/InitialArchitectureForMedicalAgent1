from typing import Optional, List

import asyncio
from fastapi import APIRouter
from pydantic import BaseModel

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


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
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
