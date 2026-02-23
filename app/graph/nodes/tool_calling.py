from app.graph.nodes.patient_tools import get_patient_history, get_patient_by_id
from app.core.llm import get_chat_llm
from langchain_core.messages import HumanMessage
from typing import Any, Dict
from langgraph.prebuilt import ToolNode
from app.core.llm import get_chat_llm
from app.core.logging import logger
from app.domain.models import AppState
from app.tools.patient_tools import get_patient_history, get_patient_by_id
import json
from app.domain.models import RetrievedDoc
# 创建工具列表
tools = [get_patient_history, get_patient_by_id]
# 创建 ToolNode
tool_node = ToolNode(tools)
# 带工具的 LLM
llm_with_tools = get_chat_llm().bind_tools(tools)
def tool_calling_node(state: AppState) -> Dict[str, Any]:
    """工具调用节点 - 判断是否需要调用工具并执行"""
    logger.info(">>> Enter node: tool_calling")
    
    query = state.messages[-1].content
    
    # 让 LLM 判断是否需要调用工具
    result = llm_with_tools.invoke(query)
    
    # 检查是否有工具调用
    if not result.tool_calls:
        logger.info("tool_calling: 无需调用工具")
        return {"need_tool_call": False}
    
    logger.info("tool_calling: 需要调用工具 %s", result.tool_calls)
    
    # 执行工具调用
    tool_result = tool_node.invoke({"messages": [result]})
    # 将 ToolMessage 转换为字符串
    tool_result_str = ""
    if tool_result and "messages" in tool_result:
        for msg in tool_result["messages"]:
            if hasattr(msg, 'content'):
                tool_result_str += msg.content + "\n"
    else:
        tool_result_str = str(tool_result)
    return {
        "need_tool_call": True,
        "medical_docs": [
            RetrievedDoc(
                id="tool_call",
                source="tool",
                title="患者病例查询结果",
                content=tool_result_str,
                score=1.0
            )
        ]
    }


# def check_if_need_tools(state: AppState) -> dict:
#     """
#     判断是否需要调用工具
#     """
#     llm_with_tools = get_chat_llm().bind_tools([get_patient_history, get_patient_by_id])
    
#     query = state.messages[-1].content
    
#     # 判断是否需要查询病例
#     result = llm_with_tools.invoke([
#         HumanMessage(content=f"判断以下问题是否需要查询患者病历：{query}")
#     ])
    
#     # 如果有工具调用，返回需要调用工具
#     if result.tool_calls:
#         return {"need_tool_call": True, "tool_calls": result.tool_calls}
    
#     return {"need_tool_call": False}
# def get_patient_history_node(state: AppState) -> dict:
#     """
#     调用工具查询患者病历
#     """
#     tool_call = state.tool_calls[0]
#     patient_id = tool_call.args["patient_id"]
#     history = get_patient_history(patient_id)
#     return {"history": history}