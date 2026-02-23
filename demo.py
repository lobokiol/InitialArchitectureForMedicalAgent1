import os
import json
from dotenv import load_dotenv
import getpass
import logging  # ✅ 新增

from typing import List, Dict, Any, Optional, Literal, Sequence, Annotated
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, List
from pydantic import BaseModel, Field

from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver

import uuid
from datetime import datetime

import redis 

print(RedisSaver)

from elasticsearch import Elasticsearch


# =========================
# 0. 日志 & 环境变量加载
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("med_rag_graph")  # ✅ 统一用这个 logger

load_dotenv(override=True)

if not os.getenv("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter your DASHSCOPE_API_KEY: ")


# =========================
# 1. 模型定义
# =========================

llm_chat = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3-max",
    temperature=0.0,
    timeout=60,
    max_retries=2,
)

llm_embedding = OpenAIEmbeddings(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="text-embedding-v2",
    deployment="text-embedding-v2",
    check_embedding_ctx_length=False,
)


# =========================
# 2. IntentResult
# =========================


class IntentResult(BaseModel):
    # ✅ 给关键字段设置默认值，防止缺字段时报错
    has_symptom: bool = False
    has_process: bool = False

    main_intent: Literal["symptom", "process", "mixed", "non_medical"] = "non_medical"

    symptom_query: Optional[str] = None
    process_query: Optional[str] = None

    need_symptom_search: bool = False
    need_process_search: bool = False

    # ✅ 允许忽略旧数据 / 序列化格式里的多余字段（lc、type、id、kwargs 等）
    model_config = ConfigDict(extra="ignore")



# =========================
# 3. RetrievedDoc
# =========================



class RetrievedDoc(BaseModel):
    id: str = ""     # 默认空字符串，防止缺字段时报错
    source: Literal["medical", "process"] = "medical"
    title: Optional[str] = None
    content: str = ""
    score: Optional[float] = None

    model_config = ConfigDict(extra="ignore")


# =========================
# 4. RelevanceResult
# =========================


class RelevanceResult(BaseModel):
    can_answer_overall: bool = False
    need_rewrite_symptom: bool = False
    need_rewrite_process: bool = False
    reason: Optional[str] = None

    model_config = ConfigDict(extra="ignore")



# =========================
# 5. AppState（修正 messages 默认值）
# =========================


class AppState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    intent_result: Optional[IntentResult] = None
    medical_docs: List[RetrievedDoc] = Field(default_factory=list)
    process_docs: List[RetrievedDoc] = Field(default_factory=list)
    relevance_result: Optional[RelevanceResult] = None
    rewrite_attempts: int = 0

    # ---------- 1) 兼容 intent_result ----------
    @field_validator("intent_result", mode="before")
    @classmethod
    def _coerce_intent_result(cls, v):
        if v is None:
            return None
        if isinstance(v, IntentResult):
            return v
        if isinstance(v, dict):
            # 兼容 Redis 存的 LC 构造器格式：{"lc": ..., "kwargs": {...}}
            if "kwargs" in v and isinstance(v["kwargs"], dict):
                v = v["kwargs"]
            try:
                return IntentResult(**v)
            except Exception:
                return None
        return None

    # ---------- 2) 兼容 medical_docs ----------
    @field_validator("medical_docs", mode="before")
    @classmethod
    def _coerce_medical_docs(cls, v):
        if v is None:
            return []
        # Redis 里会是 list
        if isinstance(v, list):
            out: List[RetrievedDoc] = []
            for item in v:
                if isinstance(item, RetrievedDoc):
                    out.append(item)
                elif isinstance(item, dict):
                    # 同样兼容 LC 构造器格式
                    if "kwargs" in item and isinstance(item["kwargs"], dict):
                        item = item["kwargs"]
                    try:
                        out.append(RetrievedDoc(**item))
                    except Exception:
                        # 某个 doc 完全脏数据就丢掉，不影响整体
                        continue
            return out
        return []

    # ---------- 3) 兼容 process_docs ----------
    @field_validator("process_docs", mode="before")
    @classmethod
    def _coerce_process_docs(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            out: List[RetrievedDoc] = []
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
        return []

    # ---------- 4) 兼容 relevance_result ----------
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


# =========================
# 全局重写次数上限（只定义一次）
# =========================

MAX_REWRITE = 2



# =========================
# 对话短期记忆长度（只保留最近 N 条消息）
# =========================

MAX_HISTORY_MSGS = 12      # 剪完后保留多少条
TRIM_TRIGGER_MSGS = 24     # 超过多少条才开始剪


# =========================
# 6. 路由函数
# =========================

def route_after_decision(state: AppState) -> str:
    ir = state.intent_result
    logger.info(">>> route_after_decision: intent_result=%s", ir)

    if not ir:
        logger.info("route_after_decision -> answer_generate (no intent_result)")
        return "answer_generate"

    if not ir.has_symptom and not ir.has_process:
        logger.info("route_after_decision -> answer_generate (no symptom & no process)")
        return "answer_generate"

    logger.info("route_after_decision -> es_rag")
    return "es_rag"


def route_after_es(state: AppState) -> str:
    ir = state.intent_result
    logger.info(">>> route_after_es: intent_result=%s", ir)

    if ir and ir.need_symptom_search:
        logger.info("route_after_es -> milvus_rag (need_symptom_search=True)")
        return "milvus_rag"

    logger.info("route_after_es -> check_docs (no symptom search needed)")
    return "check_docs"


def route_after_docs(state: AppState) -> str:
    r = state.relevance_result
    ir = state.intent_result
    logger.info(
        ">>> route_after_docs: relevance_result=%s, rewrite_attempts=%d",
        r,
        state.rewrite_attempts,
    )

    # ✅ 如果重写次数已经用完，哪怕模型还说要重写，也直接给答案，强行出环
    if state.rewrite_attempts >= MAX_REWRITE:
        logger.info("route_after_docs -> answer_generate (rewrite_attempts >= MAX_REWRITE)")
        return "answer_generate"

    # 文档已足够，正常走生成
    if r and r.can_answer_overall:
        logger.info("route_after_docs -> answer_generate (can_answer_overall=True)")
        return "answer_generate"

    # 模型认为不需要再重写了，也直接生成
    if not (r and (r.need_rewrite_symptom or r.need_rewrite_process)):
        logger.info("route_after_docs -> answer_generate (no need to rewrite)")
        return "answer_generate"

    # 否则再去 rewrite_question
    logger.info("route_after_docs -> rewrite_question (need rewrite)")
    return "rewrite_question"


# =========================
# 7. 决策节点
# =========================

DECISION_PROMPT = """
你是一个医疗问答系统的“意图识别”模块。

你必须只返回一个 json 对象（合法的 JSON），不要输出任何解释或多余文字。

字段要求与 IntentResult 一致：
- has_symptom: bool
- has_process: bool
- main_intent: "symptom" | "process" | "mixed" | "non_medical"
- symptom_query: string 或 null
- process_query: string 或 null
- need_symptom_search: bool
- need_process_search: bool

用户问题：{query}
"""


def decision_node(state: AppState) -> dict:
    logger.info(">>> Enter node: decision")
    user_query = state.messages[-1].content
    logger.info("decision_node user_query=%s", user_query)

    try:
        structured_llm = llm_chat.with_structured_output(IntentResult)
        intent = structured_llm.invoke(
            [HumanMessage(content=DECISION_PROMPT.format(query=user_query))]
        )
    except Exception:
        logger.exception("decision_node LLM 调用失败，使用兜底 intent_result")
        # 兜底：当作 non_medical，不再触发检索
        intent = IntentResult(
            has_symptom=False,
            has_process=False,
            main_intent="non_medical",
            symptom_query=None,
            process_query=None,
            need_symptom_search=False,
            need_process_search=False,
        )
        return {"intent_result": intent}

    # 根据 has_xxx 设置 need_xxx
    intent.need_symptom_search = intent.has_symptom
    intent.need_process_search = intent.has_process

    logger.info("decision_node intent_result=%s", intent)
    return {"intent_result": intent}


# =========================
# 8. ES 检索节点
# =========================

# =========================
# ES 回退：关键词抽取结果
# =========================

class EsKeywordResult(BaseModel):
    keywords: List[str] = Field(default_factory=list)


ES_KEYWORD_PROMPT = """
你是一个医院流程问答系统的“关键词抽取模块”，用于在 FAQ / 流程知识库里做宽松检索。

请从下面的问题中提取 1~5 个最关键的检索关键词，要求：
- 尽量是名词或短语，比如：“提前出院”、“预约挂号”、“住院流程”等
- 不要包含语气词、虚词（如“吗”、“呀”、“呢”等）
- 不要包含过长的整句
- 关键词之间尽量独立、有区分度

只需要返回关键词，不要解释。

用户原始问题：
{user_query}

规范化流程 query：
{process_query}
"""



ES_URL = "http://localhost:9200"
INDEX_NAME = "hospital_procedures"
es_client = Elasticsearch(ES_URL, request_timeout=30)



def _search_es_with_fallback(query: str, size: int = 5):
    """
    简化版 ES 检索：
    1. 先用 AND 做精确检索
    2. 如果 0 命中，再用 OR 放宽检索
    不再调用 LLM，只用一个 query。
    """
    for operator in ("AND", "OR"):
        must = [
            {
                "query_string": {
                    "query": query,
                    "fields": ["scene^2", "raw_text"],
                    "default_operator": operator,
                }
            }
        ]

        body = {"query": {"bool": {"must": must}}, "size": size}

        try:
            res = es_client.search(index=INDEX_NAME, body=body)
            hits = res.get("hits", {}).get("hits", [])
            logger.info(
                "es_rag_node: ES search with operator=%s, hits=%d",
                operator,
                len(hits),
            )
        except Exception:
            logger.exception("ES 查询失败 (operator=%s)", operator)
            return []

        if hits:
            # 第一轮 AND 命中就直接用 AND 的
            # 如果 AND 0 命中则尝试 OR，一旦 OR 有结果也直接用
            return hits

    # AND / OR 都没查到或都异常
    return []


def es_rag_node(state: AppState) -> dict:
    logger.info(">>> Enter node: es_rag")
    ir = state.intent_result
    logger.info("es_rag_node intent_result=%s", ir)

    # 没有流程意图就直接跳过 ES
    if not (
        ir
        and ir.has_process
        and ir.need_process_search
        and ir.process_query
        and ir.process_query.strip()
    ):
        logger.info("es_rag_node: no process search needed, skip ES")
        return {}

    query = ir.process_query.strip()
    logger.info("es_rag_node: ES query=%s", query)

    hits = _search_es_with_fallback(query, size=5)

    if not hits:
        logger.info("es_rag_node: no hits found in ES for query=%s", query)
        return {}

    docs: List[RetrievedDoc] = []
    for h in hits:
        src = h.get("_source", {})
        docs.append(
            RetrievedDoc(
                id=src.get("id", h.get("_id")),
                source="process",
                title=src.get("scene"),
                content=src.get("raw_text", ""),
                score=h.get("_score"),
            )
        )

    logger.info("es_rag_node: got %d docs", len(docs))
    return {"process_docs": docs}



# =========================
# 9. Milvus 检索节点
# =========================
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "medical_knowledge"  # 换成你实际建库用的名字
MILVUS_TOP_K = 15        # 检索时多取一些，后面再筛
MILVUS_MIN_SIM = 0.5     # COSINE 相似度阈值，后面可以根据日志再调
MILVUS_MAX_DOCS = 8      # 最终保留给下游的文档数量上限


# ✅ 显式指定 collection_name，并预留索引参数（HNSW 只是示例，你可以按实际情况调整）
milvus_store = Milvus(
    embedding_function=llm_embedding,
    connection_args={"uri": MILVUS_URI},
    collection_name=MILVUS_COLLECTION,  # ✅ 很重要，确保连到对的集合
    index_params={
        "index_type": "HNSW",           # 也可以用 "IVF_FLAT" 等
        "metric_type": "COSINE",        # ✅ 统一用 COSINE
        "params": {
            "M": 16,
            "efConstruction": 200,
        },
    },
)

# 这个 retriever 现在可以不用了，但保留也没问题，方便以后做 retriever 链
milvus_retriever = milvus_store.as_retriever(
    search_kwargs={
        "k": MILVUS_TOP_K,
        "search_params": {
            "ef": 64,  # HNSW 查询时的参数
        },
    }
)


def milvus_rag_node(state: AppState) -> dict:
    logger.info(">>> Enter node: milvus_rag")
    ir = state.intent_result
    logger.info("milvus_rag_node intent_result=%s", ir)

    # 没有症状相关的检索需求，直接跳过
    if not (ir and ir.has_symptom and ir.need_symptom_search and ir.symptom_query):
        logger.info("milvus_rag_node: no symptom search needed, skip Milvus")
        return {}

    query = ir.symptom_query.strip()
    if not query:
        logger.info("milvus_rag_node: symptom_query is empty, skip Milvus")
        return {}

    logger.info("milvus_rag_node: Milvus query=%s", query)

    try:
        # ✅ 用带分数的接口，并且一次多取一些
        docs_and_scores = milvus_store.similarity_search_with_score(
            query,
            k=MILVUS_TOP_K,
        )
    except Exception:
        # ✅ Milvus 异常时，关闭后续症状检索，避免一直重试
        logger.exception("Milvus 查询失败，关闭症状检索")
        if ir:
            if hasattr(ir, "model_copy"):
                new_ir = ir.model_copy()
            else:
                new_ir = ir.copy(deep=True)
            new_ir.need_symptom_search = False
            return {"intent_result": new_ir}
        return {}

    if not docs_and_scores:
        logger.info("milvus_rag_node: no docs returned from Milvus")
        return {}

    converted: List[RetrievedDoc] = []
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        rid = doc.metadata.get("id", "") if getattr(doc, "metadata", None) else ""
        title = doc.metadata.get("title") if getattr(doc, "metadata", None) else None

        logger.info(
            "milvus_rag_node doc[%d]: id=%s, score=%s, snippet=%s",
            i,
            rid,
            score,
            doc.page_content[:50].replace("\n", " "),
        )

        converted.append(
            RetrievedDoc(
                id=rid,
                source="medical",
                title=title,
                content=doc.page_content,
                score=float(score) if score is not None else None,
            )
        )

    # ✅ 先做相似度阈值过滤（把特别不相关的干掉）
    filtered = [
        d for d in converted
        if (d.score is None) or (d.score >= MILVUS_MIN_SIM)
    ]

    # 如果全被筛掉了，就至少保留 top1，避免完全没文档
    if not filtered and converted:
        filtered = converted[:1]

    # ✅ 再按相似度排序（COSINE 越大越好），有 score 的优先
    filtered.sort(
        key=lambda d: (d.score is None, -(d.score or 0.0))
    )

    # ✅ 最后只保留分数最高的前 MILVUS_MAX_DOCS 条
    if len(filtered) > MILVUS_MAX_DOCS:
        filtered = filtered[:MILVUS_MAX_DOCS]

    logger.info(
        "milvus_rag_node: got %d docs after filter & top%d (before=%d)",
        len(filtered),
        MILVUS_MAX_DOCS,
        len(converted),
    )

    return {"medical_docs": filtered}



# =========================
# 10. 评分节点
# =========================

RAG_EVAL_PROMPT = """
你是一个医疗问答系统的文档评估模块。

你必须只返回一个 json 对象（合法的 JSON，注意是小写的 json），不要输出任何额外说明。

这个 json 对象包含以下字段：
- can_answer_overall: bool
- need_rewrite_symptom: bool
- need_rewrite_process: bool
- reason: string 或 null

请判断：
- 当前文档是否足以回答 can_answer_overall
- 是否需要改写症状 query（need_rewrite_symptom）
- 是否需要改写流程 query（need_rewrite_process）

注意：本节点 **不负责生成重写后的 query**，只负责给出上述 json 字段。

用户问题：
{user_query}

symptom_query: {symptom_query}
process_query: {process_query}

【医学文档】
{medical_block}

【流程文档】
{process_block}
"""

def _fmt_docs(docs: List[RetrievedDoc], max_docs: int = 8) -> str:
    """
    按 score 从高到低选前 max_docs 条，并在前面带上简单的 score 信息，方便 LLM 判断权重。
    """
    if not docs:
        return "（无结果）"

    # 有 score 的优先，按 score 从高到低排；没 score 的排在后面
    docs_sorted = sorted(
        docs,
        key=lambda d: (d.score is None, -(d.score or 0.0))
    )

    selected = docs_sorted[:max_docs]

    out = []
    for i, d in enumerate(selected, 1):
        score_str = f"(score={d.score:.3f})" if d.score is not None else ""
        out.append(f"- 文档{i}{score_str}: {d.content}")
    return "\n".join(out)



def check_docs_node(state: AppState) -> dict:
    logger.info(">>> Enter node: check_docs")
    ir = state.intent_result
    user_query = state.messages[-1].content

    prompt = RAG_EVAL_PROMPT.format(
        user_query=user_query,
        symptom_query=ir.symptom_query if ir else None,
        process_query=ir.process_query if ir else None,
        medical_block=_fmt_docs(state.medical_docs),
        process_block=_fmt_docs(state.process_docs),
    )

    logger.info(
        "check_docs_node: medical_docs=%d, process_docs=%d",
        len(state.medical_docs),
        len(state.process_docs),
    )

    try:
        structured = llm_chat.with_structured_output(RelevanceResult)
        rr = structured.invoke([HumanMessage(content=prompt)])
    except Exception:
        logger.exception("check_docs_node LLM 调用失败，使用兜底 relevance_result")
        rr = RelevanceResult(
            can_answer_overall=False,
            need_rewrite_symptom=False,
            need_rewrite_process=False,
            reason="LLM 调用失败，默认认为文档不足，且不再重写。",
        )

    logger.info("check_docs_node: relevance_result=%s", rr)
    return {"relevance_result": rr}


# =========================
# 11. 重写节点
# =========================

SYMPTOM_REWRITE_PROMPT = """
请改写下面的医学检索 query，使其更适合医学知识向量检索：

用户问题：{user_query}
旧 query：{old_query}

要求：不超过 30 字，只输出 query。
"""

PROCESS_REWRITE_PROMPT = """
请改写下面的流程类检索 query，使其更适合流程知识库：

用户问题：{user_query}
旧 query：{old_query}

要求：不超过 30 字，只输出 query。
"""


def rewrite_question(state: AppState) -> dict:
    logger.info(">>> Enter node: rewrite_question (attempt=%d)", state.rewrite_attempts)
    ir = state.intent_result
    rr = state.relevance_result
    user_query = state.messages[-1].content

    attempts = state.rewrite_attempts

    # 超过最大次数：关掉检索，后面 route_after_docs 会直接走 answer_generate
    if attempts >= MAX_REWRITE:
        logger.info("rewrite_question: attempts >= MAX_REWRITE, turn off further search")
        ir_new = ir.model_copy() if hasattr(ir, "model_copy") else ir.copy(deep=True)
        ir_new.need_symptom_search = False
        ir_new.need_process_search = False
        return {
            "intent_result": ir_new,
            "rewrite_attempts": attempts,  # 不再加
        }

    # 还可以重写：拷贝一份 IntentResult 再改
    ir_new = ir.model_copy() if hasattr(ir, "model_copy") else ir.copy(deep=True)

    # ✅ 对 old_query 为 None 做兜底（退化为用 user_query）
    if ir_new.has_symptom and rr and rr.need_rewrite_symptom:
        old_symptom_q = ir_new.symptom_query or user_query
        logger.info("rewrite_question: rewriting symptom_query, old=%s", old_symptom_q)
        try:
            newq = llm_chat.invoke([
                HumanMessage(content=SYMPTOM_REWRITE_PROMPT.format(
                    user_query=user_query,
                    old_query=old_symptom_q,
                ))
            ]).content.strip()
            ir_new.symptom_query = newq
            ir_new.need_symptom_search = True
            logger.info("rewrite_question: new symptom_query=%s", newq)
        except Exception:
            logger.exception("rewrite_question 症状 query 重写失败，关闭 symptom 检索")
            ir_new.need_symptom_search = False
    else:
        ir_new.need_symptom_search = False

    if ir_new.has_process and rr and rr.need_rewrite_process:
        old_process_q = ir_new.process_query or user_query
        logger.info("rewrite_question: rewriting process_query, old=%s", old_process_q)
        try:
            newq = llm_chat.invoke([
                HumanMessage(content=PROCESS_REWRITE_PROMPT.format(
                    user_query=user_query,
                    old_query=old_process_q,
                ))
            ]).content.strip()
            ir_new.process_query = newq
            ir_new.need_process_search = True
            logger.info("rewrite_question: new process_query=%s", newq)
        except Exception:
            logger.exception("rewrite_question 流程 query 重写失败，关闭 process 检索")
            ir_new.need_process_search = False
    else:
        ir_new.need_process_search = False

    return {
        "intent_result": ir_new,
        "rewrite_attempts": attempts + 1,
    }


# =========================
# 12. 答案生成节点
# =========================

def format_history(messages: list[BaseMessage]) -> str:
    """
    把当前 state.messages 里的短期记忆全部格式化给 LLM。
    不再额外截断，长度只由 trim_history_node 控制。
    """
    if not messages:
        return "（无历史对话）"

    lines: List[str] = []
    for m in messages:
        # 这里可以只传「之前的消息」，最后一条是当前 user 问题，
        # 也可以全传，都可以。
        if isinstance(m, HumanMessage):
            role = "用户"
        elif isinstance(m, AIMessage):
            role = "助手"
        else:
            role = "系统"
        lines.append(f"{role}：{m.content}")

    return "\n".join(lines)


ANSWER_PROMPT = """
你是一个专业的医疗导诊助手。

下面是你和用户目前为止的对话历史：
{history_block}

---

用户当前问题：{user_query}

【医学文档】
{medical_block}

【流程文档】
{process_block}

回答要求：
- 优先利用医学文档和流程文档中的信息作答
- 如果文档中没有涉及、但从对话历史中可以推断（例如用户的自我介绍、之前提过的偏好等），也可以基于历史回答
- 如果仍然无法回答，要老实说明：“根据现有资料无法确定”
"""


def answer_generate_node(state: AppState) -> dict:
    logger.info(">>> Enter node: answer_generate")
    user_query = state.messages[-1].content

    history_block = format_history(state.messages)

    prompt = ANSWER_PROMPT.format(
        history_block=history_block,
        user_query=user_query,
        medical_block=_fmt_docs(state.medical_docs),
        process_block=_fmt_docs(state.process_docs)
    )

    logger.info(
        "answer_generate_node: medical_docs=%d, process_docs=%d",
        len(state.medical_docs),
        len(state.process_docs),
    )

    full_content = ""

    try:
        # 🔁 用流式接口
        for chunk in llm_chat.stream([HumanMessage(content=prompt)]):
            delta = chunk.content  # 每次过来一小段
            if delta:
                print(delta, end="", flush=True)  # 直接往控制台刷
                full_content += delta

        print("\n")  # 换行，好看一点

    except Exception:
        logger.exception("answer_generate_node LLM 调用失败，返回兜底回答")
        full_content = "抱歉，当前系统生成答案时出现了问题，请稍后再试。"
        print(full_content + "\n")

    # ⚠️ 一定要把完整内容存回 state，后面才能从 Redis 复原对话
    return {"messages": [AIMessage(content=full_content)]}



# =========================
# 13. 消息裁剪节点
# =========================

def trim_history_node(state: AppState) -> dict:
    logger.info(">>> Enter node: trim_history")

    msgs = state.messages or []
    total = len(msgs)

    # ✅ 没超过触发阈值：完全不裁剪
    if total <= TRIM_TRIGGER_MSGS:
        logger.info(
            "trim_history_node: no need to trim, total_messages=%d, trigger=%d, keep=%d",
            total,
            TRIM_TRIGGER_MSGS,
            MAX_HISTORY_MSGS,
        )
        return {"messages": msgs}

    # ✅ 超过阈值：只保留最近 MAX_HISTORY_MSGS 条
    trimmed = msgs[-MAX_HISTORY_MSGS:]

    logger.info(
        "trim_history_node: trimmed messages from %d to %d (trigger=%d, keep=%d)",
        total,
        len(trimmed),
        TRIM_TRIGGER_MSGS,
        MAX_HISTORY_MSGS,
    )

    return {"messages": trimmed}



# =========================
# 14. 图编译
# =========================


graph = StateGraph(AppState)

graph.add_node("decision", decision_node)
graph.add_node("es_rag", es_rag_node)
graph.add_node("milvus_rag", milvus_rag_node)
graph.add_node("check_docs", check_docs_node)
graph.add_node("rewrite_question", rewrite_question)
graph.add_node("answer_generate", answer_generate_node)
graph.add_node("trim_history", trim_history_node)

# 🔁 1. START 先连到 trim_history，再到 decision
graph.add_edge(START, "trim_history")
graph.add_edge("trim_history", "decision")

graph.add_conditional_edges("decision", route_after_decision, {
    "answer_generate": "answer_generate",
    "es_rag": "es_rag",
})

graph.add_conditional_edges("es_rag", route_after_es, {
    "milvus_rag": "milvus_rag",
    "check_docs": "check_docs",
})

graph.add_edge("milvus_rag", "check_docs")

graph.add_conditional_edges("check_docs", route_after_docs, {
    "answer_generate": "answer_generate",
    "rewrite_question": "rewrite_question",
})

graph.add_edge("rewrite_question", "es_rag")


# ✅ 2. 改成 answer_generate 直接结束
graph.add_edge("answer_generate", END)



# =========================
# 15. redis会话管理 
# =========================

DB_URI = "redis://localhost:6379"

# 用同一个 Redis 做用户/会话管理
redis_client = redis.Redis.from_url(DB_URI, decode_responses=True)

# Key 模板
USER_CURRENT_KEY = "user:{user_id}:current_thread"
USER_THREADS_KEY = "user:{user_id}:threads"       # sorted set: member=thread_id, score=last_active_at_ts
THREAD_META_KEY = "thread:{thread_id}:meta"       # hash: user_id, title, created_at, last_active_at, is_deleted


class SessionManager:
    """
    管理 user / thread（会话）的信息：
    - 当前会话
    - 会话列表
    - 新建 / 切换 / 删除会话
    """

    def __init__(self, client: redis.Redis):
        self.client = client

    # --------- 内部工具 ---------
    @staticmethod
    def _now_ts() -> float:
        return datetime.utcnow().timestamp()

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat()

    def _user_current_key(self, user_id: str) -> str:
        return USER_CURRENT_KEY.format(user_id=user_id)

    def _user_threads_key(self, user_id: str) -> str:
        return USER_THREADS_KEY.format(user_id=user_id)

    def _thread_meta_key(self, thread_id: str) -> str:
        return THREAD_META_KEY.format(thread_id=thread_id)

    # --------- 当前会话 ---------
    def get_current_thread(self, user_id: str) -> Optional[str]:
        return self.client.get(self._user_current_key(user_id))

    def set_current_thread(self, user_id: str, thread_id: str):
        self.client.set(self._user_current_key(user_id), thread_id)

    # --------- 创建新会话（也就是“新对话 / 软清空”） ---------
    def create_thread(self, user_id: str, title: Optional[str] = None) -> str:
        # 简单生成一个 thread_id：user_id:s:短uuid
        session_id = uuid.uuid4().hex[:8]
        thread_id = f"{user_id}:s:{session_id}"

        now_iso = self._now_iso()
        now_ts = self._now_ts()

        if not title:
            # 先用一个默认标题，后面也可以在第一次提问后更新
            title = f"对话 {now_iso[:10]}"

        # 会话元信息
        meta_key = self._thread_meta_key(thread_id)
        self.client.hset(
            meta_key,
            mapping={
                "user_id": user_id,
                "title": title,
                "created_at": now_iso,
                "last_active_at": now_iso,
                "is_deleted": "0",
            },
        )

        # 加入用户的会话集合（按 last_active_at 排序）
        self.client.zadd(self._user_threads_key(user_id), {thread_id: now_ts})

        # 设置为当前会话
        self.set_current_thread(user_id, thread_id)

        return thread_id

    # --------- 更新 last_active_at ---------
    def touch_thread(self, thread_id: str):
        meta_key = self._thread_meta_key(thread_id)
        if not self.client.exists(meta_key):
            return

        now_iso = self._now_iso()
        now_ts = self._now_ts()

        self.client.hset(meta_key, "last_active_at", now_iso)

        user_id = self.client.hget(meta_key, "user_id")
        if user_id:
            self.client.zadd(self._user_threads_key(user_id), {thread_id: now_ts})

    # --------- 列出会话列表 ---------
    def list_threads(self, user_id: str) -> list[dict]:
        """
        返回按 last_active_at 倒序排好的会话列表（排除已删除）
        每项结构：
        {
          "thread_id": str,
          "title": str,
          "created_at": str,
          "last_active_at": str,
          "is_deleted": bool
        }
        """
        key = self._user_threads_key(user_id)
        # ZREVRANGE：按 score 从大到小
        thread_ids = self.client.zrevrange(key, 0, -1)

        result = []
        for tid in thread_ids:
            meta = self.client.hgetall(self._thread_meta_key(tid))
            if not meta:
                continue
            if meta.get("is_deleted") == "1":
                # 防御：如果之前没从 ZSET 里删干净，就在这里过滤一手
                continue
            result.append(
                {
                    "thread_id": tid,
                    "title": meta.get("title", tid),
                    "created_at": meta.get("created_at", ""),
                    "last_active_at": meta.get("last_active_at", ""),
                    "is_deleted": meta.get("is_deleted") == "1",
                }
            )
        return result

    # --------- 删除会话（应用层删除） ---------
    def delete_thread(self, user_id: str, thread_id: str) -> Optional[str]:
        """
        标记会话删除 + 从用户会话列表里移除。
        返回新的 current_thread_id（如果当前会话被删，则切换；否则 None）。
        """
        meta_key = self._thread_meta_key(thread_id)
        if not self.client.exists(meta_key):
            return None

        # 标记删除
        self.client.hset(meta_key, "is_deleted", "1")

        # 从用户的 ZSET 中移除
        self.client.zrem(self._user_threads_key(user_id), thread_id)

        # 如果没删当前会话，直接返回 None
        cur = self.get_current_thread(user_id)
        if cur != thread_id:
            return None

        # 如果删的是当前会话，找一个新的会话（最近活跃的）
        threads = self.list_threads(user_id)
        if threads:
            new_thread_id = threads[0]["thread_id"]
            self.set_current_thread(user_id, new_thread_id)
            return new_thread_id

        # 如果真的一个会话都没了，就创建一个新的
        new_thread_id = self.create_thread(user_id, title="新对话")
        return new_thread_id


# =========================
# 16. 运行
# =========================


with RedisSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)

    # 实例化会话管理器
    session_manager = SessionManager(redis_client)

    if __name__ == "__main__":
        logger.info("医疗导诊助手启动")
        print("医疗导诊助手已启动，输入 quit 退出。")
        print("支持命令：/new 新对话，/list 列出对话，/switch N 切换对话，/delete N 删除对话。\n")

        user_id = input("请输入当前用户ID（比如手机号或user_id）: ").strip() or "default_user"

        # 1) 获取或创建当前 thread_id
        thread_id = session_manager.get_current_thread(user_id)
        if not thread_id:
            thread_id = session_manager.create_thread(user_id, title="默认对话")
            logger.info("为用户 %s 创建初始会话 thread_id=%s", user_id, thread_id)
        else:
            logger.info("用户 %s 继续使用已有会话 thread_id=%s", user_id, thread_id)

        print(f"当前用户：{user_id}")
        print(f"当前会话 thread_id: {thread_id}\n")

        # --------- CLI 主循环 ---------
        while True:
            q = input("User: ").strip()
            if not q:
                continue

            # ---- 退出命令 ----
            if q.lower() in {"quit", "exit"}:
                logger.info("医疗导诊助手退出 (user_id=%s)", user_id)
                break

            # ---- 新对话：/new 或 “新对话” ----
            if q in {"/new", "新对话"}:
                thread_id = session_manager.create_thread(user_id, title="新对话")
                print(f"\n✅ 已创建并切换到新对话，thread_id={thread_id}\n")
                continue

            # ---- 列出会话：/list ----
            if q in {"/list", "会话列表"}:
                threads = session_manager.list_threads(user_id)
                if not threads:
                    print("暂无会话，可以输入 /new 创建一个新对话。\n")
                    continue

                print("\n当前会话列表：")
                for idx, t in enumerate(threads, 1):
                    flag = "(当前)" if t["thread_id"] == thread_id else ""
                    print(
                        f"  [{idx}] {t['title']}  created={t['created_at']}  last={t['last_active_at']}  {flag}"
                    )
                print("")
                continue

            # ---- 切换会话：/switch N ----
            if q.startswith("/switch"):
                parts = q.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    print("用法：/switch 序号，例如 /switch 1\n")
                    continue

                idx = int(parts[1])
                threads = session_manager.list_threads(user_id)
                if not threads:
                    print("暂无会话，可以输入 /new 创建一个新对话。\n")
                    continue
                if not (1 <= idx <= len(threads)):
                    print(f"无效序号，当前共有 {len(threads)} 个会话。\n")
                    continue

                target = threads[idx - 1]
                thread_id = target["thread_id"]
                session_manager.set_current_thread(user_id, thread_id)
                print(f"\n✅ 已切换到会话 [{idx}] {target['title']}，thread_id={thread_id}\n")
                continue

            # ---- 删除会话：/delete N ----
            if q.startswith("/delete"):
                parts = q.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    print("用法：/delete 序号，例如 /delete 1\n")
                    continue

                idx = int(parts[1])
                threads = session_manager.list_threads(user_id)
                if not threads:
                    print("暂无会话，无需删除。\n")
                    continue
                if not (1 <= idx <= len(threads)):
                    print(f"无效序号，当前共有 {len(threads)} 个会话。\n")
                    continue

                target = threads[idx - 1]
                del_thread_id = target["thread_id"]

                new_cur = session_manager.delete_thread(user_id, del_thread_id)
                print(f"\n🗑️ 已删除会话 [{idx}] {target['title']} (thread_id={del_thread_id})")

                if new_cur:
                    thread_id = new_cur
                    print(f"当前会话已切换为 thread_id={thread_id}\n")
                else:
                    # 没删当前会话，当前 thread_id 不变
                    print("")
                continue

            # ---- 正常问答流程 ----
            logger.info("=== New query === (user_id=%s, thread_id=%s) %s", user_id, thread_id, q)

            inputs = {"messages": [HumanMessage(content=q)]}
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,  # 以后如果节点里要区分用户，可以用上
                }
            }

            # 让 LangGraph 跑一轮
            for chunk in app.stream(inputs, config=config):
                for node, update in chunk.items():
                    if not update:
                        continue
                    msgs = update.get("messages")
                    if msgs and isinstance(msgs[-1], AIMessage):
                        if node == "trim_history":
                            continue
                        if node == "answer_generate":
                            # answer_generate 你已经在节点内部流式打印了，这里就不再重复输出
                            continue
                        print(f"[{node}] {msgs[-1].content}\n")

            # 每轮对话结束后，更新会话的 last_active_at
            session_manager.touch_thread(thread_id)
            