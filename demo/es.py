# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import List, Optional, Dict, Any

# from elasticsearch import Elasticsearch, helpers
# from elasticsearch.exceptions import NotFoundError

# # ====== 基础配置 ======
# ES_URL = "http://localhost:9200"
# INDEX_NAME = "hospital_procedures"
# DATA_PATH = Path("./data/es医院导诊指南_clean.json")


# def get_es_client() -> Elasticsearch:
#     """初始化 ES 客户端，并打印集群信息。"""
#     es = Elasticsearch(
#         ES_URL,
#         request_timeout=30,  # 给一点超时时间，避免大请求直接挂
#     )
#     try:
#         info = es.info()
#         print(
#             "[ES] cluster:",
#             info.get("cluster_name"),
#             "version:",
#             info.get("version", {}).get("number"),
#         )
#     except Exception as e:
#         print("[ES] 无法连接到 Elasticsearch：", repr(e))
#         raise
#     return es


# def ensure_index(es: Elasticsearch, index_name: str = INDEX_NAME) -> None:
#     """如果索引不存在则创建。"""
#     if es.indices.exists(index=index_name):
#         print(f"[ES] 索引 {index_name!r} 已存在，跳过创建")
#         return

#     mapping = {
#         "mappings": {
#             "properties": {
#                 "id":          {"type": "keyword"},
#                 "hospital":    {"type": "keyword"},
#                 "scene":       {"type": "text"},
#                 "department":  {"type": "keyword"},
#                 "process_type": {"type": "keyword"},
#                 "raw_text":    {"type": "text"},
#                 "source_file": {"type": "keyword"},
#                 "page_range":  {"type": "keyword"},
#             }
#         }
#     }

#     es.indices.create(index=index_name, body=mapping)
#     print(f"[ES] 索引 {index_name!r} 已创建")


# def index_docs_if_needed(
#     es: Elasticsearch,
#     index_name: str = INDEX_NAME,
#     data_path: Path = DATA_PATH,
# ) -> None:
#     """
#     如果当前索引为空，则从 JSON 文件读取并写入 ES。
#     如果索引里已经有数据，就不重复写（你要强制重建可以删索引再跑）。
#     """
#     # 先看看索引里有多少条
#     try:
#         count = es.count(index=index_name)["count"]
#     except NotFoundError:
#         count = 0

#     print(f"[ES] 当前索引 {index_name!r} 文档数：{count}")

#     if count > 0:
#         print("[ES] 检测到已有数据，跳过写入步骤")
#         return

#     # 读取本地 JSON 数据
#     if not data_path.exists():
#         raise FileNotFoundError(f"数据文件不存在：{data_path}")

#     with data_path.open("r", encoding="utf-8") as f:
#         docs: List[Dict[str, Any]] = json.load(f)

#     print(f"[ES] 从文件读取到 {len(docs)} 条记录，开始 bulk 写入……")

#     def gen_actions():
#         for doc in docs:
#             doc_id = doc.get("id")
#             if not doc_id:
#                 continue
#             yield {
#                 "_index": index_name,
#                 "_id": doc_id,   # 用业务 id 做 _id，方便去重/更新
#                 "_source": doc,
#             }

#     helpers.bulk(es, gen_actions())
#     print(f"[ES] 写入 ES 文档数量: {len(docs)}")


# def es_keyword_search(
#     es: Elasticsearch,
#     query: str,
#     process_type: Optional[str] = None,
#     department: Optional[str] = None,
#     k: int = 5,
# ) -> List[Dict[str, Any]]:
#     """
#     用关键词在 ES 里查医院流程/FAQ。
#     - query: 用户问的问题
#     - process_type: 过滤住院 / 门诊流程等（inpatient/guide...）
#     - department: 过滤科室
#     """
#     must = [
#         {
#             "query_string": {
#                 "query": query,
#                 "fields": ["scene^2", "raw_text"],
#                 "default_operator": "AND",
#             }
#         }
#     ]
#     filters = []
#     if process_type:
#         filters.append({"term": {"process_type": process_type}})
#     if department:
#         filters.append({"term": {"department": department}})

#     body = {
#         "query": {
#             "bool": {
#                 "must": must,
#                 "filter": filters,
#             }
#         },
#         "size": k,
#     }

#     res = es.search(index=INDEX_NAME, body=body)
#     hits = res.get("hits", {}).get("hits", [])
#     return [h["_source"] for h in hits]


# if __name__ == "__main__":
#     es = get_es_client()
#     ensure_index(es, INDEX_NAME)
#     index_docs_if_needed(es, INDEX_NAME, DATA_PATH)

#     # 调试：看看前 3 条文档长什么样（可选）
#     print("\n=== 调试：索引里前 3 条文档 ===")
#     debug_res = es.search(index=INDEX_NAME, body={"query": {"match_all": {}}, "size": 3})
#     for h in debug_res["hits"]["hits"]:
#         print(h["_source"])

#     # 测试检索
#     print("\n=== 查询：提前出院 ===")
#     for hit in es_keyword_search(es, "提前出院", k=3):
#         print(hit["scene"], "=>", hit["raw_text"])

#     print("\n=== 查询：预约了为什么还要等很久 ===")
#     for hit in es_keyword_search(es, "预约了 为什么 还 要 等 很久", k=3):
#         print(hit["scene"], "=>", hit["raw_text"])

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError

# ====== 基础配置 ======
ES_URL = "http://localhost:9200"
INDEX_NAME = "hospital_procedures"
DATA_PATH = Path("./data/es医院导诊指南_clean.json")


def get_es_client() -> Elasticsearch:
    """初始化 ES 客户端，并打印集群信息。"""
    es = Elasticsearch(
        ES_URL,
        request_timeout=30,
    )
    try:
        info = es.info()
        print(
            "[ES] cluster:",
            info.get("cluster_name"),
            "version:",
            info.get("version", {}).get("number"),
        )
    except Exception as e:
        print("[ES] 无法连接到 Elasticsearch：", repr(e))
        raise
    return es


def ensure_index(es: Elasticsearch, index_name: str = INDEX_NAME) -> None:
    """如果索引不存在则创建。"""
    if es.indices.exists(index=index_name):
        print(f"[ES] 索引 {index_name!r} 已存在，跳过创建")
        return

    mapping = {
        "mappings": {
            "properties": {
                "id":          {"type": "keyword"},
                "hospital":    {"type": "keyword"},
                "scene":       {"type": "text"},
                "department":  {"type": "keyword"},
                "process_type": {"type": "keyword"},
                "raw_text":    {"type": "text"},
                "source_file": {"type": "keyword"},
                "page_range":  {"type": "keyword"},
            }
        }
    }

    es.indices.create(index=index_name, body=mapping)
    print(f"[ES] 索引 {index_name!r} 已创建")


def index_docs_if_needed(
    es: Elasticsearch,
    index_name: str = INDEX_NAME,
    data_path: Path = DATA_PATH,
) -> None:
    """
    如果当前索引为空，则从 JSON 文件读取并写入 ES。
    如果索引里已经有数据，就不重复写（你要强制重建可以删索引再跑）。
    """
    try:
        count = es.count(index=index_name)["count"]
    except NotFoundError:
        count = 0

    print(f"[ES] 当前索引 {index_name!r} 文档数：{count}")

    if count > 0:
        print("[ES] 检测到已有数据，跳过写入步骤")
        return

    # 读取本地 JSON 数据
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        docs: List[Dict[str, Any]] = json.load(f)

    print(f"[ES] 从文件读取到 {len(docs)} 条记录，开始 bulk 写入……")

    def gen_actions():
        for doc in docs:
            doc_id = doc.get("id")
            if not doc_id:
                continue
            yield {
                "_index": index_name,
                "_id": doc_id,
                "_source": doc,
            }

    helpers.bulk(es, gen_actions())
    print(f"[ES] 写入 ES 文档数量: {len(docs)}")


if __name__ == "__main__":
    es = get_es_client()
    ensure_index(es, INDEX_NAME)
    index_docs_if_needed(es, INDEX_NAME, DATA_PATH)

    # 可选：简单看一下前几条
    print("\n=== 调试：索引里前 3 条文档 ===")
    debug_res = es.search(index=INDEX_NAME, body={"query": {"match_all": {}}, "size": 3})
    for h in debug_res["hits"]["hits"]:
        print(h["_source"])
