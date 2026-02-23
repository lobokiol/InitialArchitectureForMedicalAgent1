# import os
# import json
# from pathlib import Path
# from typing import List, Dict, Any

# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
# from langchain_milvus import Milvus
# from langchain_core.documents import Document
# from pymilvus import utility, MilvusException


# # =========================
# # 0. 环境变量加载
# # =========================
# load_dotenv(override=True)

# if not os.getenv("DASHSCOPE_API_KEY"):
#     import getpass
#     os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter your DASHSCOPE_API_KEY: ")


# # =========================
# # 1. 常量配置
# # =========================
# MILVUS_URI = "http://localhost:19530"
# DATA_PATH = Path("./data/milvus数据.txt")
# PAGE_SIZE = 5000         # 每次从 Milvus 取多少 ID
# EMBED_BATCH = 25         # DashScope embedding 的 batch 限制


# # =========================
# # 2. Embedding 模型
# # =========================
# llm_embedding = OpenAIEmbeddings(
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     model="text-embedding-v2",
#     deployment="text-embedding-v2",
#     check_embedding_ctx_length=False,
# )


# # =========================
# # 3. Milvus 向量库实例
# # =========================
# vector_store = Milvus(
#     embedding_function=llm_embedding,
#     connection_args={"uri": MILVUS_URI},
#     index_params={"index_type": "FLAT", "metric_type": "L2"},
# )


# # =========================
# # 4. 分页查询所有已有 ID
# # =========================
# def load_existing_ids() -> set[str]:
#     existing = set()

#     try:
#         exists = utility.has_collection(
#             vector_store.collection_name,
#             using=vector_store._milvus_client._using,
#         )
#     except MilvusException:
#         print("⚠ 集合不存在，将在首次插入时自动创建")
#         return existing

#     if not exists:
#         print("集合不存在 → 首次插入将自动创建")
#         return existing

#     print("📌 集合已存在 → 分页查询已有业务 ID ...")

#     offset = 0

#     while True:
#         try:
#             rows = vector_store._milvus_client.query(
#                 collection_name=vector_store.collection_name,
#                 filter="",
#                 offset=offset,
#                 limit=PAGE_SIZE,
#                 output_fields=["id"],
#             )
#         except MilvusException as e:
#             print("⚠ 分页查询失败：", e)
#             break

#         if not rows:
#             break

#         for row in rows:
#             if "id" in row:
#                 existing.add(row["id"])

#         offset += PAGE_SIZE

#     print(f"📌 已加载 {len(existing)} 条 ID（来自 Milvus）")
#     return existing


# # =========================
# # 5. 构建入库的 Document 列表
# # =========================
# def build_new_documents(existing_ids: set[str]) -> List[Document]:
#     if not DATA_PATH.exists():
#         raise FileNotFoundError(f"❌ 数据文件不存在：{DATA_PATH}")

#     text = DATA_PATH.read_text(encoding="utf-8").strip()

#     if not text.startswith("["):
#         text = "[" + text.rstrip(",\n\t ") + "]"

#     records: List[Dict[str, Any]] = json.loads(text)

#     docs = []
#     for rec in records:
#         rec_id = rec.get("id")
#         if not rec_id or rec_id in existing_ids:
#             continue

#         content = rec.get("raw_question", "") + "\n\n" + rec.get("safe_answer", "")
#         metadata = {
#             "id": rec_id,
#             "departments": ",".join(rec.get("departments") or []),
#             "tags": ",".join(rec.get("tags") or []),
#             "source": rec.get("source"),
#         }

#         docs.append(Document(page_content=content, metadata=metadata))

#     print(f"📌 本次需新增：{len(docs)} 条数据")
#     return docs


# # =========================
# # 6. 分批写入 Milvus
# # =========================
# def insert_documents(docs: List[Document]):
#     if not docs:
#         print("✅ 无新增数据，不执行写入")
#         return

#     print("🚀 开始向 Milvus 写入向量 ...")

#     for i in range(0, len(docs), EMBED_BATCH):
#         batch = docs[i:i + EMBED_BATCH]
#         try:
#             ids = vector_store.add_documents(batch)
#         except Exception as e:
#             print("❌ 写入失败：", e)
#             continue

#         print(f"  - 写入 {len(batch)} 条，生成 Milvus vector IDs: {ids}")

#     print("🎉 Milvus 写入完成。")


# # =========================
# # 7. 主入口
# # =========================
# def main():
#     print("=== 开始 Milvus 增量入库（支持分页） ===")

#     existing_ids = load_existing_ids()
#     new_docs = build_new_documents(existing_ids)
#     insert_documents(new_docs)

#     print("🎉 完成所有处理！")


# if __name__ == "__main__":
#     main()



import os
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from pymilvus import utility, MilvusException


# =========================
# 0. 环境变量加载
# =========================
load_dotenv(override=True)

if not os.getenv("DASHSCOPE_API_KEY"):
    import getpass
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter your DASHSCOPE_API_KEY: ")


# =========================
# 1. 常量配置
# =========================
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "medical_knowledge"   # ✅ 和在线检索保持一致
DATA_PATH = Path("./data/milvus数据.txt")
PAGE_SIZE = 5000         # 每次从 Milvus 取多少 ID
EMBED_BATCH = 25         # DashScope embedding 的 batch 限制


# =========================
# 2. Embedding 模型
# =========================
llm_embedding = OpenAIEmbeddings(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="text-embedding-v2",
    deployment="text-embedding-v2",
    check_embedding_ctx_length=False,
)


# =========================
# 3. Milvus 向量库实例
# =========================
# ✅ 这里要跟在线检索那边保持完全一致：
#   - 同一个 collection_name
#   - 同一个 index_type / metric_type / params
vector_store = Milvus(
    embedding_function=llm_embedding,
    connection_args={"uri": MILVUS_URI},
    collection_name=MILVUS_COLLECTION,   # ✅ 显式指定集合名
    index_params={                       # ✅ 改成和检索侧一致的索引参数
        "index_type": "HNSW",            # 如需改成 IVF_FLAT，也要两边一起改
        "metric_type": "COSINE",             # 如果以后改成 IP / COSINE，两边也要同步
        "params": {
            "M": 16,
            "efConstruction": 200,
        },
    },
)


# =========================
# 4. 分页查询所有已有 ID
# =========================
def load_existing_ids() -> set[str]:
    existing = set()

    try:
        exists = utility.has_collection(
            vector_store.collection_name,
            using=vector_store._milvus_client._using,
        )
    except MilvusException:
        print("⚠ 集合不存在，将在首次插入时自动创建")
        return existing

    if not exists:
        print("集合不存在 → 首次插入将自动创建")
        return existing

    print(f"📌 集合已存在（{vector_store.collection_name}）→ 分页查询已有业务 ID ...")

    offset = 0

    while True:
        try:
            rows = vector_store._milvus_client.query(
                collection_name=vector_store.collection_name,
                filter="",
                offset=offset,
                limit=PAGE_SIZE,
                output_fields=["id"],
            )
        except MilvusException as e:
            print("⚠ 分页查询失败：", e)
            break

        if not rows:
            break

        for row in rows:
            if "id" in row:
                existing.add(row["id"])

        offset += PAGE_SIZE

    print(f"📌 已加载 {len(existing)} 条 ID（来自 Milvus）")
    return existing


# =========================
# 5. 构建入库的 Document 列表
# =========================
def build_new_documents(existing_ids: set[str]) -> List[Document]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ 数据文件不存在：{DATA_PATH}")

    text = DATA_PATH.read_text(encoding="utf-8").strip()

    if not text.startswith("["):
        text = "[" + text.rstrip(",\n\t ") + "]"

    records: List[Dict[str, Any]] = json.loads(text)

    docs = []
    for rec in records:
        rec_id = rec.get("id")
        if not rec_id or rec_id in existing_ids:
            continue

        content = rec.get("raw_question", "") + "\n\n" + rec.get("safe_answer", "")
        metadata = {
            "id": rec_id,
            "title": rec.get("raw_question", "")[:50],  # 添加 title 字段，取问题前50字
            "departments": ",".join(rec.get("departments") or []),
            "tags": ",".join(rec.get("tags") or []),
            "source": rec.get("source"),
        }

        docs.append(Document(page_content=content, metadata=metadata))

    print(f"📌 本次需新增：{len(docs)} 条数据")
    return docs


# =========================
# 6. 分批写入 Milvus
# =========================
def insert_documents(docs: List[Document]):
    if not docs:
        print("✅ 无新增数据，不执行写入")
        return

    print("🚀 开始向 Milvus 写入向量 ...")

    for i in range(0, len(docs), EMBED_BATCH):
        batch = docs[i:i + EMBED_BATCH]
        try:
            ids = vector_store.add_documents(batch)
        except Exception as e:
            print("❌ 写入失败：", e)
            continue

        print(f"  - 写入 {len(batch)} 条，生成 Milvus vector IDs: {ids}")

    print("🎉 Milvus 写入完成。")


# =========================
# 7. 主入口
# =========================
def main():
    print("=== 开始 Milvus 增量入库（支持分页） ===")

    existing_ids = load_existing_ids()
    new_docs = build_new_documents(existing_ids)
    insert_documents(new_docs)

    print("🎉 完成所有处理！")


if __name__ == "__main__":
    main()
    # from pymilvus import connections, utility
    # connections.connect(host="localhost", port="19530")
    # if utility.has_collection("medical_knowledge"):
    #     utility.drop_collection("medical_knowledge")
    #     print("✓ 集合已删除")
    # connections.disconnect("default")
    # from pymilvus import connections, utility
    # connections.connect(host="localhost", port="19530")
    # # 列出所有集合
    # print("所有集合:", utility.list_collections())
    # # 查看集合详情
    # if utility.has_collection("medical_knowledge"):
    #     print("集合存在")
    #     # 查看索引信息
    #     from pymilvus import Collection
    #     coll = Collection("medical_knowledge")
    #     coll.describe()
    # else:
    #     print("集合不存在")
    # connections.disconnect("default")