from pymilvus import Milvus, Collection
from pymilvus.orm import FieldSchema, CollectionSchema
# dropcollection.py - 正确的写法
from pymilvus import connections, utility
# 建立连接
connections.connect(host="localhost", port="19530")
# 删除集合
if utility.has_collection("medical_knowledge"):
    utility.drop_collection("medical_knowledge")
    print("✓ 集合已删除")
else:
    print("集合不存在")
connections.disconnect("default")