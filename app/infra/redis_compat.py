"""Redis 兼容工具（支持 Windows 旧版 Redis 3.x，不支持 HSET 多字段）。"""


def hset_mapping(client, key: str, mapping: dict) -> None:
    for field, value in mapping.items():
        client.hset(key, field, str(value))
