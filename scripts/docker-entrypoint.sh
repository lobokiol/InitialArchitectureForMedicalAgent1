#!/bin/bash
# Docker 入口脚本 - 初始化检查 + 启动服务

set -e

echo "🏥 Medical Triage Agent - Docker Entrypoint"
echo "============================================"

# 等待依赖服务就绪（如果设置了等待标志）
if [ "$WAIT_FOR_SERVICES" = "true" ]; then
    echo "⏳ 等待依赖服务..."
    
    # 等待 Redis
    echo "   - 检查 Redis..."
    until python -c "import redis; r = redis.from_url('$REDIS_URI'); r.ping()" 2>/dev/null; do
        echo "     Redis 未就绪，等待 2s..."
        sleep 2
    done
    echo "   ✓ Redis 就绪"
    
    # 等待 OpenSearch
    echo "   - 检查 OpenSearch..."
    until curl -f "$ES_URL/_cluster/health" >/dev/null 2>&1; do
        echo "     OpenSearch 未就绪，等待 2s..."
        sleep 2
    done
    echo "   ✓ OpenSearch 就绪"
fi

# 初始化数据目录（如果需要）
if [ ! -d "/app/data" ]; then
    mkdir -p /app/data
fi

# 检查环境变量
echo ""
echo "🔧 环境检查："
echo "   - DASHSCOPE_API_KEY: $([ -n "$DASHSCOPE_API_KEY" ] && echo '✓ 已设置' || echo '⚠ 未设置（某些功能将不可用）')"
echo "   - MODEL: ${CHAT_MODEL_NAME:-qwen-plus}"
echo ""

echo "🚀 启动服务..."
echo "============================================"

# 执行传入的命令
exec "$@"
