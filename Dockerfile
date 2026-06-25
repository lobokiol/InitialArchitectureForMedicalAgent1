# 生产级 Dockerfile - 使用 uv 作为包管理器
# 多阶段构建，优化镜像大小

# ========== 构建阶段 ==========
FROM python:3.11-slim-bookworm AS builder

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# 设置工作目录
WORKDIR /app

# 复制依赖定义
COPY pyproject.toml ./
COPY README.md ./

# 创建虚拟环境并安装依赖
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r pyproject.toml

# ========== 运行阶段 ==========
FROM python:3.11-slim-bookworm AS runtime

# 安全：使用非 root 用户运行
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制应用代码
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup cli.py ./
COPY --chown=appuser:appgroup demo/ ./demo/
COPY --chown=appuser:appgroup scripts/ ./scripts/
COPY --chown=appuser:appgroup .env.example ./

# 创建数据目录
RUN mkdir -p data && chown -R appuser:appgroup data

# 切换到非 root 用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# 暴露端口
EXPOSE 8000

# 复制入口脚本
COPY --chown=appuser:appgroup scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 启动命令
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--access-log"]
