# 本地开发服务配置 — 改路径/端口/开关只动这一文件
# 被 scripts/dev-services.ps1 加载

@{
    # OpenSearch（本地 zip，非 Docker）
    OpenSearch = @{
        Enabled     = $true
        Home        = 'esTools\opensearch-2.19.1-windows-x64\opensearch-2.19.1'
        Url         = 'http://127.0.0.1:9200'
        WaitSeconds = 90
    }

    # OpenSearch Dashboards（可选，浏览器 Dev Tools）
    Dashboards = @{
        Enabled     = $false
        Home        = 'esTools\opensearch-dashboards-2.19.1'
        Url         = 'http://127.0.0.1:5601'
        WaitSeconds = 120
    }

    # Redis（Windows 本机服务 / redis-server.exe；LangGraph checkpoint 需模块，否则回退 SQLite）
    Redis = @{
        Enabled     = $true
        Mode        = 'windows'   # windows | docker
        ServiceName = 'Redis'
        ServerExe   = 'C:\Program Files\Redis\redis-server.exe'
        ConfFile    = 'redis.windows.conf'
        Uri         = 'redis://127.0.0.1:6379'
        Port        = 6379
        WaitSeconds = 30
        ComposeFile = 'sourceData\redis\docker-compose.yaml'
    }

    # Triage 导诊记录（SQLite 单文件，无需独立进程）
    TriageDb = @{
        Enabled     = $true
        Path        = 'data\triage_sessions.db'
        InitOnStart = $true
    }

    # FastAPI + LangGraph（uv / .venv）
    Api = @{
        Enabled = $true
        Host    = '127.0.0.1'
        Port    = 8000
        Reload  = $false   # 后台启动建议 false；前台调试可改 true 并用 start-api.ps1
    }

    # Python 环境
    Python = @{
        VenvDir      = '.venv'
        UseUvInstall = $true
    }

    # 启动后验证项（可在命令行用 -SkipVerify 跳过）
    Verify = @{
        OpenSearch     = $true
        Redis          = $true
        TriageDb       = $true
        ReadyEndpoint  = $true
        RagIndexCount  = $true
        RagIndexName   = 'rag_knowledge'
        ChatSmoke      = $false   # 单条 /chat，会消耗 LLM
    }

    Logs = @{
        Dir = 'logs'
    }
}
