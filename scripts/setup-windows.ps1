# 本地 Windows 一键环境准备（需已安装 Docker Desktop + Python 3.10+）
# 用法：powershell -ExecutionPolicy Bypass -File scripts/setup-windows.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> 项目目录: $Root"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[错误] 未检测到 Docker。请先安装 Docker Desktop 并重启终端。" -ForegroundColor Red
    Write-Host "  下载: https://www.docker.com/products/docker-desktop/"
    exit 1
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[错误] 未检测到 Python。" -ForegroundColor Red
    exit 1
}

Write-Host "==> 启动 Docker 依赖 (Redis / ES / Milvus)..."
docker compose -f demo/docker-compose.local.yml up -d

Write-Host "==> 等待 Elasticsearch 就绪 (约 30-60s)..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        Invoke-RestMethod -Uri "http://localhost:9200" -TimeoutSec 3 | Out-Null
        $ready = $true
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}
if (-not $ready) {
    Write-Host "[警告] ES 尚未响应，可稍后手动检查 http://localhost:9200" -ForegroundColor Yellow
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "==> 已创建 .env，请编辑并填入 DASHSCOPE_API_KEY" -ForegroundColor Yellow
}

if (-not (Test-Path "venv")) {
    Write-Host "==> 创建 Python 虚拟环境..."
    python -m venv venv
}

Write-Host "==> 安装 Python 依赖..."
& ".\venv\Scripts\pip.exe" install -r requirements.txt

Write-Host "==> 导入 ES 指南数据..."
Push-Location demo
& "..\venv\Scripts\python.exe" es.py
Write-Host "==> 导入 Milvus 向量数据 (需 .env 中有效的 DASHSCOPE_API_KEY)..."
& "..\venv\Scripts\python.exe" milvus.py
Pop-Location

Write-Host ""
Write-Host "==> 完成。启动服务：" -ForegroundColor Green
Write-Host "  .\venv\Scripts\activate"
Write-Host "  `$env:PYTHONPATH='.'"
Write-Host "  .\venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload"
Write-Host ""
Write-Host "  另开终端运行 CLI："
Write-Host "  .\venv\Scripts\python.exe cli.py"
Write-Host ""
Write-Host "  API 文档: http://localhost:8000/docs"
