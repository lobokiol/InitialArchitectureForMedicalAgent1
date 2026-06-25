# 构建并运行 Medical Triage Agent Docker 镜像
# 用法：.\scripts\docker-build.ps1 [build|up|down|logs|shell]

param(
    [Parameter()]
    [ValidateSet("build", "up", "down", "logs", "shell", "clean", "rebuild")]
    [string]$Action = "up",
    
    [switch]$Dev,
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
Set-Location $ProjectRoot

function Test-EnvFile {
    if (-not (Test-Path ".env")) {
        Write-Host "⚠️  .env 文件不存在，复制 .env.example..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "📝 请编辑 .env 文件，设置 DASHSCOPE_API_KEY 等必要配置" -ForegroundColor Cyan
        exit 1
    }
    
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "DASHSCOPE_API_KEY=\s*(.+)") {
        $key = $matches[1].Trim()
        if ($key -and $key -ne "your-api-key-here") {
            Write-Host "✅ DASHSCOPE_API_KEY 已配置" -ForegroundColor Green
        } else {
            Write-Host "⚠️  DASHSCOPE_API_KEY 未设置或为空，请在 .env 中配置" -ForegroundColor Yellow
        }
    }
}

switch ($Action) {
    "build" {
        Write-Host "🔨 构建 Docker 镜像..." -ForegroundColor Cyan
        $cacheArg = if ($NoCache) { "--no-cache" } else { "" }
        docker build $cacheArg -t medical-triage-api:latest .
        Write-Host "✅ 构建完成" -ForegroundColor Green
    }
    
    "up" {
        Test-EnvFile
        Write-Host "🚀 启动服务（后台模式）..." -ForegroundColor Cyan
        
        $composeArgs = @("up", "-d")
        if ($Dev) {
            $composeArgs += "--build"
        }
        
        docker compose @composeArgs
        
        Write-Host ""
        Write-Host "⏳ 等待服务就绪..." -ForegroundColor Cyan
        Start-Sleep -Seconds 5
        
        # 健康检查
        $retries = 30
        $healthy = $false
        while ($retries -gt 0 -and -not $healthy) {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:8000/healthz" -TimeoutSec 5 -ErrorAction SilentlyContinue
                if ($response.status -eq "ok") {
                    $healthy = $true
                    Write-Host "✅ API 服务已就绪" -ForegroundColor Green
                }
            } catch {
                Write-Host "   等待 API 启动... ($retries)" -ForegroundColor Gray
                Start-Sleep -Seconds 2
                $retries--
            }
        }
        
        if (-not $healthy) {
            Write-Host "⚠️  API 启动超时，查看日志：docker compose logs -f api" -ForegroundColor Yellow
        } else {
            Write-Host ""
            Write-Host "🎉 服务已启动！访问地址：" -ForegroundColor Green
            Write-Host "   API:       http://localhost:8000" -ForegroundColor Cyan
            Write-Host "   Docs:      http://localhost:8000/docs" -ForegroundColor Cyan
            Write-Host "   Health:    http://localhost:8000/healthz" -ForegroundColor Cyan
            Write-Host "   Ready:     http://localhost:8000/ready" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "📋 常用命令：" -ForegroundColor Gray
            Write-Host "   查看日志：docker compose logs -f api" -ForegroundColor Gray
            Write-Host "   停止服务：docker compose down" -ForegroundColor Gray
            Write-Host "   完全清理：docker compose down -v" -ForegroundColor Gray
        }
    }
    
    "down" {
        Write-Host "🛑 停止服务..." -ForegroundColor Cyan
        docker compose down
        Write-Host "✅ 服务已停止" -ForegroundColor Green
    }
    
    "logs" {
        Write-Host "📋 查看 API 日志（Ctrl+C 退出）..." -ForegroundColor Cyan
        docker compose logs -f api
    }
    
    "shell" {
        Write-Host "🐚 进入 API 容器..." -ForegroundColor Cyan
        docker compose exec api /bin/bash
    }
    
    "clean" {
        Write-Host "🧹 清理容器和数据卷..." -ForegroundColor Cyan
        docker compose down -v
        docker system prune -f
        Write-Host "✅ 清理完成" -ForegroundColor Green
    }
    
    "rebuild" {
        Write-Host "🔄 完全重建..." -ForegroundColor Cyan
        docker compose down
        docker build --no-cache -t medical-triage-api:latest .
        docker compose up -d
        Write-Host "✅ 重建完成" -ForegroundColor Green
    }
}
