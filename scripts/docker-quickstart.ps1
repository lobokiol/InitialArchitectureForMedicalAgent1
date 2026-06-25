# Docker 快速启动脚本 - 用于面试演示
# 一键完成：检查 -> 启动 -> 验证

param(
    [switch]$SkipBuild,
    [switch]$Demo
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.BackgroundColor = "Black"
$colors = @{ "Cyan" = "Cyan"; "Green" = "Green"; "Yellow" = "Yellow"; "Red" = "Red" }

function Write-Step($msg) { Write-Host "`n▶️  $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "   ✅ $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "   ⚠️  $msg" -ForegroundColor Yellow }
function Write-Error($msg) { Write-Host "   ❌ $msg" -ForegroundColor Red }

Write-Host "`n═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "    🏥 Medical Triage Agent - Docker 快速启动" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Step 1: 检查 Docker
Write-Step "检查 Docker 环境"
try {
    $dockerVersion = docker version --format '{{.Server.Version}}' 2>$null
    Write-Ok "Docker 版本: $dockerVersion"
} catch {
    Write-Error "Docker 未运行或未安装"
    exit 1
}

# Step 2: 检查 .env
Write-Step "检查环境配置"
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Warn "已创建 .env 文件，请编辑配置 DASHSCOPE_API_KEY"
        notepad .env
        Write-Host "`n配置完成后按任意键继续..." -ForegroundColor Yellow
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    } else {
        Write-Error "找不到 .env.example"
        exit 1
    }
}

$envContent = Get-Content ".env" -Raw
if ($envContent -match "DASHSCOPE_API_KEY=\s*(.+)") {
    $key = $matches[1].Trim()
    if ($key -and $key -ne "your-api-key-here" -and $key.Length -gt 10) {
        Write-Ok "API Key 已配置"
    } else {
        Write-Warn "API Key 看起来无效，服务可能无法正常工作"
    }
} else {
    Write-Warn "未找到 DASHSCOPE_API_KEY"
}

# Step 3: 构建镜像
if (-not $SkipBuild) {
    Write-Step "构建 Docker 镜像"
    docker build -t medical-triage-api:latest . | ForEach-Object {
        if ($_ -match "^(#|Step| --->|Successfully)") {
            Write-Host "   $_" -ForegroundColor Gray
        }
    }
    Write-Ok "镜像构建完成"
} else {
    Write-Ok "跳过构建（使用现有镜像）"
}

# Step 4: 启动服务
Write-Step "启动所有服务"
docker compose up -d 2>&1 | ForEach-Object {
    if ($_ -match "(Created|Started|Running|Healthy)") {
        Write-Host "   $_" -ForegroundColor Gray
    }
}

# Step 5: 等待就绪
Write-Step "等待服务就绪（约 30-60 秒）"
$services = @("redis", "opensearch", "milvus", "api")
$maxWait = 60
$startTime = Get-Date

foreach ($service in $services) {
    $ready = $false
    $serviceStart = Get-Date
    
    while (-not $ready -and ((Get-Date) - $serviceStart).TotalSeconds -lt $maxWait) {
        $status = docker compose ps $service --format "{{.Status}}" 2>$null
        if ($status -match "healthy|running|Up") {
            $ready = $true
            Write-Ok "$service 已就绪"
        } else {
            Write-Host "   等待 $service ... $([math]::Round(((Get-Date) - $startTime).TotalSeconds))s" -ForegroundColor Gray -NoNewline
            Start-Sleep -Seconds 2
            Write-Host "`r                                                            `r" -NoNewline
        }
    }
    
    if (-not $ready) {
        Write-Warn "$service 启动较慢，继续等待..."
    }
}

# Step 6: 健康检查
Write-Step "验证服务健康"
Start-Sleep -Seconds 3

try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/healthz" -TimeoutSec 5
    if ($health.status -eq "ok") {
        Write-Ok "API 健康检查通过"
    }
} catch {
    Write-Warn "API 健康检查失败，可能仍在启动中"
}

try {
    $ready = Invoke-RestMethod -Uri "http://localhost:8000/ready" -TimeoutSec 5
    $deps = $ready.Keys -join ", "
    Write-Ok "就绪检查完成（依赖: $deps）"
    if ($ready.status -eq "degraded") {
        Write-Warn "服务处于降级模式，部分依赖可能未就绪"
    }
} catch {
    Write-Warn "就绪检查失败: $_"
}

# Step 7: 演示（可选）
if ($Demo) {
    Write-Step "运行演示测试"
    
    $testCases = @(
        @{ user_id = "demo1"; message = "我脚扭伤了，该挂什么科" },
        @{ user_id = "demo2"; message = "肚子疼，一阵一阵的绞痛" },
        @{ user_id = "demo3"; message = "你好" }  # 测试拒答
    )
    
    foreach ($case in $testCases) {
        $json = $case | ConvertTo-Json -Compress
        Write-Host "   测试: $($case.message)" -ForegroundColor Gray
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8000/chat" `
                -Method Post `
                -ContentType "application/json" `
                -Body $json `
                -TimeoutSec 30
            $reply = $response.reply.Substring(0, [Math]::Min(50, $response.reply.Length))
            Write-Ok "响应: $reply..."
        } catch {
            Write-Error "请求失败: $_"
        }
        Start-Sleep -Seconds 1
    }
}

# 完成
Write-Host "`n═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "    🎉 服务已启动！访问地址：" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "   🌐 API 地址:    http://localhost:8000" -ForegroundColor Cyan
Write-Host "   📖 API 文档:    http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "   🔍 Health:      http://localhost:8000/healthz" -ForegroundColor Cyan
Write-Host "   🔍 Ready:       http://localhost:8000/ready" -ForegroundColor Cyan
Write-Host ""
Write-Host "   📋 常用命令:" -ForegroundColor Gray
Write-Host "      查看日志:   docker compose logs -f api" -ForegroundColor Gray
Write-Host "      停止服务:   docker compose down" -ForegroundColor Gray
Write-Host "      完全清理:   docker compose down -v" -ForegroundColor Gray
Write-Host ""

if (-not $Demo) {
    Write-Host "   💡 提示: 添加 -Demo 参数运行演示测试" -ForegroundColor Yellow
}
