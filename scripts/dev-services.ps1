# 一键启动 / 验证 / 状态 / 停止本地开发依赖
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File scripts/dev-services.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/dev-services.ps1 -Action start
#   powershell -ExecutionPolicy Bypass -File scripts/dev-services.ps1 -Action verify
#   powershell -ExecutionPolicy Bypass -File scripts/dev-services.ps1 -Action status
#   powershell -ExecutionPolicy Bypass -File scripts/dev-services.ps1 -Action stop
#   powershell -ExecutionPolicy Bypass -File scripts/dev-services.ps1 -Action start -SkipVerify
#
# 配置: scripts/dev-services.config.ps1

param(
    [ValidateSet('start', 'verify', 'status', 'stop')]
    [string] $Action = 'start',
    [switch] $SkipVerify
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Cfg = . (Join-Path $Root 'scripts/dev-services.config.ps1')
$LogDir = Join-Path $Root $Cfg.Logs.Dir
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
$VenvDir = Join-Path $Root $Cfg.Python.VenvDir
$Python = Join-Path $VenvDir 'Scripts/python.exe'
$Uvicorn = Join-Path $VenvDir 'Scripts/uvicorn.exe'

function Write-Step([string]$Msg) { Write-Host "==> $Msg" -ForegroundColor Cyan }
function Write-Ok([string]$Msg)   { Write-Host "[OK] $Msg" -ForegroundColor Green }
function Write-Warn([string]$Msg) { Write-Host "[WARN] $Msg" -ForegroundColor Yellow }
function Write-Err([string]$Msg)  { Write-Host "[ERR] $Msg" -ForegroundColor Red }

function Test-PortListening([int]$Port) {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    return $null -ne $conn
}

function Get-PortFromUrl([string]$Url) {
    $uri = [Uri]$Url
    if ($uri.Port -gt 0) { return $uri.Port }
    if ($uri.Scheme -eq 'https') { return 443 }
    return 80
}

function Wait-HttpOk([string]$Url, [int]$TimeoutSec, [string]$Label) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $lastErr = $null
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-RestMethod -Uri $Url -TimeoutSec 5 | Out-Null
            return $true
        } catch {
            $lastErr = $_
            Start-Sleep -Seconds 3
        }
    }
    Write-Warn "$Label 在 ${TimeoutSec}s 内未就绪: $lastErr"
    return $false
}

function Stop-Port([int]$Port, [string]$Label) {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "  $Label : 未运行 (端口 $Port)"
        return
    }
    foreach ($c in $conns) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "  已停止 $Label (PID $($c.OwningProcess), 端口 $Port)"
    }
}

function Ensure-Venv {
    if (-not (Test-Path $Uvicorn)) {
        Write-Err ".venv 不存在。请先执行:"
        Write-Host "  cd $Root"
        Write-Host "  uv venv --python 3.11"
        Write-Host "  uv pip install -r requirements.txt"
        exit 1
    }
    if ($Cfg.Python.UseUvInstall -and (Test-Path $Uv)) {
        Write-Step "同步 Python 依赖 (uv pip install)"
        & $Uv pip install -r (Join-Path $Root 'requirements.txt') -q
    }
}

function Start-BackgroundBat([string]$Name, [string]$WorkDir, [string]$BatRelative, [int]$Port, [int]$WaitSec) {
    if (-not (Test-Path $WorkDir)) {
        Write-Warn "$Name 目录不存在，跳过: $WorkDir"
        return $false
    }
    if (Test-PortListening $Port) {
        Write-Ok "$Name 已在运行 (端口 $Port)"
        return $true
    }
    $bat = Join-Path $WorkDir $BatRelative
    if (-not (Test-Path $bat)) {
        Write-Warn "$Name 启动脚本不存在: $bat"
        return $false
    }
    $log = Join-Path $LogDir "$Name.log"
    Write-Step "启动 $Name -> 端口 $Port (日志: $log)"
    Start-Process -FilePath 'cmd.exe' -ArgumentList @(
        '/c', "cd /d `"$WorkDir`" && `"$bat`" >> `"$log`" 2>&1"
    ) -WorkingDirectory $WorkDir -WindowStyle Hidden | Out-Null
    if (Wait-HttpOk "http://127.0.0.1:$Port" $WaitSec $Name) {
        Write-Ok "$Name 已就绪"
        return $true
    }
    Write-Warn "$Name 可能仍在启动，请查看 $log"
    return $false
}

function Resolve-RedisServerExe {
    if ($Cfg.Redis.ServerExe -and (Test-Path $Cfg.Redis.ServerExe)) {
        return $Cfg.Redis.ServerExe
    }
    $cmd = Get-Command redis-server -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($candidate in @(
            'C:\Program Files\Redis\redis-server.exe',
            'C:\Program Files\Memurai\memurai.exe'
        )) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Wait-RedisPort([int]$Port, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortListening $Port) { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Start-RedisWindows {
    $port = [int]$Cfg.Redis.Port
    if (Test-PortListening $port) {
        Write-Ok "Redis 已在运行 (端口 $port)"
        return $true
    }

    $svcName = $Cfg.Redis.ServiceName
    if ($svcName) {
        $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
        if ($svc) {
            Write-Step "启动 Redis Windows 服务 ($svcName)"
            try {
                if ($svc.Status -ne 'Running') {
                    Start-Service -Name $svcName -ErrorAction Stop
                }
                if (Wait-RedisPort $port ([int]$Cfg.Redis.WaitSeconds)) {
                    Write-Ok "Redis 服务已就绪"
                    return $true
                }
                Write-Warn "Redis 服务已启动但端口 $port 未监听"
            } catch {
                Write-Warn "启动 Redis 服务失败: $_"
            }
        }
    }

    $exe = Resolve-RedisServerExe
    if (-not $exe) {
        Write-Warn "未找到 redis-server.exe，请安装 Windows Redis 或 Memurai"
        return $false
    }

    $redisDir = Split-Path $exe -Parent
    $conf = Join-Path $redisDir ($Cfg.Redis.ConfFile)
    if (-not (Test-Path $conf)) {
        $conf = Join-Path $redisDir 'memurai.conf'
    }
    $args = if (Test-Path $conf) { "`"$conf`"" } else { "--port $port" }
    $log = Join-Path $LogDir 'redis.log'
    Write-Step "启动 Redis (本机) -> 端口 $port (日志: $log)"
    Start-Process -FilePath 'cmd.exe' -ArgumentList @(
        '/c', "cd /d `"$redisDir`" && `"$exe`" $args >> `"$log`" 2>&1"
    ) -WorkingDirectory $redisDir -WindowStyle Hidden | Out-Null

    if (Wait-RedisPort $port ([int]$Cfg.Redis.WaitSeconds)) {
        Write-Ok "Redis 已就绪"
        return $true
    }
    Write-Warn "Redis 在 $($Cfg.Redis.WaitSeconds)s 内未就绪，请查看 $log"
    return $false
}

function Start-Redis {
    if (-not $Cfg.Redis.Enabled) { return $true }
    $mode = if ($Cfg.Redis.Mode) { $Cfg.Redis.Mode } else { 'windows' }
    if ($mode -eq 'docker' -and (Get-Command docker -ErrorAction SilentlyContinue)) {
        $port = [int]$Cfg.Redis.Port
        if (Test-PortListening $port) {
            Write-Ok "Redis 已在运行 (端口 $port)"
            return $true
        }
        $compose = Join-Path $Root $Cfg.Redis.ComposeFile
        if (Test-Path $compose) {
            Write-Step "启动 Redis (docker compose) -> 端口 $port"
            docker compose -f $compose up -d 2>&1 | Out-Null
            if (Wait-RedisPort $port ([int]$Cfg.Redis.WaitSeconds)) {
                Write-Ok "Redis 已就绪"
                return $true
            }
        }
    }
    return Start-RedisWindows
}

function Stop-Redis {
    if (-not $Cfg.Redis.Enabled) { return }
    $port = [int]$Cfg.Redis.Port
    $svcName = $Cfg.Redis.ServiceName
    if ($svcName) {
        $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
        if ($svc -and $svc.Status -eq 'Running') {
            Stop-Service -Name $svcName -Force -ErrorAction SilentlyContinue
            Write-Host "  已停止 Redis 服务 ($svcName)"
            return
        }
    }
    Stop-Port $port 'Redis'
}

function Init-TriageDb {
    if (-not $Cfg.TriageDb.Enabled) { return $true }
    if (-not $Cfg.TriageDb.InitOnStart) { return $true }
    Ensure-Venv
    $dbPath = Join-Path $Root $Cfg.TriageDb.Path
    Write-Step "初始化 Triage SQLite -> $($Cfg.TriageDb.Path)"
    $env:PYTHONPATH = '.'
    $py = @"
from pathlib import Path
from app.infra.triage_session_store import TriageSessionStore
p = r'$dbPath'
Path(p).parent.mkdir(parents=True, exist_ok=True)
TriageSessionStore(p).init_schema()
print(p)
"@
    & $Python -c $py
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Triage SQLite 初始化失败"
        return $false
    }
    Write-Ok "Triage SQLite 已就绪"
    return $true
}

function Start-Api {
    $port = [int]$Cfg.Api.Port
    if (-not $Cfg.Api.Enabled) { return $true }
    if (Test-PortListening $port) {
        Write-Ok "API 已在运行 (http://$($Cfg.Api.Host):$port)"
        return $true
    }
    Ensure-Venv
    $log = Join-Path $LogDir 'api.log'
    $reloadArg = if ($Cfg.Api.Reload) { '--reload --reload-dir app' } else { '' }
    Write-Step "启动 API -> http://$($Cfg.Api.Host):$port (日志: $log)"
    # UTF-8 控制台 + 无缓冲 stdout，避免 api.log 乱码且长时间不落盘
    $cmd = 'chcp 65001 >nul & set PYTHONIOENCODING=utf-8 & set PYTHONUTF8=1 & set PYTHONUNBUFFERED=1 & set PYTHONPATH=. & set HTTP_PROXY= & set HTTPS_PROXY= & set ALL_PROXY= & cd /d "' + $Root + '" & "' + $Uvicorn + '" app.main:app --host ' + $Cfg.Api.Host + ' --port ' + $port + ' ' + $reloadArg + ' >> "' + $log + '" 2>&1'
    Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $cmd -WorkingDirectory $Root -WindowStyle Hidden | Out-Null
    if (Wait-HttpOk "http://$($Cfg.Api.Host):$port/healthz" 60 'API') {
        Write-Ok "API 已就绪"
        return $true
    }
    Write-Warn "API 可能仍在启动，请查看 $log"
    return $false
}

function Invoke-Verify {
    $ok = $true
    $osPort = Get-PortFromUrl $Cfg.OpenSearch.Url

    if ($Cfg.Verify.Redis -and $Cfg.Redis.Enabled) {
        Write-Step "验证 Redis $($Cfg.Redis.Uri)"
        Ensure-Venv
        $env:PYTHONPATH = '.'
        $env:HTTP_PROXY = ''
        $env:HTTPS_PROXY = ''
        $env:ALL_PROXY = ''
        $uri = $Cfg.Redis.Uri
        $py = @"
import redis
redis.Redis.from_url('$uri', socket_connect_timeout=3).ping()
print('ping ok')
"@
        try {
            & $Python -c $py | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Ok 'Redis PING OK'
            } else {
                Write-Err 'Redis PING 失败'
                $ok = $false
            }
        } catch {
            Write-Err 'Redis 不可达'
            $ok = $false
        }
    }

    if ($Cfg.Verify.TriageDb -and $Cfg.TriageDb.Enabled) {
        $dbPath = Join-Path $Root $Cfg.TriageDb.Path
        Write-Step "验证 Triage SQLite $dbPath"
        if (-not (Test-Path $dbPath)) {
            Write-Err 'Triage SQLite 文件不存在'
            $ok = $false
        } else {
            Ensure-Venv
            $env:PYTHONPATH = '.'
            $py = @"
import sqlite3
c = sqlite3.connect(r'$dbPath')
n = c.execute('SELECT COUNT(*) FROM triage_sessions').fetchone()[0]
print(n)
"@
            try {
                $count = & $Python -c $py
                if ($LASTEXITCODE -eq 0) {
                    Write-Ok "triage_sessions 表可读，当前 $count 条记录"
                } else {
                    Write-Err 'Triage SQLite 表不可读'
                    $ok = $false
                }
            } catch {
                Write-Err 'Triage SQLite 验证失败'
                $ok = $false
            }
        }
    }

    if ($Cfg.Verify.OpenSearch -and $Cfg.OpenSearch.Enabled) {
        Write-Step "验证 OpenSearch $($Cfg.OpenSearch.Url)"
        try {
            $info = Invoke-RestMethod -Uri $Cfg.OpenSearch.Url -TimeoutSec 10
            Write-Ok "OpenSearch $($info.version.number) / $($info.cluster_name)"
        } catch {
            Write-Err "OpenSearch 不可达"
            $ok = $false
        }
    }

    if ($Cfg.Verify.ReadyEndpoint -and $Cfg.Api.Enabled) {
        $readyUrl = "http://$($Cfg.Api.Host):$($Cfg.Api.Port)/ready"
        Write-Step "验证 $readyUrl"
        try {
            $ready = Invoke-RestMethod -Uri $readyUrl -TimeoutSec 30
            if ($ready.status -eq 'ok') {
                Write-Ok "LangGraph=$($ready.langgraph.ok) OpenSearch=$($ready.opensearch.ok) Redis=$($ready.redis.ok) TriageDb=$($ready.triage_db.ok)"
            } else {
                Write-Warn "ready status=$($ready.status): $($ready | ConvertTo-Json -Compress)"
                $ok = $false
            }
        } catch {
            Write-Err "API /ready 失败"
            $ok = $false
        }
    }

    if ($Cfg.Verify.RagIndexCount -and $Cfg.OpenSearch.Enabled) {
        $idx = $Cfg.Verify.RagIndexName
        Write-Step "验证索引 $idx"
        try {
            $count = Invoke-RestMethod -Uri "$($Cfg.OpenSearch.Url)/$idx/_count" -TimeoutSec 10
            $n = $count.count
            if ($n -gt 0) {
                Write-Ok "索引 $idx 文档数: $n"
            } else {
                Write-Warn "索引 $idx 为空，可运行: .venv\Scripts\python.exe sourceData\opensearch_rag_kb.py --no-embed"
            }
        } catch {
            Write-Warn "索引 $idx 不存在或不可查，可运行入库脚本 sourceData\opensearch_rag_kb.py"
        }
    }

    if ($Cfg.Verify.ChatSmoke -and $Cfg.Api.Enabled) {
        Write-Step 'Chat smoke (calls LLM)'
        try {
            $chatUrl = 'http://{0}:{1}/chat' -f $Cfg.Api.Host, $Cfg.Api.Port
            $payload = @{ user_id = 'dev-smoke'; message = 'headache' } | ConvertTo-Json -Compress
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
            $resp = Invoke-RestMethod -Uri $chatUrl -Method POST -Body $bytes -ContentType 'application/json; charset=utf-8' -TimeoutSec 120
            $route = $resp.intent_result.triage_route
            Write-Ok "Chat smoke OK route=$route reply_len=$($resp.reply.Length)"
        } catch {
            Write-Err 'Chat smoke failed'
            $ok = $false
        }
    }

    if ($Cfg.Verify.ChatApiTests -and $Cfg.Api.Enabled) {
        Write-Step "运行 scripts/archive/test_chat_api.py"
        Ensure-Venv
        $env:PYTHONPATH = '.'
        & $Python (Join-Path $Root 'scripts/archive/test_chat_api.py') --base-url "http://$($Cfg.Api.Host):$($Cfg.Api.Port)"
        if ($LASTEXITCODE -ne 0) { $ok = $false } else { Write-Ok 'test_chat_api 全量通过' }
    }

    if ($Cfg.Dashboards.Enabled) {
        Write-Step "验证 Dashboards $($Cfg.Dashboards.Url)"
        try {
            Invoke-WebRequest -Uri $Cfg.Dashboards.Url -TimeoutSec 5 -UseBasicParsing | Out-Null
            Write-Ok 'Dashboards 可访问'
        } catch {
            Write-Warn 'Dashboards 不可访问'
        }
    }

    return $ok
}

function Show-Status {
    Write-Step '服务状态'
    $items = @(
        @{ Name = 'Redis'; Enabled = $Cfg.Redis.Enabled; Port = [int]$Cfg.Redis.Port; Url = $Cfg.Redis.Uri }
        @{ Name = 'OpenSearch'; Enabled = $Cfg.OpenSearch.Enabled; Port = (Get-PortFromUrl $Cfg.OpenSearch.Url); Url = $Cfg.OpenSearch.Url }
        @{ Name = 'Dashboards'; Enabled = $Cfg.Dashboards.Enabled; Port = (Get-PortFromUrl $Cfg.Dashboards.Url); Url = $Cfg.Dashboards.Url }
        @{ Name = 'API'; Enabled = $Cfg.Api.Enabled; Port = [int]$Cfg.Api.Port; Url = "http://$($Cfg.Api.Host):$($Cfg.Api.Port)" }
    )
    foreach ($item in $items) {
        if (-not $item.Enabled) {
            Write-Host ("  {0,-12} 已禁用" -f $item.Name)
            continue
        }
        $up = Test-PortListening $item.Port
        $state = if ($up) { '运行中' } else { '未运行' }
        $color = if ($up) { 'Green' } else { 'DarkGray' }
        Write-Host ("  {0,-12} {1,-6} {2}" -f $item.Name, $state, $item.Url) -ForegroundColor $color
    }
    if ($Cfg.TriageDb.Enabled) {
        $dbPath = Join-Path $Root $Cfg.TriageDb.Path
        $exists = Test-Path $dbPath
        $state = if ($exists) { '已就绪' } else { '未初始化' }
        $color = if ($exists) { 'Green' } else { 'DarkGray' }
        Write-Host ("  {0,-12} {1,-6} {2}" -f 'TriageDb', $state, $Cfg.TriageDb.Path) -ForegroundColor $color
    } else {
        Write-Host ("  {0,-12} 已禁用" -f 'TriageDb')
    }
    Write-Host ""
    Write-Host "  配置: scripts/dev-services.config.ps1"
    Write-Host "  日志: $LogDir"
}

function Start-All {
    if ($Cfg.Redis.Enabled) {
        Start-Redis | Out-Null
    }
    if ($Cfg.TriageDb.Enabled) {
        Init-TriageDb | Out-Null
    }
    if ($Cfg.OpenSearch.Enabled) {
        $osHome = Join-Path $Root $Cfg.OpenSearch.Home
        $osPort = Get-PortFromUrl $Cfg.OpenSearch.Url
        Start-BackgroundBat 'opensearch' $osHome 'bin\opensearch.bat' $osPort $Cfg.OpenSearch.WaitSeconds | Out-Null
    }
    if ($Cfg.Dashboards.Enabled) {
        $dbHome = Join-Path $Root $Cfg.Dashboards.Home
        $dbPort = Get-PortFromUrl $Cfg.Dashboards.Url
        Start-BackgroundBat 'dashboards' $dbHome 'bin\opensearch-dashboards.bat' $dbPort $Cfg.Dashboards.WaitSeconds | Out-Null
    }
    if ($Cfg.Api.Enabled) {
        Start-Api | Out-Null
    }
}

function Stop-All {
    Write-Step '停止服务'
    if ($Cfg.Api.Enabled) { Stop-Port ([int]$Cfg.Api.Port) 'API' }
    if ($Cfg.Dashboards.Enabled) { Stop-Port (Get-PortFromUrl $Cfg.Dashboards.Url) 'Dashboards' }
    if ($Cfg.OpenSearch.Enabled) { Stop-Port (Get-PortFromUrl $Cfg.OpenSearch.Url) 'OpenSearch' }
    if ($Cfg.Redis.Enabled) { Stop-Redis }
}

switch ($Action) {
    'start' {
        Write-Step "项目: $Root"
        Start-All
        Show-Status
        if (-not $SkipVerify) {
            Write-Host ""
            if (-not (Invoke-Verify)) { exit 1 }
        }
        Write-Host ""
        Write-Ok '开发环境已拉起。CLI: .venv\Scripts\python.exe cli.py'
        Write-Host "     API 文档: http://$($Cfg.Api.Host):$($Cfg.Api.Port)/docs"
    }
    'verify' {
        Show-Status
        if (-not (Invoke-Verify)) { exit 1 }
    }
    'status' { Show-Status }
    'stop'   { Stop-All; Show-Status }
}
