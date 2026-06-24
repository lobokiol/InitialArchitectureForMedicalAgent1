$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:PYTHONPATH = "."
$env:PYTHONUNBUFFERED = "1"
# DashScope / 本地 OpenSearch 直连，避免 SOCKS 代理导致 LLM/embedding 失败
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""

$Uv = "$env:USERPROFILE\.local\bin\uv.exe"
$Python = "$Root\.venv\Scripts\python.exe"
$Uvicorn = "$Root\.venv\Scripts\uvicorn.exe"

if (Test-Path $Uv) {
    & $Uv pip install -r requirements.txt -q
}

if (-not (Test-Path $Uvicorn)) {
    Write-Error ".venv not found. Run: uv venv --python 3.11 && uv pip install -r requirements.txt"
    exit 1
}

# 避免 start-dev 后台 API 占用 8000，导致前台终端收不到 /chat 日志
$port = 8000
for ($i = 0; $i -lt 3; $i++) {
    $pids = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    if (-not $pids) { break }
    foreach ($procId in $pids) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500
}
$left = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
if ($left) {
    Write-Warning "端口 $port 仍被占用，请先执行: .\start-dev.cmd -Action stop"
}

& $Uvicorn app.main:app --host 0.0.0.0 --port $port --reload --reload-dir app
