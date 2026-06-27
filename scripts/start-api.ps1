$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

$env:PYTHONPATH = "."
$env:PYTHONUNBUFFERED = "1"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""

$Uv = "$env:USERPROFILE\.local\bin\uv.exe"
$Uvicorn = "$Root\.venv\Scripts\uvicorn.exe"

if (Test-Path $Uv) {
    & $Uv pip install -r requirements.txt -q
}

if (-not (Test-Path $Uvicorn)) {
    Write-Error ".venv not found. Run: uv venv --python 3.11 && uv pip install -r requirements.txt"
    exit 1
}

$stopScript = Join-Path $Root 'scripts\stop-api.ps1'
if (Test-Path $stopScript) {
    & $stopScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERR] Port 8000 still in use. Run:" -ForegroundColor Red
        Write-Host "  .\start-dev.ps1 -Action stop" -ForegroundColor Yellow
        exit 1
    }
}

& $Uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
