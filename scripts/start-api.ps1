$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:PYTHONPATH = "."
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

& $Uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
