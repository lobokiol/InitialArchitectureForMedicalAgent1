$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:PYTHONPATH = "."

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
