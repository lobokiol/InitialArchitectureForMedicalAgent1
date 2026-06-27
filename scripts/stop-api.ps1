$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

$port = 8000
Write-Host "==> free port $port (LISTEN only) ..."

function Test-PortListen([int]$P) {
    return $null -ne (Get-NetTCPConnection -LocalPort $P -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1)
}

for ($i = 0; $i -lt 5; $i++) {
    if (-not (Test-PortListen $port)) { break }
    $procIds = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $procIds) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        $name = if ($p) { $p.ProcessName } else { "?" }
        taskkill /F /T /PID $procId 2>$null | Out-Null
        Write-Host "  killed PID $procId ($name)"
    }
    Start-Sleep -Milliseconds 600
}

if (Test-PortListen $port) {
    # 孤儿 socket / reload 父进程已退出但端口未释放时，结束本项目的 python/uvicorn
    $venvPy = Join-Path $Root '.venv\Scripts\python.exe'
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ExecutablePath -and (
                $_.ExecutablePath -ieq $venvPy -or
                $_.CommandLine -match 'uvicorn\s+app\.main:app'
            )
        } |
        ForEach-Object {
            taskkill /F /T /PID $_.ProcessId 2>$null | Out-Null
            Write-Host "  killed venv uvicorn PID $($_.ProcessId)"
        }
    Start-Sleep -Milliseconds 800
}

if (Test-PortListen $port) {
    Write-Warning "Port $port still in use. Run: .\start-dev.ps1 -Action stop"
    netstat -ano | findstr ":$port"
    exit 1
}

Write-Host "[OK] port $port is free" -ForegroundColor Green
exit 0
