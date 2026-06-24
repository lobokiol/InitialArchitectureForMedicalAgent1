$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$port = 8000
Write-Host "==> 清理端口 $port ..."

for ($i = 0; $i -lt 5; $i++) {
    $procIds = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    if (-not $procIds) { break }
    foreach ($procId in $procIds) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        $name = if ($p) { $p.ProcessName } else { "?" }
        taskkill /F /PID $procId 2>$null | Out-Null
        Write-Host "  已结束 PID $procId ($name)"
    }
    Start-Sleep -Milliseconds 500
}

$left = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($left) {
    Write-Warning "端口 $port 仍被占用，请用管理员 PowerShell 再执行一次，或重启终端后重试"
    netstat -ano | findstr ":$port"
    exit 1
}

Write-Host "[OK] 端口 $port 已空闲"
