# Remove Anaconda3 completely and prevent future auto-activation.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/remove-anaconda.ps1

$ErrorActionPreference = 'Stop'
$AnacondaRoot = 'D:\anaconda3'

function Write-Step([string]$Msg) { Write-Host "==> $Msg" -ForegroundColor Cyan }
function Write-Ok([string]$Msg)   { Write-Host "[OK] $Msg" -ForegroundColor Green }
function Write-Warn([string]$Msg) { Write-Host "[WARN] $Msg" -ForegroundColor Yellow }

function Remove-AnacondaPathEntries([string]$PathValue) {
    if (-not $PathValue) { return $PathValue }
    $parts = $PathValue -split ';' | Where-Object {
        $_ -and ($_ -notmatch '(?i)anaconda|(?i)\\conda\\|(?i)condabin')
    }
    return ($parts -join ';').Trim(';')
}

function Clear-CondaEnvVars([System.EnvironmentVariableTarget]$Target) {
    $names = @(
        'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'CONDA_PROMPT_MODIFIER',
        'CONDA_EXE', 'CONDA_PYTHON_EXE', '_CONDA_ROOT', '_CE_CONDA', '_CE_M'
    )
    foreach ($name in $names) {
        try {
            $val = [Environment]::GetEnvironmentVariable($name, $Target)
            if ($null -ne $val) {
                [Environment]::SetEnvironmentVariable($name, $null, $Target)
                Write-Ok "Removed env $name ($Target)"
            }
        } catch {
            Write-Warn "Could not remove env $name ($Target): $_"
        }
    }
}

function Update-PathVar([System.EnvironmentVariableTarget]$Target) {
    try {
        $current = [Environment]::GetEnvironmentVariable('Path', $Target)
        if (-not $current) { return }
        $updated = Remove-AnacondaPathEntries $current
        if ($updated -ne $current) {
            [Environment]::SetEnvironmentVariable('Path', $updated, $Target)
            Write-Ok "Cleaned Path ($Target)"
        }
    } catch {
        Write-Warn "Could not update Path ($Target): $_"
    }
}

Write-Step 'Stop processes under Anaconda (if any)'
Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -and $_.Path -like "$AnacondaRoot*"
} | ForEach-Object {
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    Write-Ok "Stopped $($_.ProcessName) (PID $($_.Id))"
}

Write-Step 'Remove conda init from PowerShell profile'
$profiles = @(
    "$HOME\Documents\WindowsPowerShell\profile.ps1",
    "$HOME\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1",
    "$HOME\Documents\PowerShell\profile.ps1",
    "$HOME\Documents\PowerShell\Microsoft.PowerShell_profile.ps1"
)
foreach ($profilePath in $profiles) {
    if (-not (Test-Path $profilePath)) { continue }
    $content = Get-Content $profilePath -Raw -ErrorAction SilentlyContinue
    if ($content -match '(?s)#region conda initialize.*?#endregion') {
        $newContent = ($content -replace '(?s)\r?\n?#region conda initialize.*?#endregion\r?\n?', '').Trim()
        if ($newContent) {
            Set-Content -Path $profilePath -Value $newContent -Encoding UTF8
        } else {
            Remove-Item $profilePath -Force
        }
        Write-Ok "Cleaned profile: $profilePath"
    }
}

Write-Step 'Remove Anaconda profiles from Windows Terminal'
$wtSettings = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
if (Test-Path $wtSettings) {
    $json = Get-Content $wtSettings -Raw | ConvertFrom-Json
    $before = @($json.profiles.list).Count
    $json.profiles.list = @($json.profiles.list | Where-Object {
        -not ($_.commandline -match '(?i)anaconda|conda-hook|activate\.bat.*anaconda')
    })
    if (@($json.profiles.list).Count -lt $before) {
        $json | ConvertTo-Json -Depth 20 | Set-Content $wtSettings -Encoding UTF8
        Write-Ok 'Removed Anaconda entries from Windows Terminal'
    }
}

Write-Step 'Clean user environment variables'
Clear-CondaEnvVars User
Update-PathVar User

Write-Step 'Clean machine environment variables (may need admin)'
try {
    Clear-CondaEnvVars Machine
    Update-PathVar Machine
} catch {
    Write-Warn "Machine env cleanup skipped (run as Administrator if anaconda remains on system Path): $_"
}

Write-Step 'Remove Anaconda config directories'
$extraDirs = @(
    "$HOME\.conda",
    "$HOME\.continuum",
    "$env:APPDATA\.anaconda",
    "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)"
)
foreach ($dir in $extraDirs) {
    if (Test-Path $dir) {
        Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Ok "Removed $dir"
    }
}

Write-Step "Remove Anaconda installation: $AnacondaRoot"
if (Test-Path $AnacondaRoot) {
    Remove-Item $AnacondaRoot -Recurse -Force -ErrorAction Stop
    Write-Ok "Deleted $AnacondaRoot"
} else {
    Write-Ok 'Anaconda directory already absent'
}

# Refresh current session PATH (best effort)
$env:Path = Remove-AnacondaPathEntries $env:Path
foreach ($name in @('CONDA_PREFIX','CONDA_DEFAULT_ENV','CONDA_EXE','CONDA_PYTHON_EXE')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Ok 'Anaconda removal finished. Open a NEW terminal — encodings/conda errors should be gone.'
Write-Host 'Project Python: D:\InitialArchitectureForMedicalAgent1\.venv\Scripts\python.exe'
