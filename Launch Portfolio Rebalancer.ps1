[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$appPath = Join-Path $repoRoot "portfolio_rebalancer_database.py"
$requirementsPath = Join-Path $repoRoot "requirements.txt"
$venvRoot = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$dependencyMarker = Join-Path $venvRoot ".requirements.sha256"

$host.UI.RawUI.WindowTitle = "Portfolio Rebalancer — Local"
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath $appPath)) {
    throw "Rebalancer source not found: $appPath"
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Preparing the private local environment (first launch only)..." -ForegroundColor Cyan
    & py -3 -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Python could not create the local environment."
    }
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip could not be prepared."
    }
}

$requirementsHash = (Get-FileHash -LiteralPath $requirementsPath -Algorithm SHA256).Hash
$installedHash = if (Test-Path -LiteralPath $dependencyMarker) {
    (Get-Content -LiteralPath $dependencyMarker -Raw).Trim()
} else {
    ""
}

if ($installedHash -ne $requirementsHash) {
    Write-Host "Installing or updating required packages..." -ForegroundColor Cyan
    & $venvPython -m pip install -r $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Package installation failed. Review the message above and relaunch."
    }
    Set-Content -LiteralPath $dependencyMarker -Value $requirementsHash -Encoding ascii
}

Write-Host "Starting Portfolio Rebalancer at http://localhost:8501" -ForegroundColor Green
Write-Host "Keep this window open. Press Ctrl+C here to stop the app." -ForegroundColor DarkGray

& $venvPython -m streamlit run $appPath `
    --server.address localhost `
    --server.port 8501 `
    --browser.gatherUsageStats false

if ($LASTEXITCODE -ne 0) {
    throw "The Portfolio Rebalancer stopped with exit code $LASTEXITCODE."
}
