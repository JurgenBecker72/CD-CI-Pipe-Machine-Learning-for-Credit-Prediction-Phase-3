# scripts/docker-run.ps1
# -----------------------------------------------------------
# Run the credit-pipeline training inside a container, mounting
# host folders for data, models, and reports so the container
# can read your Excel and write back the trained artefacts.
#
# Mental model:
#   host folder           container folder    purpose
#   <root>/data           /app/data           raw input (read) + processed (write)
#   <root>/models         /app/models         trained scaler / model pickles (write)
#   <root>/reports        /app/reports        metrics.json, plots, summaries (write)
#
# Usage:
#     .\scripts\docker-run.ps1
#     .\scripts\docker-run.ps1 -Tag v0.4.0
#     .\scripts\docker-run.ps1 -DataPath /app/data/raw/other.xlsx
# -----------------------------------------------------------

[CmdletBinding()]
param(
    [string]$Tag = "dev",
    [string]$DataPath = ""
)

$ErrorActionPreference = "Stop"

# Resolve project root (this script lives in <root>/scripts/).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$ImageName = "credit-pipeline:$Tag"

# Make sure target host folders exist (Docker would create them itself,
# but with weird perms — better to create them here, owned by you).
foreach ($d in @("data", "data/raw", "data/processed", "models", "reports")) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Force -Path $d | Out-Null
    }
}

Write-Host "Running $ImageName with mounted volumes ..." -ForegroundColor Cyan

# Build the docker run command.
$RunArgs = @(
    "run", "--rm"
    "-v", "${ProjectRoot}/data:/app/data"
    "-v", "${ProjectRoot}/models:/app/models"
    "-v", "${ProjectRoot}/reports:/app/reports"
    $ImageName
    "-m", "pipelines.run_pipeline"
)

if ($DataPath -ne "") {
    $RunArgs += "--data-path", $DataPath
}

& docker @RunArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Run FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Run complete — see metrics in reports/, models in models/" -ForegroundColor Green
