# scripts/docker-smoke.ps1
# -----------------------------------------------------------
# Smoke-test the credit-pipeline image: starts a container with
# the default CMD (a one-liner that imports sklearn / pandas and
# prints a confirmation), confirms it exits cleanly.
#
# Use this BEFORE docker-run.ps1 to catch problems with the image
# itself (missing deps, broken venv, wrong Python version).
#
# Usage:
#     .\scripts\docker-smoke.ps1
#     .\scripts\docker-smoke.ps1 -Tag v0.4.0
# -----------------------------------------------------------

[CmdletBinding()]
param(
    [string]$Tag = "dev"
)

$ErrorActionPreference = "Stop"

$ImageName = "credit-pipeline:$Tag"

Write-Host "Smoke-testing $ImageName ..." -ForegroundColor Cyan

docker run --rm $ImageName

if ($LASTEXITCODE -ne 0) {
    Write-Host "Smoke test FAILED (exit $LASTEXITCODE) — image is broken" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Smoke OK — image is healthy and ready to run the pipeline" -ForegroundColor Green
