# scripts/docker-build.ps1
# -----------------------------------------------------------
# Build the credit-pipeline training image.
#
# Usage:
#     .\scripts\docker-build.ps1
#     .\scripts\docker-build.ps1 -Tag v0.4.0
#     .\scripts\docker-build.ps1 -NoCache
#
# Tag defaults to "dev" - overridable for release builds. -NoCache forces
# a full rebuild ignoring layer cache (useful when debugging the
# Dockerfile itself).
# -----------------------------------------------------------

[CmdletBinding()]
param(
    [string]$Tag = "dev",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"

# Resolve project root (this script lives in <root>/scripts/).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$ImageName = "credit-pipeline:$Tag"
$DockerfilePath = "docker/training.Dockerfile"

Write-Host "Building $ImageName from $DockerfilePath ..." -ForegroundColor Cyan

$BuildArgs = @(
    "build"
    "-t", $ImageName
    "-f", $DockerfilePath
)

if ($NoCache) {
    $BuildArgs += "--no-cache"
    Write-Host "  --no-cache enabled (slow, full rebuild)" -ForegroundColor Yellow
}

# Build context is project root.
$BuildArgs += "."

& docker @BuildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Build OK: $ImageName" -ForegroundColor Green
docker images $ImageName
