# scripts/azure-deploy.ps1
# -----------------------------------------------------------------------------
# Idempotent one-command deploy of the credit scoring API to Azure AKS.
# Exports the production model from local MLflow, bakes it into a serving
# image, pushes the image to ACR, applies the cloud manifests against the
# AKS cluster, and waits for the Deployment to roll out.
#
# Prerequisites:
#   - Docker Desktop running, with the MLflow stack up
#     (docker compose up -d mlflow).
#   - Azure CLI authenticated to a subscription that owns the cluster
#     (az login).
#   - kubectl on PATH; az-aks-get-credentials previously run for the
#     target cluster (.\scripts\azure-deploy.ps1 -CreateInfra handles
#     this for the first-time setup case).
#
# Usage:
#   .\scripts\azure-deploy.ps1                  # build + push + apply
#   .\scripts\azure-deploy.ps1 -ImageTag v6     # explicit image tag
#   .\scripts\azure-deploy.ps1 -CreateInfra     # first-time provisioning
# -----------------------------------------------------------------------------

param(
    [string]$ResourceGroup = "credit-scoring-rg",
    [string]$Location      = "westus2",
    [string]$RegistryName  = "creditscoringjbecker",
    [string]$ClusterName   = "credit-scoring-aks",
    [string]$ImageName     = "credit-serving",
    [string]$ImageTag      = "v6",
    [switch]$CreateInfra
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot

$imageRef = "${RegistryName}.azurecr.io/${ImageName}:${ImageTag}"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

# -----------------------------------------------------------------------------
# 0. (Optional) one-time infra provisioning
# -----------------------------------------------------------------------------
if ($CreateInfra) {
    Write-Step "Creating resource group, ACR, AKS (first-time setup)"

    az group create --name $ResourceGroup --location $Location | Out-Null

    az acr create `
        --resource-group $ResourceGroup `
        --name $RegistryName `
        --sku Basic `
        --location $Location | Out-Null

    az aks create `
        --resource-group $ResourceGroup `
        --name $ClusterName `
        --location $Location `
        --node-count 1 `
        --node-vm-size Standard_B2s `
        --enable-managed-identity `
        --attach-acr $RegistryName `
        --generate-ssh-keys | Out-Null

    Write-Host "Infra provisioned. Resource group: $ResourceGroup"
}

# -----------------------------------------------------------------------------
# 1. Wire kubectl up to the AKS cluster
# -----------------------------------------------------------------------------
Write-Step "Pointing kubectl at $ClusterName"
az aks get-credentials --resource-group $ResourceGroup --name $ClusterName --overwrite-existing | Out-Null

# -----------------------------------------------------------------------------
# 2. Export the production model bundle from local MLflow
# -----------------------------------------------------------------------------
Write-Step "Exporting production model bundle from MLflow"
uv run python scripts/export_model_bundle.py

# -----------------------------------------------------------------------------
# 3. Build the serving image with the bundle baked in
# -----------------------------------------------------------------------------
Write-Step "Building $imageRef"
docker build -t $imageRef -f docker/serving.Dockerfile .

# -----------------------------------------------------------------------------
# 4. Push the image to ACR
# -----------------------------------------------------------------------------
Write-Step "Pushing to ACR"
az acr login --name $RegistryName | Out-Null
docker push $imageRef

# -----------------------------------------------------------------------------
# 5. Apply the cloud manifests
# -----------------------------------------------------------------------------
Write-Step "Applying kubernetes/cloud/ manifests"
kubectl apply -f kubernetes/cloud/

# -----------------------------------------------------------------------------
# 6. Wait for the rollout
# -----------------------------------------------------------------------------
Write-Step "Waiting for Deployment to be Ready (timeout 3 min)"
kubectl rollout status deployment/credit-serving --timeout=180s

# -----------------------------------------------------------------------------
# 7. Surface the public IP
# -----------------------------------------------------------------------------
Write-Step "Resolving LoadBalancer public IP"
$externalIp = ""
for ($i = 0; $i -lt 30; $i++) {
    $externalIp = kubectl get svc credit-serving -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
    if ($externalIp) { break }
    Start-Sleep -Seconds 10
}

Write-Host ""
Write-Host "=== Deploy complete ===" -ForegroundColor Green
Write-Host ""
if ($externalIp) {
    Write-Host "Reach the API at:" -ForegroundColor Green
    Write-Host "  http://$externalIp/healthz"
    Write-Host "  http://$externalIp/readyz"
    Write-Host "  http://$externalIp/model_info"
    Write-Host ""
    Write-Host "Score a sample applicant:"
    Write-Host "  `$payload = Get-Content scripts\smoke_payload.json -Raw"
    Write-Host "  Invoke-WebRequest -Uri http://$externalIp/v1/score ``"
    Write-Host "      -Method POST -ContentType ""application/json"" ``"
    Write-Host "      -Body `$payload -UseBasicParsing"
} else {
    Write-Host "LoadBalancer IP not yet provisioned. Re-check with:" -ForegroundColor Yellow
    Write-Host "  kubectl get svc credit-serving --watch"
}

Write-Host ""
Write-Host "When you're done, pause the cluster to stop billing:" -ForegroundColor Yellow
Write-Host "  az aks stop --resource-group $ResourceGroup --name $ClusterName --no-wait"
