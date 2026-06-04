# scripts/k8s-deploy.ps1
# -----------------------------------------------------------------------------
# Idempotent one-command deploy of the credit scoring API to local minikube.
# Builds the image, loads it into the cluster, applies every Kubernetes
# manifest in `kubernetes/`, and waits for the Deployment to be Ready.
#
# Prereqs:
#   - Docker Desktop running, with at least 4 GB / 2 CPUs allocated.
#   - kubectl, minikube on PATH.
#   - MLflow tracking server running on the host (docker compose up -d mlflow).
#   - For first run: add `127.0.0.1 credit.local` to the hosts file and run
#     `minikube tunnel` in an elevated shell (see kubernetes/README.md).
# -----------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot

$image = "credit-serving:dev"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

# -----------------------------------------------------------------------------
# 1. Cluster up
# -----------------------------------------------------------------------------
Write-Step "Ensuring minikube cluster is running"
$status = minikube status --format "{{.Host}}" 2>$null
if ($status -ne "Running") {
    Write-Host "Cluster not running -- starting it now (this takes 2-5 minutes the first time)..."
    minikube start
} else {
    Write-Host "Cluster already running."
}

# -----------------------------------------------------------------------------
# 2. Addons (idempotent -- safe to re-enable)
# -----------------------------------------------------------------------------
Write-Step "Enabling ingress + metrics-server addons"
minikube addons enable ingress | Out-Null
minikube addons enable metrics-server | Out-Null
Write-Host "Both addons enabled."

# -----------------------------------------------------------------------------
# 3. Build the serving image
# -----------------------------------------------------------------------------
Write-Step "Building $image from docker/serving.Dockerfile"
docker build -t $image -f docker/serving.Dockerfile .

# -----------------------------------------------------------------------------
# 4. Load the image into the cluster
# -----------------------------------------------------------------------------
Write-Step "Loading $image into minikube"
minikube image load $image
Write-Host "Image loaded. Verifying..."
$present = minikube image ls | Select-String credit-serving
if (-not $present) {
    throw "Image $image did not appear in `minikube image ls` after load."
}
Write-Host "Verified: $present"

# -----------------------------------------------------------------------------
# 5. Apply every manifest
# -----------------------------------------------------------------------------
Write-Step "Applying manifests from kubernetes/"
kubectl apply -f kubernetes/

# -----------------------------------------------------------------------------
# 6. Wait for the Deployment to be Ready
# -----------------------------------------------------------------------------
Write-Step "Waiting for credit-serving Deployment to be Ready (timeout 3 minutes)"
kubectl rollout status deployment/credit-serving --timeout=180s

# -----------------------------------------------------------------------------
# 7. Summary
# -----------------------------------------------------------------------------
Write-Step "Deployment complete"
Write-Host ""
Write-Host "Cluster state:" -ForegroundColor Green
kubectl get deploy,svc,ingress,hpa

Write-Host ""
Write-Host "Reach the API at:" -ForegroundColor Green
Write-Host "  http://credit.local/healthz"
Write-Host "  http://credit.local/readyz"
Write-Host "  http://credit.local/model_info"
Write-Host ""
Write-Host "If those time out, the minikube tunnel isn't running. In an admin"
Write-Host "PowerShell: minikube tunnel  (leave it running)."
