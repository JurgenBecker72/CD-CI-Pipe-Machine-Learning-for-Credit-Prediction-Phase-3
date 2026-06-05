# Cloud deployment (Azure AKS)

The same scoring API the local minikube manifests deploy, targeted at a
managed cloud cluster. Differences vs. `kubernetes/`:

| Concern | Local (`kubernetes/`) | Cloud (`kubernetes/cloud/`) |
|---|---|---|
| Cluster | minikube, single-node | AKS, 1 B2s node (autoscalable) |
| Image source | Local docker daemon (loaded via `minikube image load`) | Azure Container Registry (ACR) |
| Model bundle | Downloaded from MLflow at pod startup | Baked into the image at build time |
| External access | `minikube tunnel` + hosts file | Azure-provisioned public IP via `Service.type=LoadBalancer` |
| Cost | Free | ~$0.50/hour while running |

## What you need

* Azure CLI (`az`) and authenticated to a subscription with rights to create resources.
* `kubectl` (the same binary you use locally — context switching handles the rest).
* Docker Desktop running, with the local MLflow stack up (`docker compose up -d mlflow`) so you can extract the model bundle.

## End-to-end deploy

**One-time per machine** — create the Azure resource group, container
registry, and cluster:

```powershell
.\scripts\azure-deploy.ps1 -CreateInfra
```

(Or run the equivalent `az group create / az acr create / az aks create`
commands by hand — see `scripts/azure-deploy.ps1` for the canonical sequence.)

**Every time you want to deploy a fresh version of the API:**

```powershell
.\scripts\azure-deploy.ps1
```

The script:

1. Exports the latest `credit_scorecard@production` model from MLflow into
   `model-bundle/` via `scripts/export_model_bundle.py`.
2. Builds `creditscoringjbecker.azurecr.io/credit-serving:v{N}` from
   `docker/serving.Dockerfile` (the model bundle is baked in here).
3. Logs in to ACR and pushes the image.
4. Switches kubectl context to AKS.
5. Applies every manifest in `kubernetes/cloud/`.
6. Waits for the Deployment to roll out.
7. Prints the LoadBalancer's external IP.

## Verifying the deploy

Once the script reports the external IP (call it `$IP`):

```powershell
curl http://$IP/healthz   -UseBasicParsing   # liveness
curl http://$IP/readyz    -UseBasicParsing   # readiness + sidecars loaded
curl http://$IP/model_info -UseBasicParsing  # registered model metadata

$payload = Get-Content scripts\smoke_payload.json -Raw
Invoke-WebRequest -Uri http://$IP/v1/score `
    -Method POST -ContentType "application/json" `
    -Body $payload -UseBasicParsing |
    Select-Object -ExpandProperty Content
```

All four should return 200 and produce real scoring output.

## How the loader knows it's in offline mode

The serving image looks at the `MODEL_BUNDLE_PATH` env var. If it's set
(as it is in `kubernetes/cloud/deployment.yaml`), the loader reads
model and sidecars from local files baked into the image. If it isn't
(local-dev minikube path), the loader falls back to its original
behaviour: contact the MLflow tracking URI and download artefacts at
startup.

The two paths are otherwise identical — same `ModelBundle` dataclass,
same scoring code, same response shape. Only the source of the bytes
changes.

## Tearing it down (do this when you're done)

AKS bills per hour the node is running. Two options:

```powershell
# Pause the node (cheapest -- nodes off, control plane free, restart in 5 min)
az aks stop --resource-group credit-scoring-rg --name credit-scoring-aks --no-wait

# Or delete the whole resource group (cluster, ACR, everything)
az group delete --name credit-scoring-rg --yes --no-wait
```

`stop` is the right call if you might want to demo the service again
within a few weeks. `delete` is the right call if you're finished and
want zero ongoing cost.

## What's deliberately not here

* **Ingress + TLS.** Production would terminate TLS at an Ingress
  controller (nginx-ingress + cert-manager + Let's Encrypt) rather
  than expose HTTP via a raw LoadBalancer.
* **Cloud secrets manager.** The pod's config is passed via plain env
  vars on the Deployment. Production would use Azure Key Vault wired
  to the cluster via the External Secrets Operator.
* **Separate environments.** Just one cluster here. Production has
  dev / staging / prod with promotion gates between them.
* **GitOps controller.** `kubectl apply` is run manually by the
  deploy script. Production typically uses ArgoCD or Flux to make
  the cluster sync itself from a git repo.
* **Observability stack.** The serving image exposes Prometheus
  metrics on `/metrics`, but nothing scrapes them. Production would
  add Prometheus + Grafana + Loki + alerting.

Each is a separate afternoon's work and individually well-understood;
all five are deferred as scope choices for this build.
