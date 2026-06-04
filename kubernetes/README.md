# Kubernetes deployment

Local Kubernetes manifests for the credit scoring API. The same YAMLs apply
unchanged to a managed cloud cluster (GKE / EKS / AKS) — only the image
registry and a couple of annotations change.

## What gets deployed

| Object | File | Purpose |
|---|---|---|
| Deployment | [`deployment.yaml`](./deployment.yaml) | Runs one or more pods of `credit-serving:dev`, pointed at MLflow on the host via `host.minikube.internal:5000`, with `/healthz` and `/readyz` probes that gate traffic on model warm-up. |
| Service | [`service.yaml`](./service.yaml) | Stable in-cluster DNS name (`credit-serving.default.svc.cluster.local`) and load balancer in front of the pods. |
| Ingress | [`ingress.yaml`](./ingress.yaml) | nginx ingress controller rule routing `http://credit.local/` to the Service. |
| HorizontalPodAutoscaler | [`hpa.yaml`](./hpa.yaml) | Scales the Deployment between 1 and 3 replicas to keep average CPU at 50% of the per-pod request. |

## Prerequisites

* **Docker Desktop** — running, with at least 4 GB RAM / 2 CPUs allocated under Settings → Resources.
* **kubectl** — `kubectl version --client` should print a version >= 1.30.
* **minikube** — `minikube version` should print a version >= 1.36.
* **The MLflow tracking server** running on the host:
  ```powershell
  docker compose up -d mlflow
  ```
  Reachable at `http://localhost:5000` from your host, and from inside pods as
  `http://host.minikube.internal:5000`.

## One-command deploy

From the repository root:

```powershell
.\scripts\k8s-deploy.ps1
```

The script:

1. Starts the minikube cluster if it isn't already running.
2. Enables the `ingress` and `metrics-server` addons (idempotent).
3. Builds the serving image from `docker/serving.Dockerfile` and loads it
   into minikube's docker daemon (avoids needing a remote registry).
4. Applies every manifest in `kubernetes/`.
5. Waits for the `credit-serving` Deployment to be Ready.

## Reaching the API from the host

Two things to set up once per machine:

**Add a hostname mapping** (one-time, admin PowerShell):

```powershell
Add-Content "$env:WINDIR\System32\drivers\etc\hosts" "`n127.0.0.1 credit.local"
```

**Start the tunnel** (one terminal, admin PowerShell, keeps running):

```powershell
minikube tunnel
```

`minikube tunnel` proxies the cluster's Ingress (which binds inside Docker's
internal network) to your host's `127.0.0.1`. Required on Windows + Docker
driver; not needed on Linux with the kvm2 driver.

Then test:

```powershell
curl http://credit.local/healthz -UseBasicParsing
curl http://credit.local/readyz -UseBasicParsing
curl http://credit.local/model_info -UseBasicParsing
```

`/model_info` should return the registered model name, version, run ID,
feature count, and the persisted band-threshold + quantile-threshold sidecars.

## Verifying the autoscaler

Generate sustained load against `/v1/score` (which actually runs the model
and is much more CPU-intensive than `/healthz`) and watch replicas climb:

```powershell
# Terminal A
kubectl get hpa --watch

# Terminal B
$payload = Get-Content scripts\smoke_payload.json -Raw
while ($true) {
  Invoke-WebRequest -Uri http://credit.local/v1/score `
    -Method POST -ContentType "application/json" `
    -Body $payload -UseBasicParsing | Out-Null
}
```

Within ~60 seconds the `TARGETS` column in Terminal A will cross 50% and
the `REPLICAS` column will climb from 1 toward 3. Stop the loop with
Ctrl+C; replicas will drop back to 1 after the 60-second stabilisation
window in `hpa.yaml`.

## Tearing the cluster down

To remove just the application manifests (cluster stays up):

```powershell
kubectl delete -f kubernetes/
```

To shut the whole minikube cluster down (saves RAM):

```powershell
minikube stop
```

To delete the cluster entirely (you'll need to redeploy from scratch):

```powershell
minikube delete
```

## What's intentionally out of scope

This deployment is the production *shape* — not a production-ready stack.
Out of scope at this stage:

* **TLS** — Ingress serves HTTP only. Production would terminate TLS at
  the ingress (cert-manager + Let's Encrypt, or cloud-provided certs).
* **Secrets management** — MLflow URI is a plain env var. Production
  would use Kubernetes Secrets or an external secrets manager.
* **Image registry** — using `minikube image load`. Production would
  push to GHCR / ECR / GCR / ACR.
* **Multi-node cluster** — single-node minikube. Production clusters
  have 3+ nodes minimum for HA.
* **Cloud-native load balancer** — `Service.type: ClusterIP` + `Ingress`.
  Production on cloud usually adds `type: LoadBalancer` for the
  cloud-provided LB.
* **Persistent storage** — none required; the model is read-only and
  cached in pod memory. Production might add a PVC for log persistence
  or audit trails.

Each item is its own afternoon and is documented as future work.
