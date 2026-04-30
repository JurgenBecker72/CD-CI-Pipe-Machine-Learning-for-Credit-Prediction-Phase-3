# Learning Build — Local Production Architecture

A hands-on plan to rebuild the production architecture from `docs/images/production_architecture.png` on your own laptop, piece by piece, using free tools only. The goal is not to ship a scoring service to a real lender. The goal is to personally wire up every box on that diagram so the concepts stop being abstractions.

Target audience: you, evenings and weekends, ~4–5 weekends total.

---

## Tech stack

Everything on this list runs on your laptop. No cloud accounts, no credit card, no DevOps team. Each tool is picked to be the *smallest viable substitute* for a real production component, so you learn the concept without paying the complexity tax.

| Production concept | What you'll use locally | Why this substitute | Cost |
|---|---|---|---|
| Source control | GitHub (already set up) | Already there | Free |
| CI/CD pipeline | GitHub Actions (already set up) | Already there | Free |
| Scoring service | **FastAPI** + Uvicorn | Tiny, modern, auto-generates a Swagger UI at `/docs`, widely used in real ML deployments | Free |
| Request validation | **Pydantic** (ships with FastAPI) | Same library real services use | Free |
| Container runtime | **Docker Desktop** | Industry standard. The same `Dockerfile` you write locally is the one a cloud would use | Free for personal use |
| Container registry | **Local Docker daemon** | You don't need a remote registry until you deploy to cloud | Free |
| Model registry | **Versioned folder** (`models/v1/`, `models/v2/`, …) with a `CURRENT` pointer file | Exactly what MLflow and SageMaker do under the hood, minus the web UI | Free |
| API gateway | **Skip** — call FastAPI directly on localhost | Gateways add auth and rate limiting; neither is a learning goal | — |
| Data warehouse | **SQLite** file (`audit.db`) | A real warehouse is just a database you never delete from. SQLite is a database that lives in one file. Same concept, zero setup | Free (ships with Python) |
| Warehouse queries | **pandas** reading from SQLite | Same SQL dialect, same dataframes as real analyst work | Free |
| Training orchestration | **A plain Python script** invoked manually | A scheduled training job is just a script + cron. Skip the cron | Free |
| Monitoring dashboard | **Jupyter notebook** reading `audit.db` | The notebook *is* your dashboard. Charts are matplotlib | Free |
| Alerting | **A `print` statement** and your own eyes | Real alerting is a threshold + a notification. For learning, the threshold check is enough | Free |
| Local Kubernetes (optional) | **kind** (Kubernetes in Docker) or Docker Desktop's built-in K8s | Lets you deploy your container the same way a real cluster would, on a laptop | Free |
| HTTP client for testing | **curl** + FastAPI's Swagger UI + **httpie** | Three different ways to hit your API, so you understand what a "request" actually looks like | Free |

**What you deliberately do NOT install:** no cloud SDK (AWS/GCP/Azure), no Terraform, no Prometheus, no Grafana, no MLflow server, no Nginx, no OAuth provider, no message queue. Every one of those is important in production and every one of them is a rabbit hole that would add weeks without adding insight at this stage.

---

## Phase 1 — Scoring Service on localhost (Weekend 1)

**Goal:** send an HTTP request to your own laptop and get back a credit score.

**What you build**

A file called `service/scoring_service.py` with a FastAPI app that loads `scaler.pkl` and `calibrated_lr.pkl` on startup, exposes a `POST /score` endpoint that takes an applicant payload, runs the same preprocessing the training pipeline does, returns `{probability_of_default, score, band}`.

A Pydantic model describing the applicant payload, derived directly from `src/config.py`'s `BASE_FEATURES` list so the two can never drift apart.

A `requirements-service.txt` with just `fastapi`, `uvicorn`, `pydantic`, `joblib`, `pandas`, `scikit-learn`.

A short `service/README.md` with the three commands to start it, stop it, and hit it.

**What you learn**

How a model pickle becomes a service. How request/response JSON serialization works. What a Pydantic validation error looks like when a lender sends a malformed payload (you will break things deliberately to see the error messages — this is the point). How FastAPI's auto-generated Swagger UI at `http://localhost:8000/docs` lets you test the service without writing a client.

**Success criteria**

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d @sample_applicant.json
```

Returns something like:

```json
{"probability_of_default": 0.186, "score": 664, "band": "B"}
```

**Stretch:** add a `GET /health` endpoint, a `GET /model_info` endpoint that returns the git SHA and training date of the currently-loaded model. These are the two endpoints every real scoring service has.

**Time:** one evening if everything goes right, one weekend if Pydantic fights you on the feature list (it will, once).

---

## Phase 2 — Dockerize (Weekend 2)

**Goal:** the same scoring service, but running inside a container. This is the single most valuable skill in the plan.

**What you build**

A `Dockerfile` in the project root that starts from `python:3.11-slim`, copies the relevant source code and model pickles into the image, installs `requirements-service.txt`, and runs `uvicorn` on port 8000.

A `.dockerignore` that keeps `data/raw/*`, `archive/`, `.git/`, and friends out of the image.

A `docker-compose.yml` — this is the polite way to say `docker run` with a long list of arguments saved in a file. For now it only declares one service (`scoring`) but the file is there for when you add SQLite and the notebook later.

**What you learn**

What a container actually is (a filesystem snapshot plus a process, not a virtual machine). Why a `Dockerfile` is a recipe with layer caching. Why the order of instructions matters for build speed. How `docker build -t credit-scoring:v1 .` produces an image. How `docker run -p 8000:8000 credit-scoring:v1` starts it. How the *exact same container* will later run on Kubernetes, AWS ECS, Google Cloud Run, or anywhere else with zero code changes — the "write once, run anywhere" promise that everybody makes and almost nothing keeps except containers.

**Success criteria**

Stop your Phase 1 local Uvicorn. Start the container. The same `curl` command returns the same score. You now have proof that your scoring service is portable.

**Stretch:** hit the container from a different machine on your home network. Get `docker stats` on screen and watch CPU / memory while you run 1000 requests at it with a loop.

**Time:** one weekend. The first Dockerfile is always slow because you don't yet have the muscle memory.

---

## Phase 3 — Local model registry with versioning (Weekend 3)

**Goal:** promote a new model without rebuilding the container.

**What you build**

Restructure the `models/` folder:

```
models/
├── v1/
│   ├── scaler.pkl
│   ├── calibrated_lr.pkl
│   └── metadata.json    (git SHA, training date, AUC, KS)
├── v2/
│   └── ...
└── CURRENT              (a single line: "v2")
```

Modify the scoring service to read `CURRENT` on startup and load whichever version it points at. Add a tiny `scripts/register_model.py` that takes a `scaler.pkl` + `model.pkl` + a metadata JSON, copies them to the next free `models/vN/` slot, and optionally updates `CURRENT`. Add a `scripts/promote.py` that just writes a new value to `CURRENT`.

Mount the `models/` folder as a **volume** into the container (this is the big conceptual leap) so you can promote a model by editing one file on the host, then restart the container with `docker restart` and it loads the new version.

**What you learn**

Why models do not live inside the container image. Why the registry is its own thing. What the word "promotion" actually means in this context — flipping a pointer, not redeploying code. What a Docker volume is and why volumes are the bridge between immutable containers and mutable state. Why every real model registry (MLflow, SageMaker, Vertex AI) is fundamentally the same idea wrapped in different UIs.

**Success criteria**

Have two models sitting in `models/v1/` and `models/v2/`. They should give slightly different scores for the same applicant (you can retrain with a different random seed to get a second model). Start the container with `CURRENT=v1`, hit `/score`, note the response. Edit `CURRENT` to `v2`, `docker restart`, hit `/score` again, see the new response. You have just performed a model promotion.

**Time:** one weekend. The concepts are the easy part; the Docker volume plumbing is what takes the time.

---

## Phase 4 — Audit log and monitoring notebook (Weekend 4)

**Goal:** every score is recorded, and you can plot drift.

**What you build**

Add a SQLite database file `audit.db` at the project root. Add a middleware or a background task to the FastAPI app that writes, for every request: timestamp, request payload, response body, model version, and latency in milliseconds. One row per scored applicant.

Mount `audit.db` as another Docker volume so the container writes to a file that lives on your host, not inside the container. This way you can rebuild or restart the container without losing the audit trail — exactly the real-world pattern.

Create `notebooks/monitoring.ipynb`. It opens `audit.db` with `pd.read_sql`, shows a table of the most recent 50 scores, plots a histogram of scores over the last day, plots PSI between this week's scores and last week's, and plots latency percentiles.

Write a tiny `scripts/generate_traffic.py` that sends 500 random but plausible applicant payloads to your local scoring service in a loop, so you actually have data in `audit.db` to look at.

**What you learn**

What an audit log is for. Why it's asynchronous and out-of-band from the response path. How PSI is actually computed (it's a sum over bins, not magic). The difference between real-time metrics (latency, error rate) and batch metrics (PSI, realised bad rate). Why the monitoring notebook is both a dashboard *and* the place you'd eventually put alert thresholds.

**Success criteria**

You run `generate_traffic.py` in one terminal. In another, you open the notebook and see, in real time, the histogram of scores, the latency distribution, and the feature value ranges. You then deliberately shift one feature's distribution in `generate_traffic.py`, re-run it, and watch PSI climb in the notebook. You have simulated feature drift on your laptop.

**Stretch:** write a second notebook `retrain.ipynb` that reads `audit.db`, joins it against a fake "outcomes" table (another SQLite table you maintain by hand), and retrains the model on the joined data. This closes the full loop — the scored data becomes the training data for the next version.

**Time:** one weekend.

---

## Phase 5 — Local Kubernetes (Optional, Weekend 5)

**Goal:** deploy your container the way a real cluster would, without a cloud account.

**What you build**

Install **kind** (`brew install kind` or `choco install kind` or the equivalent on your OS). Create a local cluster with `kind create cluster`. Load your container image into the cluster with `kind load docker-image credit-scoring:v2`. Write a Kubernetes `deployment.yaml` and a `service.yaml` — two small YAML files that tell Kubernetes "run three copies of this container, load-balance across them, expose it on port 80". Apply them with `kubectl apply -f k8s/`. Hit the service with `curl`.

**What you learn**

What a Kubernetes pod is (a group of containers). What a deployment is (a rule for how many pods to keep running). What a service is (a stable network address in front of a changing set of pods). What happens when you `kubectl delete pod ...` — the deployment immediately replaces the pod you killed, which is the self-healing behaviour that makes K8s worth the complexity. What "rolling update" means — you bump the image tag in `deployment.yaml`, `kubectl apply`, and K8s replaces old pods with new ones one at a time with zero downtime.

**Success criteria**

Hit your service, get a score. In one terminal run `kubectl delete pod -l app=scoring`. In another terminal, keep hitting the service. You should see no errors — a new pod has already taken over. That is the self-healing production behaviour you want to understand.

**Honest warning:** this phase is noticeably harder than the others. YAML is frustrating, `kubectl` has a steep learning curve, and error messages are unfriendly. If you hit the wall, it is completely fine to stop here and count the first four phases as the learning win. Kubernetes is a tool you can pick up later when you have a real deployment that needs it.

**Time:** one weekend if things go smoothly, two if kind fights you on networking.

---

## Decision points along the way

A few things you will want to decide deliberately rather than drift into.

**When to stop adding features and start using it.** After Phase 2, the service is genuinely useful for any model you build in the future — not just this one. Every time you train a new scorecard, wrap it in this service shell, and you now have a deployable artefact. That alone might be the best return on this investment.

**Whether to swap Flask for FastAPI, or the model format.** You will see online tutorials that use Flask or Streamlit. Ignore them. FastAPI is what the industry has converged on for new ML services, the async support is real, and the Pydantic validation saves hours of debugging. Stick with FastAPI.

**Whether to eventually host this on a real cloud.** If after Phase 4 you want to take the next step, the cheapest path is **Google Cloud Run** or **AWS App Runner**. Both accept your exact `Dockerfile`, deploy it in about five minutes, charge you pennies per million requests, and scale to zero when nobody is using it. You would not change a single line of your scoring service code to deploy it — that is the whole point of having containerised it.

**Whether the code lives in this repo or a new one.** Honest answer: a new repo called `credit-scoring-service` would be cleaner long-term, because the training pipeline and the scoring service evolve on different cadences. But for learning purposes, keeping it all in the current repo under a `service/` subfolder is fine and means you don't have to context-switch between two repos every session.

---

## What to skip and why

Resist the urge to add any of the following before the core four phases are done:

- **Authentication / API keys.** Real in production, irrelevant for learning. Adding it early doubles Phase 1's complexity.
- **Nginx or Traefik in front of FastAPI.** Uvicorn is perfectly capable of serving traffic directly for a learning build. Reverse proxies are a production concern.
- **MLflow or Weights & Biases.** These are better than a folder-based registry, but not by enough to justify their setup overhead at this stage. Come back to them after Phase 3.
- **A real database like PostgreSQL** instead of SQLite. SQLite is a real database. It has been shipping in iOS and Android for a decade. Use it without shame.
- **Writing the whole thing in Rust or Go "for speed".** You are not latency-bound. You are learning-bound. Python is fine.
- **A web frontend.** This is an API, not a product. The Swagger UI is your frontend.

---

## Definition of done

You have built the full local stack when:

1. You can hit your own laptop with `curl` and get back a credit score in under 100 ms.
2. The service is running inside a Docker container.
3. The model it's using is loaded from a versioned folder, and you can promote a new version by editing one file and restarting.
4. Every score is logged to a SQLite file you can read with pandas.
5. You have a Jupyter notebook that shows live PSI against a baseline, and you have deliberately caused PSI to spike by shifting the inputs.
6. You can explain every arrow on `docs/images/production_architecture.png` in terms of something you personally built.

When all six are true, you understand production ML infrastructure well enough to hold your own with any DevOps engineer in the room.

---

## Rough total time

- Phase 1 (scoring service): 1 evening – 1 weekend
- Phase 2 (Docker): 1 weekend
- Phase 3 (versioning + registry): 1 weekend
- Phase 4 (audit log + monitoring notebook): 1 weekend
- Phase 5 (Kubernetes, optional): 1–2 weekends

Realistic total: four focused weekends for Phases 1–4, plus an optional fifth for K8s. Budget double that if you're also working a full week — ML infrastructure is one of those areas where the tooling breaks for reasons that have nothing to do with ML, and debugging a YAML indentation error at 11pm is a rite of passage.

---

## Next action

Phase 1 is the obvious starting point. It gives you the "localhost returns a score" moment within an evening, which is motivating enough to carry you through Phase 2. Everything else is built on top of it.

Say the word and I'll draft the `service/scoring_service.py` file for this repo — wired to your actual `src/config.py` feature list, with a sample `applicant.json` and a curl command — so you can hit your own laptop tonight.
