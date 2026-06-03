# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Serving image — slim. Holds only what the scoring service needs at runtime:
# FastAPI, sklearn, pandas, SHAP, MLflow client (no training stack, no Spark,
# no PySpark JVM dependency). Builds a separate venv from the same uv.lock so
# the runtime dependency graph is reproducible against the training image.
# -----------------------------------------------------------------------------

# ----- Stage 1: builder -- assemble the venv with uv --------------------------
FROM python:3.11-slim AS builder

# Copy uv from the official Astral image (pinned digest, no install.sh dance).
COPY --from=ghcr.io/astral-sh/uv:0.4.27 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Bring in the dependency contract (no source needed at install time).
COPY pyproject.toml uv.lock ./

# Install runtime dependencies plus the `serving` and `mlflow` extras.
# Deliberately exclude the `spark` extra and the dev group so the image
# stays small (~600MB vs ~1.5GB with Spark).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serving --extra mlflow

# ----- Stage 2: runtime -- minimal Python with the venv copied in -------------
FROM python:3.11-slim AS runtime

# Non-root user for any side-effects of writing to /tmp during scoring.
RUN useradd --create-home --shell /bin/bash serving

WORKDIR /app

# Copy the assembled venv from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Copy only the source the service needs at runtime.
COPY src/serving ./src/serving
COPY src/__init__.py ./src/__init__.py
COPY src/config.py ./src/config.py
COPY src/paths.py ./src/paths.py
COPY src/settings.py ./src/settings.py
COPY pyproject.toml ./pyproject.toml

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

USER serving

EXPOSE 8000

# Healthcheck: probe the readiness endpoint. Returns 200 once the model
# has loaded; 503 while loading.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8000/readyz', timeout=3); raise SystemExit(0 if r.status == 200 else 1)" \
        || exit 1

# Uvicorn runs the FastAPI app. One worker for predictable model memory usage;
# scale horizontally via Kubernetes replicas, not in-process workers.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
