# syntax=docker/dockerfile:1.7
# =============================================================================
# Training image for the credit pipeline.
#
# Multi-stage build:
#   * Stage 1 (builder): installs all runtime deps with uv into /opt/venv.
#                        Source is layered in last for cache efficiency.
#   * Stage 2 (runtime): slim Python base with /opt/venv + project source.
#                        Runs as a non-root user.
#
# Build:    docker build -t credit-pipeline:dev -f docker/training.Dockerfile .
# Smoke:    docker run --rm credit-pipeline:dev
# Train:    docker run --rm -v $(pwd)/data:/app/data \
#                -v $(pwd)/models:/app/models \
#                credit-pipeline:dev -m pipelines.run_pipeline
# Tests:    docker run --rm credit-pipeline:dev -m pytest -q
# =============================================================================

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.4.27

# ---------- Stage 1: builder --------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG UV_VERSION

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install build essentials only when wheels aren't available for a dep.
# Kept minimal; expand only if a sync fails for a missing toolchain.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install uv at a pinned version, not "latest", for reproducibility.
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /tmp/uv-installer.sh
RUN sh /tmp/uv-installer.sh && rm /tmp/uv-installer.sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy project metadata first so dep installation is cached separately
# from source changes.
COPY pyproject.toml uv.lock* README.md ./

# Install runtime deps only — no dev groups, no project source.
# --frozen: fail if uv.lock would need to change (production discipline).
# --no-install-project: project itself isn't installed; we run from /app.
# --no-dev: exclude development groups (lint, test, dev).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now copy the project source. This layer rebuilds whenever code changes,
# but the previous (deps) layer stays cached.
COPY src ./src
COPY pipelines ./pipelines
COPY tests ./tests


# ---------- Stage 2: runtime --------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Non-root user. UID 1000 is conventional for "first user".
RUN groupadd --system --gid 1000 app \
 && useradd  --system --uid 1000 --gid app --create-home --home-dir /home/app app

WORKDIR /app

# Copy the prepared venv and project source from the builder stage.
COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /app /app

USER app

# Default command: a smoke check confirming the image is healthy.
# Override at runtime to actually do work, e.g.:
#   docker run --rm credit-pipeline:dev -m pipelines.run_pipeline
ENTRYPOINT ["python"]
CMD ["-c", "import sys, sklearn, pandas; print(f'credit-pipeline image OK — Python {sys.version_info.major}.{sys.version_info.minor}, sklearn {sklearn.__version__}, pandas {pandas.__version__}')"]
