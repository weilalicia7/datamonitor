# =============================================================================
# SACT Scheduler — production container image
# =============================================================================
# Multi-stage build:
#   stage "builder": compile wheels from requirements.txt into a virtualenv
#   stage "runtime": slim python + the pre-built venv + the app source
#
# The runtime image runs as non-root (uid 1000), has no build tools, no cache,
# and no pip — strictly the deps + the app + gunicorn.  Expected size ~600 MB
# uncompressed (mostly numpy/scipy/pandas/ortools).
#
# Env vars honored at runtime:
#   - FLASK_SECRET_KEY            (T1.3 — required in prod)
#   - AUTH_ENABLED / AUTH_*       (T4.1)
#   - RATE_LIMIT_ENABLED          (T4.2)
#   - SESSION_COOKIE_SECURE=true  (T4.3 — MUST be true over HTTPS)
#   - LOG_FORMAT=json             (T4.4 — container log collectors)
#   - METRICS_ENABLED=true        (T4.5)
#   - OTEL_ENABLED=true           (T4.5 — optional tracing)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System build deps.  ortools / numpy / scipy wheels are available on PyPI for
# linux/amd64 so we don't need gcc; keep apt usage minimal + purge lists.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements manifest so dep installation caches properly on
# unchanged requirements.
WORKDIR /build
COPY requirements.txt ./

# Build a dedicated virtualenv inside /opt/venv.  This keeps the runtime
# image's site-packages isolated from the system python and lets us copy a
# single directory tree forward.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install 'gunicorn>=21.2'


# -----------------------------------------------------------------------------
# Stage 2 — runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    LOG_FORMAT=json \
    METRICS_ENABLED=true

# Only the bare minimum shared libraries for ortools/numpy/scipy at runtime
# (no compilers, no headers, no package manager index).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the prebuilt venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Non-root user.  1000:1000 is the convention most k8s / docker-compose setups
# bind-mount volumes as.  Home directory is /app; audit + data_cache live here.
RUN groupadd --gid 1000 sact \
    && useradd --uid 1000 --gid sact --home /app --shell /bin/bash sact \
    && mkdir -p /app/data_cache/audit \
    && chown -R sact:sact /app

WORKDIR /app
COPY --chown=sact:sact . /app

USER sact

EXPOSE 1421

# Gunicorn: 2 workers × 4 threads is a reasonable default for CPU-bound
# optimisation + I/O-bound ML routes on a 2-core pod.  Tune via env:
#   GUNICORN_WORKERS, GUNICORN_THREADS, GUNICORN_TIMEOUT
ENV GUNICORN_WORKERS=2 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=60

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://localhost:1421/health/live" || exit 1

CMD ["sh", "-c", "gunicorn \
      --bind 0.0.0.0:1421 \
      --workers ${GUNICORN_WORKERS} \
      --threads ${GUNICORN_THREADS} \
      --timeout ${GUNICORN_TIMEOUT} \
      --access-logfile - \
      --error-logfile - \
      flask_app:app"]
