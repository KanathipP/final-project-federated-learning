# --------------------------------------------------------------------
# Base image: lightweight Python with system libs we need for mne
# We use python:3.11-slim instead of CUDA-enabled images to keep it light.
# --------------------------------------------------------------------
FROM python:3.11-slim AS base

# --------------------------------------------------------------------
# Set up OS-level deps (build + runtime)
# - build-essential, gcc: needed for scipy/numpy/sklearn wheels if fallback to source
# - libopenblas-dev / liblapack-dev: linear algebra backend
# - libglib2.0-0 / libxext6 / libxrender1 / libsm6: common scientific stack deps
# - curl: optional debug/health
# We keep layer size small but realistic for mne/scikit-learn/torch CPU.
# --------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libopenblas-dev \
    liblapack-dev \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    libsm6 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------
# Create a non-root user for runtime security.
# We'll install libs as root, then drop to this user later.
# --------------------------------------------------------------------
ARG APP_USER=flwr
ARG APP_UID=1000
RUN useradd --create-home --home-dir /app --uid ${APP_UID} ${APP_USER}

# --------------------------------------------------------------------
# Working directory for our app code
# --------------------------------------------------------------------
WORKDIR /app

# --------------------------------------------------------------------
# Copy only pyproject first, install deps.
# This lets Docker cache the dependency layer unless pyproject changes.
# We also remove flwr[simulation] if you're overriding that in pyproject and
# want to control exact extras yourself (similar to your sed trick).
# --------------------------------------------------------------------
COPY pyproject.toml /app/pyproject.toml

# Optional: strip simulation extra if you had that hack in your existing Dockerfile
# If you don't need this trick anymore, you can drop this RUN.
RUN sed -i 's/.*flwr\[simulation\].*//' /app/pyproject.toml || true

# Install project dependencies (will install flwr, torch CPU, mne, etc.)
# We assume pyproject.toml declares them in [project.dependencies]
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir .

# --------------------------------------------------------------------
# Copy the actual source code.
# We copy only the Python package folder (flower_version_1) and anything else
# needed at runtime by the client/server apps.
# Data is NOT copied; that will be mounted as a volume at runtime.
# --------------------------------------------------------------------
COPY flower_version_1 /app/flower_version_1

# --------------------------------------------------------------------
# Environment variables:
# - PYTHONUNBUFFERED: logs flush immediately (useful in Docker logs)
# - DATA_ROOT: default data root inside the container
#   Each client container will get its own /app/data/<partition_id> mounted.
# --------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    DATA_ROOT=/app/data

# --------------------------------------------------------------------
# Expose a volume mount point for per-partition data.
# This doesn't create data in the image, but documents that /app/data
# is expected to be provided at runtime (-v ...:/app/data/<id>:ro).
# --------------------------------------------------------------------
VOLUME ["/app/data"]

# --------------------------------------------------------------------
# Switch to the non-root user for runtime safety.
# --------------------------------------------------------------------
USER ${APP_USER}

# --------------------------------------------------------------------
# Default entrypoint for Flower's superexec runtime.
# You'll override behavior (server vs client, partition id, etc.)
# using env vars and command args when you 'docker run'.
# --------------------------------------------------------------------
ENTRYPOINT ["flower-superexec"]
