# --------------------------------------------------------------------
# Base image: Python 3.11 slim, CPU only
# --------------------------------------------------------------------
FROM python:3.11-slim

# --------------------------------------------------------------------
# System deps for scientific stack (numpy/scipy/mne/sklearn/torch CPU)
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
# Non-root runtime user
# --------------------------------------------------------------------
ARG APP_USER=flwr
ARG APP_UID=1000
RUN useradd --create-home --home-dir /app --uid ${APP_UID} ${APP_USER}

WORKDIR /app

# --------------------------------------------------------------------
# Copy dependency spec first for layer caching
# --------------------------------------------------------------------
COPY requirements.runtime.txt /app/requirements.runtime.txt

# Upgrade pip first
RUN python -m pip install --upgrade pip

# 1. Install CPU-only torch (and optionally torchvision if you need it)
#    We explicitly use the PyTorch CPU index here only for torch.
#    If you DON'T need torchvision/torchaudio, you can drop them.
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest (flwr, mne, sklearn, numpy, scipy, etc.) from PyPI default
RUN pip install --no-cache-dir -r requirements.runtime.txt

# --------------------------------------------------------------------
# Copy project metadata and source
# We still copy pyproject.toml because Flower's superexec reads config
# from [tool.flwr.*] in that file at runtime.
# --------------------------------------------------------------------
COPY pyproject.toml /app/pyproject.toml
COPY flower_version_1 /app/flower_version_1

# --------------------------------------------------------------------
# Runtime env
# --------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    DATA_ROOT=/app/data

# Data for each partition will be mounted here by docker-compose
VOLUME ["/app/data"]

# Drop root for runtime safety
USER ${APP_USER}

# Flower entrypoint: server_app / client_app will be given via `command` in compose
ENTRYPOINT ["flower-superexec"]
