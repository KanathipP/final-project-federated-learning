# syntax=docker/dockerfile:1.7
FROM flwr/superexec:1.22.0

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PYTHONUNBUFFERED=1

WORKDIR /app

# 1) ใส่ requirements
COPY requirements.runtime.txt /deps/requirements.txt

# 2) ติดตั้ง deps เข้า venv (ปักหมุด numpy<2 แก้ MNE/GDF)
RUN --mount=type=cache,target=/root/.cache/pip \
  /python/venv/bin/pip install --upgrade pip \
  && /python/venv/bin/pip install --no-cache-dir --only-binary=:all: -r /deps/requirements.txt

# 3) ติดตั้ง Torch CPU เข้า venv
RUN --mount=type=cache,target=/root/.cache/pip \
  /python/venv/bin/pip install --no-cache-dir --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.7.1 torchvision==0.22.1

# 4) พร้อมใช้งาน
ENTRYPOINT ["flower-superexec"]
