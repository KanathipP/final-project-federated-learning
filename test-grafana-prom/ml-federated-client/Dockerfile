FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY train.py /app/train.py

RUN pip install --no-cache-dir --upgrade pip

CMD ["python", "/app/train.py"]
