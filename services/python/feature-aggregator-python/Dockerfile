FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY feature_aggregator ./feature_aggregator
COPY main.py ./

ENV FEATURE_STORE_PATH=/data/feature_store
VOLUME /data/feature_store

CMD ["python", "main.py"]
