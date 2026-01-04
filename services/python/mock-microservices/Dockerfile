FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends \
    iproute2 \
    stress-ng
rm -rf /var/lib/apt/lists/*
EOF

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services ./services

EXPOSE 8000

CMD ["uvicorn", "services.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
