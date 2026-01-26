FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
COPY api.py /app/api.py

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec ${FLY_PROCESS_GROUP}"]
