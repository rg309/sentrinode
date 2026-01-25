FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG CACHE_BUST=2
RUN echo "cache bust $CACHE_BUST"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
RUN python -m py_compile /app/app.py

EXPOSE 8080

CMD ["streamlit", "run", "/app/app.py", "--server.address=0.0.0.0", "--server.port=8080"]
