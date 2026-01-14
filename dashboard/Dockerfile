FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY app.py /app/app.py

RUN pip install --no-cache-dir \
        streamlit==1.39.0 \
        neo4j==5.20.0 \
        pandas==2.2.3 \
        numpy==1.26.4 \
        plotly==5.23.0

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
