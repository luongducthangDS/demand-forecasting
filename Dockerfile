FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY forecast/ forecast/
# Artifacts come from `python -m forecast.train`; mounted at runtime so a retrain needs no rebuild.
ENV ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
