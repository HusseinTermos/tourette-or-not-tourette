FROM python:3.10-slim

# system lib for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy artifacts + server
COPY artifacts/ ./artifacts/
COPY inference.py .

# Cloud Run provides $PORT; default to 8080 if not set
ENV PORT=8080
CMD ["sh","-c","uvicorn inference:app --host 0.0.0.0 --port ${PORT:-8080}"]