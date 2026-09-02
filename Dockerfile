FROM python:3.10-slim

# Allow logs to be written immediately to Google Cloud Logging
ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Use the PORT environment variable dynamically (Cloud Run provides this)
CMD uvicorn src.agents.agent_orchestration:app --host 0.0.0.0 --port ${PORT:-8080}
