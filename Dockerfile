FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["uvicorn", "src.agents.agent_orchestration:app", "--host", "0.0.0.0", "--port", "8080"]
