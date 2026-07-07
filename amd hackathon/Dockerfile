# Judging VM is linux/amd64 — build with:
#   docker buildx build --platform linux/amd64 -t <registry>/frugal-router:latest --push .
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .

ENTRYPOINT ["python", "agent.py"]
