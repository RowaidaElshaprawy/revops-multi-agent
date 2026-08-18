FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for psycopg2 compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for optimal layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary pgvector ragas datasets mlflow celery redis

# Copy application source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]