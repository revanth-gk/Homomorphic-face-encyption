FROM python:3.11-slim

WORKDIR /app

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY simple_server.py .

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Railway assigns PORT dynamically
EXPOSE 5000

# Use gunicorn for production with Railway's PORT variable
CMD ["sh", "-c", "python -m gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 'homomorphic_face_encryption.app:create_app()'"]
