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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements .
RUN pip install --no-cache-dir -r requirements

# Copy application code
COPY src/ ./src/
COPY simple_server.py .

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Railway assigns PORT dynamically
EXPOSE 5000

CMD ["sh", "-c", "python simple_server.py"]
