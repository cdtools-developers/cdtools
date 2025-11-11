# Multi-stage build for CDTools - Python library for ptychography and CDI reconstructions
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml setup.py requirements.txt README.md ./
COPY src/ ./src/
COPY examples/ ./examples/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Final stage - lightweight runtime image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code (optional, for development)
COPY src/ ./src/

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

