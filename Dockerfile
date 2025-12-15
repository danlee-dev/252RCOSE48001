# DocScanner AI Backend Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY ai/ ./ai/
COPY backend/ ./backend/
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

# Expose port
EXPOSE 8000

# Set working directory to backend
WORKDIR /app/backend

# Run the application (with auto migration)
CMD ["/app/start.sh"]
