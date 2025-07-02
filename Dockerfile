# Dockerfile - Corrected for LLM OCR Pipeline
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install ONLY the system dependencies you actually need
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic1 \
    curl \
    netcat-traditional \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Note: Removed tesseract-ocr since you use LLM-based OCR
# Your system uses Gemini/Mistral/GPT-4/Claude for OCR instead

# Create non-root user
RUN useradd --create-home --shell /bin/bash graphrag

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/data /app/logs /app/chroma_db_pipeline /app/ocr_outputs && \
    chown -R graphrag:graphrag /app

# Switch to non-root user
USER graphrag

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "api_server.py"]