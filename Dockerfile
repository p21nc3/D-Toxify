# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
# Upgrade pip first
RUN pip install --upgrade pip
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Verify torch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || (echo "ERROR: PyTorch not installed!" && exit 1)

# Copy application files
COPY api_server.py .
COPY rephraser.py .
COPY worker_queue.py .
COPY classifiers ./classifiers
COPY download.py .

# Create SSL directory (optional, for future use)
RUN mkdir -p ssl

# Install NLTK datasets
RUN python download.py

# Create directories for models and data
RUN mkdir -p models data results

# Expose API port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=api_server.py
ENV RUN_WITHOUT_SSL=1

# Health check (HTTP only since RUN_WITHOUT_SSL is set)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/test')" || exit 1

# Run the API server
CMD ["python", "api_server.py"]
