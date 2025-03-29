FROM python:3.9-slim

# Set argument for GPU usage (0 for CPU, 1 for GPU)
ARG USE_GPU=0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install additional CUDA dependencies if GPU is enabled
RUN if [ "$USE_GPU" = "1" ] ; then \
        apt-get update && apt-get install -y \
        nvidia-cuda-toolkit \
        && rm -rf /var/lib/apt/lists/* ; \
    fi

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN if [ "$USE_GPU" = "1" ] ; then \
        pip install --no-cache-dir -r requirements.txt && \
        pip install --no-cache-dir torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html ; \
    else \
        pip install --no-cache-dir -r requirements.txt && \
        pip install --no-cache-dir torch==2.0.1 -f https://download.pytorch.org/whl/torch_stable.html ; \
    fi

# Copy application code
COPY . .

# Create model cache directory
RUN mkdir -p /app/model_cache

# Set environment variables
ENV MODEL_CACHE_DIR=/app/model_cache
ENV PYTHONUNBUFFERED=1
ENV USE_GPU=${USE_GPU}
ENV DJANGO_SETTINGS_MODULE=worker.settings

# Expose port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300", "worker.wsgi:application"]
