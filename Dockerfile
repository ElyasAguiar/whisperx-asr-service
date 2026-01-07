# WhisperX ASR API Service Dockerfile
# Based on NVIDIA CUDA for GPU support

FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Copy requirements first for better layer caching
COPY requirements.txt /workspace/requirements.txt

# Copy application code
COPY ./app /workspace/app
COPY ./proto /workspace/proto
COPY ./scripts /workspace/scripts

# Install system dependencies (Ubuntu 24.04 has Python 3.12 by default)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set library path to prefer PyTorch's bundled cuDNN over system cuDNN
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
ENV NLTK_DATA=/.cache/nltk_data

# Install all Python dependencies
# Use --break-system-packages for Ubuntu 24.04 (PEP 668)
RUN python3 -m pip install --no-cache-dir --upgrade pip --break-system-packages && \
    pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Create cache directory
RUN mkdir -p /.cache && chmod 777 /.cache

# Generate gRPC code from proto files
RUN chmod +x /workspace/scripts/generate_grpc.sh && \
    /workspace/scripts/generate_grpc.sh

# Expose API ports (REST and gRPC)
EXPOSE 9000 50051

# Set Python path to include workspace
# ENV PYTHONPATH=/workspace:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:9000/health')" || exit 1

# Run the multi-protocol server (REST + gRPC)
CMD ["python3", "-m", "app.server"]
