FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# Python ve dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY analysis/ ./analysis/
COPY automation/ ./automation/

# Set Python path
ENV PYTHONPATH=/workspace

CMD ["bash"]
