# CUDA + PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Metadata
LABEL maintainer="HDMR Research Team"
LABEL description="HDMR-based Hyperparameter Optimization for Time Series Forecasting"

# Sistem paketleri
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /workspace

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Dizinleri oluştur
RUN mkdir -p results/benchmarks results/comparisons results/sensitivity logs

# GPU kontrolü script'i (DÜZELTILMIŞ)
RUN echo '#!/bin/bash\n\
echo "=== GPU Kontrolü ==="\n\
nvidia-smi\n\
echo ""\n\
echo "=== PyTorch CUDA ==="\n\
python -c "import torch; print(\"CUDA Available:\", torch.cuda.is_available()); print(\"CUDA Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")"\n\
echo ""\n\
echo "=== XGBoost GPU ==="\n\
python -c "import xgboost as xgb; print(\"XGBoost version:\", xgb.__version__)"\n\
echo ""\n\
echo "=== Paketler ==="\n\
python -c "import numpy, pandas, lightgbm, statsmodels; print(\"✓ Tüm paketler yüklü\")"\n\
' > /workspace/check_gpu.sh && chmod +x /workspace/check_gpu.sh

# Healthcheck
HEALTHCHECK --interval=5m --timeout=3s \
  CMD python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || exit 1

# Default komut
CMD ["python", "run_all_experiments.py", "--full"]
