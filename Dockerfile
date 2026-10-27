FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl build-essential cmake ninja-build \
    libopenblas-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps — torch CPU only (no torchvision, it conflicts with llama.cpp reqs)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build llama.cpp for GGUF conversion; strip torchvision from its requirements to
# avoid the torch/torchvision version mismatch that breaks transformers imports
RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp /app/llama.cpp && \
    cd /app/llama.cpp && \
    cmake -B build -DLLAMA_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build --config Release -j$(nproc) --target llama-quantize llama-cli 2>&1 | tail -5 && \
    grep -v "torchvision" /app/llama.cpp/requirements.txt > /tmp/llama_reqs.txt && \
    pip install --no-cache-dir -r /tmp/llama_reqs.txt && \
    pip uninstall -y torchvision 2>/dev/null || true

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Data directories
RUN mkdir -p data/datasets data/models data/output data/modelfiles

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/data/hf_cache
ENV TRANSFORMERS_CACHE=/app/data/hf_cache

CMD ["python", "backend/main.py"]
