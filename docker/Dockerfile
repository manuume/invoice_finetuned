# ── Stage 1: base ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for OpenCV + Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: dependencies ─────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .

# Note: vllm is NOT installed in Docker — it requires matching CUDA drivers.
# The container uses the transformers backend (MODEL_BACKEND=transformers).
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: app ─────────────────────────────────────────────────────────────
FROM deps AS app

COPY src/ src/
COPY configs/ configs/
COPY .env.example .env.example

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Create required runtime directories
RUN mkdir -p data/uploads data/processed reports mlruns

ENV PYTHONPATH=/app
ENV MODEL_BACKEND=transformers
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
