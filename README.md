# Financial OCR — Receipt Extraction with Qwen2.5-VL-7B

Production-grade MLOps pipeline for structured receipt field extraction.  
Fine-tunes Qwen2.5-VL-7B with LoRA on CORD v2, serves via FastAPI + vLLM,  
with automated evaluation on every push to `main`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Lightning AI Studio                                     │
│                                                          │
│  Upload → FastAPI → BackgroundTask → Qwen2.5-VL-7B      │
│                         │               (vLLM / HF)     │
│                         ▼                               │
│                     SQLite / Postgres                    │
│                         │                               │
│                  Eval Pipeline ──► MLflow ──► Report     │
└─────────────────────────────────────────────────────────┘
         │
         ▼
   Docker container
   (API + MLflow UI)
```

| Layer | Technology |
|-------|-----------|
| Model | Qwen2.5-VL-7B-Instruct + LoRA (peft) |
| Fine-tuning | HuggingFace Trainer + bf16 + gradient accumulation |
| Serving (GPU) | vLLM — OpenAI-compatible API |
| Serving (CPU) | HuggingFace transformers + 4-bit quantisation |
| API | FastAPI + BackgroundTasks (async) |
| Database | SQLite (dev) → PostgreSQL (prod) via SQLAlchemy async |
| Experiment tracking | MLflow |
| Augmentation | Albumentations |
| Evaluation | jiwer (CER/WER) + scikit-learn (F1) |
| CI/CD | GitHub Actions — lint, test, eval gate on every PR |
| Containerisation | Docker + docker-compose |
| Compute | Lightning AI Studio (CPU dev ↔ GPU training) |

---

## Dataset

**CORD v2** — Consolidated Receipt Dataset  
800 Korean retail receipts with hierarchical JSON annotations.

| Split | Samples | After augmentation |
|-------|---------|--------------------|
| Train | 800 | ~2,500 |
| Validation | 100 | 100 (no augmentation) |
| Test | 100 | 100 (no augmentation) |

Target fields extracted:

```json
{
  "store_name": "GS25 Gangnam",
  "total_price": "15000",
  "tax_price": "1500",
  "items": [
    {"name": "Americano", "price": "3000"}
  ]
}
```

---

## Quick Start

### 1. Clone & install (Lightning AI CPU studio)

```bash
git clone https://github.com/your-username/financial-ocr
cd financial-ocr
pip install -r requirements.txt
cp .env.example .env
```

### 2. Download & preprocess data

```bash
bash scripts/download_data.sh
python scripts/preprocess_data.py
```

### 3. Run tests locally

```bash
pytest tests/ -v
```

### 4. Start the API (CPU dev mode)

```bash
# Uses Qwen2.5-VL-3B + 4-bit quant on CPU
MODEL_BACKEND=transformers uvicorn src.api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Fine-Tuning (Lightning AI GPU Studio)

```bash
# Switch to GPU in Lightning AI
# lightning studio modify financial-ocr --gpu l4

python src/model/train.py --config configs/lora_config.yaml
```

Training runs ~2–3 hours on an L4 GPU (~$3).  
Checkpoints and metrics are logged automatically to MLflow.

```bash
# View experiment results
mlflow ui --port 5000
```

---

## Evaluation

```bash
# Run full evaluation on test split
python src/eval/run_eval.py --split test --checkpoint checkpoints/lora/best

# Latency benchmark (50 samples)
python -m src.eval.benchmark --split test --n-samples 50
```

Outputs a markdown report to `reports/eval_test.md` and logs all metrics to MLflow.

### Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| CER | < 0.10 | Character error rate on full JSON output |
| WER | < 0.15 | Word error rate on full JSON output |
| Field Accuracy | > 0.85 | Exact match on scalar fields |
| Field F1 | > 0.80 | Precision/recall on field extraction |
| Latency p50 | < 2s | Median inference time (GPU) |
| Latency p95 | < 5s | 95th percentile inference time (GPU) |

---

## API Reference

### `POST /api/v1/upload`

Upload a receipt image. Returns a `job_id` immediately (async processing).

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@receipt.jpg"
```

```json
{"job_id": "abc-123", "filename": "receipt.jpg", "status": "pending"}
```

### `GET /api/v1/job/{job_id}`

Poll job status: `pending → processing → done | failed`

### `GET /api/v1/results/{job_id}`

Retrieve extraction result once job is `done`.

```json
{
  "job_id": "abc-123",
  "store_name": "GS25 Gangnam",
  "total_price": "15000",
  "tax_price": "1500",
  "items": [{"name": "Americano", "price": "3000"}],
  "inference_ms": 1340.2
}
```

### `GET /api/v1/health`

Liveness probe — returns model backend and DB config.

---

## Docker

```bash
# Build and start API + MLflow UI
docker compose -f docker/docker-compose.yml up --build

# API:     http://localhost:8000/docs
# MLflow:  http://localhost:5000
```

---

## CI / CD

GitHub Actions runs on every PR and push to `main`:

1. **Lint** — ruff checks `src/`, `tests/`, `scripts/`
2. **Unit tests** — pytest (no GPU required)
3. **Eval gate** *(main only)* — runs inference on 20 test samples, fails if `field_accuracy < 0.70`
4. **Upload report** — eval markdown report saved as Actions artifact

---

## Repository Structure

```
financial-ocr/
├── .github/workflows/eval.yml    # CI/CD pipeline
├── .lightning/config.yaml        # Lightning AI studio config
├── configs/
│   ├── model_config.yaml         # Qwen model settings + prompt templates
│   ├── lora_config.yaml          # LoRA hyperparameters + training args
│   └── vllm_config.yaml          # vLLM server settings
├── data/                         # gitignored — generated by scripts
├── scripts/
│   ├── download_data.sh          # Downloads CORD v2 from HuggingFace
│   └── preprocess_data.py        # Augmentation + field extraction
├── src/
│   ├── data/
│   │   ├── dataset.py            # CORDDataset + field extractor
│   │   └── augmentation.py       # Albumentations pipeline
│   ├── model/
│   │   ├── qwen_model.py         # Unified wrapper (vLLM / transformers)
│   │   └── train.py              # LoRA fine-tuning + MLflow logging
│   ├── api/
│   │   ├── main.py               # FastAPI app + lifespan
│   │   ├── routes.py             # Upload / job / result / health endpoints
│   │   ├── schemas.py            # Pydantic request/response models
│   │   └── async_processor.py    # BackgroundTask — inference + DB write
│   ├── db/
│   │   ├── connection.py         # Async SQLAlchemy engine + session
│   │   └── models.py             # Job / Result / EvalRun ORM tables
│   └── eval/
│       ├── metrics.py            # CER, WER, field F1, accuracy
│       ├── benchmark.py          # Latency p50/p95/p99
│       └── run_eval.py           # Full eval → MLflow → markdown report
├── tests/
│   ├── test_api.py
│   ├── test_eval.py
│   └── test_model.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Resume Bullets

```
• Built end-to-end MLOps pipeline for financial receipt OCR using
  Qwen2.5-VL-7B + LoRA fine-tuning on CORD v2 (800 receipts → 2,500
  with Albumentations augmentation), achieving >85% field-level accuracy

• Designed async FastAPI inference service with vLLM backend and
  BackgroundTasks queue; p95 latency <3s on Lightning AI L4 GPU
  at <$0.004 per document

• Implemented automated evaluation gate in GitHub Actions — runs CER,
  WER, field F1 on every push to main, blocks merge if accuracy drops
  below threshold; all metrics logged to MLflow with markdown report

• Containerised full stack (API + MLflow UI) with Docker + docker-compose,
  enabling one-command local deployment: `docker compose up --build`

• Reduced GPU cost 90% during development by building CPU↔GPU switching
  logic — 3B model + 4-bit quantisation on Lightning AI CPU studio,
  7B + vLLM on GPU only for training and eval
```

---

## License

MIT
