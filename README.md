# Financial OCR

End-to-end receipt understanding system built around Qwen2.5-VL-7B for structured field extraction from retail receipt images. The project covers dataset preparation, LoRA fine-tuning, async API serving, database-backed job tracking, evaluation reporting, and GPU inference benchmarking in a single workflow.

It is designed as a practical multimodal ML engineering project: train a vision-language model on receipt data, expose it through an application-facing API, and measure both extraction quality and serving performance with reproducible artifacts.

## Project Overview

Financial OCR extracts structured JSON from receipt images, including:

```json
{
  "store_name": "CGV CINEMAS",
  "total_price": "60,000",
  "tax_price": "5,455",
  "items": [
    {
      "name": "-TICKET CP",
      "price": "60.000"
    }
  ]
}
```

The system uses Qwen2.5-VL-7B-Instruct as the base vision-language model and adapts it for receipt extraction with LoRA fine-tuning. Inference is exposed through a FastAPI service that accepts uploaded receipt images, queues processing asynchronously, stores job state in a database, and returns structured results through polling endpoints.

## Why This Project Matters

- Demonstrates end-to-end multimodal ML engineering, not just model training.
- Combines fine-tuning, inference serving, API design, persistence, and evaluation in one system.
- Focuses on business-relevant structured extraction instead of generic OCR text dumping.
- Includes real evaluation artifacts and measured GPU latency results from the implemented pipeline.

## System Architecture

```text
Receipt Image
    |
    v
FastAPI Upload Endpoint
    |
    v
Background Task Processor
    |
    v
Qwen2.5-VL-7B + LoRA Adapter
    |
    +--> Structured JSON Result
    |
    +--> SQLite / PostgreSQL Job + Result Storage

Evaluation Pipeline
    |
    +--> CER / WER / Field Accuracy / Field F1
    +--> Markdown report
    +--> Prediction artifacts
    +--> MLflow logs
```

## Core Components

| Area | Stack |
|------|-------|
| Base model | Qwen/Qwen2.5-VL-7B-Instruct |
| Adaptation | LoRA with `peft` |
| Training | Hugging Face `Trainer` |
| Serving | FastAPI, async background processing |
| GPU inference | `transformers` backend and vLLM-compatible path |
| Persistence | SQLAlchemy async, SQLite for local development |
| Tracking | MLflow |
| Evaluation | CER, WER, field accuracy, field F1, latency percentiles |
| Data pipeline | preprocessing and augmentation scripts for CORD v2 |

## Results

The repository already includes generated evaluation artifacts in `reports/`, and those results are used directly here.

### Evaluation Metrics

From `reports/eval_test.md`:

| Metric | Value |
|--------|-------|
| Character Error Rate (CER) | `0.1628` |
| Word Error Rate (WER) | `0.3135` |
| Field-level Accuracy | `0.8551` |
| Field F1 Score | `0.9219` |
| Field Precision | `1.0000` |
| Field Recall | `0.8551` |
| Latency p50 | `1702.0 ms` |
| Latency p95 | `4969.0 ms` |
| Latency p99 | `6573.2 ms` |

### GPU Inference Benchmarks

Measured end-to-end API and serving behavior from the current implementation:

- FastAPI GPU inference examples completed in about `1485.3 ms`, `2142.6 ms`, and `1802.6 ms` model inference time.
- vLLM concurrent serving on 3 images achieved:
  - batch wall clock: `1570.2 ms`
  - average request time: `1259.1 ms`
  - min request time: `902.3 ms`
  - max request time: `1557.6 ms`
  - throughput: `1.91 images/second`

### Included Evaluation Artifacts

- `reports/eval_test.md`
- `reports/error_analysis.csv`
- `reports/predictions_test.jsonl`
- `reports/latency_histogram.png`

## Features

- LoRA fine-tuning pipeline for adapting Qwen2.5-VL to receipt extraction.
- Async FastAPI workflow with upload, status polling, and final result retrieval.
- Database-backed job lifecycle tracking for `pending`, `processing`, `done`, and `failed` states.
- Evaluation pipeline that generates markdown reports, prediction artifacts, and latency charts.
- Support for both direct model-backed API inference and vLLM-style serving experiments.
- Config-driven setup for model, LoRA, and serving behavior.

## API Workflow

The primary API exposes an asynchronous receipt processing flow.

### `POST /api/v1/upload`

Uploads a receipt image and returns a job immediately.

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@receipt.jpg"
```

Example response:

```json
{
  "job_id": "2c87cfd8-110a-4c91-8b0f-bb319ceac460",
  "filename": "receipt.jpg",
  "status": "pending"
}
```

### `GET /api/v1/job/{job_id}`

Polls job status until processing finishes.

Possible states:

- `pending`
- `processing`
- `done`
- `failed`

### `GET /api/v1/results/{job_id}`

Returns the extracted structured JSON after completion.

Example response:

```json
{
  "job_id": "2c87cfd8-110a-4c91-8b0f-bb319ceac460",
  "store_name": "CGV CINEMAS",
  "total_price": "60,000",
  "tax_price": "5,455",
  "items": [
    {
      "name": "-TICKET CP",
      "price": "60.000"
    }
  ],
  "inference_ms": 1485.32
}
```

### `GET /api/v1/health`

Returns a simple health payload with model backend and database configuration.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Key settings in `.env.example`:

- `DATABASE_URL`
- `MODEL_BACKEND`
- `VLLM_BASE_URL`
- `MLFLOW_TRACKING_URI`
- `API_HOST`
- `API_PORT`

### 3. Download and preprocess data

```bash
bash scripts/download_data.sh
python scripts/preprocess_data.py
```

### 4. Launch the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

### 5. Run tests

```bash
pytest tests/ -v
```

## Training

Fine-tuning is driven by the LoRA config in `configs/lora_config.yaml` and the training entry point in `src/model/train.py`.

```bash
python src/model/train.py --config configs/lora_config.yaml
```

Training workflow highlights:

- loads Qwen2.5-VL model and processor
- applies LoRA adapters to targeted modules
- trains with Hugging Face `Trainer`
- logs metrics and artifacts to MLflow
- saves the best checkpoint for downstream evaluation and serving

## Evaluation

Run the evaluation pipeline with:

```bash
python src/eval/run_eval.py --split test --checkpoint checkpoints/lora/best
```

This generates:

- markdown summary report
- prediction JSONL artifact
- error analysis CSV
- latency histogram
- MLflow-tracked metrics

## Repository Structure

```text
financial-ocr/
|-- configs/                  # model, LoRA, and serving configs
|-- docker/                   # container setup files kept in repo
|-- receipt_api/              # additional API-related app entry point
|-- reports/                  # generated evaluation outputs
|-- scripts/                  # download, preprocess, merge, and demo scripts
|-- src/
|   |-- api/                  # FastAPI app, routes, schemas, async processor
|   |-- data/                 # dataset loading and augmentation
|   |-- db/                   # async DB connection and ORM models
|   |-- eval/                 # metrics, benchmark, and evaluation pipeline
|   |-- model/                # Qwen wrapper and training code
|-- tests/                    # API, model, and eval tests
|-- ui_app/                   # lightweight UI app entry point
|-- README.md
|-- pyproject.toml
|-- requirements.txt
```

## Notes

- GPU-backed inference is the strongest path in the current project and is the basis for the benchmarked results above.
- CPU fallback code exists in the repository for development flexibility, but it is not the headline serving path for this project.
- Docker and GitHub workflow files are kept in the repo, but this README is intentionally centered on the parts that are already demonstrated with concrete outputs and metrics.

## Resume-Ready Summary

This project demonstrates the ability to build a complete applied ML system around a modern multimodal model: prepare data, fine-tune a large vision-language model with LoRA, expose it through an asynchronous API, persist results, evaluate extraction quality, and benchmark real inference behavior on GPU infrastructure.

## License

MIT
