"""
src/api/main.py

FastAPI application.

Lifespan:
  startup  — initialise DB tables, load model into memory
  shutdown — close model / release GPU memory

Run locally:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger

load_dotenv()

from src.api.async_processor import set_model
from src.api.routes import router
from src.db.connection import init_db
from src.model.qwen_model import QwenOCRModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("Initialising database...")
    await init_db()

    logger.info("Loading OCR model...")
    checkpoint = os.getenv("LORA_CHECKPOINT")   # optional: path to fine-tuned adapter
    model = QwenOCRModel(checkpoint=checkpoint)
    set_model(model)
    logger.info("Model ready.")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Shutting down model...")
    model.close()


app = FastAPI(
    title="Financial OCR API",
    description="Receipt field extraction using Qwen2.5-VL-7B + LoRA.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
