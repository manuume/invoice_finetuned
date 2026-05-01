"""
src/api/async_processor.py

FastAPI BackgroundTask that runs OCR inference on an uploaded image,
updates the Job row, and writes the Result to the database.
"""
import json
import time
from pathlib import Path

from loguru import logger
from PIL import Image
from sqlalchemy import select

from src.db.connection import AsyncSessionLocal
from src.db.models import Job, Result
from src.model.qwen_model import QwenOCRModel

# Single model instance shared across background tasks
# Initialised once when the FastAPI app starts (see main.py lifespan)
_model: QwenOCRModel | None = None


def set_model(model: QwenOCRModel) -> None:
    global _model
    _model = model


async def process_receipt(job_id: str, image_path: str) -> None:
    """
    Background task body.
    1. Mark job as 'processing'
    2. Run inference
    3. Write Result row
    4. Mark job as 'done' (or 'failed' on error)
    """
    async with AsyncSessionLocal() as session:
        # ── Fetch job ────────────────────────────────────────────────────
        stmt = select(Job).where(Job.id == job_id)
        job = (await session.execute(stmt)).scalar_one_or_none()
        if job is None:
            logger.error(f"Job {job_id} not found in DB")
            return

        job.status = "processing"
        await session.commit()

        try:
            # ── Inference ────────────────────────────────────────────────
            image = Image.open(image_path).convert("RGB")

            t0 = time.perf_counter()
            extracted = _model.predict(image)
            inference_ms = (time.perf_counter() - t0) * 1000

            logger.info(f"Job {job_id} | inference {inference_ms:.1f} ms | {extracted}")

            # ── Persist result ────────────────────────────────────────────
            items = extracted.get("items", [])
            result = Result(
                job_id=job_id,
                store_name=extracted.get("store_name"),
                total_price=extracted.get("total_price"),
                tax_price=extracted.get("tax_price"),
                items_json=json.dumps(items, ensure_ascii=False),
                raw_output=json.dumps(extracted),
                inference_ms=round(inference_ms, 2),
            )
            session.add(result)
            job.status = "done"
            await session.commit()

        except Exception as exc:
            logger.exception(f"Job {job_id} failed: {exc}")
            job.status = "failed"
            job.error_message = str(exc)
            await session.commit()

        finally:
            # Clean up temp upload file
            try:
                Path(image_path).unlink(missing_ok=True)
            except Exception:
                pass
