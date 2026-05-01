"""
src/api/routes.py

All API endpoints.

POST /upload          — upload a receipt image, returns job_id
GET  /job/{job_id}    — poll job status
GET  /results/{job_id}— get extraction result once job is done
GET  /health          — liveness probe
"""
import json
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.async_processor import process_receipt
from src.api.schemas import (
    ExtractionResult,
    ExtractedItem,
    HealthResponse,
    JobStatusResponse,
    UploadResponse,
)
from src.db.connection import DATABASE_URL, get_session
from src.db.models import Job, Result

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_receipt(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Accepted: {', '.join(ALLOWED_TYPES)}",
        )

    job_id = str(uuid.uuid4())
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    save_path = UPLOAD_DIR / f"{job_id}{suffix}"

    contents = await file.read()
    save_path.write_bytes(contents)

    # Create Job row
    job = Job(id=job_id, filename=file.filename or "upload", status="pending")
    session.add(job)
    await session.commit()

    background_tasks.add_task(process_receipt, job_id, str(save_path))

    return UploadResponse(job_id=job_id, filename=file.filename or "upload")


# ── Job status ────────────────────────────────────────────────────────────────

@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, session: AsyncSession = Depends(get_session)):
    stmt = select(Job).where(Job.id == job_id)
    job = (await session.execute(stmt)).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=job.id,
        filename=job.filename,
        status=job.status,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# ── Result ────────────────────────────────────────────────────────────────────

@router.get("/results/{job_id}", response_model=ExtractionResult)
async def get_result(job_id: str, session: AsyncSession = Depends(get_session)):
    stmt = select(Job).where(Job.id == job_id)
    job = (await session.execute(stmt)).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not done yet. Current status: {job.status}",
        )

    stmt = select(Result).where(Result.job_id == job_id)
    result = (await session.execute(stmt)).scalar_one_or_none()
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")

    items_raw = json.loads(result.items_json or "[]")
    items = [ExtractedItem(**i) for i in items_raw if isinstance(i, dict)]

    return ExtractionResult(
        job_id=job_id,
        store_name=result.store_name,
        total_price=result.total_price,
        tax_price=result.tax_price,
        items=items,
        inference_ms=result.inference_ms,
        created_at=result.created_at,
    )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_backend=os.getenv("MODEL_BACKEND", "transformers"),
        db_url=DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,
    )
