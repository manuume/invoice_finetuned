"""
src/api/schemas.py

Pydantic v2 models used by FastAPI request / response bodies.
"""
from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str = "pending"
    message: str = "Receipt queued for processing."


# ── Job status ────────────────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: str                    # pending | processing | done | failed
    error_message: str | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime


# ── Extraction result ─────────────────────────────────────────────────────────

class ExtractedItem(BaseModel):
    name: str
    price: str


class ExtractionResult(BaseModel):
    job_id: str
    store_name: str | None = None
    total_price: str | None = None
    tax_price: str | None = None
    items: list[ExtractedItem] = Field(default_factory=list)
    inference_ms: float | None = None
    created_at: datetime.datetime


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    model_backend: str
    db_url: str
