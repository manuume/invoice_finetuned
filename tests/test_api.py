"""
tests/test_api.py

Integration tests for FastAPI routes using httpx async client.
Uses a mock model so no GPU / real inference needed.
"""
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_PREDICTION = {
    "store_name": "Test Store",
    "total_price": "12.50",
    "tax_price": "1.25",
    "items": [{"name": "Coffee", "price": "3.50"}],
}


def _make_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def mock_model():
    m = MagicMock()
    m.predict.return_value = MOCK_PREDICTION
    m.close.return_value = None
    return m


@pytest_asyncio.fixture(scope="module")
async def client(mock_model):
    """Create test client with mocked model and in-memory SQLite."""
    import os
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_api.db"
    os.environ["MODEL_BACKEND"] = "transformers"

    with patch("src.api.main.QwenOCRModel", return_value=mock_model), \
         patch("src.api.async_processor._model", mock_model):

        from src.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_upload_returns_job_id(client):
    img_bytes = _make_jpeg_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={"file": ("receipt.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_upload_rejects_bad_filetype(client):
    resp = await client.post(
        "/api/v1/upload",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_job_status_not_found(client):
    resp = await client.get("/api/v1/job/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_result_not_found_for_pending_job(client):
    img_bytes = _make_jpeg_bytes()
    upload = await client.post(
        "/api/v1/upload",
        files={"file": ("r.jpg", img_bytes, "image/jpeg")},
    )
    job_id = upload.json()["job_id"]

    # Job is pending — result not ready
    resp = await client.get(f"/api/v1/results/{job_id}")
    assert resp.status_code == 409
