"""
src/db/connection.py

Async SQLAlchemy engine.
DATABASE_URL env variable controls the backend:
  sqlite+aiosqlite:///./financial_ocr.db  (default, dev)
  postgresql+asyncpg://...                 (prod)
"""
import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite+aiosqlite:///./financial_ocr.db"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    # SQLite-specific: allow use across async tasks
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create all tables on startup."""
    from src.db import models as _  # noqa: F401 — ensures models are registered
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """FastAPI dependency — yields a DB session per request."""
    async with AsyncSessionLocal() as session:
        yield session
