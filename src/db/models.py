"""
src/db/models.py

Three tables:
  Job     — one row per uploaded receipt, tracks processing lifecycle
  Result  — extracted JSON fields, linked to Job
  EvalRun — evaluation run metrics (logged by run_eval.py)
"""
import datetime
import uuid

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.connection import Base


def _now() -> datetime.datetime:
    return datetime.datetime.utcnow()


def _uuid() -> str:
    return str(uuid.uuid4())


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(
        String(16), default="pending"
    )  # pending | processing | done | failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    result: Mapped["Result | None"] = relationship("Result", back_populates="job", uselist=False)


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), unique=True, index=True)
    store_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    total_price: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tax_price: Mapped[str | None] = mapped_column(String(64), nullable=True)
    items_json: Mapped[str | None] = mapped_column(Text, nullable=True)   # JSON list
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)   # model raw text
    inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)

    job: Mapped["Job"] = relationship("Job", back_populates="result")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    split: Mapped[str] = mapped_column(String(16))           # train / validation / test
    num_samples: Mapped[int] = mapped_column(Integer)
    cer: Mapped[float | None] = mapped_column(Float, nullable=True)
    wer: Mapped[float | None] = mapped_column(Float, nullable=True)
    field_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    field_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    p99_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_now)
