"""
Offline terminal demo for the Financial OCR FastAPI flow.

Runs the real FastAPI app in-process with TestClient, uploads three receipts
through /api/v1/upload, polls jobs, and prints structured extraction results.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from pprint import pprint


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RECEIPTS = [
    Path("data/processed/test/images/00000.jpg"),
    Path("data/processed/test/images/00001.jpg"),
    Path("data/processed/test/images/00002.jpg"),
]
CHECKPOINT = Path("checkpoints/lora/best")
POLL_INTERVAL_SECONDS = 1.0
POLL_TIMEOUT_SECONDS = 300.0


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_metrics_summary() -> None:
    report_path = Path("reports/eval_test.md")
    print(f"Report: {report_path}")
    if not report_path.exists():
        print("No evaluation report found yet.")
        return

    wanted = (
        "Character Error Rate",
        "Word Error Rate",
        "Field-level Accuracy",
        "Field F1 Score",
        "Field Precision",
        "Field Recall",
    )
    for line in report_path.read_text().splitlines():
        if any(metric in line for metric in wanted):
            print(line)


def fallback_process_if_needed(client, upload_response: dict, image_path: Path) -> None:
    job_id = upload_response["job_id"]
    status = client.get(f"/api/v1/job/{job_id}").json()["status"]
    if status != "pending":
        return

    print(f"Background task still pending for {job_id}; running direct fallback.")
    from src.api.async_processor import process_receipt

    uploaded_path = Path("data/uploads") / f"{job_id}{image_path.suffix}"
    if not uploaded_path.exists():
        print(f"Fallback skipped; uploaded file not found: {uploaded_path}")
        return
    asyncio.run(process_receipt(job_id, str(uploaded_path)))


def poll_job(client, job_id: str) -> dict:
    deadline = time.perf_counter() + POLL_TIMEOUT_SECONDS
    while time.perf_counter() < deadline:
        response = client.get(f"/api/v1/job/{job_id}")
        response.raise_for_status()
        payload = response.json()
        print(f"job_id={job_id} status={payload['status']}")
        if payload["status"] in {"done", "failed"}:
            return payload
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Timed out waiting for job {job_id}")


def main() -> None:
    os.environ.setdefault("MODEL_BACKEND", "transformers")
    os.environ.setdefault("LORA_CHECKPOINT", str(CHECKPOINT))
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./demo_offline_api.db")

    start = time.perf_counter()

    section("checkpoint verification")
    print(f"MODEL_BACKEND={os.environ['MODEL_BACKEND']}")
    print(f"LORA_CHECKPOINT={os.environ['LORA_CHECKPOINT']}")
    print(f"DATABASE_URL={os.environ['DATABASE_URL']}")
    required_checkpoint_files = [
        CHECKPOINT / "adapter_config.json",
        CHECKPOINT / "adapter_model.safetensors",
    ]
    for path in required_checkpoint_files:
        print(f"{path}: {'OK' if path.exists() else 'MISSING'}")
    for path in RECEIPTS:
        print(f"{path}: {'OK' if path.exists() else 'MISSING'}")

    missing = [path for path in required_checkpoint_files + RECEIPTS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required demo files: {missing}")

    section("metrics summary")
    print_metrics_summary()

    print("\nLoading FastAPI app and OCR model. This may take a few minutes...")
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        health.raise_for_status()
        print("\nAPI health:")
        pprint(health.json(), sort_dicts=False)

        section("uploading receipts")
        uploads = []
        for image_path in RECEIPTS:
            print(f"\nImage: {image_path}")
            upload_start = time.perf_counter()
            with image_path.open("rb") as f:
                response = client.post(
                    "/api/v1/upload",
                    files={"file": (image_path.name, f, "image/jpeg")},
                )
            upload_end = time.perf_counter()
            response.raise_for_status()
            payload = response.json()
            request_time_ms = (upload_end - upload_start) * 1000
            uploads.append(
                {
                    "image_path": image_path,
                    "payload": payload,
                    "upload_start": upload_start,
                    "request_time_ms": request_time_ms,
                }
            )
            print("Upload response:")
            pprint(payload, sort_dicts=False)
            print(f"request_time_ms: {request_time_ms:.1f}")
            fallback_process_if_needed(client, payload, image_path)

        section("polling jobs")
        final_statuses = {}
        for upload in uploads:
            job_id = upload["payload"]["job_id"]
            final_statuses[job_id] = poll_job(client, job_id)
            upload["end_to_end_time_ms"] = (
                time.perf_counter() - upload["upload_start"]
            ) * 1000

        failed = [job_id for job_id, status in final_statuses.items() if status["status"] == "failed"]
        if failed:
            print("\nFailed jobs:")
            for job_id in failed:
                pprint(final_statuses[job_id], sort_dicts=False)

        section("final extracted JSON")
        for upload in uploads:
            image_path = upload["image_path"]
            payload = upload["payload"]
            job_id = payload["job_id"]
            print(f"\nImage: {image_path}")
            print(f"job_id: {job_id}")
            print(f"request_time_ms: {upload['request_time_ms']:.1f}")
            print(f"end_to_end_time_ms: {upload['end_to_end_time_ms']:.1f}")
            if final_statuses[job_id]["status"] != "done":
                print("No result because job failed.")
                continue
            result = client.get(f"/api/v1/results/{job_id}")
            result.raise_for_status()
            result_json = result.json()
            if result_json.get("inference_ms") is not None:
                print(f"model_inference_ms: {result_json['inference_ms']:.1f}")
            pprint(result_json, sort_dicts=False)

    elapsed = time.perf_counter() - start
    section("total wall-clock time")
    print(f"total_wall_clock_ms: {elapsed * 1000:.1f}")
    print(f"total_wall_clock_seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
