"""
CPU terminal demo for Financial OCR API latency/concurrency.

This runs the real FastAPI app in-process, disables the 7B LoRA checkpoint by
default, uploads receipt images concurrently, and prints timing per receipt.
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

DEFAULT_IMAGES = [
    Path("data/processed/test/images/00000.jpg"),
    Path("data/processed/test/images/00001.jpg"),
    Path("data/processed/test/images/00002.jpg"),
]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def upload_and_wait(client, image_path: Path) -> dict:
    upload_start = time.perf_counter()
    with image_path.open("rb") as f:
        response = client.post(
            "/api/v1/upload",
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    upload_end = time.perf_counter()
    response.raise_for_status()
    upload_json = response.json()

    job_id = upload_json["job_id"]
    job_response = client.get(f"/api/v1/job/{job_id}")
    job_response.raise_for_status()
    job_json = job_response.json()

    result_json = None
    if job_json["status"] == "done":
        result_response = client.get(f"/api/v1/results/{job_id}")
        result_response.raise_for_status()
        result_json = result_response.json()

    end = time.perf_counter()
    return {
        "image": str(image_path),
        "job_id": job_id,
        "status": job_json["status"],
        "request_time_ms": (upload_end - upload_start) * 1000,
        "end_to_end_time_ms": (end - upload_start) * 1000,
        "model_inference_ms": (
            result_json.get("inference_ms") if result_json else None
        ),
        "upload_response": upload_json,
        "job_response": job_json,
        "result": result_json,
    }


async def upload_with_limits(
    client,
    image_path: Path,
    semaphore: asyncio.Semaphore,
    index: int,
) -> dict:
    await asyncio.sleep(index * 0.2)
    async with semaphore:
        return await asyncio.to_thread(upload_and_wait, client, image_path)


async def run_concurrent_uploads(client, images: list[Path]) -> list[dict]:
    semaphore = asyncio.Semaphore(2)
    tasks = [
        asyncio.create_task(upload_with_limits(client, image, semaphore, index))
        for index, image in enumerate(images)
    ]
    return await asyncio.gather(*tasks)


def main() -> None:
    os.environ["MODEL_BACKEND"] = "transformers"
    os.environ["LORA_CHECKPOINT"] = ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.environ["DATABASE_URL"] = (
        f"sqlite+aiosqlite:///./demo_cpu_concurrency_{timestamp}_{os.getpid()}.db"
    )

    images = [Path(value) for value in sys.argv[1:]] or DEFAULT_IMAGES
    for image in images:
        if not image.exists():
            raise FileNotFoundError(f"Receipt image not found: {image}")

    section("cpu compatibility")
    import torch

    print(f"MODEL_BACKEND={os.environ['MODEL_BACKEND']}")
    print(f"LORA_CHECKPOINT={os.environ['LORA_CHECKPOINT']!r}")
    print(f"DATABASE_URL={os.environ['DATABASE_URL']}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print("Expected CPU model: configs/model_config.yaml model_name_cpu")

    section("loading FastAPI app and CPU model")
    print("This can take several minutes on CPU.")
    from fastapi.testclient import TestClient
    from src.api.main import app

    total_start = time.perf_counter()
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        health.raise_for_status()
        print("API health:")
        pprint(health.json(), sort_dicts=False)

        section("concurrent uploads")
        print(f"num_images={len(images)}")
        print("concurrency=2")
        print("stagger_seconds=0.2")
        for image in images:
            print(f"image={image}")

        batch_start = time.perf_counter()
        results = asyncio.run(run_concurrent_uploads(client, images))
        batch_wall_clock_ms = (time.perf_counter() - batch_start) * 1000

    total_wall_clock_ms = (time.perf_counter() - total_start) * 1000

    section("final extracted JSON")
    for result in results:
        print(f"\nimage: {result['image']}")
        print(f"job_id: {result['job_id']}")
        print(f"status: {result['status']}")
        print(f"request_time_ms: {result['request_time_ms']:.1f}")
        print(f"end_to_end_time_ms: {result['end_to_end_time_ms']:.1f}")
        if result["model_inference_ms"] is not None:
            print(f"model_inference_ms: {result['model_inference_ms']:.1f}")
        if result["result"] is not None:
            pprint(result["result"], sort_dicts=False)
        else:
            print("No result returned.")
            pprint(result["job_response"], sort_dicts=False)

    section("latency summary")
    end_to_end = [result["end_to_end_time_ms"] for result in results]
    request_times = [result["request_time_ms"] for result in results]
    print(f"batch_wall_clock_ms: {batch_wall_clock_ms:.1f}")
    print(f"total_wall_clock_ms_including_model_load: {total_wall_clock_ms:.1f}")
    print(f"num_images: {len(results)}")
    print(f"avg_request_time_ms: {sum(request_times) / len(request_times):.1f}")
    print(f"avg_end_to_end_time_ms: {sum(end_to_end) / len(end_to_end):.1f}")
    print(f"max_end_to_end_time_ms: {max(end_to_end):.1f}")
    print(f"throughput_images_per_second: {len(results) / (batch_wall_clock_ms / 1000):.3f}")


if __name__ == "__main__":
    main()
