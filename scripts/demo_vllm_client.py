"""
Terminal client for proving Financial OCR vLLM serving.

Sends receipt images directly to the vLLM OpenAI-compatible endpoint and
prints extracted JSON plus per-image and batch latency.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Any

import httpx
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import EXTRACTION_PROMPT, SYSTEM_PROMPT


DEFAULT_IMAGES = [
    "data/processed/test/images/00000.jpg",
    "data/processed/test/images/00001.jpg",
    "data/processed/test/images/00002.jpg",
]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_output": text, "parse_error": True}


def build_payload(model: str, image_path: Path) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    b64 = pil_to_b64(image)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            },
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }


async def send_receipt(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    image_path: Path,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        payload = build_payload(model, image_path)
        start = time.perf_counter()
        response = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        request_time_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        return {
            "image": str(image_path),
            "request_time_ms": request_time_ms,
            "raw_output": content,
            "parsed_json": parse_json_response(content),
        }


async def run_demo(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    image_paths = [Path(image) for image in (args.image or DEFAULT_IMAGES)]
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Receipt image not found: {image_path}")

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        section("vLLM health")
        models_resp = await client.get(f"{base_url}/v1/models")
        models_resp.raise_for_status()
        pprint(models_resp.json(), sort_dicts=False)

        section("batch request")
        print(f"model: {args.model}")
        print(f"num_images: {len(image_paths)}")
        print(f"concurrency: {args.concurrency}")
        for image_path in image_paths:
            print(f"image: {image_path}")

        semaphore = asyncio.Semaphore(args.concurrency)
        batch_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                send_receipt(client, base_url, args.model, image_path, semaphore)
                for image_path in image_paths
            ]
        )
        batch_wall_clock_ms = (time.perf_counter() - batch_start) * 1000

    section("final extracted JSON")
    for result in results:
        print(f"\nimage: {result['image']}")
        print(f"request_time_ms: {result['request_time_ms']:.1f}")
        print("raw_output:")
        print(result["raw_output"])
        print("parsed_json:")
        pprint(result["parsed_json"], sort_dicts=False)

    request_times = [result["request_time_ms"] for result in results]
    avg_request_time_ms = sum(request_times) / len(request_times)

    section("latency")
    print(f"batch_wall_clock_ms: {batch_wall_clock_ms:.1f}")
    print(f"num_images: {len(results)}")
    print(f"avg_request_time_ms: {avg_request_time_ms:.1f}")
    print(f"min_request_time_ms: {min(request_times):.1f}")
    print(f"max_request_time_ms: {max(request_times):.1f}")
    print(f"throughput_images_per_second: {len(results) / (batch_wall_clock_ms / 1000):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--image",
        action="append",
        help="Receipt image path. Repeat for multiple images. Defaults to 3 test receipts.",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
