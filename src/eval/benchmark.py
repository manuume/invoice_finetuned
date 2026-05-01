"""
src/eval/benchmark.py

Standalone latency + throughput benchmark.
Runs inference over the test split N times and reports p50/p95/p99.

Usage:
    python -m src.eval.benchmark --split test --n-samples 50
"""
import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np
from loguru import logger
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from src.model.qwen_model import QwenOCRModel


def run_benchmark(
    split: str = "test",
    n_samples: int = 50,
    processed_dir: str = "data/processed",
) -> dict:
    meta_path = pathlib.Path(processed_dir) / split / "metadata.json"
    with open(meta_path) as f:
        records = json.load(f)

    records = records[:n_samples]
    logger.info(f"Benchmarking {len(records)} samples from '{split}' split...")

    model = QwenOCRModel()
    latencies: list[float] = []

    for i, rec in enumerate(records):
        image = Image.open(rec["image_path"]).convert("RGB")
        t0 = time.perf_counter()
        model.predict(image)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(records)} — last: {elapsed_ms:.1f} ms")

    model.close()

    results = {
        "n_samples": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 1),
        "median_ms": round(statistics.median(latencies), 1),
        "std_ms": round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0.0,
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "p99_ms": round(float(np.percentile(latencies, 99)), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "throughput_per_min": round(60_000 / statistics.mean(latencies), 1),
    }

    print("\n── Benchmark Results ────────────────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<24} {v}")
    print("─────────────────────────────────────────────────────────────────────\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()
    run_benchmark(split=args.split, n_samples=args.n_samples)
