#!/usr/bin/env bash
# scripts/download_data.sh
# Downloads CORD v2 from Hugging Face and saves raw splits to data/raw/
set -euo pipefail

echo "── Downloading CORD v2 dataset ──────────────────────────────────────────"

python - <<'EOF'
from datasets import load_dataset
import json, pathlib

RAW = pathlib.Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

print("Fetching naver-clova-ix/cord-v2 from Hugging Face...")
ds = load_dataset("naver-clova-ix/cord-v2")

for split in ["train", "validation", "test"]:
    split_dir = RAW / split
    split_dir.mkdir(exist_ok=True)
    img_dir = split_dir / "images"
    img_dir.mkdir(exist_ok=True)

    records = []
    for i, sample in enumerate(ds[split]):
        # Save image
        img_path = img_dir / f"{i:05d}.jpg"
        sample["image"].save(img_path, format="JPEG", quality=95)

        # Parse ground truth
        gt = json.loads(sample["ground_truth"])

        records.append({
            "id": i,
            "image_path": str(img_path),
            "ground_truth": gt,
        })

    meta_path = split_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  {split:12s} → {len(records):4d} samples  ({meta_path})")

print("\nDownload complete.")
EOF

echo "── Done ─────────────────────────────────────────────────────────────────"
