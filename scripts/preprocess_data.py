"""
scripts/preprocess_data.py

Reads raw CORD v2 metadata, extracts our 4 target fields,
applies Albumentations augmentation to the training split,
and writes processed splits to data/processed/.

Run:
    python scripts/preprocess_data.py
"""
import json
import pathlib
import sys

import cv2
import numpy as np
from loguru import logger
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.data.augmentation import build_augmentation_pipeline
from src.data.dataset import extract_target_fields

RAW = pathlib.Path("data/raw")
PROCESSED = pathlib.Path("data/processed")
AUGMENT_FACTOR = 3   # each training image → 3 augmented copies


def process_split(split: str, augment: bool = False) -> None:
    meta_path = RAW / split / "metadata.json"
    if not meta_path.exists():
        logger.error(f"Metadata not found: {meta_path}. Run scripts/download_data.sh first.")
        sys.exit(1)

    with open(meta_path) as f:
        records = json.load(f)

    out_dir = PROCESSED / split
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    pipeline = build_augmentation_pipeline() if augment else None
    processed = []

    for rec in records:
        img_path = pathlib.Path(rec["image_path"])
        target = extract_target_fields(rec["ground_truth"])

        # Copy original
        img = Image.open(img_path).convert("RGB")
        out_img_path = img_out / img_path.name
        img.save(out_img_path, format="JPEG", quality=95)

        processed.append({
            "id": f"{split}_{rec['id']:05d}_orig",
            "image_path": str(out_img_path),
            "target": target,
            "augmented": False,
        })

        if augment and pipeline:
            img_np = np.array(img)
            for aug_idx in range(AUGMENT_FACTOR):
                aug_result = pipeline(image=img_np)
                aug_img = Image.fromarray(aug_result["image"])
                aug_name = f"{img_path.stem}_aug{aug_idx}.jpg"
                aug_path = img_out / aug_name
                aug_img.save(aug_path, format="JPEG", quality=85)

                processed.append({
                    "id": f"{split}_{rec['id']:05d}_aug{aug_idx}",
                    "image_path": str(aug_path),
                    "target": target,
                    "augmented": True,
                })

    out_meta = out_dir / "metadata.json"
    with open(out_meta, "w") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    logger.info(f"{split:12s} → {len(processed):5d} samples  (augmented={augment})")


def main() -> None:
    logger.info("Starting preprocessing...")
    process_split("train", augment=True)
    process_split("validation", augment=False)
    process_split("test", augment=False)
    logger.info("Preprocessing complete. Outputs in data/processed/")


if __name__ == "__main__":
    main()
