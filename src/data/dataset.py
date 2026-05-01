"""
src/data/dataset.py

CORD v2 dataset loading and target field extraction.
Defines a PyTorch Dataset used by the training script.
"""
import json
import pathlib
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


# ── Field extraction ──────────────────────────────────────────────────────────

def _get_field(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    if isinstance(container, list):
        for entry in container:
            if isinstance(entry, dict) and key in entry:
                return entry[key]
    return default


def extract_target_fields(gt: str | dict) -> dict[str, Any]:
    """
    Extract our 4 target fields from CORD v2 ground_truth.
    Handles both raw JSON strings and already-parsed dicts.
    """
    # CORD v2 returns ground_truth as a JSON string → parse it first
    if isinstance(gt, str):
        gt = json.loads(gt)
        
    gt_parse = gt.get("gt_parse", gt)
    
    store_info = gt_parse.get("store_info", {})
    total_info = gt_parse.get("total", {})
    sub_total = gt_parse.get("sub_total", {})
    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    elif not isinstance(menu, list):
        menu = []
    
    items = []
    for entry in menu:
        # CORD sometimes stores menu items as JSON strings instead of dicts
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except Exception:
                continue
        if not isinstance(entry, dict):
            continue
            
        name = entry.get("nm", "")
        price = entry.get("price", "")
        if name:
            items.append({"name": name, "price": price})

    tax_price = _get_field(total_info, "tax_price", None)
    if tax_price is None:
        tax_price = _get_field(sub_total, "tax_price", None)
            
    return {
        "store_name": store_info.get("store_name", ""),
        "total_price": _get_field(total_info, "total_price", ""),
        "tax_price": tax_price,
        "items": items,
    }


def format_target_as_json(target: dict[str, Any]) -> str:
    """Serialise target dict to compact JSON string (the label for training)."""
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a financial receipt extraction assistant. "
    "Extract information accurately from receipt images. "
    "Always respond with valid JSON only — no explanation, no markdown."
)

EXTRACTION_PROMPT = (
    'Extract the following fields from this receipt image and return as JSON:\n'
    '{"store_name":"store name","total_price":"total","tax_price":"tax or null",'
    '"items":[{"name":"item","price":"price"}]}\n'
    "Return ONLY the JSON object. No extra text."
)


class CORDDataset(Dataset):
    """
    Loads pre-processed CORD v2 split and returns tokenised inputs
    ready for Qwen2.5-VL fine-tuning.
    """

    def __init__(
        self,
        split: str,
        processor: AutoProcessor,
        processed_dir: str = "data/processed",
        max_new_tokens: int = 512,
    ) -> None:
        self.processor = processor
        self.max_new_tokens = max_new_tokens

        meta_path = pathlib.Path(processed_dir) / split / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {meta_path}. "
                "Run scripts/preprocess_data.py first."
            )

        with open(meta_path) as f:
            self.records = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        target_json = format_target_as_json(rec["target"])

        # Build chat messages in Qwen2.5-VL format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            },
            {"role": "assistant", "content": target_json},
        ]

        # Tokenise — processor handles image patching + text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=False, 
            max_length=2048,)

        # Flatten batch dimension added by processor
        result = {}
        for k, v in inputs.items():
            if k == "image_grid_thw":
                result[k] = v
            else:
                result[k] = v.squeeze(0)
        return result
