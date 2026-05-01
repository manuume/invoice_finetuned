"""
Merge the trained LoRA adapter into a standalone model directory for vLLM.

Default output matches configs/vllm_config.yaml:
    checkpoints/lora/merged
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter", default="checkpoints/lora/best")
    parser.add_argument("--output", default="checkpoints/lora/merged")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    output_path = Path(args.output)
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

    print("Loading base model:")
    print(f"  {args.base_model}")
    print("Loading LoRA adapter:")
    print(f"  {adapter_path}")

    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
