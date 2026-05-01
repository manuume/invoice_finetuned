"""
src/model/qwen_model.py

Unified inference wrapper around Qwen2.5-VL-7B.
Auto-detects backend:
  - GPU available → vLLM (fast batched inference)
  - CPU only      → HuggingFace transformers + 4-bit quantization
"""
import base64
import io
import json
import os
from typing import Any

import torch
from loguru import logger
from PIL import Image

from src.data.dataset import EXTRACTION_PROMPT, SYSTEM_PROMPT


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_json_response(text: str) -> dict[str, Any]:
    """Strip markdown fences if present and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse model output as JSON: {text[:200]}")
        return {"raw_output": text, "parse_error": True}


# ── vLLM backend ──────────────────────────────────────────────────────────────

class VLLMBackend:
    """Calls a running vLLM server via OpenAI-compatible HTTP API."""

    def __init__(self, base_url: str, model: str) -> None:
        import httpx
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=60.0)
        logger.info(f"VLLMBackend → {self.base_url}")

    def predict(self, image: Image.Image) -> dict[str, Any]:
        b64 = _pil_to_b64(image)
        payload = {
            "model": self.model,
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
        resp = self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _parse_json_response(content)

    def close(self) -> None:
        self.client.close()


# ── Transformers backend ──────────────────────────────────────────────────────

class TransformersBackend:
    """
    Loads Qwen2.5-VL using HuggingFace transformers.
    Uses 4-bit quantization when on CPU to keep memory manageable.
    """

    def __init__(self, model_name: str, checkpoint: str | None = None) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        from peft import PeftModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TransformersBackend → device={self.device}, model={model_name}")

        quantization_config = None
        if self.device == "cpu":
            # 4-bit quant lets the 7B model fit in ~8 GB RAM
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        if checkpoint:
            logger.info(f"Loading LoRA adapter from {checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, checkpoint)
            self.model = self.model.merge_and_unload()

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # Strip the input tokens from output
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        raw = self.processor.decode(generated, skip_special_tokens=True)
        return _parse_json_response(raw)

    def close(self) -> None:
        pass   # model stays in memory


# ── Public factory ────────────────────────────────────────────────────────────

class QwenOCRModel:
    """
    Auto-selects backend based on MODEL_BACKEND env variable:
      MODEL_BACKEND=vllm        → VLLMBackend
      MODEL_BACKEND=transformers → TransformersBackend (default)
    """

    def __init__(self, model_name: str | None = None, checkpoint: str | None = None) -> None:
        import yaml

        cfg_path = "configs/model_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        backend_env = os.getenv("MODEL_BACKEND", "transformers")
        resolved_name = model_name or (
            cfg["model_name"] if torch.cuda.is_available() else cfg["model_name_cpu"]
        )

        if backend_env == "vllm":
            vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
            self._backend = VLLMBackend(base_url=vllm_url, model=resolved_name)
        else:
            self._backend = TransformersBackend(
                model_name=resolved_name, checkpoint=checkpoint
            )

    def predict(self, image: Image.Image) -> dict[str, Any]:
        return self._backend.predict(image)

    def close(self) -> None:
        self._backend.close()
