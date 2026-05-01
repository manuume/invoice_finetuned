"""
src/model/train.py

LoRA fine-tuning of Qwen2.5-VL-7B on CORD v2.
Tracks experiments with MLflow, saves best checkpoint by eval loss.

Usage (Lightning AI GPU studio):
    python src/model/train.py --config configs/lora_config.yaml
"""
import argparse
import os
import sys
import pathlib

import mlflow
import torch
import yaml
from loguru import logger
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from src.data.dataset import CORDDataset


# ── Collator ──────────────────────────────────────────────────────────────────

def make_collator(processor):
    """
    Pads a batch of tokenised samples and builds labels.
    Tokens that correspond to the prompt (non-assistant) are masked to -100
    so the loss only trains on the JSON output.
    """
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
        )
        attention_mask = (input_ids != pad_id).long()

        # Labels: copy input_ids but mask pad tokens
        labels = input_ids.clone()
        labels[labels == pad_id] = -100

        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # Include pixel_values / image_grid_thw if present
        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

        return result

    return collate


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("lora_config"):
        lora_config_path = pathlib.Path(cfg["lora_config"])
        if not lora_config_path.is_absolute():
            lora_config_path = pathlib.Path(config_path).parent / lora_config_path
        with open(lora_config_path) as f:
            cfg = {**yaml.safe_load(f), **cfg}

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_experiment(cfg.get("mlflow_experiment", "financial-ocr-lora"))

    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in cfg.items() if not k.startswith("mlflow")})

        # ── Load processor & model ─────────────────────────────────────────
        import yaml as _yaml
        with open("configs/model_config.yaml") as f:
            model_cfg = _yaml.safe_load(f)

        model_name = (
            model_cfg["model_name"]
            if torch.cuda.is_available()
            else model_cfg["model_name_cpu"]
        )
        logger.info(f"Loading base model: {model_name}")

        use_4bit = cfg.get("load_in_4bit", cfg.get("use_4bit", False)) or model_cfg.get(
            "load_in_4bit", model_cfg.get("use_4bit", False)
        )
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        max_seq_length = cfg.get("max_seq_length", cfg.get("model_max_length"))
        if max_seq_length:
            processor.tokenizer.model_max_length = int(max_seq_length)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # ── Apply LoRA ────────────────────────────────────────────────────
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias=cfg["bias"],
            target_modules=cfg["target_modules"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        mlflow.log_metric("trainable_params_pct", round(100 * trainable / total, 2))

        # ── Datasets ──────────────────────────────────────────────────────
        train_ds = CORDDataset("train", processor)
        val_ds = CORDDataset("validation", processor)
        logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

        # ── Training args ─────────────────────────────────────────────────
        training_args = TrainingArguments(
            output_dir=cfg["output_dir"],
            num_train_epochs=cfg["num_train_epochs"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            warmup_steps=cfg.get("warmup_steps", 30),  # Use warmup_steps (config has 30)
            weight_decay=cfg["weight_decay"],
            bf16=cfg.get("bf16", False) and torch.cuda.is_available(),
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
            fp16=False,
            logging_steps=cfg["logging_steps"],
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_steps=cfg["save_steps"],
            max_steps=cfg.get("max_steps", -1),
            save_total_limit=cfg["save_total_limit"],
            load_best_model_at_end=cfg["load_best_model_at_end"],
            metric_for_best_model=cfg["metric_for_best_model"],
            report_to="none",          # we log manually to MLflow
            dataloader_num_workers=2,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=make_collator(processor),
        )

        # ── Train ─────────────────────────────────────────────────────────
        logger.info("Starting training...")
        train_result = trainer.train()

        # ── Log final metrics ─────────────────────────────────────────────
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime_s": train_result.metrics.get("train_runtime", 0),
        })

        # ── Save best checkpoint ───────────────────────────────────────────
        best_ckpt = pathlib.Path(cfg["output_dir"]) / "best"
        trainer.save_model(str(best_ckpt))
        processor.save_pretrained(str(best_ckpt))
        mlflow.log_artifacts(str(best_ckpt), artifact_path="model")
        logger.info(f"Best checkpoint saved to {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    args = parser.parse_args()
    main(args.config)
