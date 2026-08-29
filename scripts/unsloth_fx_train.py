#!/usr/bin/env python3
"""Run local Unsloth QLoRA training for redacted FX trajectories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import unsloth
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel, is_bfloat16_supported


MODEL_NAME = "unsloth/LFM2.5-8B-A1B"
LFM_TARGET_MODULES = [
    "w1",
    "w2",
    "w3",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "in_proj",
]
MIN_CURATED_RECORDS = 20


def default_dataset_path() -> Path:
    configured = os.environ.get("FX_TRAJECTORY_DATASET")
    if configured:
        return Path(configured).expanduser()

    candidates = (
        Path("/mnt/c/Users/Hudson/.cache/fx-lfm-trajectories.jsonl"),
        Path.home() / ".cache" / "fx-lfm-trajectories.jsonl",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"dataset not found: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc

        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise SystemExit(f"record {line_number} has no messages list")
        for message in messages:
            if not isinstance(message, dict):
                raise SystemExit(f"record {line_number} contains a non-object message")
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise SystemExit(f"record {line_number} contains an unsupported role")
            if not isinstance(message.get("content"), str):
                raise SystemExit(f"record {line_number} contains non-string content")
        records.append(record)

    if not records:
        raise SystemExit(f"dataset is empty: {path}")
    return records


def normalized_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in record["messages"]:
        normalized = {
            "role": message["role"],
            "content": message["content"],
        }
        if message["role"] == "tool" and message.get("name"):
            normalized["name"] = message["name"]
        messages.append(normalized)
    return messages


def make_dataset(records: list[dict[str, Any]], tokenizer: Any) -> Dataset:
    rows = []
    for index, record in enumerate(records):
        text = tokenizer.apply_chat_template(
            normalized_messages(record),
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"example_id": str(record.get("metadata", {}).get("example_id", index)), "text": text})
    return Dataset.from_list(rows)


def split_dataset(dataset: Dataset, eval_ratio: float) -> tuple[Dataset, Dataset | None]:
    if len(dataset) < 5 or eval_ratio <= 0:
        return dataset, None
    split = dataset.train_test_split(test_size=eval_ratio, seed=42)
    return split["train"], split["test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument("--model", default=os.environ.get("FX_UNSLOTH_MODEL", MODEL_NAME))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("FX_UNSLOTH_OUTPUT", Path.home() / ".cache" / "fx-lfm-qlora")),
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="run two training steps and save only the adapter")
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument("--export-gguf", action="store_true")
    parser.add_argument("--quantization-method", default="q8_0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.dataset)
    if len(records) < MIN_CURATED_RECORDS and not args.allow_small_dataset:
        raise SystemExit(
            f"refusing to train on {len(records)} records; curate at least "
            f"{MIN_CURATED_RECORDS} or pass --allow-small-dataset for a smoke check"
        )
    if len(records) < 50:
        print(f"warning: {len(records)} records; result is not a quality fine-tune")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model}")
    print(f"dataset={args.dataset} records={len(records)}")
    print(f"output_dir={args.output_dir}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
        use_gradient_checkpointing="unsloth",
    )
    dataset = make_dataset(records, tokenizer)
    train_dataset, eval_dataset = split_dataset(dataset, args.eval_ratio)

    model = FastModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=LFM_TARGET_MODULES,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    bf16 = torch.cuda.is_available() and is_bfloat16_supported()
    max_steps = args.max_steps or (2 if args.smoke else -1)
    adapter_dir = args.output_dir / "adapter"
    training_args = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        optim="adamw_8bit",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="steps" if eval_dataset is not None and not args.smoke else "no",
        eval_steps=1,
        report_to="none",
        seed=42,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not args.smoke else None,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dataset": str(args.dataset),
                "records": len(records),
                "max_seq_length": args.max_seq_length,
                "max_steps": max_steps,
                "epochs": args.epochs,
                "lora_rank": args.lora_rank,
                "target_modules": LFM_TARGET_MODULES,
                "bf16": bf16,
                "smoke": args.smoke,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"adapter={adapter_dir}")

    if args.export_gguf:
        if args.smoke:
            raise SystemExit("refusing GGUF export from --smoke run")
        merged_dir = args.output_dir / "merged-16bit"
        gguf_dir = args.output_dir / "gguf"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=args.quantization_method,
        )
        print(f"merged={merged_dir}")
        print(f"gguf={gguf_dir}")


if __name__ == "__main__":
    main()
