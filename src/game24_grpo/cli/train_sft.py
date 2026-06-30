from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from game24_grpo.config import DEFAULT_PROMPT_TEMPLATE
from game24_grpo.prompting import build_prompt
from game24_grpo.training import _build_peft_config, _resolve_torch_dtype


@dataclass
class SftLoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class SftConfig:
    model_name: str
    train_path: str
    output_dir: str
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    torch_dtype: str | None = "float32"
    max_steps: int = 100
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    save_steps: int = 100
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_length: int = 384
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"
    report_to: list[str] = field(default_factory=lambda: ["none"])
    lora: SftLoraConfig = field(default_factory=SftLoraConfig)


class SftDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _load_config(path: str | Path) -> SftConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    lora = payload.pop("lora", {})
    return SftConfig(lora=SftLoraConfig(**lora), **payload)


def _render_prompt(prompt: str, tokenizer: AutoTokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        return prompt


def _load_sft_dataset(config: SftConfig, tokenizer: AutoTokenizer) -> Dataset:
    rows: list[dict[str, Any]] = []
    with Path(config.train_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            prompt = build_prompt(
                numbers=raw["numbers"],
                target=int(raw.get("target", 24)),
                prompt_template=config.prompt_template,
            )
            rendered_prompt = _render_prompt(prompt, tokenizer)
            completion = raw["completion"] + tokenizer.eos_token
            prompt_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + completion_ids)[: config.max_length]
            labels = [-100] * len(prompt_ids) + completion_ids
            labels = labels[: config.max_length]
            rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run solver-label SFT for 24-game.")
    parser.add_argument("--config", required=True, help="Path to SFT YAML config.")
    args = parser.parse_args()

    config = _load_config(args.config)
    print(f"[sft] loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[sft] loading dataset: {config.train_path}")
    dataset = _load_sft_dataset(config, tokenizer)
    print(f"[sft] dataset ready: {len(dataset)} rows")

    model_kwargs = {}
    torch_dtype = _resolve_torch_dtype(config.torch_dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        **model_kwargs,
    )
    peft_config = _build_peft_config(config)  # type: ignore[arg-type]
    if peft_config is not None:
        try:
            from peft import get_peft_model
        except ImportError as exc:
            raise ImportError("LoRA SFT requires the optional 'peft' package.") from exc
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        report_to=config.report_to,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=SftDataCollator(tokenizer.pad_token_id),
    )
    print("[sft] starting training")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("[sft] training finished")


if __name__ == "__main__":
    main()
