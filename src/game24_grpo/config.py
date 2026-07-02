from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROMPT_TEMPLATE = (
    "You are solving an arithmetic puzzle.\n"
    "Use the numbers {numbers} exactly once each.\n"
    "Your goal is to make {target}.\n"
    "You may only use +, -, *, /, and parentheses.\n"
    "First reason in <think></think>, then provide the final expression in <answer></answer>.\n"
    "Keep the <think> section brief.\n"
    "In <answer>, output exactly one expression and nothing else.\n"
    "Do not include words, explanations, or an equals sign."
)


@dataclass
class DataConfig:
    train_path: str
    eval_path: str
    unsolvable_eval_path: str | None = None
    tot_nonoverlap_path: str | None = None
    tot_easy100_path: str | None = None
    id_train_path: str | None = None
    id_test_path: str | None = None
    tot_non_easy_path: str | None = None
    tot_hard_path: str | None = None
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    max_prompt_length: int = 256
    max_completion_length: int = 192


@dataclass
class RewardWeights:
    format: float = 0.05
    valid_expression: float = 0.15
    proximity: float = 0.15
    correct: float = 2.0
    number_mismatch_penalty: float = -1.0
    missing_answer_penalty: float = 0.0


REWARD_VARIANTS: dict[str, RewardWeights] = {
    "soft": RewardWeights(
        format=0.50,
        valid_expression=0.10,
        proximity=0.15,
        correct=5.0,
        number_mismatch_penalty=-0.5,
        missing_answer_penalty=-0.5,
    ),
    "hard": RewardWeights(
        format=0.05,
        valid_expression=0.0,
        proximity=0.0,
        correct=5.0,
        number_mismatch_penalty=-0.5,
        missing_answer_penalty=-0.5,
    ),
}


@dataclass
class LoraConfig:
    enabled: bool = False
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
class TrainConfig:
    model_name: str
    output_dir: str
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    max_steps: int | None = None
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 100
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    generation_batch_size: int | None = None
    max_prompt_length: int = 256
    max_completion_length: int = 192
    beta: float = 0.04
    temperature: float = 0.8
    top_p: float = 0.95
    optim: str = "adamw_torch"
    torch_dtype: str | None = None
    train_limit: int | None = None
    report_to: list[str] = field(default_factory=lambda: ["none"])
    reward_variant: str | None = None
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    lora: LoraConfig = field(default_factory=LoraConfig)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_data_config(path: str | Path) -> DataConfig:
    payload = _load_yaml(path)
    return DataConfig(**payload)


def load_train_config(path: str | Path) -> TrainConfig:
    payload = _load_yaml(path)
    reward_variant = payload.get("reward_variant")
    reward_weights = payload.pop("reward_weights", {})
    if reward_variant is not None:
        if reward_weights:
            raise ValueError("set either reward_variant or reward_weights, not both")
        if reward_variant not in REWARD_VARIANTS:
            valid = ", ".join(sorted(REWARD_VARIANTS))
            raise ValueError(f"unsupported reward_variant={reward_variant!r}; choose one of {valid}")
        reward_weights = REWARD_VARIANTS[reward_variant].__dict__
    lora = payload.pop("lora", {})
    return TrainConfig(
        reward_weights=RewardWeights(**reward_weights),
        lora=LoraConfig(**lora),
        **payload,
    )
