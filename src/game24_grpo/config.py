from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROMPT_TEMPLATE = (
    "You are solving the 24-game.\n"
    "Use the numbers {numbers} exactly once each.\n"
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
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    max_prompt_length: int = 256
    max_completion_length: int = 192


@dataclass
class RewardWeights:
    format: float = 0.1
    valid_expression: float = 0.2
    correct: float = 1.0


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    bf16: bool = True
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
    max_prompt_length: int = 256
    max_completion_length: int = 192
    beta: float = 0.04
    temperature: float = 0.8
    top_p: float = 0.95
    report_to: list[str] = field(default_factory=lambda: ["none"])
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_data_config(path: str | Path) -> DataConfig:
    payload = _load_yaml(path)
    return DataConfig(**payload)


def load_train_config(path: str | Path) -> TrainConfig:
    payload = _load_yaml(path)
    reward_weights = payload.pop("reward_weights", {})
    return TrainConfig(reward_weights=RewardWeights(**reward_weights), **payload)
