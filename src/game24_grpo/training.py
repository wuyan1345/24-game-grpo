from __future__ import annotations

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

from game24_grpo.config import DataConfig, TrainConfig
from game24_grpo.data import load_jsonl_dataset
from game24_grpo.rewards import Game24Reward, RewardConfig


def build_trainer(train_config: TrainConfig, data_config: DataConfig) -> GRPOTrainer:
    print(f"[train] loading train dataset: {data_config.train_path}")
    dataset = load_jsonl_dataset(data_config.train_path, data_config.prompt_template)
    print(f"[train] train dataset ready: {len(dataset)} rows")
    print(f"[train] loading tokenizer: {train_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    reward = Game24Reward(
        RewardConfig(
            format_weight=train_config.reward_weights.format,
            valid_expression_weight=train_config.reward_weights.valid_expression,
            correct_weight=train_config.reward_weights.correct,
        )
    )
    print("[train] building GRPO config")
    trainer_kwargs = {
        "output_dir": train_config.output_dir,
        "bf16": train_config.bf16,
        "max_steps": train_config.max_steps if train_config.max_steps is not None else -1,
        "learning_rate": train_config.learning_rate,
        "weight_decay": train_config.weight_decay,
        "warmup_ratio": train_config.warmup_ratio,
        "lr_scheduler_type": train_config.lr_scheduler_type,
        "logging_steps": train_config.logging_steps,
        "save_steps": train_config.save_steps,
        "eval_steps": train_config.eval_steps,
        "num_train_epochs": train_config.num_train_epochs,
        "per_device_train_batch_size": train_config.per_device_train_batch_size,
        "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
        "num_generations": train_config.num_generations,
        "generation_batch_size": train_config.generation_batch_size,
        "max_prompt_length": train_config.max_prompt_length,
        "max_completion_length": train_config.max_completion_length,
        "beta": train_config.beta,
        "temperature": train_config.temperature,
        "top_p": train_config.top_p,
        "report_to": train_config.report_to,
    }
    supported_fields = GRPOConfig.__dataclass_fields__.keys()
    trainer_config = GRPOConfig(**{key: value for key, value in trainer_kwargs.items() if key in supported_fields})
    print("[train] instantiating GRPO trainer")
    return GRPOTrainer(
        model=train_config.model_name,
        reward_funcs=reward,
        args=trainer_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
