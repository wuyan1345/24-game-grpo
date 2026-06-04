from __future__ import annotations

import argparse

from game24_grpo.config import load_data_config, load_train_config
from game24_grpo.training import build_trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 24-game policy with TRL GRPOTrainer.")
    parser.add_argument("--config", required=True, help="Path to the GRPO training config YAML.")
    parser.add_argument("--data-config", required=True, help="Path to the data config YAML.")
    args = parser.parse_args()
    print(f"[train] loading train config: {args.config}")
    print(f"[train] loading data config: {args.data_config}")

    train_config = load_train_config(args.config)
    data_config = load_data_config(args.data_config)
    print(
        f"[train] building trainer: model={train_config.model_name}, "
        f"train_path={data_config.train_path}, output_dir={train_config.output_dir}"
    )
    trainer = build_trainer(train_config=train_config, data_config=data_config)
    print("[train] trainer ready, starting training loop")
    trainer.train()
    print("[train] training finished")


if __name__ == "__main__":
    main()
