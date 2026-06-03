#!/usr/bin/env bash
set -euo pipefail

python -m game24_grpo.cli.train_grpo \
  --config configs/grpo.example.yaml \
  --data-config configs/data.example.yaml
