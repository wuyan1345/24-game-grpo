# 24-Game GRPO

Minimal scaffold for:

- preparing 24-game train and eval splits
- running baseline evaluation for `Qwen/Qwen2.5-1.5B-Instruct`
- training with `trl.GRPOTrainer`

Target output format:

```text
<think>...</think>
<answer>(11 - 5) * (8 / 2)</answer>
```

## Use

```bash
pip install -e .[dev]
```

Build or rebuild processed data:

```bash
PYTHONPATH=src python -m game24_grpo.cli.build_data \
  --use-local-hard-eval \
  --output-dir data/processed
```

Run unit tests:

```bash
PYTHONPATH=src pytest
```

Run a baseline smoke test:

```bash
PYTHONPATH=src python -m game24_grpo.cli.evaluate \
  --model models/qwen25-1.5b-instruct-hf \
  --data-config configs/data.example.yaml \
  --split eval \
  --limit 20 \
  --max-new-tokens 192 \
  --output outputs/baseline_eval_limit20_gpu.json
```

Run GRPO training:

```bash
PYTHONPATH=src python -m game24_grpo.cli.train_grpo \
  --config configs/grpo.example.yaml \
  --data-config configs/data.example.yaml
```

Main files:

- `configs/data.example.yaml`: dataset paths and the prompt template used for both training and evaluation
- `configs/grpo.example.yaml`: baseline GRPO hyperparameters
- `src/game24_grpo/cli/build_data.py`: data preparation entrypoint
- `src/game24_grpo/cli/evaluate.py`: baseline evaluation entrypoint
- `src/game24_grpo/cli/train_grpo.py`: GRPO training entrypoint

## Next

1. rerun the full `eval` and `unsolvable_eval` baselines with the cleaned prompt
2. inspect the reward curves during the first GRPO run
3. tune prompt length and generation length if truncation reappears
