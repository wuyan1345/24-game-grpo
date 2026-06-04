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

If direct Hugging Face access is unavailable, you can use a mirror for model downloads:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then download the model snapshot locally:

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir models/qwen25-1.5b-instruct-hf
```

Build processed data from the remote datasets mentioned in `task.md`:

```bash
PYTHONPATH=src python -m game24_grpo.cli.build_data \
  --output-dir data/processed
```

This always produces:

- `train.jsonl`: all `solvable=True` records from `nlile/24-game`
- `eval.jsonl`: the 100 hard ToT records selected by `--hard-start-index/--hard-end-index`
- `unsolvable_eval.jsonl`: all `solvable=False` records from `nlile/24-game`

When available, the script also writes:

- `tot_nonoverlap.jsonl`: the remaining ToT records that do not overlap with the `nlile` solvable training set

The processed JSONL files are intentionally minimal and currently store only:

- `numbers`
- `target`
- `solvable`

If remote dataset access is unavailable, you can override the Hugging Face rows endpoint if your mirror provides one:

```bash
export HF_DATASETS_ROWS_URL=https://datasets-server.hf-mirror.com/rows
```

If you want to generate a local dataset for extension work, you can build the full 24-game space locally and split it automatically into train/eval:

```bash
PYTHONPATH=src python -m game24_grpo.cli.build_data \
  --generate-local \
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
