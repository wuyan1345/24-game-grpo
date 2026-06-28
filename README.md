# 24-Game GRPO

This repository trains and evaluates `Qwen/Qwen2.5-1.5B-Instruct` for the 24-point game with verifiable rewards and GRPO.

Expected response format:

```text
<think>...</think><answer>...</answer>
```

The `<answer>` content must be only one arithmetic expression using the four input numbers exactly once, operators `+ - * /`, and parentheses. The verifier checks syntax, number use, and whether the expression evaluates to 24 within `1e-6`.

## Setup

Install dependencies on the remote GPU server from inside this repository:

```bash
python -m pip install -e ".[dev]"
```

The current server notes use a P100 GPU, so examples default to FP32. Avoid BF16/FP16 on that server unless the hardware changes and a smoke evaluation proves generation is normal.

If direct Hugging Face access is unavailable:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Data

The committed processed files are:

- `data/processed/train.jsonl`: solvable training examples.
- `data/processed/eval.jsonl`: Tree-of-Thoughts hard holdout, indices 900-1000 from `test-time-compute/game-of-24`.
- `data/processed/unsolvable_eval.jsonl`: unsolvable examples when available.

Rebuild from Hugging Face datasets:

```bash
game24-build-data --output-dir data/processed
```

Build a generated local split for debugging the pipeline:

```bash
game24-build-data --generate-local --output-dir data/processed
```

## Evaluation

Run the untrained Qwen baseline once before any GRPO training. Save detailed JSON results under `outputs/results/`:

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 game24-eval \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data-config configs/data.example.yaml \
  --split eval \
  --dtype float32 \
  --max-new-tokens 192 \
  --experiment-id base_eval_full_qwen25_15b_fp32 \
  --overlap-filter "Tree-of-Thoughts hard holdout excludes nlile solvable train keys" \
  --difficulty-filter "indices 900-1000 hard"
```

For an unsolvable false-positive check:

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 game24-eval \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data-config configs/data.example.yaml \
  --split unsolvable_eval \
  --dtype float32 \
  --max-new-tokens 192 \
  --experiment-id base_unsolvable_qwen25_15b_fp32 \
  --difficulty-filter "nlile solvable=False"
```

Result JSON records include experiment metadata, generation settings, aggregate metrics, and per-example completions/verifier details. Default filenames use:

```text
outputs/results/<YYYYMMDD_HHMMSS>_<experiment_id>.json
```

Important metrics:

- `solved_rate`
- `format_valid_rate`
- `expression_valid_rate`
- `number_use_valid_rate`
- `value_correct_rate`
- `unsolvable_false_positive_rate`

Sampling or best-of-N evaluation is available for test-time compute:

```bash
game24-eval \
  --model outputs/grpo-qwen25-15b/checkpoint-final \
  --data-config configs/data.example.yaml \
  --split eval \
  --dtype float32 \
  --temperature 0.7 \
  --top-p 0.95 \
  --num-samples 8 \
  --experiment-id grpo_best_of_8_hard
```

## Training

Do not start GRPO until the base FP32 evaluation has been run, logged, and explicitly approved for training.

After approval, run a short GRPO pilot:

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 game24-train \
  --config configs/grpo.example.yaml \
  --data-config configs/data.example.yaml
```

`configs/grpo.example.yaml` sets `bf16: false` for the current P100 server. Keep checkpoints and large run artifacts under `outputs/` or `runs/`, and commit only code, configs, scripts, and lightweight metric summaries unless large artifacts are explicitly requested.
