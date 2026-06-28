# 24-Game GRPO

This repository trains and evaluates `Qwen/Qwen2.5-1.5B-Instruct` for the 24-point
game with verifiable rewards and GRPO.

Expected response format:

```text
<think>...</think><answer>...</answer>
```

The `<answer>` content must be only one arithmetic expression using the four input
numbers exactly once, operators `+ - * /`, and parentheses. The verifier checks
syntax, number use, and whether the expression evaluates to 24 within `1e-6`.

## Setup

Install dependencies on the remote GPU server from inside this repository:

```bash
python -m pip install -e ".[dev]"
```

The current server notes use a P100 GPU. Evaluation defaults to FP32 because FP16
generation was observed to produce invalid repeated punctuation on this hardware.
The first GRPO pilot uses FP16 LoRA to fit memory, then evaluates the saved adapter
with FP32.

If direct Hugging Face access is unavailable:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Data

The processed files are:

- `data/processed/train.jsonl`: solvable training examples.
- `data/processed/eval.jsonl`: Tree-of-Thoughts hard holdout, indices 900-1000.
- `data/processed/unsolvable_eval.jsonl`: unsolvable examples or generated fallback.

Rebuild from Hugging Face datasets:

```bash
game24-build-data --output-dir data/processed
```

Build a generated local split for debugging the pipeline:

```bash
game24-build-data --generate-local --output-dir data/processed
```

## Evaluation

Run the untrained Qwen baseline once before any GRPO training:

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
  --difficulty-filter "nlile solvable=False or generated unsolvable fallback"
```

Result JSON records include experiment metadata, generation settings, aggregate
metrics, and per-example completions/verifier details. Default filenames use:

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
  --model outputs/grpo-qwen25-15b-pilot/checkpoint-10 \
  --data-config configs/data.example.yaml \
  --split eval \
  --dtype float32 \
  --temperature 0.7 \
  --top-p 0.95 \
  --num-samples 8 \
  --trained \
  --baseline-type grpo_lora_pilot \
  --experiment-id grpo_lora_pilot_best_of_8_hard
```

## Training

The base FP32 evaluation has been run and logged. Start with the short LoRA GRPO
pilot:

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 game24-train \
  --config configs/grpo.pilot.yaml \
  --data-config configs/data.example.yaml
```

Evaluate the pilot adapter checkpoint:

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 game24-eval \
  --model outputs/grpo-qwen25-15b-pilot/checkpoint-10 \
  --data-config configs/data.example.yaml \
  --split eval \
  --dtype float32 \
  --max-new-tokens 192 \
  --trained \
  --baseline-type grpo_lora_pilot \
  --experiment-id grpo_lora_pilot_hard_fp32
```

For the required unsolvable false-positive check:

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 game24-eval \
  --model outputs/grpo-qwen25-15b-pilot/checkpoint-10 \
  --data-config configs/data.example.yaml \
  --split unsolvable_eval \
  --dtype float32 \
  --max-new-tokens 192 \
  --trained \
  --baseline-type grpo_lora_pilot \
  --experiment-id grpo_lora_pilot_unsolvable_fp32
```

`configs/grpo.example.yaml` keeps full-model GRPO settings, while
`configs/grpo.pilot.yaml` is the memory-conscious P100 pilot. Keep checkpoints and
large run artifacts under `outputs/` or `runs/`, and commit only code, configs,
scripts, and lightweight metric summaries unless large artifacts are explicitly
requested.
