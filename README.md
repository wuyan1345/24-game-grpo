# 24-Game GRPO

Minimal scaffold for:

- building processed datasets
- running baseline evaluation for `Qwen/Qwen2.5-1.5B-Instruct`
- training with `trl.GRPOTrainer`

Install dependencies first:

```bash
pip install -e .[dev]
```

## 1. Download model

If direct Hugging Face access is unavailable, use a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Download the model to a local directory:

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir models/qwen25-1.5b-instruct-hf
```

The evaluation examples below assume the model is in:

```text
models/qwen25-1.5b-instruct-hf
```

## 2. Build data

The processed datasets are normally committed with the repository, so in most cases you do not need to rebuild them.

Build processed data from the remote datasets:

```bash
game24-build-data \
  --output-dir data/processed
```

If you want to generate a local dataset for extension work:

```bash
game24-build-data \
  --generate-local \
  --output-dir data/processed
```

## 3. Evaluate

example evaluation on the eval set:

```bash
game24-eval \
  --model models/qwen25-1.5b-instruct-hf \
  --data-config configs/data.example.yaml \
  --split eval \
  --limit 20 \
  --max-new-tokens 192 \
  --model models/qwen25-1.5b-instruct-hf \
  --data-config configs/data.example.yaml \
  --split eval \
  --limit 20 \
  --max-new-tokens 192 \
  --output outputs/baseline_eval_limit20_gpu.json
```

## 4. Train

Run GRPO training:

```bash
game24-train \
  --config configs/grpo.example.yaml \
  --data-config configs/data.example.yaml
```