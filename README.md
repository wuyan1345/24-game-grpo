# 24-Game RLVR/GRPO

This repository runs one reproducible 24-game experiment chain for
`Qwen/Qwen2.5-1.5B-Instruct`.

The required response format is:

```text
<think>...</think><answer>...</answer>
```

For solvable puzzles, `<answer>` must contain exactly one arithmetic expression
using the four input numbers once, operators `+ - * /`, and parentheses. For
unsolvable puzzles, `<answer>` must be exactly `NO_SOLUTION`.

## Data

Build the fixed final splits on the remote server:

```bash
HF_ENDPOINT=https://hf-mirror.com game24-build-data --output-dir data/processed
```

The canonical files are:

- `id_train.jsonl`: deterministic `5/6` split of `nlile/24-game solvable=True`.
- `id_test.jsonl`: deterministic held-out `1/6` split of `nlile/24-game solvable=True`.
- `tot_non_easy.jsonl`: 100 non-easy `test-time-compute/game-of-24` rows, rank `>=101`, filtered against ID train and hard rows.
- `tot_hard.jsonl`: Tree-of-Thoughts hard rows, 0-based indices `[900, 1000)`.
- `unsolvable_eval.jsonl`: 50 `nlile/24-game solvable=False` rows, with exhaustive generated fallback if needed.

Compatibility aliases are also written:

- `train.jsonl` -> ID train.
- `eval.jsonl` -> ID test.
- `tot_nonoverlap.jsonl` -> non-easy split.

Build the two SFT label sets:

```bash
game24-build-sft --input data/processed/id_train.jsonl \
  --unsolvable-input data/processed/unsolvable_eval.jsonl \
  --unsolvable-limit 50 \
  --label-mode reference \
  --output data/processed/sft_fixed_train.jsonl

game24-build-sft --input data/processed/id_train.jsonl \
  --unsolvable-input data/processed/unsolvable_eval.jsonl \
  --unsolvable-limit 50 \
  --label-mode solver \
  --output data/processed/sft_solver_train.jsonl
```

## Training

Run only the canonical configs for the final chain:

```bash
game24-train --config configs/grpo.base_soft.yaml --data-config configs/data.example.yaml
game24-train --config configs/grpo.base_hard.yaml --data-config configs/data.example.yaml
game24-train-sft --config configs/sft.fixed.yaml
game24-train-sft --config configs/sft.solver.yaml
game24-train --config configs/grpo.solver_sft_soft.yaml --data-config configs/data.example.yaml
game24-train --config configs/grpo.solver_sft_hard.yaml --data-config configs/data.example.yaml
```

All GRPO configs use `max_steps: 100` and only two reward variants:

- `soft`: format, validity, number-use penalty, proximity, and exact correctness.
- `hard`: strict exact-verifier reward with minimal format support and invalid-answer penalties.

## Evaluation

Every reported model/checkpoint is evaluated on exactly four splits:

- `id_test`
- `tot_non_easy`
- `tot_hard`
- `unsolvable_eval`

For SFT and SFT+GRPO checkpoints, run best-of-1/4/8 verifier selection:

```bash
game24-eval --model outputs/final/solver-sft-lora \
  --data-config configs/data.example.yaml \
  --split tot_non_easy \
  --dtype float32 \
  --temperature 0.7 \
  --top-p 0.95 \
  --num-samples 8 \
  --trained \
  --baseline-type solver_sft \
  --experiment-id solver_sft_bestof8_tot_non_easy
```

Detailed JSON records are saved as:

```text
outputs/results/<YYYYMMDD_HHMMSS>_<experiment_id>.json
```

Each JSON includes generation config, split/filter notes, aggregate metrics, and
per-example verifier details. Update `/home/wuyan/study/nlp/final/logs.md` after
each training or evaluation batch.
