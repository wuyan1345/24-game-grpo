# Easy100 SFT4-Uns50 Chain Snapshot

Snapshot time: 2026-07-01 02:20 CST

Remote host/path: `nlp`, `/home/ma-user/work/24-game-grpo`

This snapshot records the current easy100 optimization chain after replacing
the solver SFT base with the selected `4epoch + uns50` LoRA checkpoint.

## Checkpoints

| Role | Remote path | Size | Notes |
|---|---|---:|---|
| Selected SFT base | `outputs/easy100/solver-sft-4epoch-uns50-r64-lr5e5-lora` | 4.5G | 4 epochs, `sft_solver_train.jsonl`, 1135 solvable + 50 generated unsolvable labels, LoRA r64 alpha128, lr `5e-5`. |
| Hard GRPO | `outputs/easy100/solver-sft4-uns50-r64-lr5e5-hard-grpo-100/checkpoint-100` | 861M | 100-step GRPO from selected SFT, hard reward, lr `3e-5`, max completion 64. Best current checkpoint in this chain. |
| Soft GRPO | `outputs/easy100/solver-sft4-uns50-r64-lr5e5-soft-grpo-100/checkpoint-100` | 861M | 100-step GRPO from selected SFT, soft reward, lr `3e-5`, max completion 64. Collapsed to `NO_SOLUTION`. |
| Low-lr hard GRPO | `outputs/easy100/solver-sft4-uns50-r64-lr5e5-hard-grpo-lr8e7-100/checkpoint-100` | 861M | 100-step GRPO from selected SFT, hard reward, lr `8e-7`, max completion 96. |

## Result Files

The JSON result records are stored both in `outputs/results/` and in this
snapshot directory.

| Model | Samples | Result JSON | Solved | Format | Expr valid | Number use | Value correct |
|---|---:|---|---:|---:|---:|---:|---:|
| SFT 4epoch uns50 r64 lr5e-5 | 1 | `20260630_230321_easy100_sft4_uns50_r64_lr5e5_bestof1.json` | 0.2000 | 0.9200 | 0.6600 | 0.6600 | 0.2000 |
| SFT 4epoch uns50 r64 lr5e-5 | 8 | `20260630_230924_easy100_sft4_uns50_r64_lr5e5_bestof8_t07.json` | 0.5000 | 0.8800 | 0.7100 | 0.7100 | 0.5000 |
| SFT4-uns50 hard GRPO 100 | 1 | `20260701_001327_easy100_sft4_uns50_hard_grpo100_bestof1.json` | 0.2700 | 1.0000 | 0.7700 | 0.7700 | 0.2700 |
| SFT4-uns50 hard GRPO 100 | 8 | `20260701_001858_easy100_sft4_uns50_hard_grpo100_bestof8_t07.json` | 0.5700 | 0.9900 | 0.7600 | 0.7600 | 0.5700 |
| SFT4-uns50 soft GRPO 100 | 1 | `20260701_002735_easy100_sft4_uns50_soft_grpo100_bestof1.json` | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| SFT4-uns50 soft GRPO 100 | 8 | `20260701_003339_easy100_sft4_uns50_soft_grpo100_bestof8_t07.json` | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| SFT4-uns50 hard GRPO lr8e-7 100 | 1 | `20260701_004140_easy100_sft4_uns50_hard_grpo_lr8e7_100_bestof1.json` | 0.1900 | 0.9200 | 0.6600 | 0.6600 | 0.1900 |
| SFT4-uns50 hard GRPO lr8e-7 100 | 8 | `20260701_004803_easy100_sft4_uns50_hard_grpo_lr8e7_100_bestof8_t07.json` | 0.5300 | 0.9300 | 0.7100 | 0.7100 | 0.5300 |

## Commands

Train the selected SFT base:

```bash
cd /home/ma-user/work/24-game-grpo
/usr/bin/time -p env HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 \
  python -m game24_grpo.cli.train_sft \
  --config configs/sft.solver_4epoch_uns50_r64_lr5e5.yaml
```

Train the GRPO checkpoints from the selected SFT base:

```bash
cd /home/ma-user/work/24-game-grpo

/usr/bin/time -p env HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 \
  python -m game24_grpo.cli.train_grpo \
  --config configs/grpo.solver_sft_hard.yaml \
  --data-config configs/data.example.yaml

/usr/bin/time -p env HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 \
  python -m game24_grpo.cli.train_grpo \
  --config configs/grpo.solver_sft_soft.yaml \
  --data-config configs/data.example.yaml

/usr/bin/time -p env HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 \
  python -m game24_grpo.cli.train_grpo \
  --config configs/grpo.solver_sft_2epoch_uns100_hard_100.yaml \
  --data-config configs/data.example.yaml
```

Evaluate greedy b1:

```bash
game24-eval \
  --model <checkpoint-or-adapter-dir> \
  --data-config configs/data.example.yaml \
  --split tot_easy100 \
  --dtype float32 \
  --max-new-tokens 192 \
  --experiment-id <experiment_id> \
  --baseline-type <baseline_type> \
  --trained \
  --checkpoint <checkpoint-or-adapter-dir> \
  --reward-variant <none|hard|soft> \
  --overlap-filter "ToT ranks 1-100 easy slice; auxiliary easy100" \
  --difficulty-filter "easy100; greedy"
```

Evaluate verifier best-of-8:

```bash
game24-eval \
  --model <checkpoint-or-adapter-dir> \
  --data-config configs/data.example.yaml \
  --split tot_easy100 \
  --dtype float32 \
  --max-new-tokens 192 \
  --temperature 0.7 \
  --top-p 0.95 \
  --num-samples 8 \
  --experiment-id <experiment_id> \
  --baseline-type <baseline_type> \
  --trained \
  --checkpoint <checkpoint-or-adapter-dir> \
  --reward-variant <none|hard|soft> \
  --overlap-filter "ToT ranks 1-100 easy slice; auxiliary easy100" \
  --difficulty-filter "easy100; best-of-8 t0.7 p0.95"
```

Do not run b12 for this chain unless the experiment plan changes explicitly.

## Analysis

The selected SFT base is the best current SFT-only checkpoint under the current
no-b12 policy: b1 solves 20/100 and b8 solves 50/100 on `tot_easy100`.

The standard hard GRPO run improves the selected SFT base on both decoding
settings: b1 improves from 0.20 to 0.27, and b8 improves from 0.50 to 0.57.
This is the best checkpoint in the current snapshot.

The soft GRPO run is invalid for this chain. Both b1 and b8 solve 0/100. The
per-example records show the model outputs `NO_SOLUTION` for all 100 solvable
easy examples. The format is valid, but `NO_SOLUTION` is not an arithmetic
expression, so expression validity, number use, and value correctness are all
zero.

The low-lr hard GRPO run is stable but weaker than standard hard GRPO. It
reaches b8 0.53, which is above the SFT b8 baseline but below standard hard
GRPO b8 0.57.

Current snapshot decision: use
`outputs/easy100/solver-sft4-uns50-r64-lr5e5-hard-grpo-100/checkpoint-100`
as the best easy100 checkpoint from this chain.
