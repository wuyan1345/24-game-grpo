from __future__ import annotations

import argparse

from game24_grpo.config import load_data_config
from game24_grpo.evaluation import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline evaluation for a 24-game model.")
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument("--data-config", required=True, help="Path to the data config YAML.")
    parser.add_argument("--split", choices=["eval", "unsolvable_eval"], default="eval")
    parser.add_argument("--limit", type=int, help="Optional number of examples to evaluate.")
    parser.add_argument("--output", help="Optional path to save detailed JSON results.")
    parser.add_argument("--max-new-tokens", type=int, help="Optional override for generation length.")
    args = parser.parse_args()
    print(f"[evaluate] loading data config: {args.data_config}")

    data_config = load_data_config(args.data_config)
    dataset_path = data_config.eval_path if args.split == "eval" else data_config.unsolvable_eval_path
    if dataset_path is None:
        raise ValueError(f"split {args.split} is not configured")
    print(
        f"[evaluate] starting evaluation: split={args.split}, model={args.model}, "
        f"dataset={dataset_path}, limit={args.limit or 'all'}"
    )

    metrics = evaluate_model(
        model_name=args.model,
        dataset_path=dataset_path,
        prompt_template=data_config.prompt_template,
        max_new_tokens=args.max_new_tokens or data_config.max_completion_length,
        limit=args.limit,
        output_path=args.output,
    )
    if args.output:
        print(f"[evaluate] wrote detailed output: {args.output}")
    print(f"total={metrics.total}")
    print(f"solve_rate={metrics.solve_rate:.4f}")
    print(f"format_pass_rate={metrics.format_pass_rate:.4f}")
    print(f"valid_expression_rate={metrics.valid_expression_rate:.4f}")
    print(f"hallucination_rate={metrics.hallucination_rate:.4f}")


if __name__ == "__main__":
    main()
