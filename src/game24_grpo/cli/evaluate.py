from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from game24_grpo.config import load_data_config
from game24_grpo.evaluation import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline evaluation for a 24-game model.")
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument("--data-config", required=True, help="Path to the data config YAML.")
    parser.add_argument(
        "--split",
        choices=[
            "train",
            "eval",
            "id_train",
            "id_test",
            "tot_easy100",
            "tot_non_easy",
            "tot_hard",
            "unsolvable_eval",
            "tot_nonoverlap",
        ],
        default="id_test",
    )
    parser.add_argument("--limit", type=int, help="Optional number of examples to evaluate.")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Optional starting row offset within the selected split.",
    )
    parser.add_argument("--output", help="Optional path to save detailed JSON results.")
    parser.add_argument(
        "--output-dir",
        default="outputs/results",
        help="Directory for default result JSON files.",
    )
    parser.add_argument(
        "--experiment-id",
        default="eval",
        help="Stable identifier used in result metadata and filenames.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Optional override for generation length.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
    )
    parser.add_argument("--temperature", type=float, help="Enable sampling with this temperature.")
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p value used only when sampling is enabled.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Samples per problem; best correct sample is selected.",
    )
    parser.add_argument("--baseline-type", default="untrained_qwen")
    parser.add_argument(
        "--trained",
        action="store_true",
        help="Mark the evaluated model as trained in result metadata.",
    )
    parser.add_argument("--checkpoint", help="Checkpoint path or identifier for result metadata.")
    parser.add_argument("--reward-variant", default="none")
    parser.add_argument("--overlap-filter", default="")
    parser.add_argument("--difficulty-filter", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    print(f"[evaluate] loading data config: {args.data_config}")

    data_config = load_data_config(args.data_config)
    dataset_paths = {
        "train": data_config.train_path,
        "eval": data_config.eval_path,
        "id_train": data_config.id_train_path or data_config.train_path,
        "id_test": data_config.id_test_path or data_config.eval_path,
        "tot_easy100": data_config.tot_easy100_path,
        "tot_non_easy": data_config.tot_non_easy_path or data_config.tot_nonoverlap_path,
        "tot_hard": data_config.tot_hard_path,
        "unsolvable_eval": data_config.unsolvable_eval_path,
        "tot_nonoverlap": data_config.tot_nonoverlap_path,
    }
    dataset_path = dataset_paths[args.split]
    if dataset_path is None:
        raise ValueError(f"split {args.split} is not configured")
    print(
        f"[evaluate] starting evaluation: split={args.split}, model={args.model}, "
        f"dataset={dataset_path}, offset={args.offset}, limit={args.limit or 'all'}"
    )
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(args.output_dir) / f"{timestamp}_{args.experiment_id}.json")

    metrics = evaluate_model(
        model_name=args.model,
        dataset_path=dataset_path,
        prompt_template=data_config.prompt_template,
        split=args.split,
        max_new_tokens=args.max_new_tokens or data_config.max_completion_length,
        limit=args.limit,
        offset=args.offset,
        output_path=output_path,
        experiment_id=args.experiment_id,
        dtype=args.dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        baseline_type=args.baseline_type,
        trained=args.trained,
        checkpoint=args.checkpoint,
        reward_variant=args.reward_variant,
        overlap_filter=args.overlap_filter,
        difficulty_filter=args.difficulty_filter,
        notes=args.notes,
    )
    print(f"[evaluate] wrote detailed output: {output_path}")
    print(f"total={metrics.total}")
    print(f"solve_rate={metrics.solve_rate:.4f}")
    print(f"format_pass_rate={metrics.format_pass_rate:.4f}")
    print(f"valid_expression_rate={metrics.valid_expression_rate:.4f}")
    print(f"number_use_valid_rate={metrics.number_use_valid_rate:.4f}")
    print(f"value_correct_rate={metrics.value_correct_rate:.4f}")
    print(f"unsolvable_false_positive_rate={metrics.unsolvable_false_positive_rate:.4f}")


if __name__ == "__main__":
    main()
