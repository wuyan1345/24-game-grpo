from __future__ import annotations

import argparse
import json
from pathlib import Path

from game24_grpo.solver import solve_24
from game24_grpo.verifier import verify_completion


def _format_numbers(numbers: list[int]) -> str:
    return ", ".join(str(number) for number in numbers)


def build_solvable_completion(expression: str, numbers: list[int], target: int) -> str:
    formatted_numbers = _format_numbers(numbers)
    return (
        f"<think>Use the expression {expression}. It uses {formatted_numbers} "
        f"exactly once and evaluates to {target}.</think>"
        f"<answer>{expression}</answer>"
    )


def build_unsolvable_completion(numbers: list[int], target: int) -> str:
    formatted_numbers = _format_numbers(numbers)
    return (
        f"<think>No valid expression exists using {formatted_numbers} exactly once "
        f"to reach {target}.</think>"
        "<answer>NO_SOLUTION</answer>"
    )


def _normalize_reference_solution(row: dict) -> str | None:
    solution = row.get("reference_solution")
    if solution:
        return str(solution)
    solutions = row.get("all_reference_solutions") or []
    if solutions:
        return str(solutions[0])
    return None


def _select_expression(row: dict, label_mode: str) -> str | None:
    if label_mode == "solver":
        return solve_24([int(item) for item in row["numbers"]])
    if label_mode == "reference":
        return _normalize_reference_solution(row)
    raise ValueError(f"unsupported label_mode: {label_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build solver-labeled SFT data for 24-game.")
    parser.add_argument("--input", required=True, help="Input JSONL with numbers/target rows.")
    parser.add_argument("--output", required=True, help="Output solver-labeled JSONL path.")
    parser.add_argument("--limit", type=int, help="Optional number of rows to label.")
    parser.add_argument(
        "--label-mode",
        choices=["solver", "reference"],
        default="solver",
        help="Use exhaustive solver labels or trusted source reference labels for solvable rows.",
    )
    parser.add_argument(
        "--unsolvable-input",
        help="Optional JSONL of unsolvable rows to append with NO_SOLUTION labels.",
    )
    parser.add_argument(
        "--unsolvable-limit",
        type=int,
        help="Optional number of unsolvable rows to append.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w",
        encoding="utf-8",
    ) as sink:
        for line_index, line in enumerate(source):
            if args.limit is not None and line_index >= args.limit:
                break
            row = json.loads(line)
            numbers = [int(item) for item in row["numbers"]]
            target = int(row.get("target", 24))
            expression = _select_expression(row, args.label_mode)
            if expression is None:
                skipped += 1
                continue
            completion = build_solvable_completion(expression, numbers, target)
            verification = verify_completion(completion, numbers)
            if not verification.is_correct:
                skipped += 1
                continue
            sink.write(
                json.dumps(
                    {
                        "numbers": numbers,
                        "target": target,
                        "solution": expression,
                        "completion": completion,
                        "label_mode": args.label_mode,
                        "source": row.get("source", ""),
                        "puzzle_key": row.get("puzzle_key", ""),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            written += 1

        if args.unsolvable_input is not None:
            unsolvable_path = Path(args.unsolvable_input)
            with unsolvable_path.open("r", encoding="utf-8") as unsolvable_source:
                for line_index, line in enumerate(unsolvable_source):
                    if args.unsolvable_limit is not None and line_index >= args.unsolvable_limit:
                        break
                    row = json.loads(line)
                    numbers = [int(item) for item in row["numbers"]]
                    target = int(row.get("target", 24))
                    completion = build_unsolvable_completion(numbers, target)
                    sink.write(
                        json.dumps(
                            {
                                "numbers": numbers,
                                "target": target,
                                "solution": None,
                                "completion": completion,
                                "solvable": False,
                                "label_mode": args.label_mode,
                                "source": row.get("source", ""),
                                "puzzle_key": row.get("puzzle_key", ""),
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    written += 1

    print(f"[build-sft] wrote {written} rows to {output_path}; skipped={skipped}")


if __name__ == "__main__":
    main()
