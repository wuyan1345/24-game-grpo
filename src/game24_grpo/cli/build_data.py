from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

from game24_grpo.solver import solve_24

DEFAULT_NLILE_DATASET = "nlile/24-game"
DEFAULT_TOT_DATASET = "test-time-compute/game-of-24"
DEFAULT_DATASET_CONFIG = "default"
ROWS_API_URL = "https://datasets-server.huggingface.co/rows"
TOT_HARD_EVAL_PATH = Path(__file__).resolve().parent.parent / "assets" / "tot_hard_eval.csv"


@dataclass(frozen=True)
class PreparedRecord:
    numbers: list[int]
    target: int
    solvable: bool
    reference_solution: str | None
    all_reference_solutions: list[str]
    source: str
    puzzle_key: str
    metadata: dict[str, Any]


def parse_numbers(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str):
        return [int(item) for item in value.strip().split()]
    raise TypeError(f"unsupported numbers field type: {type(value).__name__}")


def canonical_key(numbers: Iterable[int]) -> str:
    return " ".join(str(value) for value in sorted(int(item) for item in numbers))


def parse_percentage(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        numeric = float(value)
        return numeric / 100.0 if numeric > 1.0 else numeric
    text = str(value).strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    numeric = float(text)
    return numeric / 100.0 if numeric > 1.0 else numeric


def normalize_solution_text(solution: str) -> str:
    return solution.replace("×", "*").replace("÷", "/")


def normalize_nlile_row(row: dict[str, Any]) -> PreparedRecord:
    numbers = parse_numbers(row["numbers"])
    solutions = [normalize_solution_text(item) for item in row.get("solutions", [])]
    return PreparedRecord(
        numbers=numbers,
        target=24,
        solvable=bool(row["solvable"]),
        reference_solution=solutions[0] if solutions else None,
        all_reference_solutions=solutions,
        source=DEFAULT_NLILE_DATASET,
        puzzle_key=canonical_key(numbers),
        metadata={
            "amt": row.get("amt"),
            "solved_rate": row.get("solved_rate"),
            "mean_time": row.get("mean_time"),
            "std_time": row.get("std_time"),
        },
    )


def normalize_tot_row(row: dict[str, Any]) -> PreparedRecord:
    numbers = parse_numbers(row["Puzzles"])
    rank = int(row["Rank"])
    solved_rate = parse_percentage(row.get("Solved rate"))
    return PreparedRecord(
        numbers=numbers,
        target=24,
        solvable=True,
        reference_solution=None,
        all_reference_solutions=[],
        source=DEFAULT_TOT_DATASET,
        puzzle_key=canonical_key(numbers),
        metadata={
            "rank": rank,
            "amt_seconds": row.get("AMT (s)"),
            "solved_rate": solved_rate,
            "one_sigma_mean_seconds": row.get("1-sigma Mean (s)"),
            "one_sigma_std_seconds": row.get("1-sigma STD (s)"),
        },
    )


def serialize_record(record: PreparedRecord) -> dict[str, Any]:
    return asdict(record)


def write_jsonl(path: Path, records: list[PreparedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(serialize_record(record), ensure_ascii=True) + "\n")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def fetch_dataset_rows(
    dataset: str,
    config: str,
    split: str,
    batch_size: int = 100,
) -> list[dict[str, Any]]:
    offset = 0
    all_rows: list[dict[str, Any]] = []
    while True:
        query = urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": batch_size,
            }
        )
        with urlopen(f"{ROWS_API_URL}?{query}") as response:
            payload = json.loads(response.read().decode("utf-8"))

        rows = [item["row"] for item in payload.get("rows", [])]
        all_rows.extend(rows)
        offset += len(rows)

        total = payload.get("num_rows_total")
        if not rows or (isinstance(total, int) and offset >= total):
            break
    return all_rows


def load_rows_from_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
        if rows and isinstance(rows[0], dict) and "row" in rows[0]:
            return [item["row"] for item in rows]
        return rows
    raise ValueError(f"unsupported row dump format: {path}")


def load_remote_datasets(
    nlile_dataset: str,
    tot_dataset: str,
    nlile_config: str,
    tot_config: str,
    nlile_split: str,
    tot_split: str,
) -> tuple[list[PreparedRecord], list[PreparedRecord]]:
    nlile_rows = fetch_dataset_rows(dataset=nlile_dataset, config=nlile_config, split=nlile_split)
    tot_rows = fetch_dataset_rows(dataset=tot_dataset, config=tot_config, split=tot_split)
    nlile_records = [normalize_nlile_row(row) for row in nlile_rows]
    tot_records = [normalize_tot_row(row) for row in tot_rows]
    tot_records.sort(key=lambda item: int(item.metadata["rank"]))
    return nlile_records, tot_records


def load_prepared_source_rows(
    nlile_rows_json: str | Path,
    tot_rows_json: str | Path,
) -> tuple[list[PreparedRecord], list[PreparedRecord]]:
    nlile_rows = load_rows_from_json(nlile_rows_json)
    tot_rows = load_rows_from_json(tot_rows_json)
    nlile_records = [normalize_nlile_row(row) for row in nlile_rows]
    tot_records = [normalize_tot_row(row) for row in tot_rows]
    tot_records.sort(key=lambda item: int(item.metadata["rank"]))
    return nlile_records, tot_records


def load_tot_hard_eval_rows(path: str | Path = TOT_HARD_EVAL_PATH) -> list[PreparedRecord]:
    records: list[PreparedRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(normalize_tot_row(dict(row)))
    return records


def generate_full_game24_records() -> list[PreparedRecord]:
    records: list[PreparedRecord] = []
    for numbers_tuple in combinations_with_replacement(range(1, 14), 4):
        numbers = list(numbers_tuple)
        solution = solve_24(numbers)
        records.append(
            PreparedRecord(
                numbers=numbers,
                target=24,
                solvable=solution is not None,
                reference_solution=solution,
                all_reference_solutions=[solution] if solution is not None else [],
                source="generated/full-24-combinations",
                puzzle_key=canonical_key(numbers),
                metadata={},
            )
        )
    return records


def select_holdout_records(
    tot_records: list[PreparedRecord],
    hard_start_index: int,
    hard_end_index: int,
) -> list[PreparedRecord]:
    if hard_start_index < 0 or hard_end_index <= hard_start_index:
        raise ValueError("hard holdout indices must satisfy 0 <= start < end")
    if hard_end_index > len(tot_records):
        raise ValueError("hard holdout end index exceeds ToT dataset size")
    return tot_records[hard_start_index:hard_end_index]


def prepare_splits(
    nlile_records: list[PreparedRecord],
    tot_records: list[PreparedRecord],
    hard_start_index: int,
    hard_end_index: int,
) -> dict[str, list[PreparedRecord]]:
    hard_eval_records = select_holdout_records(tot_records, hard_start_index, hard_end_index)
    hard_eval_keys = {record.puzzle_key for record in hard_eval_records}

    train_records = [
        record
        for record in nlile_records
        if record.solvable and record.puzzle_key not in hard_eval_keys
    ]
    unsolvable_records = [record for record in nlile_records if not record.solvable]
    tot_nonoverlap_records = [record for record in tot_records if record.puzzle_key in hard_eval_keys]

    return {
        "train": train_records,
        "eval": hard_eval_records,
        "unsolvable_eval": unsolvable_records,
        "tot_full": tot_records,
        "tot_nonoverlap": tot_nonoverlap_records,
    }


def summarize_splits(
    splits: dict[str, list[PreparedRecord]],
    hard_start_index: int,
    hard_end_index: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "hard_holdout": {
            "start_index": hard_start_index,
            "end_index_exclusive": hard_end_index,
            "size": len(splits["eval"]),
        },
        "counts": {name: len(records) for name, records in splits.items()},
    }
    if splits["eval"]:
        solved_rates = [
            record.metadata.get("solved_rate")
            for record in splits["eval"]
            if isinstance(record.metadata.get("solved_rate"), float)
        ]
        ranks = [
            int(record.metadata["rank"])
            for record in splits["eval"]
            if record.metadata.get("rank") is not None
        ]
        summary["eval_stats"] = {
            "min_rank": min(ranks) if ranks else None,
            "max_rank": max(ranks) if ranks else None,
            "min_solved_rate": min(solved_rates) if solved_rates else None,
            "max_solved_rate": max(solved_rates) if solved_rates else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 24-game datasets into JSONL files.")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--nlile-dataset", default=DEFAULT_NLILE_DATASET)
    parser.add_argument("--tot-dataset", default=DEFAULT_TOT_DATASET)
    parser.add_argument("--nlile-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--tot-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--nlile-rows-json", help="Optional local JSON dump of nlile rows.")
    parser.add_argument("--tot-rows-json", help="Optional local JSON dump of ToT rows.")
    parser.add_argument("--nlile-split", default="train")
    parser.add_argument("--tot-split", default="train")
    parser.add_argument(
        "--use-local-hard-eval",
        action="store_true",
        help="Use the bundled ToT hard-eval CSV and generate the full 24-game space algorithmically.",
    )
    parser.add_argument(
        "--hard-start-index",
        type=int,
        default=900,
        help="0-based inclusive start index for the Tree-of-Thoughts hard holdout.",
    )
    parser.add_argument(
        "--hard-end-index",
        type=int,
        default=1000,
        help="0-based exclusive end index for the Tree-of-Thoughts hard holdout.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if bool(args.nlile_rows_json) != bool(args.tot_rows_json):
        raise ValueError("provide both --nlile-rows-json and --tot-rows-json together")

    hard_start_index = args.hard_start_index
    hard_end_index = args.hard_end_index

    if args.use_local_hard_eval:
        nlile_records = generate_full_game24_records()
        tot_records = load_tot_hard_eval_rows()
        hard_start_index = 0
        hard_end_index = len(tot_records)
    elif args.nlile_rows_json and args.tot_rows_json:
        nlile_records, tot_records = load_prepared_source_rows(
            nlile_rows_json=args.nlile_rows_json,
            tot_rows_json=args.tot_rows_json,
        )
    else:
        nlile_records, tot_records = load_remote_datasets(
            nlile_dataset=args.nlile_dataset,
            tot_dataset=args.tot_dataset,
            nlile_config=args.nlile_config,
            tot_config=args.tot_config,
            nlile_split=args.nlile_split,
            tot_split=args.tot_split,
        )
    splits = prepare_splits(
        nlile_records=nlile_records,
        tot_records=tot_records,
        hard_start_index=hard_start_index,
        hard_end_index=hard_end_index,
    )

    write_jsonl(output_dir / "train.jsonl", splits["train"])
    write_jsonl(output_dir / "eval.jsonl", splits["eval"])
    write_jsonl(output_dir / "unsolvable_eval.jsonl", splits["unsolvable_eval"])
    if not args.use_local_hard_eval:
        write_jsonl(output_dir / "tot_full.jsonl", splits["tot_full"])
    write_jsonl(output_dir / "tot_nonoverlap.jsonl", splits["tot_nonoverlap"])
    write_summary(
        output_dir / "summary.json",
        summarize_splits(splits, hard_start_index, hard_end_index),
    )


if __name__ == "__main__":
    main()
