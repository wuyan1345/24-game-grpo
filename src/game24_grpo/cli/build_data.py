from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
ROWS_API_URL = os.environ.get("HF_DATASETS_ROWS_URL", "https://datasets-server.huggingface.co/rows")


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


def _log(message: str) -> None:
    print(f"[build-data] {message}")


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


def normalize_jsonl_row(row: dict[str, Any], source: str) -> PreparedRecord:
    numbers = parse_numbers(row["numbers"])
    solution = row.get("reference_solution")
    solutions = row.get("all_reference_solutions") or ([solution] if solution else [])
    return PreparedRecord(
        numbers=numbers,
        target=int(row.get("target", 24)),
        solvable=bool(row.get("solvable", True)),
        reference_solution=solution,
        all_reference_solutions=[str(item) for item in solutions if item],
        source=row.get("source") or source,
        puzzle_key=row.get("puzzle_key") or canonical_key(numbers),
        metadata=row.get("metadata", {}),
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
    return {
        "numbers": record.numbers,
        "target": record.target,
        "solvable": record.solvable,
        "source": record.source,
        "puzzle_key": record.puzzle_key,
        "reference_solution": record.reference_solution,
        "all_reference_solutions": record.all_reference_solutions,
        "metadata": record.metadata,
    }


def write_jsonl(path: Path, records: list[PreparedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(serialize_record(record), ensure_ascii=True) + "\n")


def fetch_dataset_rows(
    dataset: str,
    config: str,
    split: str,
    batch_size: int = 100,
) -> list[dict[str, Any]]:
    _log(f"fetching rows: dataset={dataset}, config={config}, split={split}")
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
        _log(f"fetched {offset} rows from {dataset}")

        total = payload.get("num_rows_total")
        if not rows or (isinstance(total, int) and offset >= total):
            break
    return all_rows


def load_remote_datasets(
    nlile_dataset: str,
    tot_dataset: str,
    nlile_config: str,
    tot_config: str,
    nlile_split: str,
    tot_split: str,
) -> tuple[list[PreparedRecord], list[PreparedRecord]]:
    _log("loading remote datasets from Hugging Face rows API")
    nlile_rows = fetch_dataset_rows(dataset=nlile_dataset, config=nlile_config, split=nlile_split)
    tot_rows = fetch_dataset_rows(dataset=tot_dataset, config=tot_config, split=tot_split)
    nlile_records = [normalize_nlile_row(row) for row in nlile_rows]
    tot_records = [normalize_tot_row(row) for row in tot_rows]
    tot_records.sort(key=lambda item: int(item.metadata["rank"]))
    return nlile_records, tot_records


def load_jsonl_records(path: str | Path, source: str) -> list[PreparedRecord]:
    records: list[PreparedRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(normalize_jsonl_row(json.loads(line), source=source))
    return records


def load_tot_csv_records(path: str | Path) -> list[PreparedRecord]:
    records: list[PreparedRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(normalize_tot_row(row))
    records.sort(key=lambda item: int(item.metadata["rank"]))
    return records


def generate_full_game24_records() -> list[PreparedRecord]:
    _log("generating full 24-game space locally")
    records: list[PreparedRecord] = []
    for index, numbers_tuple in enumerate(combinations_with_replacement(range(1, 14), 4), start=1):
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
        if index % 250 == 0:
            _log(f"generated {index} combinations")
    return records


def split_generated_records(
    records: list[PreparedRecord],
    eval_size: int,
    seed: int,
) -> dict[str, list[PreparedRecord]]:
    solvable_records = [record for record in records if record.solvable]
    unsolvable_records = [record for record in records if not record.solvable]
    shuffled = list(solvable_records)
    random.Random(seed).shuffle(shuffled)
    actual_eval_size = min(eval_size, len(shuffled))
    eval_records = shuffled[:actual_eval_size]
    train_records = shuffled[actual_eval_size:]
    return {
        "id_train": train_records,
        "id_test": eval_records,
        "unsolvable_eval": unsolvable_records,
        "tot_non_easy": [],
        "tot_hard": [],
    }


def generate_unsolvable_records(limit: int) -> list[PreparedRecord]:
    if limit < 0:
        raise ValueError("unsolvable fallback limit must be non-negative")
    records: list[PreparedRecord] = []
    for numbers_tuple in combinations_with_replacement(range(1, 14), 4):
        numbers = list(numbers_tuple)
        if solve_24(numbers) is not None:
            continue
        records.append(
            PreparedRecord(
                numbers=numbers,
                target=24,
                solvable=False,
                reference_solution=None,
                all_reference_solutions=[],
                source="generated/unsolvable-24-combinations",
                puzzle_key=canonical_key(numbers),
                metadata={"fallback": True},
            )
        )
        if len(records) >= limit:
            break
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
    min_unsolvable_eval_size: int = 50,
    id_test_fraction: float = 1.0 / 6.0,
    non_easy_start_rank: int = 101,
    non_easy_size: int = 100,
) -> dict[str, list[PreparedRecord]]:
    tot_hard_records = select_holdout_records(tot_records, hard_start_index, hard_end_index)
    hard_eval_keys = {record.puzzle_key for record in tot_hard_records}

    solvable_records = sorted(
        (record for record in nlile_records if record.solvable),
        key=lambda record: (record.puzzle_key, record.numbers),
    )
    id_test_size = max(1, round(len(solvable_records) * id_test_fraction))
    id_test_keys = {
        record.puzzle_key for index, record in enumerate(solvable_records) if index % 6 == 5
    }
    if len(id_test_keys) < id_test_size:
        for record in solvable_records:
            id_test_keys.add(record.puzzle_key)
            if len(id_test_keys) >= id_test_size:
                break
    id_train_records = [record for record in solvable_records if record.puzzle_key not in id_test_keys]
    id_test_records = [record for record in solvable_records if record.puzzle_key in id_test_keys]
    id_train_keys = {record.puzzle_key for record in id_train_records}

    unsolvable_records = [record for record in nlile_records if not record.solvable]
    if len(unsolvable_records) < min_unsolvable_eval_size:
        existing_keys = {record.puzzle_key for record in unsolvable_records}
        _log(
            f"unsolvable_eval has {len(unsolvable_records)} rows; "
            f"generating fallback rows to reach {min_unsolvable_eval_size}"
        )
        fallback_records = [
            record
            for record in generate_unsolvable_records(min_unsolvable_eval_size + len(existing_keys))
            if record.puzzle_key not in existing_keys
        ]
        needed_fallback_rows = min_unsolvable_eval_size - len(unsolvable_records)
        unsolvable_records.extend(fallback_records[:needed_fallback_rows])
    unsolvable_records = unsolvable_records[:min_unsolvable_eval_size]

    tot_non_easy_records: list[PreparedRecord] = []
    for record in tot_records:
        rank = int(record.metadata["rank"])
        if rank < non_easy_start_rank:
            continue
        if record.puzzle_key in hard_eval_keys or record.puzzle_key in id_train_keys:
            continue
        tot_non_easy_records.append(record)
        if len(tot_non_easy_records) >= non_easy_size:
            break

    return {
        "id_train": id_train_records,
        "id_test": id_test_records,
        "tot_non_easy": tot_non_easy_records,
        "tot_hard": tot_hard_records,
        "unsolvable_eval": unsolvable_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 24-game datasets into JSONL files.")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--nlile-dataset", default=DEFAULT_NLILE_DATASET)
    parser.add_argument("--tot-dataset", default=DEFAULT_TOT_DATASET)
    parser.add_argument("--nlile-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--tot-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--nlile-split", default="train")
    parser.add_argument("--tot-split", default="train")
    parser.add_argument(
        "--generate-local",
        action="store_true",
        help=(
            "Generate a local dataset algorithmically and split it into train/eval "
            "without using remote datasets."
        ),
    )
    parser.add_argument(
        "--nlile-jsonl-fallback",
        help="Use an existing nlile-format JSONL file instead of the HF rows API.",
    )
    parser.add_argument(
        "--tot-csv-fallback",
        help="Use an existing official ToT CSV file instead of the HF rows API.",
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
    parser.add_argument(
        "--generated-eval-size",
        type=int,
        default=100,
        help="Number of locally generated solvable records to hold out for eval.",
    )
    parser.add_argument(
        "--generated-seed",
        type=int,
        default=42,
        help="Random seed for deterministic local train/eval splitting.",
    )
    parser.add_argument(
        "--min-unsolvable-eval-size",
        type=int,
        default=50,
        help="Minimum unsolvable eval rows; generated unsolvable puzzles fill any shortage.",
    )
    parser.add_argument("--non-easy-start-rank", type=int, default=101)
    parser.add_argument("--non-easy-size", type=int, default=100)
    args = parser.parse_args()
    _log(f"starting dataset build: output_dir={args.output_dir}")

    output_dir = Path(args.output_dir)

    hard_start_index = args.hard_start_index
    hard_end_index = args.hard_end_index

    if args.generate_local:
        _log("using local generation mode")
        generated_records = generate_full_game24_records()
        _log(
            "splitting generated records with "
            f"eval_size={args.generated_eval_size}, seed={args.generated_seed}"
        )
        splits = split_generated_records(
            generated_records,
            eval_size=args.generated_eval_size,
            seed=args.generated_seed,
        )
        hard_start_index = 0
        hard_end_index = len(splits["eval"])
    elif args.nlile_jsonl_fallback or args.tot_csv_fallback:
        if not args.nlile_jsonl_fallback or not args.tot_csv_fallback:
            raise ValueError("set both --nlile-jsonl-fallback and --tot-csv-fallback")
        _log("using local fallback dataset files")
        nlile_records = load_jsonl_records(
            args.nlile_jsonl_fallback,
            source=DEFAULT_NLILE_DATASET,
        )
        tot_records = load_tot_csv_records(args.tot_csv_fallback)
    else:
        _log("using remote dataset mode")
        nlile_records, tot_records = load_remote_datasets(
            nlile_dataset=args.nlile_dataset,
            tot_dataset=args.tot_dataset,
            nlile_config=args.nlile_config,
            tot_config=args.tot_config,
            nlile_split=args.nlile_split,
            tot_split=args.tot_split,
        )
    if not args.generate_local:
        _log(
            f"source sizes: nlile={len(nlile_records)}, tot={len(tot_records)}, "
            f"hard_range=[{hard_start_index}, {hard_end_index})"
        )
        splits = prepare_splits(
            nlile_records=nlile_records,
            tot_records=tot_records,
            hard_start_index=hard_start_index,
            hard_end_index=hard_end_index,
            min_unsolvable_eval_size=args.min_unsolvable_eval_size,
            non_easy_start_rank=args.non_easy_start_rank,
            non_easy_size=args.non_easy_size,
        )
    _log(
        "writing splits: "
        + ", ".join(f"{name}={len(records)}" for name, records in splits.items())
    )

    write_jsonl(output_dir / "id_train.jsonl", splits["id_train"])
    write_jsonl(output_dir / "id_test.jsonl", splits["id_test"])
    write_jsonl(output_dir / "tot_non_easy.jsonl", splits["tot_non_easy"])
    write_jsonl(output_dir / "tot_hard.jsonl", splits["tot_hard"])
    write_jsonl(output_dir / "unsolvable_eval.jsonl", splits["unsolvable_eval"])
    write_jsonl(output_dir / "train.jsonl", splits["id_train"])
    write_jsonl(output_dir / "eval.jsonl", splits["id_test"])
    if splits["tot_non_easy"]:
        write_jsonl(output_dir / "tot_nonoverlap.jsonl", splits["tot_non_easy"])
    _log("dataset build complete")


if __name__ == "__main__":
    main()
