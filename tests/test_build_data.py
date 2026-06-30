from game24_grpo.cli.build_data import (
    canonical_key,
    generate_unsolvable_records,
    normalize_nlile_row,
    normalize_tot_row,
    prepare_splits,
    split_generated_records,
)
from game24_grpo.solver import solve_24


def test_normalize_nlile_row_converts_symbols() -> None:
    row = {
        "numbers": [1, 1, 2, 6],
        "solutions": ["(1+1)\u00f76\u00f72", "(1+1)\u00f76\u00f72"],
        "solvable": True,
        "amt": 5.0,
        "solved_rate": 0.9,
        "mean_time": 5.5,
        "std_time": 1.2,
    }
    record = normalize_nlile_row(row)
    assert record.reference_solution == "(1+1)/6/2"
    assert record.puzzle_key == "1 1 2 6"


def test_normalize_tot_row_parses_percentage() -> None:
    row = {
        "Rank": 901,
        "Puzzles": "1 3 4 6",
        "AMT (s)": 12.3,
        "Solved rate": "80.50%",
        "1-sigma Mean (s)": 11.8,
        "1-sigma STD (s)": 2.1,
    }
    record = normalize_tot_row(row)
    assert record.numbers == [1, 3, 4, 6]
    assert record.metadata["solved_rate"] == 0.805


def test_prepare_splits_builds_id_holdout_and_tot_splits() -> None:
    nlile_records = [
        normalize_nlile_row(
            {
                "numbers": [1, 1, 2, 6],
                "solutions": ["(1+1)*6*2"],
                "solvable": True,
            }
        ),
        normalize_nlile_row(
            {
                "numbers": [3, 3, 8, 8],
                "solutions": [],
                "solvable": False,
            }
        ),
        normalize_nlile_row(
            {
                "numbers": [4, 4, 10, 10],
                "solutions": ["(10*10-4)/4"],
                "solvable": True,
            }
        ),
    ]
    tot_records = [
        normalize_tot_row({"Rank": 101, "Puzzles": "1 1 2 6", "Solved rate": "99.2%"}),
        normalize_tot_row({"Rank": 901, "Puzzles": "4 4 10 10", "Solved rate": "85.0%"}),
        normalize_tot_row({"Rank": 102, "Puzzles": "1 3 4 6", "Solved rate": "75.0%"}),
    ]
    splits = prepare_splits(
        nlile_records,
        tot_records,
        hard_start_index=1,
        hard_end_index=2,
        min_unsolvable_eval_size=1,
        non_easy_size=1,
    )
    train_keys = {record.puzzle_key for record in splits["id_train"]}
    test_keys = {record.puzzle_key for record in splits["id_test"]}
    non_easy_keys = {record.puzzle_key for record in splits["tot_non_easy"]}
    hard_keys = {record.puzzle_key for record in splits["tot_hard"]}
    assert len(train_keys) == 1
    assert len(test_keys) == 1
    assert canonical_key([4, 4, 10, 10]) in hard_keys
    assert non_easy_keys
    assert non_easy_keys.isdisjoint(train_keys)
    assert non_easy_keys.isdisjoint(hard_keys)


def test_prepare_splits_generates_unsolvable_fallback_when_source_has_too_few() -> None:
    nlile_records = [
        normalize_nlile_row(
            {
                "numbers": [4, 4, 10, 10],
                "solutions": ["(10*10-4)/4"],
                "solvable": True,
            }
        )
    ]
    tot_records = [
        normalize_tot_row({"Rank": 1, "Puzzles": "4 4 10 10", "Solved rate": "85.0%"}),
    ]
    splits = prepare_splits(
        nlile_records,
        tot_records,
        hard_start_index=0,
        hard_end_index=1,
        min_unsolvable_eval_size=5,
    )
    assert len(splits["unsolvable_eval"]) == 5
    assert all(record.solvable is False for record in splits["unsolvable_eval"])
    assert all(solve_24(record.numbers) is None for record in splits["unsolvable_eval"])


def test_generate_unsolvable_records_returns_requested_unsolvable_prefix() -> None:
    records = generate_unsolvable_records(3)
    assert len(records) == 3
    assert all(record.source == "generated/unsolvable-24-combinations" for record in records)
    assert all(solve_24(record.numbers) is None for record in records)


def test_split_generated_records_holds_out_eval_subset_deterministically() -> None:
    records = [
        normalize_nlile_row({"numbers": [1, 1, 2, 6], "solutions": ["a"], "solvable": True}),
        normalize_nlile_row({"numbers": [1, 1, 3, 8], "solutions": ["a"], "solvable": True}),
        normalize_nlile_row({"numbers": [1, 1, 4, 8], "solutions": ["a"], "solvable": True}),
        normalize_nlile_row({"numbers": [3, 3, 8, 8], "solutions": [], "solvable": False}),
    ]
    splits = split_generated_records(records, eval_size=2, seed=123)
    assert len(splits["id_train"]) == 1
    assert len(splits["id_test"]) == 2
    assert len(splits["unsolvable_eval"]) == 1
    assert splits["tot_non_easy"] == []
    assert splits["tot_hard"] == []
