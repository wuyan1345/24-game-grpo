from game24_grpo.cli.build_data import (
    canonical_key,
    normalize_nlile_row,
    normalize_tot_row,
    prepare_splits,
)


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


def test_prepare_splits_excludes_eval_holdout_from_train() -> None:
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
        normalize_tot_row({"Rank": 1, "Puzzles": "1 1 2 6", "Solved rate": "99.2%"}),
        normalize_tot_row({"Rank": 2, "Puzzles": "4 4 10 10", "Solved rate": "85.0%"}),
    ]
    splits = prepare_splits(nlile_records, tot_records, hard_start_index=1, hard_end_index=2)
    train_keys = {record.puzzle_key for record in splits["train"]}
    eval_keys = {record.puzzle_key for record in splits["eval"]}
    assert canonical_key([4, 4, 10, 10]) in eval_keys
    assert canonical_key([4, 4, 10, 10]) not in train_keys
    assert canonical_key([1, 1, 2, 6]) in train_keys
