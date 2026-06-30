import json

from game24_grpo.cli.build_sft_data import (
    build_solvable_completion,
    build_unsolvable_completion,
    main,
)
from game24_grpo.verifier import verify_completion


def test_build_solvable_completion_mentions_expression_numbers_and_target() -> None:
    numbers = [4, 4, 10, 10]
    expression = "((10 * 10) - 4) / 4"

    completion = build_solvable_completion(expression, numbers, 24)

    assert f"Use the expression {expression}." in completion
    assert "It uses 4, 4, 10, 10 exactly once" in completion
    assert "evaluates to 24" in completion
    assert completion.endswith(f"<answer>{expression}</answer>")
    assert verify_completion(completion, numbers).is_correct is True


def test_build_unsolvable_completion_uses_no_solution() -> None:
    completion = build_unsolvable_completion([1, 1, 1, 1], 24)

    assert "No valid expression exists using 1, 1, 1, 1 exactly once" in completion
    assert completion.endswith("<answer>NO_SOLUTION</answer>")


def test_build_sft_data_cli_writes_new_completion_template(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "train.jsonl"
    unsolvable_path = tmp_path / "unsolvable.jsonl"
    output_path = tmp_path / "sft.jsonl"
    input_path.write_text(json.dumps({"numbers": [4, 4, 10, 10], "target": 24}) + "\n")
    unsolvable_path.write_text(json.dumps({"numbers": [1, 1, 1, 1], "target": 24}) + "\n")

    monkeypatch.setattr(
        "sys.argv",
        [
            "game24-build-sft",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--unsolvable-input",
            str(unsolvable_path),
            "--unsolvable-limit",
            "1",
        ],
    )

    main()

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 2
    assert "Use the expression" in rows[0]["completion"]
    assert rows[0]["solution"] in rows[0]["completion"]
    assert rows[0]["completion"].endswith(f"<answer>{rows[0]['solution']}</answer>")
    assert rows[1]["solution"] is None
    assert rows[1]["completion"].endswith("<answer>NO_SOLUTION</answer>")
