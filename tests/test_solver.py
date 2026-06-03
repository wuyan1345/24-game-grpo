from game24_grpo.solver import solve_24
from game24_grpo.verifier import verify_completion


def test_solver_finds_valid_expression() -> None:
    expression = solve_24([4, 4, 10, 10])
    assert expression is not None
    completion = f"<think>search</think><answer>{expression}</answer>"
    result = verify_completion(completion, [4, 4, 10, 10])
    assert result.is_correct is True


def test_solver_returns_none_for_unsolvable_case() -> None:
    assert solve_24([1, 1, 1, 1]) is None
