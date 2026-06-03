from game24_grpo.verifier import verify_completion


def test_verifier_accepts_correct_expression() -> None:
    completion = "<think>Try a direct construction.</think><answer>(10 * 10 - 4) / 4</answer>"
    result = verify_completion(completion, [4, 4, 10, 10])
    assert result.is_correct is True


def test_verifier_rejects_wrong_numbers() -> None:
    completion = "<think>Incorrect number reuse.</think><answer>(10 * 10 - 5) / 5</answer>"
    result = verify_completion(completion, [4, 4, 10, 10])
    assert result.used_numbers_match is False
    assert result.is_correct is False
