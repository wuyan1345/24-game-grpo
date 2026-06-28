from game24_grpo.rewards import Game24Reward, RewardConfig


def _reward(completion: str, numbers: list[int]) -> float:
    reward = Game24Reward(RewardConfig())
    return reward([completion], [numbers])[0]


def test_reward_correct_expression_dominates() -> None:
    score = _reward(
        "<think>check</think><answer>(10 * 10 - 4) / 4</answer>",
        [4, 4, 10, 10],
    )
    assert score > 2.0


def test_reward_penalizes_number_mismatch() -> None:
    score = _reward(
        "<think>bad numbers</think><answer>(10 * 10 - 5) / 5</answer>",
        [4, 4, 10, 10],
    )
    assert score < 0.0


def test_reward_gives_small_shaping_for_wrong_but_valid_expression() -> None:
    wrong_valid = _reward(
        "<think>valid but wrong</think><answer>(10 - 4) * (6 / 5)</answer>",
        [4, 5, 6, 10],
    )
    correct = _reward(
        "<think>valid and correct</think><answer>(10 - 6) * (5 + 1)</answer>",
        [1, 5, 6, 10],
    )
    assert 0.0 < wrong_valid < 0.4
    assert correct > wrong_valid + 1.5


def test_reward_does_not_reward_malformed_expression_as_valid() -> None:
    score = _reward(
        "<think>bad syntax</think><answer>(10 + 4 *</answer>",
        [4, 4, 10, 10],
    )
    assert score <= RewardConfig().format_weight


def test_reward_missing_format_can_still_penalize_number_mismatch() -> None:
    score = _reward("<answer>10 + 10 + 5 - 1</answer>", [4, 4, 10, 10])
    assert score == RewardConfig().number_mismatch_penalty
