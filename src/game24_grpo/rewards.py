from __future__ import annotations

import re
from dataclasses import dataclass

from game24_grpo.verifier import verify_completion


THINK_ANSWER_PATTERN = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)


@dataclass
class RewardConfig:
    format_weight: float = 0.1
    valid_expression_weight: float = 0.2
    correct_weight: float = 1.0


class Game24Reward:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self.__name__ = self.__class__.__name__

    def __call__(self, completions: list[str], numbers: list[list[int]], **_: object) -> list[float]:
        rewards: list[float] = []
        for completion, puzzle_numbers in zip(completions, numbers, strict=True):
            score = 0.0
            if THINK_ANSWER_PATTERN.match(completion):
                score += self.config.format_weight

            verification = verify_completion(completion, puzzle_numbers)
            if verification.is_valid_expression:
                score += self.config.valid_expression_weight
            if verification.is_correct:
                score += self.config.correct_weight

            rewards.append(score)
        return rewards
