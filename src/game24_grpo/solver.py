from __future__ import annotations

from fractions import Fraction
from itertools import combinations


TARGET = Fraction(24, 1)


def solve_24(numbers: list[int]) -> str | None:
    values = tuple((Fraction(number, 1), str(number)) for number in numbers)
    return _search(values)


def _search(state: tuple[tuple[Fraction, str], ...]) -> str | None:
    if len(state) == 1:
        value, expression = state[0]
        return expression if value == TARGET else None

    for left_index, right_index in combinations(range(len(state)), 2):
        left_value, left_expression = state[left_index]
        right_value, right_expression = state[right_index]
        remaining = [state[index] for index in range(len(state)) if index not in (left_index, right_index)]

        candidates = [
            (left_value + right_value, f"({left_expression} + {right_expression})"),
            (left_value - right_value, f"({left_expression} - {right_expression})"),
            (right_value - left_value, f"({right_expression} - {left_expression})"),
            (left_value * right_value, f"({left_expression} * {right_expression})"),
        ]
        if right_value != 0:
            candidates.append((left_value / right_value, f"({left_expression} / {right_expression})"))
        if left_value != 0:
            candidates.append((right_value / left_value, f"({right_expression} / {left_expression})"))

        for value, expression in candidates:
            next_state = tuple(remaining + [(value, expression)])
            result = _search(next_state)
            if result is not None:
                return result
    return None
