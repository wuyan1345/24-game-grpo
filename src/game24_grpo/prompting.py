from __future__ import annotations


def build_prompt(numbers: list[int], target: int, prompt_template: str) -> str:
    rendered_numbers = ", ".join(str(value) for value in numbers)
    return prompt_template.format(numbers=rendered_numbers, target=target)
