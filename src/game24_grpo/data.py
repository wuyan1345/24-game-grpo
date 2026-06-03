from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from game24_grpo.prompting import build_prompt


def load_jsonl_dataset(path: str | Path, prompt_template: str) -> Dataset:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            numbers = raw["numbers"]
            records.append(
                {
                    "prompt": build_prompt(numbers=numbers, prompt_template=prompt_template),
                    "numbers": numbers,
                    "target": raw.get("target", 24),
                    "solvable": raw.get("solvable", True),
                    "reference_solution": raw.get("reference_solution"),
                    "source": raw.get("source"),
                }
            )
    return Dataset.from_list(records)
